"""天机导轨真机评估；默认影子推理，--check-only 不连接任何设备。"""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import time
import subprocess
import yaml

import dill
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from threadpoolctl import threadpool_info, threadpool_limits

from diffusion_policy.third_party import bimanual_teleop

from diffusion_policy.env_runner.tianji_rail_real_runner import TianjiRailRealRunner, write_json
from diffusion_policy.real_world.policy_inference_process import PolicyInferenceProcess
from diffusion_policy.real_world.tianji_policy_runtime import INFERENCE_MODES
from diffusion_policy.real_world.tianji_rail_env import (
    PROJECT_ROOT, TianjiRailEnv, load_deployment_config, preprocess_observation,
)


def load_policy(checkpoint, device):
    # Local checkpoints contain the trusted training config and normalizer state.
    payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    torch.manual_seed(int(cfg.training.seed))
    np.random.seed(int(cfg.training.seed))
    expected = {"camera_0": (3,), "camera_1": (3,), "camera_2": (3,),
                "robot_eef_pose": (12,), "hand_joint": (40,)}
    if set(cfg.task.shape_meta.obs) != set(expected) or list(cfg.task.shape_meta.action.shape) != [52]:
        raise ValueError("Checkpoint must use three RGB + 12-D flange + 40-D hand / 52-D EEF actions")
    for key, prefix in expected.items():
        if tuple(cfg.task.shape_meta.obs[key].shape)[:len(prefix)] != prefix:
            raise ValueError(f"Checkpoint shape mismatch: {key}")
    if int(cfg.get("n_latency_steps", 0)) != 0:
        raise ValueError("Only checkpoints with n_latency_steps=0 are supported")
    policy = hydra.utils.instantiate(cfg.policy)
    key = "ema_model" if cfg.training.use_ema else "model"
    policy.load_state_dict(payload["state_dicts"][key])
    policy.to(device).eval()
    return policy, cfg, key


def check_inference(policy, cfg, iterations=3):
    obs = {f"camera_{i}": np.zeros((policy.n_obs_steps, 480, 640, 3), np.uint8) for i in range(3)}
    obs.update(robot_eef_pose=np.zeros((policy.n_obs_steps, 12), np.float32),
               hand_joint=np.zeros((policy.n_obs_steps, 40), np.float32))
    prepared = preprocess_observation(obs, cfg.task.shape_meta, cfg.task.dataset)
    tensor_obs = {key: torch.from_numpy(value).unsqueeze(0).to(policy.device) for key, value in prepared.items()}
    durations = []
    with torch.inference_mode():
        for i in range(iterations + 1):
            start = time.monotonic_ns()
            result = policy.predict_action(tensor_obs)["action"].detach().cpu().numpy()
            if result.shape != (1, policy.n_action_steps, 52) or not np.isfinite(result).all():
                raise ValueError("Checkpoint prediction shape/finite check failed")
            if i:
                durations.append((time.monotonic_ns() - start) / 1e6)
    policy.reset()
    return {"batch_size": 1, "n_obs_steps": policy.n_obs_steps, "n_action_steps": policy.n_action_steps,
            "image_shapes": {key: list(value.shape) for key, value in prepared.items() if key.startswith("camera")},
            "inference_ms": durations, "p95_ms": float(np.percentile(durations, 95)),
            "scope": "synthetic images, checkpoint preprocessing/normalizer/EMA, no device or end-to-end latency claim"}


@threadpool_limits.wrap(limits=1, user_api="blas")
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "diffusion_policy/config/tianji_rail_real_eval.yaml")
    parser.add_argument("--device", help="覆盖配置中的模型设备，例如 cpu 或 cuda:0")
    parser.add_argument("--inference-mode", choices=INFERENCE_MODES,
                        help="覆盖配置中的推理模式：async 推理与执行并行；sync 推理期间保持，动作块执行完再观测")
    parser.add_argument("--output", type=Path, help="新建会话目录，不覆盖已有目录")
    parser.add_argument("--check-only", action="store_true", help="只重载模型与模拟输入推理，不连接设备")
    parser.add_argument("--motion", action="store_true", help="允许手动接合；必须已填写工作空间")
    args = parser.parse_args(argv)
    config = load_deployment_config(args.config)
    device = args.device or config["device"]
    config["inference_mode"] = args.inference_mode or config["inference_mode"]
    output = args.output or Path(config["output_dir"]) / time.strftime("%Y%m%d-%H%M%S")
    if output.exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")
    if args.motion and not args.check_only and any(box is None for box in config["workspace"].values()):
        parser.error("运动前请填写 config.workspace 的左右臂 min/max")
    # Small servo matrices must not compete with GPU dispatch through a large
    # BLAS pool. Limit PyTorch as well, before model initialization/preflight.
    torch.set_num_threads(1)
    policy, cfg, state_key = load_policy(args.checkpoint, device)
    if config["inference_mode"] == "async" and config["replanning_steps"] >= policy.n_action_steps:
        raise ValueError("replanning_steps must be smaller than checkpoint n_action_steps")
    if config["history_s"] <= (policy.n_obs_steps - 1) / config["policy_hz"] + config["max_state_gap_s"]:
        raise ValueError("history_s cannot cover checkpoint observation window")
    preflight = check_inference(policy, cfg)
    metadata = {"checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _checkpoint_hash(args.checkpoint),
        "state_dict": state_key, "device": device, "seed": torch.initial_seed(), "deployment": config,
        "checkpoint_config": OmegaConf.to_container(cfg, resolve=True), "preflight": preflight,
        "compute_threads": {"torch": torch.get_num_threads(), "blas": [
            {key: pool[key] for key in ("internal_api", "prefix", "num_threads")}
            for pool in threadpool_info() if pool["user_api"] == "blas"]},
        "versions": {name: version(name) for name in ("torch", "torchvision", "numpy", "scipy",
             "zarr", "numcodecs", "quadprog", "hydra-core", "diffusers", "huggingface-hub",
             "robomimic", "dill", "pyrealsense2", "wuji-sdk")}}
    metadata["vendored_dependencies"] = json.loads(Path(bimanual_teleop.__file__).with_name("manifest.json").read_text())
    metadata["device_configs"] = {key: yaml.safe_load(Path(config[key]).read_text())
                                  for key in ("arm_config", "hand_config", "camera_config")}
    metadata["source_revisions"] = {}
    for name, root in (("diffusion_policy", PROJECT_ROOT),):
        revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
        changes = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True)
        metadata["source_revisions"][name] = {"commit": revision.stdout.strip(), "dirty": bool(changes.stdout.strip())}
    if args.check_only:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "preflight.json", metadata)
        print(f"模型重载和 batch=1 推理通过：{output / 'preflight.json'}")
        return 0
    n_obs_steps = policy.n_obs_steps
    # Release the parent's model before the child creates its CUDA context.
    policy.to("cpu")
    del policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    inference = PolicyInferenceProcess(args.checkpoint, device)
    metadata["preflight"] = inference.preflight
    metadata["inference_backend"] = "spawn_process"
    env = None
    try:
        env = TianjiRailEnv(config, n_obs_steps, motion=args.motion)
        runner = TianjiRailRealRunner(env, inference, cfg, metadata, output)
        env.start()
        metadata["camera_metadata"] = env.cameras.metadata
        metadata["hand_metadata"] = {side: hand.metadata for side, hand in env.hands.items()}
        metadata["sdk_session"] = env.session.metadata
        metadata["kinematics"] = {"model_sha256": env.kinematics.model.digest,
                                  "library_path": str(env.kinematics.library_path)}
        write_json(output / "session.json", metadata)
        print(f"推理模式：{config['inference_mode']}")
        report = runner.run()
        print(f"运动试验 {report['motion_trials']} 次，成功率 {report['success_rate']}；记录：{output}")
    except Exception as error:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "failure.json", {"error": str(error), "monotonic_ns": time.monotonic_ns()})
        raise
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            inference.close()
    return 0


def _checkpoint_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
