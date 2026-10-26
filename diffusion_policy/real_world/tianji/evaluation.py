"""天机导轨真机评估；默认影子推理，--check-only 不连接任何设备。"""

import json
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf
from threadpoolctl import threadpool_info, threadpool_limits

from diffusion_policy.common.file_util import file_sha256 as _checkpoint_hash
from diffusion_policy.common.paths import PROJECT_ROOT
from diffusion_policy.env_runner.tianji_rail_real_runner import (
    TianjiRailRealRunner,
    write_json,
)
from diffusion_policy.real_world.tianji.config import load_deployment_config
from diffusion_policy.real_world.tianji.env import TianjiRailEnv
from diffusion_policy.real_world.dataset_observation_source import DatasetObservationSource
from diffusion_policy.real_world.tianji.inference import PolicyInferenceProcess
from diffusion_policy.real_world.tianji.trace import (
    SessionTrace,
    runtime_files,
    write_provenance,
)
from diffusion_policy.third_party import bimanual_teleop

from .checkpoint import (
    check_inference,
    checkpoint_action_space,
    load_policy,
    validate_policy_frequency,
)


@threadpool_limits.wrap(limits=1, user_api="blas")
def run(args):
    config = load_deployment_config(args.config, args.overrides)
    device = args.device or config["device"]
    config["inference_mode"] = args.inference_mode or config["inference_mode"]
    dataset_path = getattr(args, "dataset", None)
    if dataset_path is not None and config["inference_mode"] != "sync":
        raise ValueError("数据集观测部署须使用 --inference-mode sync")
    if dataset_path is not None and args.trace:
        raise ValueError("数据集观测部署暂不支持 --trace；可使用 --record-observations")
    output = args.output or Path(config["output_dir"]) / time.strftime("%Y%m%d-%H%M%S")
    if output.exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")
    if (
        args.motion
        and not args.check_only
        and any(box is None for box in config["workspace"].values())
    ):
        raise ValueError("运动前请填写 config.workspace 的左右臂 min/max")
    # Small servo matrices must not compete with GPU dispatch through a large
    # BLAS pool. Limit PyTorch as well, before model initialization/preflight.
    torch.set_num_threads(1)
    policy, cfg, state_key = load_policy(args.checkpoint, device)
    validate_policy_frequency(cfg, config["policy_hz"])
    config["action_space"] = checkpoint_action_space(cfg)
    if config["action_space"] == "joint" and args.trace:
        raise ValueError("--trace 的决策记录只支持末端版 checkpoint")
    if config["inference_mode"] == "sync" and (
        policy.n_obs_steps not in (1, 2) or policy.n_action_steps < 2
    ):
        raise ValueError(
            "Synchronous inference requires checkpoint n_obs_steps in (1, 2) and n_action_steps >= 2"
        )
    if (
        config["inference_mode"] == "async"
        and config["replanning_steps"] >= policy.n_action_steps
    ):
        raise ValueError("replanning_steps must be smaller than checkpoint n_action_steps")
    if (
        config["history_s"]
        <= (policy.n_obs_steps - 1) / config["policy_hz"] + config["max_state_gap_s"]
    ):
        raise ValueError("history_s cannot cover checkpoint observation window")
    observation_source = None
    if dataset_path is not None:
        observation_source = DatasetObservationSource(
            cfg,
            dataset_path,
            split=args.dataset_split,
            mode=args.dataset_mode or "auto",
            episode=args.dataset_episode,
            segment=args.dataset_segment,
            max_windows=args.dataset_max_windows,
            n_obs_steps=int(policy.n_obs_steps),
            n_action_steps=int(policy.n_action_steps),
        )
    preflight = (
        check_inference(policy, cfg)
        if observation_source is None
        else check_inference(policy, cfg, observation=observation_source.read()[0])
    )
    metadata = _session_metadata(args, cfg, config, device, state_key, preflight)
    metadata["observation_source"] = (
        {"kind": "real"} if observation_source is None else observation_source.metadata
    )
    print(f"checkpoint sha256 {metadata['checkpoint_sha256']}")
    if args.trace and args.trace_capacity < 1:
        raise ValueError("--trace-capacity 须为正整数")
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
    inference = PolicyInferenceProcess(args.checkpoint, device, capture_rng=args.trace)
    if observation_source is not None:
        metadata["dataset_preflight"] = preflight
    return _run_session(
        args, cfg, config, metadata, inference, output, n_obs_steps, observation_source
    )


def _run_session(
    args, cfg, config, metadata, inference, output, n_obs_steps, observation_source=None
):
    metadata["preflight"] = inference.preflight
    metadata["inference_backend"] = "spawn_process"
    env = None
    trace = None
    try:
        env = TianjiRailEnv(config, n_obs_steps, motion=args.motion)
        if args.trace:
            trace = SessionTrace(decision_capacity=args.trace_capacity)
            env.enable_trace(trace)
        env.record_observations = args.record_observations and trace is None
        runner = TianjiRailRealRunner(
            env,
            inference,
            cfg,
            metadata,
            output,
            record_observations=args.record_observations,
            trace=trace,
        )
        runner.observation_source = observation_source
        if trace is not None:
            files = runtime_files()
            files["eval_tianji_rail"] = Path(__file__).resolve()
            write_provenance(output, files, PROJECT_ROOT)
        env.start()
        metadata["camera_metadata"] = env.cameras.metadata
        metadata["hand_metadata"] = {side: hand.metadata for side, hand in env.hands.items()}
        metadata["sdk_session"] = env.session.metadata
        metadata["kinematics"] = {
            "model_sha256": env.kinematics.model.digest,
            "library_path": str(env.kinematics.library_path),
        }
        write_json(output / "session.json", metadata)
        print(f"推理模式：{config['inference_mode']}")
        report = runner.run()
        print(
            f"运动试验 {report['motion_trials']} 次，成功率 {report['success_rate']}；记录：{output}"
        )
    except Exception as error:
        output.mkdir(parents=True, exist_ok=True)
        write_json(
            output / "failure.json", {"error": str(error), "monotonic_ns": time.monotonic_ns()}
        )
        raise
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            inference.close()
    return 0


def _session_metadata(args, cfg, config, device, state_key, preflight):
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _checkpoint_hash(args.checkpoint),
        "state_dict": state_key,
        "device": device,
        "seed": torch.initial_seed(),
        "deployment": config,
        "checkpoint_config": OmegaConf.to_container(cfg, resolve=True),
        "preflight": preflight,
        "record_observations": bool(args.record_observations),
        "compute_threads": {
            "torch": torch.get_num_threads(),
            "blas": [
                {key: pool[key] for key in ("internal_api", "prefix", "num_threads")}
                for pool in threadpool_info()
                if pool["user_api"] == "blas"
            ],
        },
        "versions": {
            name: version(name)
            for name in (
                "torch",
                "torchvision",
                "numpy",
                "scipy",
                "zarr",
                "numcodecs",
                "quadprog",
                "hydra-core",
                "diffusers",
                "huggingface-hub",
                "robomimic",
                "dill",
                "pyrealsense2",
                "wuji-sdk",
            )
        },
    }
    metadata["config_overrides"] = list(args.overrides)
    metadata["prediction_seed"] = args.seed
    if args.seed is not None:
        metadata["seed"] = int(args.seed)
    metadata["vendored_dependencies"] = json.loads(
        Path(bimanual_teleop.__file__).with_name("manifest.json").read_text()
    )
    metadata["device_configs"] = {
        key: yaml.safe_load(Path(config[key]).read_text())
        for key in ("arm_config", "hand_config", "camera_config")
    }
    metadata["source_revisions"] = {"diffusion_policy": _source_revision()}
    return metadata


def _source_revision():
    revision = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    changes = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
        capture_output=True,
        text=True,
    )
    return {
        "commit": revision.stdout.strip(),
        "dirty": bool(changes.stdout.strip()),
    }
