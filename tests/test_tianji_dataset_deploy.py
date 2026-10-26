"""Dataset policy inputs with the real synchronous runner/runtime and fake devices."""

import json
from itertools import product
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import hydra
import numpy as np
from omegaconf import OmegaConf
from diffusion_policy.common.tianji_rail_pose import eef_action_to_6d
import torch
import zarr

from diffusion_policy.real_world.tianji import evaluation as entry
from diffusion_policy.cli.tianji.eval import main as eval_main
from diffusion_policy.env_runner.tianji_rail_real_runner import TianjiRailRealRunner
from diffusion_policy.real_world.dataset_observation_source import DatasetObservationSource
from diffusion_policy.real_world.tianji_policy_runtime import TianjiPolicyRuntime
from tests.test_tianji_rail_real_eval import FakeArm, FakeHand, config
from scripts.export_tianji_episodes import export_episodes


RUNNER = "diffusion_policy.env_runner.tianji_rail_real_runner"


def write_dataset(path, space="joint"):
    root = zarr.open_group(str(path), mode="w")
    root.attrs.update(action_space=space, schema_version=2, target_fps=30)
    data, meta = root.create_group("data"), root.create_group("meta")
    meta.array("episode_ends", np.array([48, 96], np.int64))
    meta.array("segment_ends", np.array([24, 48, 96], np.int64))
    meta.attrs["segments"] = [{"source_episode": value} for value in ("a", "a", "b")]
    data.array("timestamp", np.arange(96, dtype=np.float64) / 30)
    dim, arm_key, arm_width = (
        (54, "robot_joint", 14) if space == "joint" else (52, "robot_eef_pose", 12)
    )
    data.array("action", np.zeros((96, dim), np.float32))
    data.array(
        arm_key, np.repeat(np.arange(96, dtype=np.float32)[:, None] / 1000, arm_width, axis=1)
    )
    data.array("hand_joint", np.zeros((96, 40), np.float32))
    pixels = np.arange(96, dtype=np.uint8)[:, None, None, None] + np.zeros(
        (1, 12, 16, 3), np.uint8
    )
    for i in range(3):
        data.array(f"camera_{i}", pixels + i, chunks=(1, 12, 16, 3))


def checkpoint_cfg(space="joint", steps=1):
    dim, arm_key, arm_width = (
        (54, "robot_joint", 14) if space == "joint" else (58, "robot_eef_pose", 14)
    )
    shape = {
        "obs": {f"camera_{i}": {"shape": [3, 12, 16], "type": "rgb"} for i in range(3)},
        "action": {"shape": [dim]},
    }
    shape["obs"].update({arm_key: {"shape": [arm_width]}, "hand_joint": {"shape": [40]}})
    return OmegaConf.create(
        {
            "task": {
                "shape_meta": shape,
                "sample_hz": 30,
                "dataset": {
                    "_target_": "diffusion_policy.dataset.tianji_rail_image_dataset.TianjiRailImageDataset",
                    "dataset_path": "unused.zarr",
                    "shape_meta": shape,
                    "horizon": 16,
                    "n_obs_steps": steps,
                    "seed": 42,
                    "val_ratio": 0.5,
                    "image_shape": [3, 12, 16],
                    "resize_enabled": False,
                    "sample_hz": 30,
                    "action_space": space,
                },
            }
        }
    )


class DatasetDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "joint.zarr"
        write_dataset(self.path)

    def source(self, cfg=None, **kwargs):
        cfg = cfg or checkpoint_cfg()
        return DatasetObservationSource(
            cfg,
            self.path,
            n_obs_steps=int(cfg.task.dataset.n_obs_steps),
            n_action_steps=8,
            **kwargs,
        )

    def test_source_split_frame_order_copying_and_reset_for_both_spaces_and_counts(self):
        for space in ("eef", "joint"):
            path = self.root / f"{space}-counts.zarr"
            write_dataset(path, space)
            for steps in (1, 2):
                cfg = checkpoint_cfg(space, steps)
                base_cfg = OmegaConf.to_container(cfg.task.dataset, resolve=True)
                base_cfg["dataset_path"] = str(path)
                base = hydra.utils.instantiate(base_cfg)
                sources = []
                for split in ("train", "val"):
                    with self.subTest(space=space, steps=steps, split=split):
                        source = DatasetObservationSource(
                            cfg, path, split=split, n_obs_steps=steps, n_action_steps=8
                        )
                        mask = base.train_mask if split == "train" else base.val_mask
                        self.assertTrue(mask[source.metadata["segment"]])
                        sources.append(source.metadata["source_episode"])
                        obs, _, alignment = source.read()
                        first = source.starts[0]
                        self.assertEqual(
                            alignment["dataset_frames"], list(range(first, first + steps))
                        )
                        for key in ("camera_0", "camera_1", "camera_2"):
                            self.assertEqual(obs[key].shape, (steps, 12, 16, 3))
                        np.testing.assert_array_equal(
                            obs["camera_0"][:, 0, 0, 0], np.arange(first, first + steps)
                        )
                        obs["camera_0"].fill(255)
                        self.assertEqual(source.read()[0]["camera_0"][0, 0, 0, 0], first)
                        source.advance()
                        self.assertEqual(source.read()[2]["dataset_frames"][0], first + 8)
                        source.reset()
                        self.assertEqual(source.read()[2]["dataset_frames"][0], first)
                self.assertNotEqual(sources[0], sources[1])

    def test_wrong_split_episode_segment_and_empty_validation_are_rejected(self):
        train = self.source()
        val = self.source(split="val")
        with self.assertRaisesRegex(ValueError, "no windows in split=train"):
            self.source(episode=val.metadata["source_episode"])
        with self.assertRaisesRegex(ValueError, "no windows in split=train"):
            self.source(segment=val.metadata["segment"])
        self.assertEqual(
            self.source(episode="1").metadata["source_episode"],
            train.metadata["source_episode"],
        )
        cfg = checkpoint_cfg()
        cfg.task.dataset.val_ratio = 0
        with self.assertRaisesRegex(ValueError, "No independent val windows"):
            self.source(cfg, split="val")
        with self.assertRaisesRegex(ValueError, "positive integer"):
            self.source(max_windows=0)
        cfg = checkpoint_cfg("eef")
        with self.assertRaisesRegex(
            Exception, "Missing or invalid observation: robot_eef_pose"
        ):
            self.source(cfg)

    def test_shared_observations_select_checkpoint_fields_without_using_stored_actions(self):
        for stored_space in ("joint", "eef"):
            path = self.root / f"shared-{stored_space}.zarr"
            write_dataset(path, stored_space)
            root = zarr.open_group(str(path), mode="a")
            for key, width in (("robot_joint", 14), ("robot_eef_pose", 12)):
                if key not in root["data"]:
                    root["data"].array(key, np.full((96, width), 0.25, np.float32))
            episode = export_episodes(path, self.root / f"shared-export-{stored_space}", ["a"])[
                0
            ]["path"]
            for space, steps, selected_path in product(
                ("joint", "eef"), (1, 2), (path, episode)
            ):
                with self.subTest(
                    stored=stored_space, policy=space, steps=steps, path=selected_path
                ):
                    cfg = checkpoint_cfg(space, steps)
                    before = OmegaConf.to_container(cfg, resolve=True)
                    source = DatasetObservationSource(
                        cfg, selected_path, split="val", n_obs_steps=steps, n_action_steps=8
                    )
                    obs, _, _ = source.read()
                    key = "robot_joint" if space == "joint" else "robot_eef_pose"
                    unused = "robot_eef_pose" if space == "joint" else "robot_joint"
                    np.testing.assert_array_equal(obs[key], source.dataset.data[key][:steps])
                    self.assertNotIn(unused, obs)
                    self.assertNotIn("action", obs)
                    self.assertEqual(source.metadata["action_space"], space)
                    self.assertEqual(source.metadata["dataset_action_space"], stored_space)
                    self.assertFalse(source.metadata["dataset_actions_used"])
                    self.assertEqual(OmegaConf.to_container(cfg, resolve=True), before)
                    if space != stored_space:
                        training_cfg = dict(before["task"]["dataset"], dataset_path=str(path))
                        with self.assertRaisesRegex(
                            Exception, "does not match configured action_space"
                        ):
                            hydra.utils.instantiate(training_cfg)

    def test_shared_observation_source_still_rejects_malformed_action_schemas(self):
        cfg = checkpoint_cfg()
        cfg.task.dataset.shape_meta.action.shape = [52]
        with self.assertRaisesRegex(ValueError, "Checkpoint action shape"):
            self.source(cfg)
        root = zarr.open_group(str(self.path), mode="a")
        root.attrs["action_space"] = "both"
        with self.assertRaisesRegex(ValueError, "action_space must be eef or joint"):
            self.source()
        root.attrs["action_space"] = "joint"
        del root["data/action"]
        root["data"].array("action", np.zeros((96, 52), np.float32))
        with self.assertRaisesRegex(Exception, "requires 54-D policy actions"):
            self.source()

    def test_episode_exports_preserve_all_arrays_boundaries_and_original_split(self):
        outputs = export_episodes(self.path, self.root / "exported", ["a", "b"], "rail")
        original = zarr.open_group(str(self.path), mode="r")
        cfg = checkpoint_cfg()
        for item, split in zip(outputs, ("val", "train")):
            with self.subTest(split=split):
                root = zarr.open_group(item["path"], mode="r")
                low, high = item["source_start_frame"], item["source_end_frame"]
                for key in original["data"].array_keys():
                    np.testing.assert_array_equal(
                        root["data"][key][:], original["data"][key][low:high]
                    )
                np.testing.assert_array_equal(root["meta/episode_ends"][:], [high - low])
                for steps in (1, 2):
                    current = checkpoint_cfg(steps=steps)
                    source = DatasetObservationSource(
                        current,
                        item["path"],
                        mode="episode",
                        split=split,
                        n_obs_steps=steps,
                        n_action_steps=8,
                    )
                    self.assertEqual(source.metadata["dataset_mode"], "episode")
                    self.assertEqual(source.metadata["split"], split)
                    self.assertTrue(source.metadata["split_verified"])
                    self.assertEqual(source.metadata["val_ratio"], 0.5)
                    self.assertEqual(source.dataset.n_source_episodes, 1)
                    self.assertEqual(current.task.dataset.val_ratio, 0.5)
                    self.assertEqual(
                        source.read()[2]["source_dataset_frames"], list(range(low, low + steps))
                    )
                auto = DatasetObservationSource(
                    cfg,
                    item["path"],
                    n_obs_steps=1,
                    n_action_steps=8,
                    episode=str(item["source_episode_index"]),
                )
                self.assertEqual(auto.metadata["split"], split)
        self.assertEqual(
            zarr.open_group(outputs[0]["path"], mode="r")["meta/segment_ends"][:].tolist(),
            [24, 48],
        )

    def test_episode_split_mismatch_full_mode_and_incomplete_exports_are_rejected(self):
        item = export_episodes(self.path, self.root / "exported", ["a"], "rail")[0]
        cfg = checkpoint_cfg()
        with self.assertRaisesRegex(ValueError, "belongs to split=val"):
            DatasetObservationSource(
                cfg, item["path"], split="train", n_obs_steps=1, n_action_steps=8
            )
        with self.assertRaisesRegex(ValueError, "episode export"):
            DatasetObservationSource(
                cfg, item["path"], mode="full", n_obs_steps=1, n_action_steps=8
            )
        with self.assertRaisesRegex(ValueError, "exactly one source episode"):
            DatasetObservationSource(
                cfg, self.path, mode="episode", n_obs_steps=1, n_action_steps=8
            )
        root = zarr.open_group(item["path"], mode="a")
        provenance = dict(root.attrs["episode_export"])
        provenance["complete"] = False
        root.attrs["episode_export"] = provenance
        with self.assertRaisesRegex(ValueError, "incomplete episode export"):
            DatasetObservationSource(cfg, item["path"], n_obs_steps=1, n_action_steps=8)
        provenance["complete"] = True
        provenance["source_end_frame"] += 1
        root.attrs["episode_export"] = provenance
        with self.assertRaisesRegex(ValueError, "does not match"):
            DatasetObservationSource(cfg, item["path"], n_obs_steps=1, n_action_steps=8)

    def test_unmarked_episode_can_replay_but_cannot_claim_a_verified_split(self):
        item = export_episodes(self.path, self.root / "exported", ["a"], "rail")[0]
        root = zarr.open_group(item["path"], mode="a")
        del root.attrs["episode_export"]
        cfg = checkpoint_cfg()
        source = DatasetObservationSource(cfg, item["path"], n_obs_steps=1, n_action_steps=8)
        self.assertEqual(source.metadata["split"], "unknown")
        self.assertFalse(source.metadata["split_verified"])
        self.assertEqual(source.read()[2]["dataset_frames"], [0])
        for split in ("train", "val"):
            with self.assertRaisesRegex(ValueError, "Cannot verify train/val"):
                DatasetObservationSource(
                    cfg, item["path"], split=split, n_obs_steps=1, n_action_steps=8
                )

    def test_episode_uses_current_checkpoint_split_seed_not_a_new_local_holdout(self):
        item = export_episodes(self.path, self.root / "exported", ["a"], "rail")[0]
        cfg = checkpoint_cfg()
        cfg.task.dataset.seed = 43
        source = DatasetObservationSource(cfg, item["path"], n_obs_steps=1, n_action_steps=8)
        self.assertEqual(source.metadata["split"], "train")
        self.assertEqual(source.metadata["split_seed"], 43)
        self.assertEqual(source.metadata["episode_export"]["source_episodes"], ["a", "b"])

    def test_check_only_reads_dataset_and_never_constructs_devices_or_worker(self):
        root = zarr.open_group(str(self.path), mode="a")
        root["data"].array("robot_eef_pose", np.zeros((96, 12), np.float32))
        episode = export_episodes(self.path, self.root / "check-episode", ["a"], "rail")[0][
            "path"
        ]
        for space, steps in product(("joint", "eef"), (1, 2)):
            cfg = checkpoint_cfg(space, steps)
            dim, key, width = (
                (54, "robot_joint", 14) if space == "joint" else (58, "robot_eef_pose", 14)
            )
            seen = []

            def predict(obs):
                seen.append(obs["camera_0"][0, :, 0, 0, 0].cpu().numpy())
                self.assertEqual(obs[key].shape, (1, steps, width))
                np.testing.assert_allclose(
                    obs["camera_1"][0, :, 0, 0, 0].cpu().numpy(), (np.arange(steps) + 1) / 255
                )
                return {"action": torch.zeros(1, 8, dim)}

            policy = SimpleNamespace(
                n_obs_steps=steps,
                n_action_steps=8,
                device="cpu",
                reset=lambda: None,
                predict_action=predict,
            )
            output = self.root / f"check-{space}-{steps}"
            with patch.object(
                entry, "load_policy", return_value=(policy, cfg, "ema_model")
            ), patch.object(entry, "_checkpoint_hash", return_value="offline"), patch.object(
                entry, "version", return_value="offline"
            ), patch.object(
                entry, "TianjiRailEnv"
            ) as env, patch.object(
                entry, "PolicyInferenceProcess"
            ) as worker:
                self.assertEqual(
                    eval_main(
                        [
                            "--checkpoint",
                            "unused.ckpt",
                            "--dataset",
                            episode,
                            "--dataset-mode",
                            "episode",
                            "--dataset-split",
                            "val",
                            "--inference-mode",
                            "sync",
                            "--check-only",
                            "--device",
                            "cpu",
                            "--output",
                            str(output),
                        ]
                    ),
                    0,
                )
                env.assert_not_called()
                worker.assert_not_called()
            metadata = json.loads((output / "preflight.json").read_text())
            self.assertEqual(metadata["observation_source"]["split"], "val")
            self.assertEqual(metadata["observation_source"]["dataset_mode"], "episode")
            self.assertEqual(metadata["observation_source"]["action_space"], space)
            self.assertEqual(metadata["observation_source"]["dataset_action_space"], "joint")
            self.assertTrue(metadata["preflight"]["scope"].startswith("dataset observations"))
            self.assertEqual(len(seen), 4)
            np.testing.assert_allclose(seen[0], np.arange(steps) / 255)

    def test_dataset_options_reject_async_missing_dataset_and_trace_before_model_loading(self):
        for options in (
            ("--dataset-split", "val"),
            ("--dataset-mode", "episode"),
            ("--dataset", str(self.path)),
            ("--dataset", str(self.path), "--inference-mode", "sync", "--trace"),
        ):
            with self.subTest(options=options), patch.object(entry, "load_policy") as model:
                with self.assertRaises((SystemExit, ValueError)):
                    eval_main(["--checkpoint", "unused.ckpt", *options])
                model.assert_not_called()

    def test_motion_runner_consumes_all_steps_before_advancing_and_restarts_from_first_window(
        self,
    ):
        for space in ("eef", "joint"):
            path = self.root / f"{space}-motion.zarr"
            write_dataset(path, space)
            episode = export_episodes(path, self.root / f"{space}-episodes", ["b"], "rail")[0][
                "path"
            ]
            for steps, storage in product((1, 2), ("full", "episode")):
                with self.subTest(space=space, steps=steps, storage=storage):
                    cfg = checkpoint_cfg(space, steps)
                    source = DatasetObservationSource(
                        cfg,
                        path if storage == "full" else episode,
                        n_obs_steps=steps,
                        n_action_steps=8,
                        max_windows=2,
                    )
                    deployment = config() | {
                        "inference_mode": "sync",
                        "action_space": space,
                        "trial_timeout_s": 3.0,
                    }
                    dim = 54 if space == "joint" else 52
                    now, events, inputs, chunk_inputs = [1_000_000_000], [], [], []
                    env = SimpleNamespace(
                        config=deployment, motion=True, error=None, recorder=None
                    )

                    def emit(event, **values):
                        events.append((event, values))
                        if env.recorder is not None:
                            env.recorder.event(event, monotonic_ns=now[0], **values)

                    runtime = TianjiPolicyRuntime(
                        FakeArm(),
                        {side: FakeHand(side) for side in ("left", "right")},
                        deployment,
                        lambda: None,
                        emit,
                        clock_ns=lambda: now[0],
                    )
                    self.addCleanup(runtime.close)
                    env.runtime, env.emit = runtime, emit

                    def engage():
                        runtime.arm.driver.engaged = True
                        runtime.seed(np.zeros(dim), arm_ready=True)

                    def completion():
                        if runtime.active:
                            self.assertEqual(source.index, chunk_inputs[-1])
                            now[0] += 70_000_000  # Late ticks must still consume every action.
                            runtime.tick("arm")
                            runtime.tick("hand")
                        return runtime.sync_observation_window()

                    def execute(actions, anchor):
                        chunk_inputs.append(source.index)
                        return runtime.submit(actions, anchor)

                    def predict(obs):
                        inputs.append(obs["camera_0"][0, :, 0, 0, 0].cpu().numpy())
                        self.assertEqual(obs["hand_joint"].shape, (1, steps, 40))
                        action = (
                            torch.from_numpy(
                                eef_action_to_6d(np.zeros((1, 8, dim), np.float32))
                            )
                            if space == "eef"
                            else torch.zeros(1, 8, dim)
                        )
                        action[0, :, 0] = torch.arange(8) * 0.001
                        return {"action": action}

                    env.engage, env.pause = engage, runtime.pause
                    env.sync_observation_window, env.exec_actions = completion, execute
                    env.get_obs = lambda: self.fail(
                        "Dataset policy path read live observations"
                    )
                    env.latest_measured = lambda: {"scope": "fake device feedback"}
                    policy = SimpleNamespace(
                        n_action_steps=8,
                        device="cpu",
                        reset=lambda: None,
                        predict_action=predict,
                    )
                    output = self.root / f"motion-{space}-{steps}-{storage}"
                    runner = TianjiRailRealRunner(
                        env,
                        policy,
                        cfg,
                        {"observation_source": source.metadata},
                        output,
                        record_observations=True,
                    )
                    runner.observation_source = source
                    terminal = SimpleNamespace(
                        isatty=lambda: True,
                        readline=lambda: "s\n" if runner._count < 2 else "q\n",
                    )

                    def ready(*args):
                        return ([terminal], [], []) if runner._trial is None else ([], [], [])

                    runner._begin()
                    watchdog = threading.Timer(5, runner._stop.set)
                    watchdog.start()
                    try:
                        with patch(f"{RUNNER}.sys.stdin", terminal), patch(
                            f"{RUNNER}.select.select", side_effect=ready
                        ):
                            report = runner.run()
                    finally:
                        watchdog.cancel()
                    self.assertIsNone(runner.error)
                    self.assertEqual(
                        [trial["outcome"] for trial in report["trials"]],
                        ["playback_complete"] * 2,
                    )
                    self.assertEqual(chunk_inputs, [0, 1, 0, 1])
                    self.assertEqual(len(inputs), 4)
                    np.testing.assert_allclose(inputs[0], inputs[2])
                    np.testing.assert_allclose(inputs[1], inputs[3])
                    for kind, side in (("arm", None), ("hand", "left"), ("hand", "right")):
                        consumed = [
                            v["step_id"]
                            for e, v in events
                            if e == "submission"
                            and v["mode"] == "policy"
                            and v["kind"] == kind
                            and v["side"] == side
                        ]
                        self.assertEqual(consumed, list(range(8)) * 4)
                    with np.load(output / "trial_0001/obs_00.npz") as recorded:
                        self.assertIn(
                            "robot_joint" if space == "joint" else "robot_eef_pose",
                            recorded.files,
                        )
                    timing = json.loads((output / "trial_0001/timing.json").read_text())
                    self.assertIn("dataset_read_to_prediction_return_ms", timing)
                    self.assertNotIn("observation_age_at_prediction_return_ms", timing)
                    self.assertEqual(report["success_rate"], 0)


if __name__ == "__main__":
    unittest.main()
