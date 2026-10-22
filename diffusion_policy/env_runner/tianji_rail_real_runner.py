"""Trial event logging and inference. The operator loop remains responsive to stop."""

import json
import select
import sys
import threading
import time
from pathlib import Path

import numpy as np

from diffusion_policy.common.image_preprocessing import preprocess_observation
from diffusion_policy.common.tianji_rail_pose import policy_actions_to_controller
from diffusion_policy.real_world.tianji.observation import ObservationNotReady
from diffusion_policy.real_world.tianji.recording import (
    TrialRecorder,
    summarize_trials,
    write_json,
)
from diffusion_policy.real_world.tianji.recording import timing_report as timing_report


class TianjiRailRealRunner:
    def __init__(
        self,
        env,
        policy,
        checkpoint_config,
        metadata,
        output_dir,
        *,
        record_observations=False,
        trace=None,
    ):
        self.env, self.policy, self.cfg = env, policy, checkpoint_config
        self.record_observations = bool(record_observations) and trace is None
        self.trace = trace
        self.metadata = metadata
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        if any(self.output.glob("trial_*")) or (self.output / "session.json").exists():
            raise FileExistsError(f"Evaluation output already contains a trial: {self.output}")
        write_json(self.output / "session.json", metadata)
        self._stop = threading.Event()
        self._enabled = threading.Event()
        self._inference_lock = threading.Lock()
        self._generation = 0
        self._snapshot_index = 0
        self._trial = None
        self._started = None
        self._count = 0
        self.error = None

    def _begin(self):
        if self._trial is not None:
            return
        self._count += 1
        self._snapshot_index = 0
        self._trial = TrialRecorder(
            self.output / f"trial_{self._count:04d}",
            {
                **self.metadata,
                "mode": "motion" if self.env.motion else "shadow",
                "record_observations": self.record_observations,
                "trace": self.trace is not None,
                "started_monotonic_ns": time.monotonic_ns(),
            },
            self.env.config["record_queue_capacity"],
        )
        self.env.recorder = self._trial
        self._generation += 1
        self.error = None
        try:
            if self.trace is not None and not self.trace.decisions_ready:
                obs, _, _ = self.env.get_obs()
                self.trace.allocate_decisions(obs, self._n_action_steps(), capture_real=False)
                print(
                    f"完整记录缓冲：{self.trace.decision_capacity} 次决策，图像约 {self.trace.image_mebibytes:.0f} MiB。"
                    "缓冲满会把本段标为无效，不会停住控制。"
                )
            if self.trace is not None:
                self.trace.begin_trial()
            self._started = time.monotonic()
            if self.env.motion:
                self.env.engage()
            # Inference/reset are serialized inside the worker, never in the control loops.
            with self._inference_lock:
                if self.metadata.get("prediction_seed") is not None:
                    self.policy.reseed(int(self.metadata["prediction_seed"]))
                self._enabled.set()
        except Exception as error:
            self._end("device_abort", str(error))
            print(f"试验未能开始：{error}")

    def _end(self, outcome, reason=None):
        with self._inference_lock:
            self._enabled.clear()
            self._generation += 1
            try:
                self.env.pause(reason or outcome)
            finally:
                self.env.recorder = None
                trial, self._trial = self._trial, None
        if trial is not None:
            trial.close(outcome, reason)
            if self.trace is not None:
                self._flush_trace(trial.path)
            summarize_trials(self.output)

    def _infer(self):
        import torch

        generation = -1
        anchor_previous = None
        sync = self.env.config["inference_mode"] == "sync"
        try:
            while not self._stop.is_set():
                if not self._enabled.wait(timeout=0.05):
                    continue
                current = self._generation
                if generation != current:
                    self.policy.reset()
                    generation, anchor_previous = current, None
                completed = None
                if sync:
                    completed = self.env.chunk_completed_ns()
                    if completed is None:
                        self._stop.wait(0.002)
                        continue
                    # Full alignment competes with the arm loop for the GIL; skip it until a fresh anchor exists.
                    latest = self.env.latest_anchor_ns()
                    if latest is None or latest <= completed:
                        self._stop.wait(0.003)
                        continue
                started = time.monotonic_ns()
                try:
                    obs, anchor, alignment = self.env.get_obs()
                except ObservationNotReady as error:
                    # In sync mode the runtime is holding here, so not-ready observations are retried.
                    if self.env.motion and self.env.runtime.active:
                        raise RuntimeError(str(error)) from error
                    self._stop.wait(0.01)
                    continue
                observed = time.monotonic_ns()
                if sync:
                    # The observation must show the robot after the previous chunk, not during it.
                    if anchor <= completed:
                        self._stop.wait(0.003)
                        continue
                elif (
                    anchor_previous is not None
                    and anchor - anchor_previous
                    < self.env.config["replanning_steps"] * 1e9 / self.env.config["policy_hz"]
                ):
                    self._stop.wait(0.003)
                    continue
                process_predict = getattr(self.policy, "predict_observation", None)
                process_timing = None
                if process_predict is not None:
                    prediction = process_predict(obs)
                    process_timing = self.policy.last_timing
                    preprocessed = process_timing["preprocessed_ns"]
                    inputs_ready = process_timing["inputs_ready_ns"]
                else:
                    prepared = preprocess_observation(
                        obs, self.cfg.task.shape_meta, self.cfg.task.dataset
                    )
                    preprocessed = time.monotonic_ns()
                    tensor_obs = {
                        key: torch.from_numpy(value).unsqueeze(0).to(self.policy.device)
                        for key, value in prepared.items()
                    }
                    inputs_ready = time.monotonic_ns()
                    with torch.inference_mode():
                        prediction = self.policy.predict_action(tensor_obs)
                actions = prediction["action"][0].detach().cpu().numpy()
                actions = policy_actions_to_controller(
                    actions, self.env.config.get("action_space", "eef")
                )
                returned = time.monotonic_ns()
                # A stopped/previous trial cannot publish into a new trial.
                with self._inference_lock:
                    if generation != self._generation or not self._enabled.is_set():
                        continue
                    prediction_id = None
                    if self.trace is not None:
                        prediction_id = self.trace.store_decision(
                            obs=obs,
                            alignment=alignment,
                            anchor_ns=anchor,
                            actions=actions,
                            started_ns=started,
                            observed_ns=observed,
                            preprocessed_ns=preprocessed,
                            inputs_ready_ns=inputs_ready,
                            returned_ns=returned,
                            rng=getattr(self.policy, "last_rng", None),
                            control_input="real",
                            live_capture_started_ns=started,
                            live_capture_returned_ns=observed,
                        )
                    observation = {} if self.trace is not None else self._observation_record(obs)
                    if prediction_id is not None:
                        observation = {**observation, "prediction_id": prediction_id}
                    self.env.emit(
                        "prediction",
                        anchor_ns=anchor,
                        started_ns=started,
                        returned_ns=returned,
                        observed_ns=observed,
                        preprocessed_ns=preprocessed,
                        inputs_ready_ns=inputs_ready,
                        process_timing=process_timing,
                        actions=actions,
                        alignment=alignment,
                        chunk_completed_ns=completed,
                        **observation,
                    )
                    if prediction_id is None:
                        dropped = self.env.exec_actions(actions, anchor)
                    else:
                        dropped = self.env.exec_actions(
                            actions, anchor, prediction_id=prediction_id
                        )
                    if self.trace is None:
                        self._save_observation_window(obs, anchor)
                    self.env.emit(
                        "prediction_coverage",
                        dropped_steps=dropped,
                        remaining_steps=len(actions) - dropped,
                    )
                anchor_previous = anchor
        except Exception as error:
            with self._inference_lock:
                if generation == self._generation:
                    self.error = str(error)
                    self._enabled.clear()
                    self.env.runtime.pause(error, fault=True)

    def _n_action_steps(self):
        preflight = getattr(self.policy, "preflight", None)
        if isinstance(preflight, dict) and "n_action_steps" in preflight:
            return int(preflight["n_action_steps"])
        return int(self.policy.n_action_steps)

    def _flush_trace(self, path):
        try:
            meta = self.trace.flush(path)
        except Exception as error:
            (Path(path) / "trace_flush_error.txt").write_text(str(error))
            return
        trial_path = Path(path) / "trial.json"
        if trial_path.is_file() and meta is not None:
            trial = json.loads(trial_path.read_text())
            trial["trace_valid"] = meta["valid"]
            trial["trace_invalid_reason"] = meta["invalid_reason"]
            trial["trace_dropped"] = meta["dropped"]
            write_json(trial_path, trial)
            print(
                "完整记录有效。"
                if meta["valid"]
                else f"完整记录无效：{meta['invalid_reason']}。不能据此判定根因。"
            )

    def _observation_record(self, obs):
        if not self.record_observations:
            return {}
        index = self._snapshot_index
        name = f"obs_{index:02d}" if index < 3 else f"obs_recent_{index % 3:02d}"
        record = {
            "observation_file": f"{name}.npz",
            "prediction_index": index,
            "input_robot_eef_pose": np.asarray(obs["robot_eef_pose"], dtype=np.float32),
            "input_hand_joint": np.asarray(obs["hand_joint"], dtype=np.float32),
            "measured": self.env.latest_measured(),
        }
        if "robot_joint" in obs:
            record["input_robot_joint"] = np.asarray(obs["robot_joint"], dtype=np.float32)
        return record

    def _save_observation_window(self, obs, anchor):
        if not self.record_observations or self._trial is None:
            return
        index = self._snapshot_index
        self._snapshot_index += 1
        name = f"obs_{index:02d}" if index < 3 else f"obs_recent_{index % 3:02d}"
        extra = {"robot_joint": obs["robot_joint"]} if "robot_joint" in obs else {}
        self._trial.snapshot(
            name,
            anchor_ns=anchor,
            prediction_index=index,
            camera_0=obs["camera_0"],
            camera_1=obs["camera_1"],
            camera_2=obs["camera_2"],
            robot_eef_pose=obs["robot_eef_pose"],
            hand_joint=obs["hand_joint"],
            **extra,
        )

    def run(self):
        thread = threading.Thread(target=self._infer, name="policy-inference", daemon=True)
        thread.start()
        if not self.env.motion:
            self._begin()
        print("s 开始试验；p 暂停并记接管；y 成功；n 失败；q 退出（终端输入后按 Enter）。")
        print(
            "默认影子模式，不使能设备。"
            if not self.env.motion
            else "运动模式：每次仍需手动 s 开始。"
        )
        if self.trace is not None and self.env.motion:
            print(
                "出现明显偏离就可以停，不必凑满试验时限。有人递送导轨或扶机器人时，用 p 记下接管时间。"
            )
        try:
            while not self._stop.wait(0.02):
                if self._trial is not None:
                    problem = self.error or self.env.error or self.env.runtime.fault
                    if not thread.is_alive():
                        problem = problem or "Inference worker exited"
                    if problem:
                        self._end("device_abort", problem)
                        print(f"试验中止：{problem}")
                        if not self.env.motion:
                            break
                    elif (
                        self._started is not None
                        and time.monotonic() - self._started >= self.env.config["trial_timeout_s"]
                    ):
                        self._end("timeout" if self.env.motion else "shadow", "Trial time limit")
                        if not self.env.motion:
                            break
                if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
                    command = sys.stdin.readline().strip().lower()
                    if command == "s":
                        if not thread.is_alive():
                            print("推理线程已退出，请退出并重新启动入口。")
                        else:
                            self._begin()
                    elif command in ("p", "y", "n"):
                        self._end(
                            {"p": "takeover", "y": "success", "n": "failure"}[command]
                            if self.env.motion
                            else "shadow",
                            f"Operator {command}",
                        )
                    elif command == "q":
                        break
        finally:
            self._stop.set()
            self._end("takeover" if self.env.motion else "shadow", "Runner exit")
            thread.join(timeout=2)
        return summarize_trials(self.output)
