"""Visualize main-camera CNOS detection and its timer; never connect robot actuators."""
from pathlib import Path
import tempfile
import time

import numpy as np

from rail_preinsert.core import digest, read_json
from rail_receive.logic import DetectionGate, load_config
from rail_receive.perception import ReceivePerception
from rail_receive.visualization import PreviewWindow


def main(duration_s=None):
    config = load_config(Path(__file__).resolve().with_name("receive.yaml"))
    window = PreviewWindow()
    if not window.enabled:
        raise RuntimeError("需要在有图形桌面的本机终端运行检测预览。")
    serial = read_json(config["calibration"])["camera_serial"]
    mesh_digest = digest(config["mesh"])
    gate = DetectionGate(config, serial, mesh_digest)
    print("只观察检测，不连接机械臂或灵巧手；计时达到阈值也不会闭合。Q / 空格 / Esc / X 退出。", flush=True)
    with tempfile.TemporaryDirectory(prefix="rail-detection-view-") as directory:
        directory = Path(directory) / "perception"
        perception = None
        vision_error = None
        deadline = time.monotonic()+duration_s if duration_s is not None else float("inf")
        try:
            perception = ReceivePerception(config, directory, serial)
            while time.monotonic() < deadline:
                record = None
                if vision_error is None:
                    try:
                        perception.publish_context({"timestamp_ns": time.monotonic_ns()}, np.eye(4), enabled=True, epoch=0)
                        record = perception.latest()
                    except (RuntimeError, OSError) as error:
                        vision_error = str(error)
                        print(f"检测预览异常：{vision_error}；窗口保留，按 Q 退出。", flush=True)
                decision = gate.update(record, time.monotonic_ns())
                if decision["reason"] == "already_triggered":
                    decision["stable_s"] = config["detection_duration_s"]
                elif gate.fired and not decision["ready"]:
                    # Unlike a robot grasp, a viewer may show repeated detection cycles.
                    gate = DetectionGate(config, serial, mesh_digest)
                ready = gate.fired
                key = window.update(directory / "preview.png", receiving={
                    "config": config, "stage": "monitor_ready" if ready else "waiting_rail",
                    "decision": decision, "record": record, "monitor_only": True, "vision_error": vision_error})
                if key in ("q", " ", "\x1b", "x"):
                    break
                time.sleep(.02)
        finally:
            try:
                if perception is not None:
                    perception.close()
            finally:
                window.close()


if __name__ == "__main__":
    main()
