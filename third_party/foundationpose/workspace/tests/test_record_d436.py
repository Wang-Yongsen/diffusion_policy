from datetime import datetime
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from rail_pose.data import Scene

spec = importlib.util.spec_from_file_location(
    "record_d436", Path(__file__).resolve().parents[1] / "scripts" / "record_d436.py")
record = importlib.util.module_from_spec(spec)
spec.loader.exec_module(record)


def fake_realsense(model="Intel RealSense D436", coeffs=(0.0,) * 5, distortion="distortion.inverse_brown_conrady"):
    enum = lambda *names: SimpleNamespace(**{n: n for n in names})  # noqa: E731
    camera_info = enum("serial_number", "name", "firmware_version")
    info = {"serial_number": "405622076394", "name": model, "firmware_version": "5.17"}
    device = SimpleNamespace(get_info=info.get, query_sensors=lambda: [],
                             first_depth_sensor=lambda: SimpleNamespace(get_depth_scale=lambda: .001))
    intrinsics = SimpleNamespace(fx=388., fy=387., ppx=332., ppy=260., model=distortion, coeffs=list(coeffs))
    stream = SimpleNamespace(get_intrinsics=lambda: intrinsics, get_extrinsics_to=lambda _: SimpleNamespace(
        rotation=[1, 0, 0, 0, 1, 0, 0, 0, 1], translation=[.015, 0, 0]))
    profile = SimpleNamespace(get_device=lambda: device,
                              get_stream=lambda _: SimpleNamespace(as_video_stream_profile=lambda: stream))
    counter = iter(range(1, 10_000))

    def frameset(_timeout=None):
        number = next(counter)
        number += number > 20  # one dropped color frame after warmup
        image = lambda data: SimpleNamespace(  # noqa: E731
            get_data=lambda: data, get_frame_number=lambda: number, get_timestamp=lambda: number * 33.3,
            get_frame_timestamp_domain=lambda: "system_time")
        rgb = np.full((480, 640, 3), number % 255, np.uint8)
        depth = np.full((480, 640), 800, np.uint16)
        return SimpleNamespace(get_color_frame=lambda: image(rgb), get_depth_frame=lambda: image(depth))

    pipeline = SimpleNamespace(start=lambda _: profile, wait_for_frames=frameset, stop=lambda: None)
    return SimpleNamespace(
        camera_info=camera_info, stream=enum("color", "depth"), format=enum("rgb8", "z16"),
        option=enum("global_time_enabled"), timestamp_domain=enum("global_time"),
        context=lambda: SimpleNamespace(query_devices=lambda: [device]),
        config=lambda: SimpleNamespace(enable_device=lambda _: None, enable_stream=lambda *a: None),
        pipeline=lambda _: pipeline, align=lambda _: SimpleNamespace(process=lambda f: f))


def test_episode_name_uses_local_recording_time(tmp_path):
    assert record.episode_directory(tmp_path, datetime(2026, 10, 19, 17, 53, 4)).name == \
        "rail_task_episode_20261019_175304"


def test_unsupported_rgb_distortion_is_rejected():
    K = np.array([[388., 0, 332], [0, 387, 260], [0, 0, 1]])
    assert record.rectification(K, "distortion.inverse_brown_conrady", [0] * 5, (640, 480)) is None
    assert record.rectification(K, "distortion.inverse_brown_conrady", [-.05, .06, 0, 0, -.02], (640, 480))
    with pytest.raises(RuntimeError, match="rectification"):
        record.rectification(K, "distortion.kannala_brandt4", [.1, 0, 0, 0, 0], (640, 480))


def test_recording_round_trips_as_workspace_scene(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake_realsense(coeffs=(-.05, .06, .001, -.0006, -.02)))
    assert record.main(["--root", str(tmp_path), "--frames", "12", "--workers", "2"]) == 0
    (directory,) = tmp_path.iterdir()
    assert directory.name.startswith("rail_task_episode_")
    scene = Scene(directory)
    assert len(scene) == 12
    frame = scene.frame(0)
    assert frame.rgb.shape == (480, 640, 3) and frame.depth_m.dtype == np.float32
    assert np.isclose(frame.depth_m[240, 320], .8)
    summary = scene.manifest["recording"]
    assert summary["frames"] == 12 and summary["dropped_color_frames"] == 1
    assert scene.manifest["camera"]["rgb_rectified"] and scene.manifest["camera"]["serial"] == "405622076394"


def test_missing_camera_reports_before_creating_episode(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pyrealsense2", SimpleNamespace(
        camera_info=SimpleNamespace(serial_number="s", name="n"),
        context=lambda: SimpleNamespace(query_devices=lambda: [])))
    with pytest.raises(RuntimeError, match="No RealSense"):
        record.main(["--root", str(tmp_path), "--frames", "1"])
    assert not any(tmp_path.iterdir())
