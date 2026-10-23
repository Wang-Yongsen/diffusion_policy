from dataclasses import replace
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import trimesh

from rail_pose.camera import RealSenseCamera
from rail_pose.data import Frame
from rail_pose import ui


def test_no_camera_is_reported_before_pipeline_start(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyrealsense2", SimpleNamespace(
        context=lambda: SimpleNamespace(query_devices=lambda: [])))
    with pytest.raises(RuntimeError, match="No RealSense"):
        RealSenseCamera({"serial": "expected"}).start()


def test_gui_required_before_window_creation(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    with pytest.raises(RuntimeError, match="desktop terminal"):
        ui.require_display()


def test_closed_camera_does_not_return_a_cached_frame():
    camera = RealSenseCamera({})
    camera.latest = SimpleNamespace(frame_id="cached")
    camera.close()
    with pytest.raises(RuntimeError, match="Camera is closed"):
        camera.next()


def test_live_quit_cancels_worker_and_writes_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "require_display", lambda: None)
    for name in ["namedWindow", "imshow", "destroyWindow"]:
        monkeypatch.setattr(cv2, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(cv2, "waitKey", lambda *args: ord("q"))
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *args: 1)
    frame = Frame("000001", np.zeros((20, 30, 3), np.uint8), np.ones((20, 30), np.float32),
                  np.array([[100, 0, 15], [0, 100, 10], [0, 0, 1]], float), time.monotonic_ns())
    camera = SimpleNamespace(metadata={}, next=lambda **kwargs: frame)
    class Engine:
        initialized = False
        config = {"max_depth_m": 3}
        mesh = trimesh.creation.box(extents=[.1, .02, .01])
        cancelled = False
        def cancel_pending(self):
            self.cancelled = True
    engine = Engine()
    ui.run_live(camera, engine, tmp_path, .5)
    assert engine.cancelled
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "scene/scene.json").is_file()
