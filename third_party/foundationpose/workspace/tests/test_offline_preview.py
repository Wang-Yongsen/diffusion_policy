import json
import cv2
import numpy as np
import pytest
import trimesh

from rail_pose import cli, data, engine, ui
from rail_pose.common import load_config
from rail_pose.data import Frame


def test_offline_preview_requires_desktop_before_creating_output(tmp_path, monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    args = cli.parser().parse_args(["track", "--scene", str(tmp_path / "missing"), "--preview"])
    with pytest.raises(RuntimeError, match="desktop terminal"):
        cli.run_offline(args, load_config())


@pytest.mark.parametrize("quit_early", [True, False])
def test_preview_can_exit_early_or_hold_final_frame_and_save_results(tmp_path, monkeypatch, quit_early):
    frames = [Frame(str(i), np.zeros((80, 100, 3), np.uint8), np.ones((80, 100), np.float32),
                    np.array([[100, 0, 50], [0, 100, 40], [0, 0, 1]], float), i)
              for i in range(2)]
    class Scene:
        directory = tmp_path
        def __init__(self, path):
            pass
        def __len__(self):
            return len(frames)
        def frame(self, index):
            return frames[index]
    inferred = []
    class Engine:
        mesh = trimesh.creation.box(extents=[.1, .02, .01])
        mesh_path = tmp_path / "model.obj"
        def __init__(self, *args, **kwargs):
            pass
        def infer(self, frame):
            inferred.append(frame.frame_id)
            pose = np.eye(4)
            pose[2, 3] = 1
            return pose, {"valid": True, "mode": "register" if len(inferred) == 1 else "track",
                          "processing_s": .01, "torch_peak_allocated_mib": 1,
                          "torch_peak_reserved_mib": 1, "frame_id": frame.frame_id}
    monkeypatch.setattr(data, "Scene", Scene)
    monkeypatch.setattr(engine, "PoseEngine", Engine)
    monkeypatch.setattr(cli, "new_run", lambda *args: tmp_path)
    monkeypatch.setattr(ui, "require_display", lambda: None)
    closed = []
    shown = []
    monkeypatch.setattr(cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(cv2, "imshow", lambda title, image: shown.append(image.copy()))
    monkeypatch.setattr(cv2, "destroyWindow", lambda title: closed.append(title))
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *args: 1)
    keys = iter([ord("q")] if quit_early else [-1, -1, ord("q")])
    monkeypatch.setattr(cv2, "waitKey", lambda *args: next(keys))
    args = cli.parser().parse_args(["track", "--scene", "unused", "--frames", "2", "--preview"])
    assert cli.run_offline(args, load_config()) == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    expected = 1 if quit_early else 2
    assert summary["frames"] == summary["valid_frames"] == expected
    assert summary["requested_frames"] == 2
    assert summary["stopped_by_user"] == quit_early
    assert len(inferred) == expected
    assert len(list((tmp_path / "poses").glob("*.txt"))) == expected
    assert len((tmp_path / "poses.jsonl").read_text().splitlines()) == expected
    assert shown and closed == [ui.OfflinePreview.title]
