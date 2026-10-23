import json
from dataclasses import replace
import numpy as np
import pytest

from rail_pose.data import Frame, SceneWriter, Scene, depth_to_metres


def example():
    return Frame("000001", np.zeros((10, 20, 3), np.uint8), np.ones((10, 20), np.float32) * .8,
                 np.array([[100, 0, 10], [0, 100, 5], [0, 0, 1]], float), 10,
                 np.ones((10, 20), bool))


def test_raw_scale_and_invalid_samples():
    raw = np.array([[0, 8000, 40000]], np.uint16)
    converted = depth_to_metres(raw, .0001, 3)
    np.testing.assert_allclose(converted, [[0, .8, 0]])


def test_round_trip_retains_metric_depth_calibration_mask_and_time(tmp_path):
    frame = example()
    writer = SceneWriter(tmp_path / "scene", {"serial": "fixture"})
    writer.write(frame)
    result = Scene(tmp_path / "scene").frame(0)
    np.testing.assert_array_equal(result.rgb, frame.rgb)
    np.testing.assert_array_equal(result.depth_m, frame.depth_m)
    np.testing.assert_array_equal(result.K, frame.K)
    np.testing.assert_array_equal(result.mask, frame.mask)
    assert result.timestamp_ns == frame.timestamp_ns


@pytest.mark.parametrize("change", [
    {"depth_m": np.ones((10, 20), np.float32) * 800},
    {"depth_m": np.ones((9, 20), np.float32)},
    {"depth_m": np.ones((10, 20), np.uint16)},
    {"depth_m": np.full((10, 20), np.nan, np.float32)},
    {"K": np.eye(4)}, {"mask": np.ones((20, 10), bool)}, {"frame_id": "../outside"},
])
def test_invalid_input_rejected(change):
    with pytest.raises(ValueError):
        replace(example(), **change).validate()


def test_registration_rejects_empty_mask_and_missing_target_depth():
    frame = example()
    with pytest.raises(ValueError):
        replace(frame, mask=np.zeros_like(frame.mask)).registration_mask()
    with pytest.raises(ValueError):
        replace(frame, depth_m=np.zeros_like(frame.depth_m)).registration_mask()


def test_duplicate_frame_cannot_overwrite_scene(tmp_path):
    writer = SceneWriter(tmp_path / "scene", {})
    writer.write(example())
    with pytest.raises(ValueError):
        writer.write(example())
    assert len(Scene(tmp_path / "scene")) == 1


def test_wrong_manifest_unit_is_rejected(tmp_path):
    writer = SceneWriter(tmp_path / "scene", {})
    writer.write(example())
    path = tmp_path / "scene/scene.json"
    manifest = json.loads(path.read_text())
    manifest["depth_unit"] = "mm"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        Scene(tmp_path / "scene")


def test_reinitialization_can_add_mask_without_overwriting_rgb_or_depth(tmp_path):
    frame = replace(example(), mask=None)
    writer = SceneWriter(tmp_path / "scene", {})
    writer.write(frame)
    rgb = (tmp_path / "scene/rgb/000001.png").read_bytes()
    depth = (tmp_path / "scene/depth/000001.npy").read_bytes()
    writer.attach_mask(example())
    assert (tmp_path / "scene/rgb/000001.png").read_bytes() == rgb
    assert (tmp_path / "scene/depth/000001.npy").read_bytes() == depth
    np.testing.assert_array_equal(Scene(tmp_path / "scene").frame(0).mask, example().mask)
