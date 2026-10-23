import numpy as np
import pytest

from rail_pose.episode import STAMP, depth_layout, pair_streams, register_depth

K = np.array([[100, 0, 31.5], [0, 100, 23.5], [0, 0, 1]], float)


def test_identity_registration_preserves_depth():
    depth = np.full((48, 64), .8, np.float32)
    depth[10:20, 30:40] = .5
    np.testing.assert_allclose(register_depth(depth, K, K, np.eye(3), np.zeros(3), depth.shape), depth)


def test_translated_point_lands_at_projected_pixel_and_nearest_surface_wins():
    depth = np.zeros((48, 64), np.float32)
    depth[23, 31] = 1.0  # optical centre ray (31.5 - .5)
    depth[23, 40] = .5
    t = np.array([.05, 0, 0])
    result = register_depth(depth, K, K, np.eye(3), t, depth.shape)
    assert result[23, 36] == pytest.approx(1.0)  # 31 + 100 * .05 / 1.0
    assert result[23, 50] == pytest.approx(.5)   # 40 + 100 * .05 / .5
    background = np.full((48, 64), 2.0, np.float32)
    background[23, 31] = 1.0
    shifted = register_depth(background, K, K, np.eye(3), t, depth.shape)
    assert shifted[23, 36] == pytest.approx(1.0)


def test_depth_layout_reads_spool_or_intrinsics():
    intrinsics = {"height": 480, "width": 640}
    assert depth_layout({"spool": {"depth_shape": [480, 640], "depth_dtype": "uint16"}}, intrinsics) == (480, 640)
    assert depth_layout({}, intrinsics) == (480, 640)
    with pytest.raises(ValueError):
        depth_layout({"spool": {"depth_shape": [480, 640], "depth_dtype": "float32"}}, intrinsics)


def test_pairing_rejects_distant_depth():
    rgb = np.array([(0, 1, 0.), (33_000_000, 2, 0.)], STAMP)
    depth = np.array([(10_000, 7, 0.), (33_020_000, 8, 0.)], STAMP)
    nearest, offset = pair_streams(rgb, depth, 1_000_000)
    np.testing.assert_array_equal(nearest, [0, 1])
    np.testing.assert_array_equal(offset, [10_000, 20_000])
    with pytest.raises(ValueError):
        pair_streams(rgb, depth, 5_000)
