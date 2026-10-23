import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from rail_pose.geometry import validate_pose, pose_difference, project


def test_pose_projection_direction_and_translation():
    pose = np.eye(4)
    pose[2, 3] = 2
    K = np.array([[100, 0, 10], [0, 100, 20], [0, 0, 1]], float)
    np.testing.assert_allclose(project([[0, 0, 0], [1, 0, 0]], pose, K), [[10, 20], [60, 20]])


def test_rigid_pose_validation_and_geodesic_jump():
    previous = np.eye(4)
    current = np.eye(4)
    current[:3, :3] = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    current[0, 3] = .05
    validate_pose(current)
    distance, angle = pose_difference(previous, current)
    assert distance == pytest.approx(.05)
    assert angle == pytest.approx(30)
    current[0, 0] *= 2
    with pytest.raises(ValueError):
        validate_pose(current)
