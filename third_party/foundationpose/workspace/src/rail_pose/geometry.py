from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def validate_pose(pose: np.ndarray) -> None:
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("pose must be a finite 4×4 transform")
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError("invalid homogeneous transform")
    R = pose[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=2e-3) or abs(np.linalg.det(R) - 1) > 2e-3:
        raise ValueError("pose rotation is not in SO(3)")


def pose_difference(previous, current):
    distance = float(np.linalg.norm(current[:3, 3] - previous[:3, 3]))
    angle = float(np.rad2deg(Rotation.from_matrix(current[:3, :3] @ previous[:3, :3].T).magnitude()))
    return distance, angle


def project(points, pose, K):
    camera = np.asarray(points) @ pose[:3, :3].T + pose[:3, 3]
    if np.any(camera[:, 2] <= .001):
        return None
    pixels = camera @ K.T
    return pixels[:, :2] / pixels[:, 2, None]


def overlay(rgb, mesh, pose, K, label="", color=(40, 220, 80)):
    import cv2
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if pose is not None:
        low, high = mesh.bounds
        corners = np.array([[x, y, z] for x in [low[0], high[0]]
                            for y in [low[1], high[1]] for z in [low[2], high[2]]])
        pixels = project(corners, pose, K)
        if pixels is not None:
            pixels = np.clip(pixels, -10000, 10000).astype(int)
            for i in range(8):
                for bit in [1, 2, 4]:
                    j = i ^ bit
                    if i < j:
                        cv2.line(image, tuple(pixels[i]), tuple(pixels[j]), color, 1, cv2.LINE_AA)
        axes = np.vstack([np.zeros(3), np.eye(3) * min(mesh.extents.max() / 4, .06)])
        pixels = project(axes, pose, K)
        if pixels is not None:
            pixels = np.clip(pixels, -10000, 10000).astype(int)
            for axis, bgr in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)], 1):
                cv2.arrowedLine(image, tuple(pixels[0]), tuple(pixels[axis]), bgr, 2, cv2.LINE_AA)
        xyz = pose[:3, 3]
        cv2.putText(image, f"CAD origin [m]: {xyz[0]:.3f} {xyz[1]:.3f} {xyz[2]:.3f}",
                    (10, image.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
    if label:
        cv2.rectangle(image, (0, 0), (image.shape[1], 34), (30, 30, 30), -1)
        cv2.putText(image, label[:90], (10, 23), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    return image
