"""URDF/SDK frame registration and link poses; no mesh intersection checks."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .core import digest, matrix, read_json


class RobotGeometry:
    def __init__(self, config, kinematics):
        from diffusion_policy.third_party.bimanual_teleop.kinematics import RobotModel
        self.robot = RobotModel(Path(config["urdf"]))
        self.kinematics = kinematics
        zero = np.zeros(54)
        # The SDK base includes its first DH offset. The URDF first joint frame
        # is not that base: align through flange FK, then verify nonzero poses.
        self.T_B_W = matrix(kinematics.fk("left", tuple(zero[:7]))) @ np.linalg.inv(
            self.robot.flange("left", zero, world=True))
        self.registration_error_m = 0.
        for q7 in [np.radians([30, -60, -34, -52, 30, 12, 4]),
                   np.radians([25, -55, -30, -45, 20, 10, 2])]:
            q = zero.copy(); q[:7] = q7
            predicted = self.T_B_W @ self.robot.flange("left", q, world=True)
            actual = matrix(kinematics.fk("left", tuple(q7)))
            error = np.linalg.norm(predicted[:3, 3] - actual[:3, 3])
            angle = Rotation.from_matrix(predicted[:3, :3] @ actual[:3, :3].T).magnitude()
            self.registration_error_m = max(self.registration_error_m, float(error))
            if error > .0001 or angle > .0002:
                raise ValueError("URDF does not agree with official FK after fixed-base registration")
        if digest(config["urdf"]) != read_json(config["manifest"])["robot_urdf_sha256"]:
            raise ValueError("URDF differs from its model manifest")

    def poses(self, q):
        values = dict(zip(self.robot.names, q))
        cache = {}

        def visit(link):
            if link in cache:
                return cache[link]
            if link not in self.robot.by_child:
                T = self.T_B_W @ self.robot.world
            else:
                joint = self.robot.by_child[link]
                T = visit(joint.parent) @ joint.origin
                if joint.kind == "revolute":
                    R = np.eye(4)
                    R[:3, :3] = Rotation.from_rotvec(joint.axis * values[joint.name]).as_matrix()
                    T = T @ R
            cache[link] = T
            return T
        return {link: visit(link) for link in set(self.robot.by_child) | {joint.parent for joint in self.robot.joints.values()}}
