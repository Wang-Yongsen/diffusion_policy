"""Measured histories and fake devices for rail control regressions."""

import threading
from types import SimpleNamespace

import numpy as np

from diffusion_policy.common.paths import PROJECT_ROOT
from diffusion_policy.real_world.tianji.config import load_deployment_config
from diffusion_policy.third_party.bimanual_teleop.devices.wuji.adapter import (
    JOINT_NAMES,
)
from diffusion_policy.third_party.bimanual_teleop.types import JointTarget, Submission


def config():
    result = load_deployment_config(
        PROJECT_ROOT / "diffusion_policy/config/tianji_rail_real_eval.yaml"
    )
    # Existing regression cases exercise historical 30 Hz queue boundaries.
    result["policy_hz"], result["replanning_steps"] = 30, 2
    result["workspace"] = {
        side: {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]} for side in ("left", "right")
    }
    return result


def checkpoint_cfg():
    shape_meta = {
        "obs": {
            key: {"shape": [3, 16, 16], "type": "rgb"}
            for key in ("camera_0", "camera_1", "camera_2")
        }
    }
    shape_meta["obs"].update(robot_eef_pose={"shape": [14]}, hand_joint={"shape": [40]})
    return SimpleNamespace(
        task=SimpleNamespace(shape_meta=shape_meta, dataset={"resize_enabled": False})
    )


def histories():
    result = {
        f"camera_{i}": [
            (1_000_000_000, np.full((16, 16, 3), i, np.uint8)),
            (1_033_333_333, np.full((16, 16, 3), 10 + i, np.uint8)),
        ]
        for i in range(3)
    }
    for kind, size in (("arms", 6), ("hands", 20)):
        for side in ("left", "right"):
            result[f"{kind}/{side}"] = [
                (stamp, np.zeros(size)) for stamp in range(990_000_000, 1_050_000_001, 10_000_000)
            ]
    return result


class FakeArm:
    def __init__(self):
        self.driver = SimpleNamespace(engaged=True, profile=SimpleNamespace(profile_id="arm"))
        self.commands = []
        self.stopped = threading.Event()

    def submit(self, target):
        self.commands.append(target)
        return Submission(target.command_id, True)

    def request_hold(self, reason):
        self.driver.engaged = False
        self.stopped.set()


class FakeHand:
    def __init__(self, side):
        self.enabled = True
        self.device_id = side
        self.profile = SimpleNamespace(profile_id="hand")
        self.last_target = JointTarget(JOINT_NAMES, (0.0,) * 20)
        self.commands = []
        self.accepted = True

    def submit(self, target):
        self.commands.append(target)
        if self.accepted:
            self.last_target = target.payload
        return Submission(
            target.command_id, self.accepted, None if self.accepted else "fake reject"
        )

    def request_hold(self, reason):
        pass

    def health(self, **kwargs):
        return SimpleNamespace(ready=True)
