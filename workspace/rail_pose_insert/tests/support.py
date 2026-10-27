"""Core offline regression coverage; all actuator and camera owners are mocked."""

from pathlib import Path
import sys
import threading
from types import SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
from rail_common.core import (
    digest,
    write_json,
)
from rail_receive.above import AboveTransfer
from rail_receive.logic import load_config
from diffusion_policy.third_party.bimanual_teleop.types import Pose


def config():
    return load_config(HERE / "receive.yaml")


def T(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    result = np.eye(4)
    result[:3, 3] = xyz
    result[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    return result


class ToyKinematics:
    """Full SE(3) toy for coordinate tests; every call asserts right-arm routing."""

    def __init__(self):
        arm = NS(
            lower_rad=(-4.0,) * 7,
            upper_rad=(4.0,) * 7,
            max_joint_velocity_rad_s=(1.0,) * 7,
            limits_native=[(0, 0, 0, 60)] * 7,
        )
        self.model = NS(arm=lambda side: arm, digest="toy")

    def fk(self, side, q):
        assert side == "right"
        return Pose(
            "tianji_right_base",
            "tianji_right_flange",
            tuple(q[:3]),
            tuple(Rotation.from_rotvec(q[3:6]).as_quat()),
        )

    def ik(self, side, target, seed):
        assert side == "right" and target.parent_frame == "tianji_right_base"
        return (*target.position_m, *Rotation.from_quat(target.orientation_xyzw).as_rotvec(), 0.0)


class AboveTests:

    def fixture(self, directory):
        c = load_config(HERE / "receive.yaml")
        c["move_above"].update(
            target=str(Path(directory) / "target.json"), samples=2, settle_s=0.001
        )
        geo = NS(
            T_B_C=T([0.1, -0.2, 0.3], [0.2, -0.1, 0.3]),
            calibration={"camera_serial": "test"},
            mesh_digest=digest(c["mesh"]),
        )
        goal = T([0.1, 0.2, 0.4], [0.1, 0.1, 0.1])
        write_json(
            c["move_above"]["target"],
            {
                "unit": "m",
                "camera_serial": "test",
                "mesh_sha256": geo.mesh_digest,
                "T_C_O_goal": goal.tolist(),
            },
        )
        for key in ("sam_prompt", "sam_prompt_preplace"):
            c[key] = str(Path(directory) / (key + ".json"))
            write_json(
                c[key],
                dict(
                    camera_serial="test",
                    image_size=[640, 480],
                    box_xyxy=[20, 30, 500, 400],
                    point_coords=[],
                    point_labels=[],
                ),
            )
        transfer = AboveTransfer(c, geo, ToyKinematics(), directory)
        hw = Mock(paused=False)
        hw.lock = threading.RLock()
        return (transfer, hw)

    def record(self, transfer, stamp, camera_pose):
        return {
            "valid": True,
            "timestamp_ns": stamp,
            "T_C_CAD": camera_pose.tolist(),
            "camera_serial": "test",
            "mesh_sha256": transfer.geometry.mesh_digest,
            "timestamp_source": "sdk_global_timestamp_mapped_to_host_monotonic",
        }
