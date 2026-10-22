"""Generate explicitly synthetic integration inputs; never use them on hardware."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[2]))

from rail_preinsert.core import PROJECT, digest, load_config, matrix, write_json
from diffusion_policy.third_party.bimanual_teleop.devices.tianji.model import TianjiKinematics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    config = load_config(HERE / "config.yaml")
    episode = PROJECT / "bimanual-teleop/recordings/rail_place3/episode_000034"
    info = json.loads((episode / "episode.json").read_text())
    stamp = info["start_ns"] + int(9.25e9)
    joints = []
    for kind, width in [("arms", 13), ("hands", 20)]:
        for side in ["left", "right"]:
            values = np.fromfile(episode / f"raw_spool/streams/{kind}__{side}.bin",
                                 dtype=[("t", "<i8"), ("sequence", "<i8"), ("v", "<f8", width)])
            selected = values[np.argmin(np.abs(values["t"] - stamp))]["v"]
            joints.extend(selected[:7] if kind == "arms" else selected)
    q = np.asarray(joints)
    k = TianjiKinematics()
    initial = matrix(k.fk("left", tuple(q[:7])))
    # Construct a centred rail orientation and a nearby reachable endpoint.
    # This is a coordinate/planner fixture, not a reconstructed physical grasp.
    R_B_O = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    import trimesh
    centre_cad = trimesh.load(config["mesh"], force="mesh").bounds.mean(axis=0)
    grasp = np.eye(4)
    grasp[:3, :3] = initial[:3, :3].T @ R_B_O
    grasp[:3, 3] = np.array([0., 0., .22]) - grasp[:3, :3] @ centre_cad
    object_initial = initial @ grasp
    q_goal = q[:7] + np.array([.03, -.02, .01, 0., .01, 0., 0.])
    pre = matrix(k.fk("left", tuple(q_goal))) @ grasp
    goal = pre.copy(); goal[:3, 3] -= .02 * goal[:3, 2]
    calibration = {"parent_frame": "tianji_left_base", "child_frame": "head_color_optical",
                   "unit": "m", "camera_serial": "synthetic-offline", "T_B_C": np.eye(4).tolist(), "test_data": True}
    target = {"unit": "m", "camera_serial": "synthetic-offline", "T_C_O_goal": goal.tolist(),
              "mesh_sha256": digest(config["mesh"]), "test_data": True}
    observation = {"unit": "m", "frame": "T_C_CAD", "camera_serial": "synthetic-offline",
                   "mesh_sha256": digest(config["mesh"]), "test_data": True,
                   "records": [{"timestamp_ns": t, "valid": True, "T_C_CAD": object_initial.tolist()}
                               for t in [1_010_000_000, 1_020_000_000, 1_030_000_000]]}
    state = {"timestamp_ns": 1_050_000_000, "full_joints": q.tolist(), "model_sha256": k.model.digest,
             "left_joint_history": [[t, q[:7].tolist()] for t in range(1_000_000_000, 1_060_000_000, 10_000_000)],
             "test_data": True,
             "note": "Recorded joint configuration, synthetic grasp/camera/target/environment. Not a hardware success."}
    scene = {"parent_frame": "tianji_left_base", "unit": "m", "test_data": True, "obstacles": []}
    for name, value in [("calibration", calibration), ("target", target), ("observation", observation),
                        ("state", state), ("scene", scene)]:
        write_json(out / f"{name}.json", value)
    for key, filename in [("calibration", "calibration"), ("target", "target"), ("scene", "scene")]:
        config[key] = str(out / f"{filename}.json")
    config.pop("config_path", None)
    (out / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True))
    print(out)


if __name__ == "__main__":
    main()
