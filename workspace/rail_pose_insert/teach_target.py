"""Record the fully seated rail target using the head camera; no robot motion."""

import argparse
from pathlib import Path
import tempfile

from rail_common.core import HERE, digest, load_calibration, mean_pose, write_json
from rail_receive.logic import load_config


def teach(config, args):
    from rail_common.perception import Perception
    from rail_common.core import observation_poses
    from rail_common.pose_alignment import RailPoseAlignment

    if Path(args.output).exists():
        raise FileExistsError("Target already exists; choose a new output rather than overwriting")
    calibration, _ = load_calibration(config["calibration"])
    with tempfile.TemporaryDirectory(prefix="rail-teach-") as directory:
        perception = Perception(
            config, Path(directory) / "perception", calibration["camera_serial"]
        )
        try:
            print("导轨正确就位后，在相机窗口标注可见导轨区域。", flush=True)
            observation = perception.measure(config["move_above"]["samples"])
        finally:
            perception.close()
    records, poses = observation_poses(
        observation, calibration, config["mesh"], alignment=RailPoseAlignment.from_config(config)
    )
    target = {
        "schema_version": 1,
        "unit": "m",
        "frame": "T_C_CAD",
        "camera_serial": calibration["camera_serial"],
        "mesh_sha256": digest(config["mesh"]),
        "T_C_O_goal": mean_pose(poses).tolist(),
        "test_data": False,
        "pose_alignment": [r.get("pose_alignment") for r in records],
        "observation": observation,
        "calibration_sha256_at_teach": digest(config["calibration"]),
    }
    if Path(args.output).exists():
        raise FileExistsError("Target already exists; choose a new output rather than overwriting")
    write_json(args.output, target)
    print(f"目标位姿已保存：{Path(args.output).resolve()}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "receive.yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.output is None:
            args.output = Path(config["config_path"]).parent / config["move_above"]["target"]
        teach(config, args)
        return 0
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print(f"目标示教失败：{error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
