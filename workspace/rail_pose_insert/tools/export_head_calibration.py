"""Convert a measured external-camera result to the experiment's SDK base, offline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1]), str(HERE.parents[2])]

from rail_preinsert.core import digest, load_config, read_json, transform


def export(config, result_path, output, camera_serial, kinematics):
    from rail_preinsert.geometry import RobotGeometry

    result_path, output = Path(result_path).resolve(), Path(output).resolve()
    report = read_json(result_path)
    if (report.get("schema") != "external_camera_result_v1"
            or not report.get("passed") or not report.get("quality", {}).get("passed")
            or report.get("synthetic", False) or report.get("test_data", False)):
        raise ValueError("Only measured external-camera results that passed validation are accepted")
    if report.get("camera", {}).get("serial") != camera_serial:
        raise ValueError("Result is not from the requested head camera")
    if report.get("urdf_sha256") != digest(config["urdf"]):
        raise ValueError("Calibration and experiment URDF differ")
    side = report.get("side")
    if side not in ("left", "right") or report.get("reference_frame") != "project_world":
        raise ValueError("Unknown external-camera reference frame")
    if output.exists():
        raise FileExistsError("Calibration already exists; choose a new output rather than overwriting")

    # Reuse the same official-FK registration and nonzero-pose checks as planning.
    # RobotModel.base_pose is the first-joint frame, not the SDK arm base.
    geometry = RobotGeometry(config, kinematics)
    base = transform(report["T_world_robot_base"])
    if not np.allclose(base, geometry.robot.base_pose(side), atol=1e-8, rtol=0):
        raise ValueError("Calibration and experiment project-world conventions differ")
    camera = transform(report["T_world_camera_opencv"])
    if not np.allclose(base @ transform(report["T_base_camera_opencv"]), camera,
                       atol=1e-8, rtol=0):
        raise ValueError("External result's world/base transforms are inconsistent")
    payload = {
        "parent_frame": "tianji_left_base", "child_frame": "head_color_optical", "unit": "m",
        "camera_serial": camera_serial,
        "T_B_C": transform(geometry.T_B_W @ camera).tolist(),
        "source_result": str(result_path), "source_result_sha256": digest(result_path),
        "urdf_sha256": report["urdf_sha256"], "sdk_model_sha256": kinematics.model.digest,
        "wrist_calibration_sha256": report["wrist_calibration_sha256"],
        "validation": report["quality"].get("validation", {}),
        "note": "Measured optical camera to SDK left base; validation does not prove absolute accuracy.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="camera_calibration external solve result.json")
    parser.add_argument("--camera-serial", required=True, help="Actual head color camera serial")
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--output", type=Path, default=HERE / "calibrations/head_to_left_base.json")
    args = parser.parse_args()
    from diffusion_policy.third_party.bimanual_teleop.devices.tianji.model import TianjiKinematics
    export(load_config(args.config), args.result, args.output, args.camera_serial, TianjiKinematics())
    print(f"头部彩色相机到 SDK 左臂基座的外参已保存：{args.output.resolve()}")


if __name__ == "__main__":
    main()
