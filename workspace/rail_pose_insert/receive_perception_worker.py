"""Camera preview and postgrasp SAM prompt/pose estimation; no actuator access."""

from __future__ import annotations
import argparse
from dataclasses import replace
import gc
from pathlib import Path
import sys
import time
from rail_common.core import HERE, digest, load_target, read_json, transform, write_json
from rail_common.pose_alignment import RailPoseAlignment
from rail_receive.logic import load_config
from rail_common.camera import Camera


def pose_engine(args, directory, config=None):
    """Imported only after grasping, when the controller requests held mode."""
    root = args.foundationpose_root.resolve()
    sys.path[:0] = [str(root), str(root / "workspace/src")]
    from rail_pose.engine import PoseEngine
    from rail_common.body_orientation_prior import BodyOrientationPrior, PriorPoseEngine

    prior = BodyOrientationPrior.from_config(config or {}, camera_serial=args.camera_serial)
    engine = PoseEngine(
        args.mesh.resolve(), read_json(root / "workspace/configs/default.json"), directory / "debug"
    )
    return PriorPoseEngine(engine, prior) if prior is not None else engine


def pose_reference(config, camera_serial):
    """Transport supplies a validated reference; preview can use a matching goal."""
    reference = config.get("pose_preview", {}).get("T_C_O_goal")
    if reference is not None:
        return (reference, None)
    path = Path(config["config_path"]).parent / config["move_above"]["target"]
    try:
        _, reference = load_target(path, {"camera_serial": camera_serial}, config["mesh"])
        return (reference, None)
    except (OSError, ValueError, KeyError) as error:
        return (None, str(error))


def serve(args, config):
    """Preview before closing; prompted SAM then FoundationPose in held mode."""
    import numpy as np
    from rail_receive.sam_prompt import SamPromptDetector
    from rail_receive.visualization import save_preview

    directory = args.serve.resolve()
    camera = detector = engine = None
    try:
        sys.path.insert(0, str(args.foundationpose_root.resolve() / "workspace/src"))
        camera = Camera(args.camera_serial, rgb_only=True)
        using_rgb_only = True
        save_preview(directory, camera.frame(), None, None, reason="waiting for grasp")
        identity = {
            "camera_serial": args.camera_serial,
            "mesh_sha256": digest(args.mesh),
            "replay_only": False,
        }
        alignment = RailPoseAlignment.from_config(config)
        reference, reference_error = (
            pose_reference(config, args.camera_serial) if alignment else (None, None)
        )
        write_json(directory / "ready.json", {"ready": True, "mode": "camera_preview", **identity})
        previous, generation, active = (None, None, False)
        predicted_pose = None
        phase = "camera_preview"

        def progress(reason, epoch, details=None):
            write_json(
                directory / "latest.json",
                {
                    **(details or {}),
                    "valid": False,
                    "reason": reason,
                    "generation": generation,
                    "context_epoch": epoch,
                    **identity,
                },
            )

        while True:
            path = directory / "request.json"
            if path.is_file():
                request = read_json(path)
                if request["id"] != previous:
                    previous = request["id"]
                    if request["command"] == "shutdown":
                        break
                    if request["command"] == "stop":
                        active = False
                    elif request["command"] == "initialize":
                        predicted_pose = (
                            transform(request["T_C_O_predicted"], "T_C_O_predicted")
                            if "T_C_O_predicted" in request
                            else None
                        )
                        if engine is not None:
                            engine = None
                            gc.collect()
                            import torch

                            torch.cuda.empty_cache()
                            detector = None
                        generation, active = (request["id"], True)
                    else:
                        raise ValueError("Unknown receiving perception command")
            context_path = directory / "context.json"
            context = read_json(context_path) if context_path.is_file() else None
            if not active or not context or (not context.get("enabled")):
                try:
                    save_preview(
                        directory,
                        camera.frame(rgb_only=True),
                        None,
                        None,
                        reason="detection inactive",
                    )
                except (ValueError, RuntimeError):
                    pass
                time.sleep(0.1)
                continue
            epoch = context["epoch"]
            frame = None
            try:
                if using_rgb_only:
                    phase = "opening_rgbd"
                    progress(phase, epoch)
                    camera.close()
                    camera = Camera(args.camera_serial)
                    using_rgb_only = False
                phase = "acquiring_rgbd"
                frame = camera.frame()
                if engine is None:
                    if detector is None:
                        phase = "loading_sam"
                        progress(phase, epoch)
                        gc.collect()
                        if predicted_pose is None:
                            detector = SamPromptDetector(config, camera_serial=args.camera_serial)
                        else:
                            detector = SamPromptDetector(
                                config, camera_serial=args.camera_serial, stage="preplace"
                            )
                    frame = camera.frame()
                    phase = "segmenting_held_rail"
                    progress(phase, epoch)
                    mask, record = detector.detect(frame)
                    if predicted_pose is not None:
                        record.update(
                            prompt_source="saved_preplace_prompt",
                            T_C_O_predicted=predicted_pose.tolist(),
                        )
                    if mask is None:
                        raise RuntimeError(
                            "SAM returned an empty mask; adjust the box/points with set_sam_prompt.py"
                        )
                    else:
                        phase = "initializing_pose"
                        progress(phase, epoch, record)
                        detector.close()
                        detector = None
                        pose_config = config
                        if predicted_pose is not None:
                            pose_config = dict(
                                config,
                                body_orientation_prior={
                                    **config.get("body_orientation_prior", {}),
                                    "enabled": False,
                                },
                            )
                        engine = pose_engine(args, directory, pose_config)
                        pose, record = engine.infer(replace(frame, mask=mask), initialize=True)
                else:
                    pose, record = engine.infer(frame, initialize=not engine.initialized)
                if engine is not None:
                    alignment_text = ""
                    if pose is not None:
                        phase = "aligning_pose"
                        if alignment is None or reference is None:
                            raise ValueError(
                                f"Cannot display an aligned rail pose without a matching goal and alignment policy: {reference_error or 'missing policy'}"
                            )
                        record = alignment.align_record(
                            {**record, "T_C_CAD": pose.tolist()}, reference
                        )
                        pose = np.asarray(record["T_C_CAD"])
                        info = record["pose_alignment"]
                        angles = "/".join((f"{v:.1f}" for v in info["relative_xyz_deg"]))
                        status = (
                            "within 30 deg" if info["within_typical_range"] else "30-45 deg range"
                        )
                        alignment_text = f"Aligned | goal-relative XYZ deg: {angles} | {status}"
                    save_preview(
                        directory,
                        frame,
                        engine.mesh,
                        pose,
                        None,
                        alignment_text or record.get("reason", ""),
                    )
            except (ValueError, RuntimeError, OSError) as error:
                pose = None
                if phase == "aligning_pose" and frame is not None:
                    save_preview(
                        directory, frame, None, None, reason="No aligned pose: " + str(error)
                    )
                if engine is not None:
                    engine.reset()
                    engine = None
                record = {
                    "valid": False,
                    "detected_in_roi": False,
                    "reason": "perception_frame_failed",
                    "detail": str(error),
                }
            record.update(
                identity,
                generation=generation,
                timestamp_ns=int(frame.timestamp_ns) if frame is not None else time.monotonic_ns(),
                context_epoch=epoch,
                timestamp_source=(
                    frame.metadata["timestamp_source"] if frame is not None else "camera_failure"
                ),
                output_monotonic_ns=time.monotonic_ns(),
            )
            if predicted_pose is not None:
                record.update(
                    pose_source="preplace_registration",
                    reestimate_request=True,
                    body_orientation_prior_enabled=False,
                )
            write_json(directory / "latest.json", record)
    except Exception as error:
        write_json(directory / "ready.json", {"ready": False, "error": str(error)})
        raise
    finally:
        if detector is not None:
            detector.close()
        if camera is not None:
            camera.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "receive.yaml")
    parser.add_argument("--foundationpose-root", type=Path)
    parser.add_argument("--mesh", type=Path)
    parser.add_argument("--camera-serial")
    parser.add_argument("--serve", type=Path)
    args = parser.parse_args()
    config = read_json(args.config) if args.config.suffix == ".json" else load_config(args.config)
    args.foundationpose_root = args.foundationpose_root or Path(config["foundationpose_root"])
    args.mesh = args.mesh or Path(config["mesh"])
    if not args.serve or not args.camera_serial:
        parser.error("Live worker requires --serve and --camera-serial")
    serve(args, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
