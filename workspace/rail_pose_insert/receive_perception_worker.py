"""Camera preview and postgrasp SAM prompt/pose estimation; no actuator access."""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import sys
import time

from rail_preinsert.core import HERE, digest, load_target, read_json, transform, write_json
from rail_preinsert.pose_alignment import RailPoseAlignment
from rail_receive.logic import load_config
from perception_worker import Camera


def components(args, config, directory):
    root = args.foundationpose_root.resolve()
    sys.path[:0] = [str(root), str(root / "workspace/src")]
    from rail_pose.engine import PoseEngine
    from rail_receive.automatic import RailProposer, AutomaticTracker
    engine = PoseEngine(args.mesh.resolve(), read_json(root / "workspace/configs/default.json"), directory / "debug")
    tracker = AutomaticTracker(engine, RailProposer(config, engine.mesh.extents), config["automatic_detection"])
    return engine, tracker


def save_frame(directory, camera, engine, tracker, frame, pose, writer, attempt):
    from PIL import Image
    from rail_pose.geometry import overlay
    from rail_pose.data import SceneWriter
    if tracker.initial_mask is not None:
        attempt += 1
        target = directory / f"attempt_{attempt:04d}"
        writer = SceneWriter(target / "scene", {"serial": camera.serial, "K": frame.K.tolist()})
        frame = replace(frame, mask=tracker.initial_mask)
        Image.fromarray(overlay(frame.rgb, engine.mesh, pose, frame.K)).save(target / "initial_pose.png")
        Image.fromarray(frame.mask.astype("uint8")*255).save(target / "automatic_mask.png")
    if writer is not None:
        writer.write(frame)
    return writer, attempt


def pose_engine(args, directory, config=None):
    """Imported only after grasping, when the controller requests held mode."""
    root = args.foundationpose_root.resolve()
    sys.path[:0] = [str(root), str(root / "workspace/src")]
    from rail_pose.engine import PoseEngine
    from rail_preinsert.body_orientation_prior import BodyOrientationPrior, PriorPoseEngine
    prior = BodyOrientationPrior.from_config(config or {}, camera_serial=args.camera_serial)
    engine = PoseEngine(args.mesh.resolve(), read_json(root / "workspace/configs/default.json"), directory / "debug")
    return PriorPoseEngine(engine, prior) if prior is not None else engine


def pose_reference(config, camera_serial):
    """Transport supplies a validated reference; preview can use a matching goal."""
    reference = config.get("pose_preview", {}).get("T_C_O_goal")
    if reference is not None:
        return reference, None
    path = Path(config["config_path"]).parent / config["move_above"]["target"]
    try:
        _, reference = load_target(path, {"camera_serial": camera_serial}, config["mesh"])
        return reference, None
    except (OSError, ValueError, KeyError) as error:
        # Receiving/preview may run without a taught goal. It never moves a rail.
        return None, str(error)


def serve(args, config):
    """Preview before closing; prompted SAM then FoundationPose in held mode."""
    import numpy as np
    from types import SimpleNamespace
    from rail_receive.cnos import CnosDetector
    from rail_receive.sam_prompt import SamPromptDetector, projected_prompt
    from rail_receive.visualization import save_preview, save_pose_snapshot, save_mask_snapshot
    directory = args.serve.resolve()
    camera = detector = engine = None
    try:
        sys.path.insert(0, str(args.foundationpose_root.resolve() / "workspace/src"))
        camera = Camera(args.camera_serial, rgb_only=True)
        using_rgb_only = True
        save_preview(directory, camera.frame(), None, None, reason="waiting for grasp")
        detector_mode = None
        identity = {"camera_serial": args.camera_serial, "mesh_sha256": digest(args.mesh), "replay_only": False}
        alignment = RailPoseAlignment.from_config(config)
        reference, reference_error = pose_reference(config, args.camera_serial) if alignment else (None, None)
        write_json(directory / "ready.json", {"ready": True, "mode": "camera_preview", **identity})
        previous, generation, active = None, None, False
        predicted_pose, mesh_bounds = None, None
        writer, attempt = None, 0
        once_complete, once_started = False, None
        phase = "camera_preview"

        def progress(reason, epoch, details=None):
            write_json(directory / "latest.json", {**(details or {}), "valid": False, "reason": reason,
                "generation": generation, "context_epoch": epoch, **identity})

        def preplace_prompt(frame):
            nonlocal mesh_bounds
            if mesh_bounds is None:
                # Load geometry on CPU: the old pose engine must be freed
                # before SAM, and SAM must be freed before the next engine.
                import trimesh
                mesh_bounds = trimesh.load(args.mesh.resolve(), force="mesh").bounds
            return projected_prompt(mesh_bounds, predicted_pose, frame.K,
                                    config["receive_image_size"], args.camera_serial)

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
                        predicted_pose = (transform(request["T_C_O_predicted"], "T_C_O_predicted")
                                          if "T_C_O_predicted" in request else None)
                        # T means new SAM segmentation + pose registration even
                        # while the continuous held-mode tracker is active.
                        if engine is not None:
                            engine = None
                            gc.collect()
                            import torch
                            torch.cuda.empty_cache()
                        if detector is not None and detector_mode == "held":
                            detector.close()
                            detector = None
                        generation, active, writer = request["id"], True, None
                        once_complete, once_started = False, None
                        (directory / generation).mkdir(exist_ok=True)
                    else:
                        raise ValueError("Unknown receiving perception command")
            context_path = directory / "context.json"
            context = read_json(context_path) if context_path.is_file() else None
            if not active or not context or not context.get("enabled"):
                # Keep showing the main camera while opening/holding, with CNOS stopped.
                try:
                    save_preview(directory, camera.frame(rgb_only=True), None, None, reason="detection inactive")
                except (ValueError, RuntimeError):
                    pass
                time.sleep(.1)
                continue
            mode, epoch = context.get("mode", "receive"), context["epoch"]
            if mode == "held_once" and once_complete:
                # Keep the same window and camera alive without running either model.
                try:
                    save_preview(directory, camera.frame(rgb_only=True), None, None, reason="pose complete; camera only")
                except (ValueError, RuntimeError):
                    pass
                time.sleep(.1)
                continue
            pose, initial_mask, frame = None, None, None
            try:
                if mode == "held_once":
                    once_started = once_started or time.monotonic()
                    if time.monotonic()-once_started > config["perception_timeout_s"]:
                        raise RuntimeError("No prompted rail mask found before pose timeout")
                if mode != "receive" and using_rgb_only:
                    phase = "opening_rgbd"
                    progress(phase, epoch)
                    camera.close()
                    camera = Camera(args.camera_serial)
                    using_rgb_only = False
                phase = "acquiring_rgbd" if mode != "receive" else "acquiring_rgb"
                frame = camera.frame()
                if mode == "receive":
                    if detector is None:
                        detector = CnosDetector(config)
                        detector_mode = "receive"
                    mask, record = detector.detect(frame)
                    # There is deliberately no FoundationPose inference here.
                    save_preview(directory, frame, None, None, reason="rail detected" if record["valid"] else "searching rail",
                                 roi=config["receive_roi_xyxy"], detection=record)
                else:
                    if engine is None:
                        if detector is None or detector_mode != "held":
                            phase = "loading_sam"
                            progress(phase, epoch)
                            if detector is not None:
                                detector.close()
                            gc.collect()
                            if predicted_pose is None:
                                detector = SamPromptDetector(config, camera_serial=args.camera_serial)
                            else:
                                detector = SamPromptDetector(config, camera_serial=args.camera_serial,
                                                             prompt=preplace_prompt(frame))
                            detector_mode = "held"
                        # Capture after model loading so the prompt is applied
                        # to the latest RGB-D image, rather than an old frame.
                        frame = camera.frame()
                        if predicted_pose is not None:
                            # Recompute from the current frame's intrinsics;
                            # no receive-position bbox or pixel points survive.
                            detector.set_prompt(preplace_prompt(frame))
                        phase = "segmenting_held_rail"
                        progress(phase, epoch)
                        mask, record = detector.detect(frame)
                        if predicted_pose is not None:
                            record.update(prompt_source="projected_preplace_pose",
                                          T_C_O_predicted=predicted_pose.tolist())
                        if mask is None:
                            raise RuntimeError("SAM returned an empty mask; adjust the box/points with set_sam_prompt.py")
                        else:
                            snapshot = Path(generation) / "mask.png"
                            save_mask_snapshot(directory / snapshot, frame, mask, record)
                            record["mask_snapshot"] = str(snapshot)
                            write_json(directory / generation / "sam_detection.json", record)
                            phase = "initializing_pose"
                            progress(phase, epoch, record)
                            # Release the detector GPU allocation before loading pose networks.
                            detector.close()
                            detector = None
                            pose_config = config
                            if predicted_pose is not None:
                                # Fine alignment is measured in the taught CAD
                                # frame without an additional body-axis bias.
                                pose_config = dict(config, body_orientation_prior={
                                    **config.get("body_orientation_prior", {}), "enabled": False})
                            engine = pose_engine(args, directory, pose_config)
                            pose, record = engine.infer(replace(frame, mask=mask), initialize=True)
                            initial_mask = mask
                            if mode == "held_once":
                                # Model loading/registration may take seconds: report a fresh image.
                                frame = camera.frame()
                                pose, record = engine.infer(frame, initialize=False)
                    else:
                        pose, record = engine.infer(frame, initialize=not engine.initialized)
                    if engine is not None:
                        alignment_text = ""
                        if pose is not None:
                            phase = "aligning_pose"
                            if alignment is None or reference is None:
                                raise ValueError("Cannot display an aligned rail pose without a matching "
                                                 f"goal and alignment policy: {reference_error or 'missing policy'}")
                            # Keep tracking state raw; publish and draw only a
                            # hypothesis that actually passes the per-axis prior.
                            record = alignment.align_record({**record, "T_C_CAD": pose.tolist()}, reference)
                            pose = np.asarray(record["T_C_CAD"])
                            info = record["pose_alignment"]
                            angles = "/".join(f"{v:.1f}" for v in info["relative_xyz_deg"])
                            status = "within 30 deg" if info["within_typical_range"] else "30-45 deg range"
                            alignment_text = f"Aligned | goal-relative XYZ deg: {angles} | {status}"
                        save_preview(directory, frame, engine.mesh, pose, None,
                                     alignment_text or record.get("reason", ""))
                        if pose is not None and mode == "held_once":
                            snapshot = Path(generation) / "pose.png"
                            save_pose_snapshot(directory / snapshot, frame, engine.mesh, pose, reason=alignment_text)
                            record["pose_snapshot"] = str(snapshot)
            except (ValueError, RuntimeError, OSError) as error:
                pose = None
                if phase == "aligning_pose" and frame is not None:
                    save_preview(directory, frame, None, None, reason="No aligned pose: " + str(error))
                if mode == "held_once":
                    once_complete, engine = True, None
                    if detector is not None:
                        detector.close()
                        detector = None
                    gc.collect()
                    import torch
                    torch.cuda.empty_cache()
                    camera.close()
                    camera = Camera(args.camera_serial, rgb_only=True)
                    using_rgb_only = True
                    record = {"valid": False, "reason": "held_pose_unavailable", "phase": phase, "detail": str(error)}
                elif engine is not None:
                    engine.reset()
                    # A lost pose needs a new prompted mask.
                    engine = None
                    # Load SAM on the next iteration, after this exception's
                    # traceback has released its reference to the pose engine.
                if mode != "held_once":
                    record = {"valid": False, "detected_in_roi": False, "reason": "perception_frame_failed", "detail": str(error)}
            record.update(identity, generation=generation,
                          timestamp_ns=int(frame.timestamp_ns) if frame is not None else time.monotonic_ns(),
                          context_epoch=epoch,
                          timestamp_source=frame.metadata["timestamp_source"] if frame is not None else "camera_failure",
                          output_monotonic_ns=time.monotonic_ns())
            if predicted_pose is not None:
                record.update(pose_source="preplace_registration", reestimate_request=True,
                              body_orientation_prior_enabled=False)
            write_json(directory / "latest.json", record)
            if mode != "receive":
                # Keep every accepted/rejected estimate for jitter diagnosis,
                # including the three frames preceding a controller pause.
                with (directory / generation / "pose_records.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
            if pose is not None and mode != "held_once":
                writer, attempt = save_frame(directory / generation, camera, engine,
                    SimpleNamespace(initial_mask=initial_mask), frame, pose, writer, attempt)
            if pose is not None and mode == "held_once":
                once_complete, engine = True, None
                gc.collect()
                import torch
                torch.cuda.empty_cache()
                # Keep a working RGB-D pipeline open for T; idle previews read
                # color only and do not register depth or run either model.
    except Exception as error:
        write_json(directory / "ready.json", {"ready": False, "error": str(error)})
        raise
    finally:
        if detector is not None:
            detector.close()
        if camera is not None:
            camera.close()


def replay(args, config):
    """Explicit simulated palm; stored manual masks are discarded, never a live grasp input."""
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    engine, tracker = components(args, config, directory)
    from rail_pose.data import Scene
    from rail_preinsert.core import transform
    scene = Scene(args.scene.resolve())
    palm = transform(read_json(args.palm_transform)["T_C_G"])
    camera = type("ReplayCamera", (), {"serial": scene.manifest["camera"].get("serial", "offline")})()
    writer, attempt, records = None, 0, []
    for index in range(len(scene)):
        source = scene.frame(index)
        frame = replace(source, rgb=source.rgb.copy(), depth_m=source.depth_m.copy(), mask=None)
        pose, record = tracker.step(frame, palm, 0)
        record.update(frame_id=frame.frame_id, timestamp_ns=frame.timestamp_ns, replay_only=True,
                      simulated_palm=True, stored_mask_used=False)
        records.append(record)
        if pose is not None:
            writer, attempt = save_frame(directory, camera, engine, tracker, frame, pose, writer, attempt)
        from rail_receive.visualization import save_preview
        save_preview(directory, frame, engine.mesh, pose, config.get("pose_preview"), record.get("reason", ""))
    result = {"replay_only": True, "simulated_palm": True, "stored_mask_used": False, "records": records}
    write_json(directory / "observation.json", result)
    print(json.dumps({"valid_frames": sum(r.get("valid") is True for r in records),
                      "total_frames": len(records), "output": str(directory)}, indent=2))
    return 0 if any(r.get("valid") for r in records) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "receive.yaml")
    parser.add_argument("--foundationpose-root", type=Path)
    parser.add_argument("--mesh", type=Path)
    parser.add_argument("--camera-serial")
    parser.add_argument("--serve", type=Path)
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--palm-transform", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pose-references", type=Path, help="离线叠图的 T_C_O_goal / T_C_O_above JSON")
    args = parser.parse_args()
    config = read_json(args.config) if args.config.suffix == ".json" else load_config(args.config)
    if args.pose_references:
        config["pose_preview"] = read_json(args.pose_references)
    args.foundationpose_root = args.foundationpose_root or Path(config["foundationpose_root"])
    args.mesh = args.mesh or Path(config["mesh"])
    if args.scene:
        if not args.palm_transform or not args.output or args.serve:
            parser.error("Replay requires --palm-transform and --output, and cannot use --serve")
        return replay(args, config)
    if not args.serve or not args.camera_serial:
        parser.error("Live worker requires --serve and --camera-serial")
    serve(args, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
