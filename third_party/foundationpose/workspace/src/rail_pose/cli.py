from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
import traceback

from .common import WORKSPACE, REPOSITORY, resolve, save_json, load_config, new_run


def parser():
    result = argparse.ArgumentParser(description="Standalone FoundationPose real-CAD workspace")
    result.add_argument("--config", default="configs/default.json")
    sub = result.add_subparsers(dest="command", required=True)
    model = sub.add_parser("prepare-model", help="convert real CAD to a metric mesh")
    model.add_argument("--source", required=True)
    model.add_argument("--source-unit", choices=["m", "mm"])
    model.add_argument("--output", default="input/model/rail.obj")
    demo = sub.add_parser("import-demo", help="convert official mm-PNG data to workspace format")
    demo.add_argument("--source", required=True)
    demo.add_argument("--output", required=True)
    demo.add_argument("--max-frames", type=int, default=30)
    episode = sub.add_parser("import-episode", help="convert a bimanual-teleop recording camera to workspace format")
    episode.add_argument("--source", required=True, help="episode directory containing episode.json")
    episode.add_argument("--camera", default="camera_0")
    episode.add_argument("--output", required=True)
    episode.add_argument("--start", type=int, default=0)
    episode.add_argument("--max-frames", type=int)
    bag = sub.add_parser("import-bag", help="convert a RealSense ROS 2 capture.db3 recording to workspace format")
    bag.add_argument("--source", required=True, help="directory containing capture.db3, or the bag file")
    bag.add_argument("--output", required=True)
    bag.add_argument("--start", type=int, default=0)
    bag.add_argument("--max-frames", type=int)
    for command in ["estimate", "track"]:
        p = sub.add_parser(command, help="run saved metric RGB-D data")
        p.add_argument("--scene", required=True)
        p.add_argument("--mesh")
        p.add_argument("--output")
        p.add_argument("--index", type=int, default=0)
        p.add_argument("--frames", type=int, default=30)
        p.add_argument("--reverse", action="store_true", help="track backward in time from --index")
        p.add_argument("--batch-size", type=int)
        p.add_argument("--reference", action="store_true", help="use unmodified upstream predictors")
        p.add_argument("--preview", action="store_true", help="show pose overlays in a desktop window")
    annotation = sub.add_parser("annotate", help="draw a saved-frame target mask in a desktop")
    annotation.add_argument("--scene", required=True)
    annotation.add_argument("--index", type=int, default=0)
    annotation.add_argument("--polygon", help="headless mask from pixel vertices 'x,y x,y ...' on the saved RGB")
    for command in ["capture", "live"]:
        p = sub.add_parser(command, help="use the local PC's RealSense camera")
        p.add_argument("--serial")
        p.add_argument("--output")
        if command == "capture":
            p.add_argument("--frames", type=int, default=120)
            p.add_argument("--preview", action="store_true")
            p.add_argument("--annotate-first", action="store_true")
        else:
            p.add_argument("--mesh")
            p.add_argument("--batch-size", type=int)
    sub.add_parser("check-env", help="verify core imports, GPU and required model files")
    package = sub.add_parser("package", help="build the local-PC deployment archive")
    package.add_argument("--output", default="deployment/foundationpose-rail.tar.gz")
    return result


def run_offline(args, config):
    import cv2
    import numpy as np
    from .data import Scene
    from .engine import PoseEngine, failure_record
    from .geometry import overlay
    if args.preview:
        from .ui import OfflinePreview, require_display
        require_display()
    scene = Scene(resolve(args.scene))
    output = new_run(args.command, args.output)
    config["pose_batch_size"] = args.batch_size or config["pose_batch_size"]
    if config["pose_batch_size"] < 1:
        raise ValueError("--batch-size must be positive")
    engine = PoseEngine(resolve(args.mesh or config["model"]), config, output / "debug", low_memory=not args.reference)
    if not 0 <= args.index < len(scene):
        raise ValueError(f"--index must be within the scene's {len(scene)} frames")
    if args.command == "estimate":
        indices = [args.index]
    elif args.reverse:
        indices = range(args.index, max(-1, args.index - args.frames), -1)
    else:
        indices = range(args.index, min(len(scene), args.index + args.frames))
    records = []
    stopped_by_user = False
    (output / "overlays").mkdir()
    (output / "poses").mkdir()
    save_json(output / "settings.json", {"config": config, "scene": str(scene.directory),
                                           "mesh": str(engine.mesh_path), "reference": args.reference,
                                           "first_index": args.index, "reverse": bool(getattr(args, "reverse", False))})
    preview_context = OfflinePreview() if args.preview else nullcontext()
    with preview_context as preview, (output / "poses.jsonl").open("w", encoding="utf-8") as stream:
        for index in indices:
            frame = scene.frame(index)
            try:
                pose, record = engine.infer(frame)
                np.savetxt(output / "poses" / f"{frame.frame_id}.txt", pose)
            except (ValueError, RuntimeError) as exc:
                engine.reset()
                pose, record = None, failure_record(frame, exc)
            image = overlay(frame.rgb, engine.mesh, pose, frame.K,
                            f"{frame.frame_id} {record.get('mode', 'FAILED')}")
            cv2.imwrite(str(output / "overlays" / f"{frame.frame_id}.png"), image)
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            records.append(record)
            if preview is not None and not preview.show(image):
                stopped_by_user = True
                break
        if preview is not None and records and not stopped_by_user:
            preview.hold()
    valid = [x for x in records if x["valid"]]
    tracks = [x["processing_s"] for x in valid if x["mode"] == "track"]
    summary = {"frames": len(records), "valid_frames": len(valid),
               "requested_frames": len(indices), "stopped_by_user": stopped_by_user,
               "failed_frames": len(records) - len(valid),
               "initialization_s": [x["processing_s"] for x in valid if x["mode"] == "register"],
               "mean_tracking_s": float(np.mean(tracks)) if tracks else None,
               "mean_tracking_fps": 1 / float(np.mean(tracks)) if tracks else None,
               "torch_peak_allocated_mib": max((x["torch_peak_allocated_mib"] for x in valid), default=0),
               "torch_peak_reserved_mib": max((x["torch_peak_reserved_mib"] for x in valid), default=0),
               "physical_accuracy_verified": False}
    save_json(output / "summary.json", summary)
    print(json.dumps({"output": str(output), **summary}, ensure_ascii=False, indent=2))
    return 0 if len(valid) == len(records) and records else 1


def run_capture(args, config):
    import numpy as np
    from .camera import RealSenseCamera
    from .data import SceneWriter
    from .ui import annotate, require_display
    settings = dict(config["camera"])
    if args.serial:
        settings["serial"] = args.serial
    if args.preview or args.annotate_first:
        require_display()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    output = resolve(args.output or f"datasets/capture-{stamp}")
    last_id = None
    with RealSenseCamera(settings, config["max_depth_m"]) as camera:
        writer = SceneWriter(output, camera.metadata, config["max_depth_m"])
        try:
            for index in range(args.frames):
                frame = camera.next(after_id=last_id)
                last_id = frame.frame_id
                if index == 0 and args.annotate_first:
                    mask = annotate(frame.rgb)
                    if mask is None:
                        raise RuntimeError("Mask annotation cancelled")
                    frame = replace(frame, mask=mask)
                writer.write(frame)
                if args.preview:
                    import cv2
                    cv2.imshow("RGB-D capture | q: stop", cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 255 == ord("q"):
                        break
        finally:
            if args.preview:
                import cv2
                cv2.destroyAllWindows()
    print(json.dumps({"output": str(output), "frames": len(writer.seen)}))
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "prepare-model":
            from .model import prepare_model
            result = prepare_model(resolve(args.source), resolve(args.output), args.source_unit)
            print(json.dumps({k: v for k, v in result.items() if k != "cad_audit"}, ensure_ascii=False, indent=2))
            return 0
        if args.command == "import-demo":
            from .data import import_official_scene
            print(import_official_scene(resolve(args.source), resolve(args.output), args.max_frames))
            return 0
        if args.command == "import-episode":
            from .episode import import_episode
            print(import_episode(resolve(args.source), resolve(args.output), args.camera,
                                 config["max_depth_m"], args.start, args.max_frames))
            return 0
        if args.command == "import-bag":
            from .bag import import_realsense_bag
            print(import_realsense_bag(resolve(args.source), resolve(args.output),
                                       config["max_depth_m"], args.start, args.max_frames))
            return 0
        if args.command in ["estimate", "track"]:
            return run_offline(args, config)
        if args.command == "annotate":
            from PIL import Image
            from .data import Scene
            from .ui import annotate
            scene = Scene(resolve(args.scene))
            frame = scene.frame(args.index)
            if args.polygon:
                import cv2
                import numpy as np
                points = np.array([[float(v) for v in p.split(",")] for p in args.polygon.split()])
                if points.ndim != 2 or points.shape != (len(points), 2) or len(points) < 3:
                    raise ValueError("--polygon needs at least three 'x,y' vertices")
                mask = np.zeros(frame.rgb.shape[:2], np.uint8)
                cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
                mask = mask.astype(bool)
            else:
                mask = annotate(frame.rgb)
            if mask is None or not mask.any():
                return 1
            path = scene.directory / "masks" / f"{frame.frame_id}.png"
            Image.fromarray(mask.astype("uint8") * 255).save(path)
            print(path)
            return 0
        if args.command == "capture":
            if args.frames < 1:
                raise ValueError("--frames must be positive")
            return run_capture(args, config)
        if args.command == "live":
            from .camera import RealSenseCamera
            from .engine import PoseEngine
            from .ui import run_live, require_display
            require_display()
            config["pose_batch_size"] = args.batch_size or config["pose_batch_size"]
            output = new_run("live", args.output)
            engine = PoseEngine(resolve(args.mesh or config["model"]), config, output / "debug")
            settings = dict(config["camera"])
            if args.serial:
                settings["serial"] = args.serial
            with RealSenseCamera(settings, config["max_depth_m"]) as camera:
                run_live(camera, engine, output, config["max_frame_age_s"])
            print(output)
            return 0
        if args.command == "check-env":
            from .diagnostics import check_environment
            report = check_environment()
            save_json(WORKSPACE / "outputs/environment-check.json", report)
            print(json.dumps(report, indent=2))
            return 0 if report["ready"] else 1
        if args.command == "package":
            from .packaging import package
            print(package(resolve(args.output)))
            return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 1
