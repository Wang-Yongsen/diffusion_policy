"""Run in the shared DP environment; owns one rectified head RGB-D stream."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import time


from rail_common.core import digest, read_json, write_json
from rail_common.camera import Camera


def measure(camera, engine, directory, samples):
    from rail_pose.ui import annotate

    directory.mkdir(parents=True, exist_ok=False)
    frame = camera.frame()
    mask = annotate(frame.rgb, title="Rail: mark visible rail; exclude hands/holes; Enter accept")
    if mask is None:
        raise RuntimeError("Mask annotation cancelled")
    frame = replace(frame, mask=mask)
    engine.reset()
    records = []
    for index in range(samples):
        if index:
            frame = camera.frame()
        T, record = engine.infer(frame, initialize=index == 0)
        record["timestamp_source"] = frame.metadata.get("timestamp_source", "offline_scene_time")
        record["output_monotonic_ns"] = time.monotonic_ns()
        records.append(record)
    return {
        "camera_serial": camera.serial,
        "unit": "m",
        "frame": "T_C_CAD",
        "mesh_sha256": digest(engine.mesh_path),
        "test_data": False,
        "records": records,
        "replay_only": bool(getattr(camera, "offline", False)),
        "source_directory": str(directory),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundationpose-root", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--camera-serial", required=True)
    parser.add_argument("--serve", type=Path, required=True)
    args = parser.parse_args()
    directory = args.serve.resolve()
    root = args.foundationpose_root.resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "workspace/src"))
    camera = None
    try:
        from rail_pose.engine import PoseEngine

        config = read_json(root / "workspace/configs/default.json")
        engine = PoseEngine(args.mesh.resolve(), config, directory / "debug")
        camera = Camera(args.camera_serial)
        write_json(directory / "ready.json", {"ready": True, "camera_serial": args.camera_serial})
        previous = None
        while True:
            path = directory / "request.json"
            if not path.is_file():
                time.sleep(0.03)
                continue
            request = read_json(path)
            if request["id"] == previous:
                time.sleep(0.03)
                continue
            previous = request["id"]
            if request["command"] == "shutdown":
                break
            try:
                if request["command"] != "measure" or not 1 <= int(request["samples"]) <= 10:
                    raise ValueError("Unsupported perception request")
                observation = measure(camera, engine, directory / previous, int(request["samples"]))
                write_json(
                    directory / "response.json",
                    {"id": previous, "ok": True, "observation": observation},
                )
            except Exception as error:
                write_json(
                    directory / "response.json", {"id": previous, "ok": False, "error": str(error)}
                )
    except Exception as error:
        write_json(directory / "ready.json", {"ready": False, "error": str(error)})
        raise
    finally:
        if camera is not None:
            camera.close()


if __name__ == "__main__":
    main()
