"""Run in the shared DP environment; owns one rectified head RGB-D stream."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import time

import numpy as np

from rail_preinsert.core import digest, read_json, write_json


class Camera:
    def __init__(self, serial, *, rgb_only=False):
        import cv2
        import pyrealsense2 as rs
        from rail_pose.episode import RECTIFIABLE
        self.rs, self.cv2, self.rgb_only = rs, cv2, rgb_only
        self.pipeline = rs.pipeline()
        config = rs.config(); config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        if not rgb_only:
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.offset_ns = time.time_ns() - time.monotonic_ns()
        try:
            profile = self.pipeline.start(config)
            for sensor in profile.get_device().query_sensors():
                if sensor.supports(rs.option.global_time_enabled):
                    sensor.set_option(rs.option.global_time_enabled, 1)
            color = profile.get_stream(rs.stream.color).as_video_stream_profile()
            ci = color.get_intrinsics()
            self.K = np.array([[ci.fx, 0, ci.ppx], [0, ci.fy, ci.ppy], [0, 0, 1]])
            self.remap = None
            if np.max(np.abs(ci.coeffs)) > 1e-8:
                supported = RECTIFIABLE | {"distortion.brown_conrady", "distortion.modified_brown_conrady"}
                if str(ci.model) not in supported:
                    raise ValueError(f"Unsupported RGB distortion: {ci.model}")
                self.remap = cv2.initUndistortRectifyMap(self.K, np.asarray(ci.coeffs), None,
                                                       self.K, (640, 480), cv2.CV_32FC1)
            if not rgb_only:
                depth = profile.get_stream(rs.stream.depth).as_video_stream_profile()
                di = depth.get_intrinsics()
                if np.max(np.abs(di.coeffs)) > 1e-8:
                    raise ValueError("Depth distortion requires an explicit rectification model")
                self.depth_K = np.array([[di.fx, 0, di.ppx], [0, di.fy, di.ppy], [0, 0, 1]])
                extrinsics = depth.get_extrinsics_to(color)
                self.R = np.asarray(extrinsics.rotation).reshape(3, 3).T
                self.t = np.asarray(extrinsics.translation)
                self.scale = profile.get_device().first_depth_sensor().get_depth_scale()
            self.serial = serial
            for index in range(15):
                # Cold RGB-D startup can exceed 2 s even though subsequent frames
                # arrive normally. Keep the longer wait confined to startup.
                self.pipeline.wait_for_frames(10000 if index == 0 else 2000)
        except BaseException:
            self.close(); raise

    def frame(self, *, rgb_only=None):
        from rail_pose.data import Frame, depth_to_metres
        rs = self.rs
        color_only = self.rgb_only if rgb_only is None else rgb_only
        deadline = time.monotonic() + 2.
        while True:
            remaining_ms = max(1, int((deadline-time.monotonic())*1000))
            frames = self.pipeline.wait_for_frames(remaining_ms)
            color = frames.get_color_frame()
            if not color:
                raise ValueError("Incomplete color frameset")
            if color_only:
                break
            depth = frames.get_depth_frame()
            if not depth or depth.get_frame_timestamp_domain() != rs.timestamp_domain.global_time:
                raise ValueError("RGB-D must use GLOBAL_TIME for robot-state synchronization")
            offset = depth.get_timestamp() - color.get_timestamp()
            if abs(offset) <= 5:
                break
            # A stream switch can yield an unmatched first frameset. Discard it
            # within the existing timeout; never register mismatched RGB/depth.
            if time.monotonic() >= deadline:
                raise ValueError("RGB/depth timestamp separation exceeds 5 ms")
        if color.get_frame_timestamp_domain() != rs.timestamp_domain.global_time:
            raise ValueError("Camera must use GLOBAL_TIME for robot-state synchronization")
        stamp = round(color.get_timestamp()*1e6) - self.offset_ns
        received = time.monotonic_ns()
        if not received - 2_000_000_000 <= stamp <= received + 10_000_000:
            raise ValueError("RealSense/global host clock mapping is inconsistent")
        rgb = np.asanyarray(color.get_data()).copy()
        if self.remap is not None:
            rgb = self.cv2.remap(rgb, *self.remap, interpolation=self.cv2.INTER_LINEAR)
        if color_only:
            return Frame(str(color.get_frame_number()), rgb, np.zeros(rgb.shape[:2], np.float32),
                         self.K.copy(), stamp, metadata={
                             "timestamp_source": "sdk_global_timestamp_mapped_to_host_monotonic",
                             "received_monotonic_ns": received, "rgb_only": True})
        from rail_pose.episode import register_depth
        raw_depth = depth_to_metres(np.asanyarray(depth.get_data()), self.scale, 3.)
        registered = register_depth(raw_depth, self.depth_K, self.K, self.R, self.t, rgb.shape[:2])
        registered[registered > 3.] = 0
        return Frame(str(color.get_frame_number()), rgb, registered, self.K.copy(), stamp,
                     metadata={"timestamp_source": "sdk_global_timestamp_mapped_to_host_monotonic",
                               "depth_minus_rgb_ms": offset, "received_monotonic_ns": received})

    def close(self):
        try:
            self.pipeline.stop()
        except RuntimeError:
            pass


def measure(camera, engine, directory, samples):
    from PIL import Image
    from rail_pose.geometry import overlay
    from rail_pose.ui import annotate
    from rail_pose.data import SceneWriter
    directory.mkdir(parents=True, exist_ok=False)
    frame = camera.frame()
    mask = frame.mask if getattr(camera, "offline", False) else annotate(
        frame.rgb, title="Rail: mark visible rail; exclude hands/holes; Enter accept")
    if mask is None:
        raise RuntimeError("Mask annotation cancelled")
    frame = replace(frame, mask=mask)
    writer = SceneWriter(directory / "scene", {"serial": camera.serial, "K": frame.K.tolist()})
    engine.reset()
    records = []
    for index in range(samples):
        if index:
            frame = camera.frame()
        writer.write(frame)
        T, record = engine.infer(frame, initialize=index == 0)
        record["timestamp_source"] = frame.metadata.get("timestamp_source", "offline_scene_time")
        record["output_monotonic_ns"] = time.monotonic_ns()
        records.append(record)
        Image.fromarray(overlay(frame.rgb, engine.mesh, T, frame.K)).save(directory / f"pose_{index:02d}.png")
    return {"camera_serial": camera.serial, "unit": "m", "frame": "T_C_CAD",
            "mesh_sha256": digest(engine.mesh_path), "test_data": False, "records": records,
            "replay_only": bool(getattr(camera, "offline", False)),
            "source_directory": str(directory)}


class SceneCamera:
    offline = True

    def __init__(self, path, index):
        from rail_pose.data import Scene
        self.scene, self.index = Scene(path), index
        self.serial = self.scene.manifest["camera"].get("serial", "offline-scene")

    def frame(self):
        if self.index >= len(self.scene):
            raise ValueError("Offline RGB-D scene ended")
        value = self.scene.frame(self.index)
        self.index += 1
        return replace(value, rgb=value.rgb.copy(), depth_m=value.depth_m.copy())

    def close(self):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundationpose-root", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--camera-serial")
    parser.add_argument("--serve", type=Path)
    parser.add_argument("--scene", type=Path, help="离线 RGB-D 场景；不访问相机或机器人")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.scene:
        if args.serve or not args.output or not 1 <= args.samples <= 10:
            parser.error("--scene requires --output and 1..10 samples; cannot combine with --serve")
        directory = args.output.resolve(); directory.mkdir(parents=True, exist_ok=False)
    else:
        if not args.serve or not args.camera_serial:
            parser.error("Live worker requires --serve and --camera-serial")
        directory = args.serve.resolve()
    root = args.foundationpose_root.resolve()
    sys.path.insert(0, str(root)); sys.path.insert(0, str(root / "workspace/src"))
    camera = None
    try:
        from rail_pose.engine import PoseEngine
        config = read_json(root / "workspace/configs/default.json")
        engine = PoseEngine(args.mesh.resolve(), config, directory / "debug")
        if args.scene:
            camera = SceneCamera(args.scene.resolve(), args.index)
            observation = measure(camera, engine, directory / "measurement", args.samples)
            write_json(directory / "observation.json", observation)
            print(f"Offline observation saved: {directory / 'observation.json'}", flush=True)
            return
        camera = Camera(args.camera_serial)
        write_json(directory / "ready.json", {"ready": True, "camera_serial": args.camera_serial})
        previous = None
        while True:
            path = directory / "request.json"
            if not path.is_file():
                time.sleep(.03); continue
            request = read_json(path)
            if request["id"] == previous:
                time.sleep(.03); continue
            previous = request["id"]
            if request["command"] == "shutdown":
                break
            try:
                if request["command"] != "measure" or not 1 <= int(request["samples"]) <= 10:
                    raise ValueError("Unsupported perception request")
                observation = measure(camera, engine, directory / previous, int(request["samples"]))
                write_json(directory / "response.json", {"id": previous, "ok": True, "observation": observation})
            except Exception as error:
                write_json(directory / "response.json", {"id": previous, "ok": False, "error": str(error)})
    except Exception as error:
        write_json(directory / "ready.json", {"ready": False, "error": str(error)})
        raise
    finally:
        if camera is not None:
            camera.close()


if __name__ == "__main__":
    main()
