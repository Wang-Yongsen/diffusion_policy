"""Record one RealSense D436 RGB-D episode into datasets/rail_task_episode_<local time> (workspace scene format)."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from queue import Queue
import shutil
import sys
from threading import Lock, Thread
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rail_pose.common import load_config, resolve, save_json  # noqa: E402
from rail_pose.data import Frame, depth_to_metres  # noqa: E402

PREFIX = "rail_task_episode_"
# librealsense 2.59 projects all three models with the OpenCV forward polynomial.
RECTIFIABLE = {"brown_conrady", "modified_brown_conrady", "inverse_brown_conrady"}


def episode_directory(root: Path, when: datetime) -> Path:
    return root / f"{PREFIX}{when.strftime('%Y%m%d_%H%M%S')}"


def rectification(K: np.ndarray, model: str, coefficients, size):
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if not np.any(np.abs(coefficients) > 1e-8):
        return None
    if model.removeprefix("distortion.") not in RECTIFIABLE:
        raise RuntimeError(f"RGB distortion {model} needs an explicit rectification model")
    return cv2.initUndistortRectifyMap(K, coefficients, None, K, size, cv2.CV_32FC1)


class EpisodeWriter:
    """Encodes frames on worker threads; scene.json lists the successfully written frames in capture order."""
    def __init__(self, directory: Path, camera: dict, max_depth_m: float, workers=4, capacity=90):
        directory.mkdir(parents=True, exist_ok=False)
        for name in ("rgb", "depth", "masks"):
            (directory / name).mkdir()
        self.directory, self.max_depth_m = directory, max_depth_m
        self.manifest = {
            "format_version": 1, "color_order": "RGB", "depth_unit": "m",
            "depth_format": "float32_npy", "max_depth_m": max_depth_m,
            "camera": camera, "frames": [],
        }
        save_json(directory / "scene.json", self.manifest)
        self.queue = Queue(capacity)
        self.lock = Lock()
        self.written: set[str] = set()
        self.errors: list[Exception] = []
        self.max_backlog = 0
        self.threads = [Thread(target=self._work, name=f"episode-writer-{i}", daemon=True) for i in range(workers)]
        for thread in self.threads:
            thread.start()

    @property
    def backlog(self) -> int:
        return self.queue.qsize()

    def write(self, frame: Frame) -> None:
        if self.errors:
            raise RuntimeError(f"Frame writing failed: {self.errors[0]}") from self.errors[0]
        self.manifest["frames"].append({
            "frame_id": frame.frame_id, "timestamp_ns": int(frame.timestamp_ns),
            "K": frame.K.tolist(), "metadata": frame.metadata,
        })
        self.queue.put(frame)
        self.max_backlog = max(self.max_backlog, self.queue.qsize())

    def _work(self):
        while (frame := self.queue.get()) is not None:
            try:
                frame.validate(self.max_depth_m)
                bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
                if not cv2.imwrite(str(self.directory / "rgb" / f"{frame.frame_id}.png"), bgr,
                                   [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                    raise OSError(f"could not write RGB frame {frame.frame_id}")
                np.save(self.directory / "depth" / f"{frame.frame_id}.npy", frame.depth_m.astype(np.float32),
                        allow_pickle=False)
                with self.lock:
                    self.written.add(frame.frame_id)
            except Exception as exc:
                with self.lock:
                    self.errors.append(exc)

    def close(self, recording: dict) -> int:
        for _ in self.threads:
            self.queue.put(None)
        for thread in self.threads:
            thread.join()
        self.manifest["frames"] = [x for x in self.manifest["frames"] if x["frame_id"] in self.written]
        self.manifest["recording"] = {**recording, "frames": len(self.manifest["frames"]),
                                      "write_failures": len(self.errors), "max_writer_backlog": self.max_backlog}
        save_json(self.directory / "scene.json", self.manifest)
        if self.errors:
            raise RuntimeError(f"{len(self.errors)} frames failed to write: {self.errors[0]}") from self.errors[0]
        return len(self.manifest["frames"])


class D436Camera:
    """Delivers every aligned, rectified RGB-D frameset in order, unlike the latest-frame RealSenseCamera."""
    def __init__(self, serial=None, width=640, height=480, fps=30, max_depth_m=3.0, warmup=15):
        self.serial, self.width, self.height, self.fps = serial, width, height, fps
        self.max_depth_m, self.warmup = max_depth_m, warmup
        self.pipeline = None
        self.last_number = None
        self.dropped = 0
        before = time.monotonic_ns()
        wall = time.time_ns()
        after = time.monotonic_ns()
        self.wall_minus_monotonic_ns = wall - (before + after) // 2

    def start(self):
        import pyrealsense2 as rs
        self.rs = rs
        context = rs.context()
        devices = {d.get_info(rs.camera_info.serial_number): d.get_info(rs.camera_info.name)
                   for d in context.query_devices()}
        if not devices:
            raise RuntimeError("No RealSense camera found; connect the D436 to this PC")
        if self.serial and self.serial not in devices:
            raise RuntimeError(f"Requested camera {self.serial} not found; available: {devices}")
        if not self.serial and len(devices) != 1:
            raise RuntimeError(f"Select --serial when more than one camera is connected: {devices}")
        serial = self.serial or next(iter(devices))
        if "D436" not in devices[serial]:
            print(f"WARNING: camera {serial} reports model '{devices[serial]}', not D436", file=sys.stderr)
        settings = rs.config()
        settings.enable_device(serial)
        settings.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        settings.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.pipeline = rs.pipeline(context)
        try:
            profile = self.pipeline.start(settings)
            device = profile.get_device()
            for sensor in device.query_sensors():
                if sensor.supports(rs.option.global_time_enabled):
                    sensor.set_option(rs.option.global_time_enabled, 1)
            self.depth_scale = device.first_depth_sensor().get_depth_scale()
            color = profile.get_stream(rs.stream.color).as_video_stream_profile()
            depth = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            intr = color.get_intrinsics()
            self.K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float64)
            self.remap = rectification(self.K, str(intr.model), intr.coeffs, (self.width, self.height))
            extr = depth.get_extrinsics_to(color)
            self.metadata = {
                "serial": serial, "model": devices[serial],
                "firmware": device.get_info(rs.camera_info.firmware_version),
                "width": self.width, "height": self.height, "fps": self.fps, "K": self.K.tolist(),
                "depth_scale_m": self.depth_scale, "depth_aligned_to": "RGB",
                "depth_registration": "librealsense rs.align to color, then the RGB rectification map (nearest)",
                "original_color_distortion": {"model": str(intr.model), "coefficients": list(intr.coeffs)},
                "rgb_rectified": self.remap is not None,
                "depth_to_color": {"rotation_column_major": list(extr.rotation), "translation_m": list(extr.translation)},
            }
            self.align = rs.align(rs.stream.color)
            for _ in range(self.warmup):
                self.pipeline.wait_for_frames(5000)
            return self
        except Exception:
            self.close()
            raise

    def read(self, timeout_ms=2000) -> Frame | None:
        frames = self.align.process(self.pipeline.wait_for_frames(timeout_ms))
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color or not depth:
            return None
        number = color.get_frame_number()
        if self.last_number is not None:
            if number <= self.last_number:
                return None
            self.dropped += number - self.last_number - 1
        self.last_number = number
        rgb = np.asanyarray(color.get_data()).copy()
        depth_m = depth_to_metres(np.asanyarray(depth.get_data()), self.depth_scale, self.max_depth_m)
        if self.remap is not None:
            rgb = cv2.remap(rgb, *self.remap, interpolation=cv2.INTER_LINEAR)
            depth_m = cv2.remap(depth_m, *self.remap, interpolation=cv2.INTER_NEAREST)
        received = time.monotonic_ns()
        timestamp, timestamp_source = received, "host_received_monotonic"
        if color.get_frame_timestamp_domain() == self.rs.timestamp_domain.global_time:
            timestamp = round(color.get_timestamp() * 1_000_000) - self.wall_minus_monotonic_ns
            timestamp_source = "sdk_global_timestamp_mapped_to_host_monotonic"
            if not received - 5_000_000_000 <= timestamp <= received + 10_000_000:
                raise RuntimeError("SDK/global host time mapping is inconsistent")
        return Frame(f"{number:08d}", rgb, depth_m, self.K.copy(), timestamp, metadata={
            "received_monotonic_ns": received, "timestamp_source": timestamp_source,
            "color_frame_number": number, "depth_frame_number": depth.get_frame_number(),
            "color_timestamp_ms": color.get_timestamp(), "depth_timestamp_ms": depth.get_timestamp(),
            "color_timestamp_domain": str(color.get_frame_timestamp_domain()),
            "depth_timestamp_domain": str(depth.get_frame_timestamp_domain())})

    def close(self):
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass
            self.pipeline = None


def preview_image(frame: Frame, status: str, recording: bool, depth_view_m=2.0) -> np.ndarray:
    depth = np.clip(frame.depth_m / depth_view_m * 255, 0, 255).astype(np.uint8)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_TURBO)
    depth[frame.depth_m <= 0] = 0
    image = np.hstack([cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR), depth])
    colour = (0, 0, 255) if recording else (0, 200, 255)
    if recording:
        cv2.circle(image, (20, 22), 8, colour, -1)
    cv2.putText(image, status, (36, 28), cv2.FONT_HERSHEY_SIMPLEX, .6, colour, 2, cv2.LINE_AA)
    return image


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--serial", help="camera serial; required when several RealSense cameras are connected")
    result.add_argument("--root", default="datasets", help="parent directory, relative to workspace/")
    result.add_argument("--duration", type=float, help="stop after this many seconds")
    result.add_argument("--frames", type=int, help="stop after this many frames")
    result.add_argument("--width", type=int, default=640)
    result.add_argument("--height", type=int, default=480)
    result.add_argument("--fps", type=int, default=30)
    result.add_argument("--preview", action="store_true", help="show RGB and depth; q/Esc stops")
    result.add_argument("--wait-start", action="store_true", help="with --preview, start recording on space")
    result.add_argument("--workers", type=int, default=4, help="PNG/NPY writer threads")
    result.add_argument("--config", default="configs/default.json")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.wait_start and not args.preview:
        raise SystemExit("--wait-start needs --preview")
    if (args.duration is not None and args.duration <= 0) or (args.frames is not None and args.frames < 1):
        raise SystemExit("--duration and --frames must be positive")
    if args.preview:
        from rail_pose.ui import require_display
        require_display()
    max_depth_m = load_config(args.config)["max_depth_m"]
    camera = D436Camera(args.serial, args.width, args.height, args.fps, max_depth_m)
    window = "D436 episode | space: start  q/Esc: stop"
    writer, started, started_at, stop_reason = None, None, None, "frames_or_duration"
    camera.start()
    print(f"Camera {camera.metadata['model']} {camera.metadata['serial']} ready; "
          f"RGB rectified: {camera.metadata['rgb_rectified']}", flush=True)
    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue
            if writer is None and not args.wait_start:
                started_at = datetime.now()
                writer = EpisodeWriter(episode_directory(resolve(args.root), started_at), camera.metadata,
                                       max_depth_m, args.workers)
                started, dropped_before = time.monotonic(), camera.dropped
                print(f"Recording to {writer.directory}", flush=True)
            if writer is not None:
                writer.write(frame)
            count = len(writer.manifest["frames"]) if writer else 0
            elapsed = time.monotonic() - started if writer else 0.0
            if args.preview:
                status = (f"REC {elapsed:6.1f}s  {count} frames  dropped {camera.dropped - dropped_before}  "
                          f"backlog {writer.backlog}" if writer else "READY - press space to record")
                cv2.imshow(window, preview_image(frame, status, writer is not None))
                key = cv2.waitKey(1) & 255
                if key in (ord("q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    stop_reason = "user_quit"
                    break
                if key == ord(" ") and writer is None:
                    args.wait_start = False
            if writer and ((args.frames and count >= args.frames) or (args.duration and elapsed >= args.duration)):
                break
    except KeyboardInterrupt:
        stop_reason = "ctrl_c"
    finally:
        stopped = time.monotonic()
        camera.close()
        if args.preview:
            cv2.destroyAllWindows()
    if writer is None:
        print("Nothing recorded")
        return 1
    print("Flushing frames to disk...", flush=True)
    frames = writer.close({
        "started_at_local": started_at.isoformat(timespec="milliseconds"),
        "duration_s": round(stopped - started, 3), "stop_reason": stop_reason,
        "dropped_color_frames": camera.dropped - dropped_before,
    })
    if frames == 0:
        shutil.rmtree(writer.directory)
        print("Nothing recorded")
        return 1
    print(json.dumps({"output": str(writer.directory), **writer.manifest["recording"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
