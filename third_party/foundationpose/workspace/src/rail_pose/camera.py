from __future__ import annotations

from threading import Condition, Event, Thread
import time
import numpy as np

from .data import Frame, depth_to_metres


class RealSenseCamera:
    """One independent RGB-D camera; owns no robot connection."""
    def __init__(self, settings: dict, max_depth_m=3.0):
        self.settings = settings
        self.max_depth_m = max_depth_m
        self.condition = Condition()
        self.stop_event = Event()
        self.latest = None
        self.error = None
        self.pipeline = None
        self.thread = None
        self.metadata = {}
        before = time.monotonic_ns()
        wall = time.time_ns()
        after = time.monotonic_ns()
        self.wall_minus_monotonic_ns = wall - (before + after) // 2

    def start(self):
        import pyrealsense2 as rs
        import cv2
        self.rs, self.cv2 = rs, cv2
        context = rs.context()
        devices = list(context.query_devices())
        requested = self.settings.get("serial")
        serials = [d.get_info(rs.camera_info.serial_number) for d in devices]
        if not devices:
            raise RuntimeError("No RealSense camera found; connect the D435 on the local PC")
        if requested and requested not in serials:
            raise RuntimeError(f"Requested camera {requested} not found; available serials: {serials}")
        if not requested and len(devices) != 1:
            raise RuntimeError("Select --serial when more than one camera is connected")
        serial = requested or serials[0]
        settings = rs.config()
        settings.enable_device(serial)
        width, height, fps = (self.settings[k] for k in ["width", "height", "fps"])
        settings.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        settings.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
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
            coefficients = np.asarray(intr.coeffs, dtype=np.float64)
            self.remap = None
            if np.any(np.abs(coefficients) > 1e-8):
                if intr.model not in (rs.distortion.brown_conrady, rs.distortion.modified_brown_conrady):
                    raise RuntimeError(f"RGB distortion {intr.model} needs an explicit rectification model")
                self.remap = cv2.initUndistortRectifyMap(self.K, coefficients, None, self.K,
                                                       (width, height), cv2.CV_32FC1)
            extr = depth.get_extrinsics_to(color)
            self.metadata = {
                "serial": serial, "model": device.get_info(rs.camera_info.name),
                "width": width, "height": height, "fps": fps, "K": self.K.tolist(),
                "depth_scale_m": self.depth_scale, "depth_aligned_to": "RGB",
                "original_color_distortion": {"model": str(intr.model), "coefficients": intr.coeffs},
                "rgb_rectified": self.remap is not None,
                "depth_to_color": {"rotation_column_major": list(extr.rotation), "translation_m": list(extr.translation)},
            }
            self.align = rs.align(rs.stream.color)
            self.thread = Thread(target=self._capture, name="rail-rgbd", daemon=True)
            self.thread.start()
            self.next(timeout=10)
            return self
        except Exception:
            self.close()
            raise

    def _capture(self):
        warmup = 15
        try:
            while not self.stop_event.is_set():
                frames = self.pipeline.wait_for_frames(1000)
                if warmup:
                    warmup -= 1
                    continue
                frames = self.align.process(frames)
                color, depth = frames.get_color_frame(), frames.get_depth_frame()
                if not color or not depth:
                    continue
                rgb = np.asanyarray(color.get_data()).copy()
                depth_m = depth_to_metres(np.asanyarray(depth.get_data()), self.depth_scale, self.max_depth_m)
                if self.remap is not None:
                    rgb = self.cv2.remap(rgb, *self.remap, interpolation=self.cv2.INTER_LINEAR)
                    depth_m = self.cv2.remap(depth_m, *self.remap, interpolation=self.cv2.INTER_NEAREST)
                received = time.monotonic_ns()
                timestamp = received
                timestamp_source = "host_received_monotonic"
                if color.get_frame_timestamp_domain() == self.rs.timestamp_domain.global_time:
                    timestamp = round(color.get_timestamp() * 1_000_000) - self.wall_minus_monotonic_ns
                    timestamp_source = "sdk_global_timestamp_mapped_to_host_monotonic"
                    if not received - 5_000_000_000 <= timestamp <= received + 10_000_000:
                        raise RuntimeError("SDK/global host time mapping is inconsistent")
                frame = Frame(
                    f"{color.get_frame_number():08d}", rgb, depth_m, self.K.copy(), timestamp,
                    metadata={"received_monotonic_ns": received,
                              "timestamp_source": timestamp_source,
                              "color_timestamp_ms": color.get_timestamp(), "depth_timestamp_ms": depth.get_timestamp(),
                              "color_timestamp_domain": str(color.get_frame_timestamp_domain()),
                              "depth_timestamp_domain": str(depth.get_frame_timestamp_domain())},
                )
                frame.validate(self.max_depth_m)
                with self.condition:
                    self.latest = frame
                    self.condition.notify_all()
        except Exception as exc:
            if not self.stop_event.is_set():
                with self.condition:
                    self.error = exc
                    self.condition.notify_all()

    def next(self, after_id=None, timeout=2.0):
        deadline = time.monotonic() + timeout
        with self.condition:
            if self.stop_event.is_set():
                raise RuntimeError("Camera is closed")
            while self.latest is None or self.latest.frame_id == after_id:
                if self.error:
                    raise RuntimeError(f"Camera acquisition failed: {self.error}") from self.error
                if self.stop_event.is_set():
                    raise RuntimeError("Camera is closed")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("No fresh RGB-D frame")
                self.condition.wait(remaining)
            if self.error:
                raise RuntimeError(f"Camera acquisition failed: {self.error}") from self.error
            return self.latest

    def close(self):
        self.stop_event.set()
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass
            self.pipeline = None
        with self.condition:
            self.condition.notify_all()
        if self.thread and self.thread is not __import__("threading").current_thread():
            self.thread.join(timeout=3)

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.close()
