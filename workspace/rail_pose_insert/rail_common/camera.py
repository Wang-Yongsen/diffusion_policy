"""Rectified head camera with timestamps matched to robot feedback."""

import time
import numpy as np


class Camera:
    def __init__(self, serial, *, rgb_only=False):
        import cv2
        import pyrealsense2 as rs
        from rail_pose.episode import RECTIFIABLE

        self.rs, self.cv2, self.rgb_only = rs, cv2, rgb_only
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
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
                supported = RECTIFIABLE | {
                    "distortion.brown_conrady",
                    "distortion.modified_brown_conrady",
                }
                if str(ci.model) not in supported:
                    raise ValueError(f"Unsupported RGB distortion: {ci.model}")
                self.remap = cv2.initUndistortRectifyMap(
                    self.K, np.asarray(ci.coeffs), None, self.K, (640, 480), cv2.CV_32FC1
                )
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
            self.close()
            raise

    def frame(self, *, rgb_only=None):
        from rail_pose.data import Frame, depth_to_metres

        rs = self.rs
        color_only = self.rgb_only if rgb_only is None else rgb_only
        deadline = time.monotonic() + 2.0
        while True:
            remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
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
        stamp = round(color.get_timestamp() * 1e6) - self.offset_ns
        received = time.monotonic_ns()
        if not received - 2_000_000_000 <= stamp <= received + 10_000_000:
            raise ValueError("RealSense/global host clock mapping is inconsistent")
        rgb = np.asanyarray(color.get_data()).copy()
        if self.remap is not None:
            rgb = self.cv2.remap(rgb, *self.remap, interpolation=self.cv2.INTER_LINEAR)
        if color_only:
            return Frame(
                str(color.get_frame_number()),
                rgb,
                np.zeros(rgb.shape[:2], np.float32),
                self.K.copy(),
                stamp,
                metadata={
                    "timestamp_source": "sdk_global_timestamp_mapped_to_host_monotonic",
                    "received_monotonic_ns": received,
                    "rgb_only": True,
                },
            )
        from rail_pose.episode import register_depth

        raw_depth = depth_to_metres(np.asanyarray(depth.get_data()), self.scale, 3.0)
        registered = register_depth(raw_depth, self.depth_K, self.K, self.R, self.t, rgb.shape[:2])
        registered[registered > 3.0] = 0
        return Frame(
            str(color.get_frame_number()),
            rgb,
            registered,
            self.K.copy(),
            stamp,
            metadata={
                "timestamp_source": "sdk_global_timestamp_mapped_to_host_monotonic",
                "depth_minus_rgb_ms": offset,
                "received_monotonic_ns": received,
            },
        )

    def close(self):
        try:
            self.pipeline.stop()
        except RuntimeError:
            pass
