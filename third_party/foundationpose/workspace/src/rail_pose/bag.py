"""Convert a RealSense ROS 2 sqlite bag (capture.db3) into a workspace RGB-D scene."""
from __future__ import annotations

import json
import sqlite3
import struct
from pathlib import Path

import numpy as np

from .data import Frame, SceneWriter, depth_to_metres
from .episode import RECTIFIABLE, register_depth

COLOR_TOPIC = "/device_0/sensor_1/Color_0/image/data"
DEPTH_TOPIC = "/device_0/sensor_0/Depth_0/image/data"
COLOR_INFO = "/device_0/sensor_1/Color_0/image/camera_info"
DEPTH_INFO = "/device_0/sensor_0/Depth_0/image/camera_info"
COLOR_TF = "/device_0/sensor_1/Color_0/tf/ref_0"


def _align(position: int, size: int = 4) -> int:
    return (position + size - 1) & ~(size - 1)


def _string(buffer: bytes, position: int) -> tuple[str, int]:
    """Read a CDR string. Do not pad afterwards; the next field supplies its own alignment."""
    position = _align(position)
    length = struct.unpack_from("<I", buffer, position)[0]
    position += 4
    text = buffer[position:position + max(length - 1, 0)].decode("ascii") if length else ""
    return text, position + length


def parse_camera_info(buffer: bytes) -> dict:
    """Decode one sensor_msgs/CameraInfo CDR blob from a RealSense bag."""
    position = 4
    position += 8
    frame, position = _string(buffer, position)
    height, width = struct.unpack_from("<II", buffer, _align(position))
    position = _align(position) + 8
    model, position = _string(buffer, position)
    count = struct.unpack_from("<I", buffer, _align(position))[0]
    position = _align(position) + 4
    coeffs = struct.unpack_from("<" + "d" * count, buffer, position)
    position += 8 * count
    k = struct.unpack_from("<" + "d" * 9, buffer, position)
    return {
        "frame": frame, "width": width, "height": height, "model": model,
        "fx": k[0], "fy": k[4], "ppx": k[2], "ppy": k[5], "coeffs": list(coeffs),
    }


def parse_image(buffer: bytes) -> tuple[dict, memoryview]:
    """Decode one sensor_msgs/Image CDR blob; pixels are a view into the blob."""
    position = 4
    position += 8
    frame, position = _string(buffer, position)
    height, width = struct.unpack_from("<II", buffer, _align(position))
    position = _align(position) + 8
    encoding, position = _string(buffer, position)
    bigendian = buffer[position]
    position = _align(position + 1)
    step = struct.unpack_from("<I", buffer, position)[0]
    position += 4
    length = struct.unpack_from("<I", buffer, position)[0]
    position += 4
    if length != height * step:
        raise ValueError(f"image payload {length} does not match {height}×{step}")
    return {"frame": frame, "height": height, "width": width, "encoding": encoding,
            "bigendian": bigendian, "step": step}, memoryview(buffer)[position:position + length]


def parse_tf_string(buffer: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Read a RealSense ref_0 string: column-major rotation and metres."""
    text = _string(buffer, 4)[0]
    fields = dict(part.split("=", 1) for part in text.split(";"))
    rotation = np.array([float(x) for x in fields["rotation"].split(",")], dtype=np.float64).reshape(3, 3).T
    translation = np.array([float(x) for x in fields["translation"].split(",")], dtype=np.float64)
    return rotation, translation


def _topic_rows(connection, name: str):
    return connection.execute(
        "SELECT m.timestamp, m.data FROM messages m JOIN topics t ON m.topic_id = t.id "
        "WHERE t.name = ? ORDER BY m.id", (name,),
    ).fetchall()


def _one(connection, name: str) -> bytes:
    rows = _topic_rows(connection, name)
    if not rows:
        raise ValueError(f"bag has no topic {name}")
    return rows[0][1]


def _image_array(payload: memoryview, info: dict) -> np.ndarray:
    if info["encoding"] == "bgr8":
        return np.frombuffer(payload, np.uint8).reshape(info["height"], info["width"], 3)[..., ::-1].copy()
    if info["encoding"] in ("mono16", "16UC1"):
        dtype = ">u2" if info["bigendian"] else "<u2"
        return np.frombuffer(payload, dtype).reshape(info["height"], info["width"]).astype(np.uint16)
    raise ValueError(f"unsupported image encoding {info['encoding']}")


def import_realsense_bag(source: Path, destination: Path, max_depth_m=3.0,
                         start=0, max_frames=None, max_offset_ms=33.0) -> Path:
    import cv2
    source = Path(source)
    bag = source / "capture.db3" if source.is_dir() else source
    if not bag.is_file():
        raise ValueError(f"expected capture.db3 under {source}")
    meta_path = bag.parent / "metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    connection = sqlite3.connect(f"file:{bag}?mode=ro", uri=True)
    try:
        color_info = parse_camera_info(_one(connection, COLOR_INFO))
        depth_info = parse_camera_info(_one(connection, DEPTH_INFO))
        rotation, translation = parse_tf_string(_one(connection, COLOR_TF))
        color_rows = _topic_rows(connection, COLOR_TOPIC)
        depth_rows = _topic_rows(connection, DEPTH_TOPIC)
    finally:
        connection.close()
    if not color_rows or not depth_rows:
        raise ValueError("bag is missing color or depth images")
    if np.any(np.abs(depth_info["coeffs"]) > 1e-8):
        raise ValueError("depth stream distortion is not supported")
    width, height = color_info["width"], color_info["height"]
    K = np.array([[color_info["fx"], 0, color_info["ppx"]],
                  [0, color_info["fy"], color_info["ppy"]], [0, 0, 1]], dtype=np.float64)
    depth_K = np.array([[depth_info["fx"], 0, depth_info["ppx"]],
                        [0, depth_info["fy"], depth_info["ppy"]], [0, 0, 1]], dtype=np.float64)
    coefficients = np.asarray(color_info["coeffs"], dtype=np.float64)
    remap = None
    if np.any(np.abs(coefficients) > 1e-8):
        model = "distortion." + color_info["model"]
        if color_info["model"] == "plumb_bob":
            model = "distortion.inverse_brown_conrady"
        if model not in RECTIFIABLE and color_info["model"] not in ("plumb_bob", "inverse_brown_conrady"):
            raise ValueError(f"RGB distortion {color_info['model']} needs an explicit rectification model")
        remap = cv2.initUndistortRectifyMap(K, coefficients, None, K, (width, height), cv2.CV_32FC1)
    scale = float(metadata.get("depth_scale_m", 0.001))
    rgb_stamps = np.array([(stamp, i, 0.0) for i, (stamp, _) in enumerate(color_rows)],
                          dtype=np.dtype([("time_ns", "<i8"), ("sequence", "<i8"), ("source_time_ms", "<f8")]))
    depth_stamps = np.array([(stamp, i, 0.0) for i, (stamp, _) in enumerate(depth_rows)],
                            dtype=rgb_stamps.dtype)
    position = np.clip(np.searchsorted(depth_stamps["time_ns"], rgb_stamps["time_ns"]), 1, max(len(depth_stamps) - 1, 1))
    before = np.maximum(position - 1, 0)
    nearest = np.where(
        np.abs(depth_stamps["time_ns"][before] - rgb_stamps["time_ns"])
        <= np.abs(depth_stamps["time_ns"][position] - rgb_stamps["time_ns"]),
        before, position)
    offset = depth_stamps["time_ns"][nearest] - rgb_stamps["time_ns"]
    keep = np.abs(offset) <= round(max_offset_ms * 1e6)
    if not keep.any():
        raise ValueError(f"no RGB/depth pair is within {max_offset_ms} ms")
    stop = len(color_rows) if max_frames is None else min(len(color_rows), start + max_frames)
    writer = SceneWriter(destination, {
        "source": str(bag), "serial": metadata.get("serial"), "model": metadata.get("camera"),
        "width": width, "height": height, "fps": metadata.get("fps"), "K": K.tolist(),
        "depth_scale_m": scale, "depth_aligned_to": "RGB",
        "depth_registration": "depth pixel footprints splatted into rectified RGB; nearest surface kept",
        "depth_intrinsics": depth_info, "depth_to_color": {
            "rotation_row_major": rotation.ravel().tolist(), "translation_m": translation.tolist(),
        },
        "original_color_distortion": {"model": color_info["model"], "coefficients": color_info["coeffs"]},
        "rgb_rectified": remap is not None, "rgb_source": COLOR_TOPIC,
        "timestamp_source": "ros2 bag message timestamp",
        "first_index": start, "source_frames": len(color_rows),
    }, max_depth_m)
    depth_cache: dict[int, np.ndarray] = {}
    for index in range(start, stop):
        if not keep[index]:
            continue
        color_info_i, color_payload = parse_image(color_rows[index][1])
        rgb = _image_array(color_payload, color_info_i)
        if remap is not None:
            rgb = cv2.remap(rgb, *remap, interpolation=cv2.INTER_LINEAR)
        depth_index = int(nearest[index])
        if depth_index not in depth_cache:
            depth_info_i, depth_payload = parse_image(depth_rows[depth_index][1])
            raw = _image_array(depth_payload, depth_info_i)
            depth_m = depth_to_metres(raw, scale, max_depth_m)
            registered = register_depth(depth_m, depth_K, K, rotation, translation, (height, width))
            registered[registered > max_depth_m] = 0
            depth_cache.clear()
            depth_cache[depth_index] = registered
        writer.write(Frame(
            f"{index:08d}", rgb, depth_cache[depth_index], K.copy(), int(color_rows[index][0]),
            metadata={"source_index": index, "depth_index": depth_index,
                      "depth_minus_rgb_ns": int(offset[index]),
                      "timestamp_source": "ros2_bag_message_timestamp"}))
    return destination
