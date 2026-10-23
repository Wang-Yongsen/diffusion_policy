"""Convert one bimanual-teleop recording camera into a workspace RGB-D scene."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .data import Frame, SceneWriter, depth_to_metres

STAMP = np.dtype([("time_ns", "<i8"), ("sequence", "<i8"), ("source_time_ms", "<f8")])
# librealsense 2.59 projects this model with the OpenCV forward polynomial.
RECTIFIABLE = {"distortion.inverse_brown_conrady"}


def intrinsic_matrix(value: dict) -> np.ndarray:
    return np.array([[value["fx"], 0, value["ppx"]], [0, value["fy"], value["ppy"]], [0, 0, 1]], dtype=np.float64)


def register_depth(depth_m, depth_K, color_K, R, t, shape):
    """Splat every depth pixel footprint into the pinhole color grid and keep the nearest surface."""
    height, width = shape
    v, u = np.nonzero(depth_m > 0)
    z = depth_m[v, u].astype(np.float64)

    def to_color(offset):
        x = (u + offset - depth_K[0, 2]) / depth_K[0, 0] * z
        y = (v + offset - depth_K[1, 2]) / depth_K[1, 1] * z
        point = np.stack([x, y, z], axis=1) @ R.T + t
        with np.errstate(divide="ignore", invalid="ignore"):
            return point, point[:, :2] / point[:, 2:] * color_K[[0, 1], [0, 1]] + color_K[:2, 2]

    centre, pixel = to_color(0)
    keep = centre[:, 2] > 0
    pixel = np.floor(pixel[keep] + .5).astype(np.int64)
    # A footprint fills the color pixels whose centres it contains, never fewer than its own centre pixel.
    low = np.minimum(np.ceil(to_color(-.5)[1][keep]).astype(np.int64), pixel)
    high = np.maximum(np.ceil(to_color(.5)[1][keep]).astype(np.int64) - 1, pixel)
    value = centre[keep, 2].astype(np.float32)
    span = int(min(max((high - low).max(initial=0), 0) + 1, 4))
    result = np.full(height * width, np.inf, np.float32)
    for dy in range(span):
        for dx in range(span):
            x, y = low[:, 0] + dx, low[:, 1] + dy
            inside = (x <= high[:, 0]) & (y <= high[:, 1]) & (x >= 0) & (y >= 0) & (x < width) & (y < height)
            np.minimum.at(result, y[inside] * width + x[inside], value[inside])
    result[np.isinf(result)] = 0
    return result.reshape(height, width)


def depth_layout(document, depth_info):
    """Return (height, width) of the uint16 raw depth frames.

    Older recordings store this under ``spool``. Newer ones only keep the
    depth camera intrinsics; the raw file size is checked against that shape.
    """
    spool = document.get("spool")
    if spool:
        if spool["depth_dtype"] != "uint16":
            raise ValueError("expected uint16 raw depth")
        return tuple(spool["depth_shape"])
    return int(depth_info["height"]), int(depth_info["width"])


def pair_streams(rgb, depth, max_offset_ns):
    """Match each RGB frame to the nearest depth frame in host monotonic time."""
    position = np.clip(np.searchsorted(depth["time_ns"], rgb["time_ns"]), 1, len(depth) - 1)
    before = position - 1
    nearest = np.where(np.abs(depth["time_ns"][before] - rgb["time_ns"]) <= np.abs(depth["time_ns"][position] - rgb["time_ns"]),
                       before, position)
    offset = depth["time_ns"][nearest] - rgb["time_ns"]
    if np.abs(offset).max(initial=0) > max_offset_ns:
        raise ValueError(f"RGB/depth offset {np.abs(offset).max() / 1e6:.3f} ms exceeds the pairing limit")
    if len(np.unique(nearest)) != len(nearest):
        raise ValueError("several RGB frames map to one depth frame")
    return nearest, offset


def import_episode(source: Path, destination: Path, camera="camera_0", max_depth_m=3.0,
                   start=0, max_frames=None, max_offset_ms=5.0) -> Path:
    import cv2
    document = json.loads((source / "episode.json").read_text(encoding="utf-8"))
    info = document["metadata"]["cameras"][camera]
    if "depth_intrinsics" not in info:
        raise ValueError(f"{camera} was recorded without depth")
    if info.get("image_rotation_deg", 0) != 0:
        raise ValueError("depth registration expects unrotated RGB")
    color, depth_info = info["rgb_intrinsics"], info["depth_intrinsics"]
    width, height = color["width"], color["height"]
    if np.any(np.abs(depth_info["coeffs"]) > 1e-8):
        raise ValueError("depth stream distortion is not supported")
    K, depth_K = intrinsic_matrix(color), intrinsic_matrix(depth_info)
    coefficients = np.asarray(color["coeffs"], dtype=np.float64)
    remap = None
    if np.any(np.abs(coefficients) > 1e-8):
        if color["model"] not in RECTIFIABLE:
            raise ValueError(f"RGB distortion {color['model']} needs an explicit rectification model")
        remap = cv2.initUndistortRectifyMap(K, coefficients, None, K, (width, height), cv2.CV_32FC1)
    extrinsics = info["depth_to_rgb"]
    R = np.asarray(extrinsics["rotation"], dtype=np.float64).reshape(3, 3).T  # librealsense column-major
    t = np.asarray(extrinsics["translation"], dtype=np.float64)

    cameras = source / "raw_spool" / "cameras"
    rgb_stamps = np.fromfile(cameras / f"{camera}_rgb.bin", dtype=STAMP)
    depth_stamps = np.fromfile(cameras / f"{camera}_depth.bin", dtype=STAMP)
    shape = depth_layout(document, depth_info)
    raw = np.memmap(cameras / f"{camera}_depth.raw", dtype="<u2", mode="r")
    if raw.size != len(depth_stamps) * shape[0] * shape[1]:
        raise ValueError("raw depth size does not match its timestamp count")
    raw = raw.reshape(len(depth_stamps), *shape)
    nearest, offset = pair_streams(rgb_stamps, depth_stamps, round(max_offset_ms * 1e6))
    video = cv2.VideoCapture(str(source / f"{camera}.mp4"))
    if int(video.get(cv2.CAP_PROP_FRAME_COUNT)) != len(rgb_stamps):
        raise ValueError("video frame count does not match RGB timestamps")
    stop = len(rgb_stamps) if max_frames is None else min(len(rgb_stamps), start + max_frames)
    writer = SceneWriter(destination, {
        "source": str(source), "camera": camera, "serial": info["serial"], "model": info["model"],
        "width": width, "height": height, "fps": info["fps"], "K": K.tolist(),
        "depth_scale_m": info["depth_scale"], "depth_aligned_to": "RGB",
        "depth_registration": "depth pixel footprints splatted into rectified RGB; nearest surface kept",
        "depth_intrinsics": depth_info, "depth_to_color": {"rotation_row_major": R.ravel().tolist(), "translation_m": t.tolist()},
        "original_color_distortion": {"model": color["model"], "coefficients": color["coeffs"]},
        "rgb_rectified": remap is not None, "rgb_source": f"{camera}.mp4 (H.264 yuv420p)",
        "timestamp_source": document["metadata"].get("camera_time"), "task": document["metadata"]["recording"].get("task"),
        "first_index": start, "source_frames": len(rgb_stamps),
    }, max_depth_m)
    try:
        for index in range(stop):
            ok, bgr = video.read()
            if not ok:
                raise ValueError(f"video ended at frame {index}")
            if index < start:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if remap is not None:
                rgb = cv2.remap(rgb, *remap, interpolation=cv2.INTER_LINEAR)
            depth_index = nearest[index]
            depth_m = depth_to_metres(raw[depth_index], info["depth_scale"], max_depth_m)
            registered = register_depth(depth_m, depth_K, K, R, t, (height, width))
            registered[registered > max_depth_m] = 0
            stamp = rgb_stamps[index]
            writer.write(Frame(
                f"{int(stamp['sequence']):08d}", rgb, registered, K.copy(), int(stamp["time_ns"]),
                metadata={"source_index": index, "depth_index": int(depth_index),
                          "depth_sequence": int(depth_stamps[depth_index]["sequence"]),
                          "depth_minus_rgb_ns": int(offset[index]),
                          "rgb_source_time_ms": float(stamp["source_time_ms"]),
                          "timestamp_source": "sdk_global_timestamp_mapped_to_host_monotonic"}))
    finally:
        video.release()
    return destination
