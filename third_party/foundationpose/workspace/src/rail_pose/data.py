from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import re
import numpy as np
from PIL import Image

from .common import save_json


@dataclass(frozen=True)
class Frame:
    frame_id: str
    rgb: np.ndarray
    depth_m: np.ndarray
    K: np.ndarray
    timestamp_ns: int
    mask: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def validate(self, max_depth_m: float = 3.0) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.frame_id):
            raise ValueError("frame_id must be a filename-safe identifier")
        if self.rgb.dtype != np.uint8 or self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError("RGB must be uint8 H×W×3 in RGB order")
        if self.depth_m.shape != self.rgb.shape[:2] or self.depth_m.dtype.kind != "f":
            raise ValueError("depth must be a floating-point metre array aligned to RGB")
        if not np.isfinite(self.depth_m).all() or (self.depth_m < 0).any():
            raise ValueError("depth contains negative or non-finite values")
        if self.depth_m.max(initial=0) > max_depth_m:
            raise ValueError("depth exceeds configured metre range; check units and clipping")
        K = np.asarray(self.K)
        if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError("invalid camera intrinsics")
        if not np.allclose(K[2], [0, 0, 1]):
            raise ValueError("K must use the pinhole intrinsic convention")
        if self.mask is not None and self.mask.shape != self.depth_m.shape:
            raise ValueError("mask must share the RGB pixel grid")
        if self.timestamp_ns < 0:
            raise ValueError("timestamp must be nonnegative")

    def registration_mask(self, minimum: int = 64) -> np.ndarray:
        if self.mask is None or not np.any(self.mask):
            raise ValueError("registration requires a nonempty target mask")
        if np.count_nonzero(self.mask.astype(bool) & (self.depth_m >= .001)) < minimum:
            raise ValueError("not enough valid target depth for registration")
        return self.mask.astype(bool)


def depth_to_metres(raw: np.ndarray, scale: float, max_depth_m: float) -> np.ndarray:
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("depth_scale must be positive")
    depth = np.asarray(raw, dtype=np.float32) * np.float32(scale)
    depth[(~np.isfinite(depth)) | (depth < .001) | (depth > max_depth_m)] = 0
    return depth


class SceneWriter:
    def __init__(self, directory: Path, camera: dict, max_depth_m: float = 3.0):
        self.directory = directory
        self.max_depth_m = max_depth_m
        directory.mkdir(parents=True, exist_ok=False)
        for name in ("rgb", "depth", "masks"):
            (directory / name).mkdir()
        self.seen: set[str] = set()
        self.manifest = {
            "format_version": 1, "color_order": "RGB", "depth_unit": "m",
            "depth_format": "float32_npy", "max_depth_m": max_depth_m,
            "camera": camera, "frames": [],
        }
        save_json(directory / "scene.json", self.manifest)

    def write(self, frame: Frame) -> None:
        frame.validate(self.max_depth_m)
        if frame.frame_id in self.seen:
            raise ValueError("duplicate frame_id would overwrite recorded data")
        self.seen.add(frame.frame_id)
        Image.fromarray(frame.rgb).save(self.directory / "rgb" / f"{frame.frame_id}.png")
        np.save(self.directory / "depth" / f"{frame.frame_id}.npy", frame.depth_m.astype(np.float32), allow_pickle=False)
        if frame.mask is not None:
            Image.fromarray(frame.mask.astype(np.uint8) * 255).save(self.directory / "masks" / f"{frame.frame_id}.png")
        self.manifest["frames"].append({
            "frame_id": frame.frame_id, "timestamp_ns": int(frame.timestamp_ns),
            "K": frame.K.tolist(), "metadata": frame.metadata,
        })
        save_json(self.directory / "scene.json", self.manifest)

    def attach_mask(self, frame: Frame) -> None:
        frame.validate(self.max_depth_m)
        if frame.frame_id not in self.seen or frame.mask is None:
            raise ValueError("mask attachment requires an existing frame and a mask")
        Image.fromarray(frame.mask.astype(np.uint8) * 255).save(
            self.directory / "masks" / f"{frame.frame_id}.png")


class Scene:
    def __init__(self, directory: Path):
        self.directory = directory
        self.manifest = json.loads((directory / "scene.json").read_text(encoding="utf-8"))
        if self.manifest.get("format_version") != 1 or self.manifest.get("depth_unit") != "m":
            raise ValueError("expected workspace scene format 1 with metre depth")
        if self.manifest.get("color_order") != "RGB" or self.manifest.get("depth_format") != "float32_npy":
            raise ValueError("unsupported color/depth encoding")
        self.records = self.manifest["frames"]
        ids = [x["frame_id"] for x in self.records]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("scene must contain frames with unique IDs")
        if any(not re.fullmatch(r"[A-Za-z0-9_-]+", x) for x in ids):
            raise ValueError("invalid scene frame ID")

    def __len__(self):
        return len(self.records)

    def frame(self, index: int) -> Frame:
        record = self.records[index]
        name = record["frame_id"]
        mask_path = self.directory / "masks" / f"{name}.png"
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0 if mask_path.exists() else None
        result = Frame(
            name, np.asarray(Image.open(self.directory / "rgb" / f"{name}.png").convert("RGB")),
            np.load(self.directory / "depth" / f"{name}.npy", allow_pickle=False),
            np.asarray(record["K"], dtype=np.float64), int(record["timestamp_ns"]),
            mask, record.get("metadata", {}),
        )
        result.validate(self.manifest["max_depth_m"])
        return result


def import_official_scene(source: Path, destination: Path, max_frames: int = 30) -> Path:
    """The official Ycbineoat demo explicitly encodes PNG depth in millimetres."""
    K = np.loadtxt(source / "cam_K.txt").reshape(3, 3)
    rgb_files = sorted((source / "rgb").glob("*.png"))[:max_frames]
    if not rgb_files:
        raise ValueError("official scene has no RGB frames")
    writer = SceneWriter(destination, {"K": K.tolist(), "source": str(source), "depth_source_unit": "mm"})
    for index, rgb_path in enumerate(rgb_files):
        name = rgb_path.stem
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        raw = np.asarray(Image.open(source / "depth" / rgb_path.name))
        depth = depth_to_metres(raw, .001, 3.0)
        mask_path = source / "masks" / rgb_path.name
        mask = np.asarray(Image.open(mask_path).convert("RGB")).any(axis=2) if mask_path.exists() else None
        writer.write(Frame(name, rgb, depth, K.copy(), index * 33_333_333, mask,
                           {"source": "official_demo", "time_source": "synthetic_index_at_30hz"}))
    return destination
