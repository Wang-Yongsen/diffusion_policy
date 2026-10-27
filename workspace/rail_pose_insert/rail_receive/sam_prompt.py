"""One local, officially prompted SAM mask; no CAD matching or candidate ranking."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np

CHECKPOINT = (
    Path(__file__).resolve().parents[3] / "third_party/cnos/pretrained/"
    "segment-anything/sam_vit_h_4b8939.pth"
)
PROMPT_KEYS = {"initial": "sam_prompt", "preplace": "sam_prompt_preplace"}
PROMPT_NAMES = {"initial": "初始分割框", "preplace": "导轨台上方分割框"}


def prompt_path(config, stage="initial"):
    if stage not in PROMPT_KEYS:
        raise ValueError(f"Unknown SAM prompt stage: {stage}")
    path = config.get(PROMPT_KEYS[stage])
    if not path:
        raise ValueError(f"Missing {PROMPT_KEYS[stage]} for {PROMPT_NAMES[stage]}")
    return Path(path)


def _size(value, name):
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(type(v) is not int or v <= 0 for v in value)
    ):
        raise ValueError(f"{name} must be [width, height] with positive integer pixels")
    return list(value)


def _coordinates(value, shape, name):
    try:
        raw = np.asarray(value)
        if (
            raw.dtype.kind not in "iuf"
            or raw.shape != shape
            or any(isinstance(v, (bool, np.bool_)) for v in np.asarray(value, dtype=object).flat)
        ):
            raise ValueError
        result = raw.astype(np.float64)
        if not np.isfinite(result).all():
            raise ValueError
        return result
    except (ValueError, TypeError, OverflowError):
        raise ValueError(
            f"{name} must contain finite pixel coordinates with shape {shape}"
        ) from None


def validate_prompt(prompt, image_size, camera_serial=None):
    """Validate a prompt in original image coordinates, returning copied JSON data.

    Box coordinates follow half-open XYXY bounds. Points must lie inside that
    box. Empty point lists mean box-only SAM inference; no point is invented at
    the box center, which may be occupied by the robot hand.
    """
    if not isinstance(prompt, dict):
        raise ValueError("SAM prompt JSON must be an object")
    expected_size = _size(image_size, "image_size")
    if _size(prompt.get("image_size"), "prompt image_size") != expected_size:
        raise ValueError("SAM prompt image_size does not match the camera image_size")
    serial = prompt.get("camera_serial")
    if not isinstance(serial, str) or not serial:
        raise ValueError("SAM prompt camera_serial must be a nonempty string")
    if camera_serial is not None and serial != str(camera_serial):
        raise ValueError("SAM prompt camera_serial does not match the active camera")
    box = _coordinates(prompt.get("box_xyxy"), (4,), "box_xyxy")
    x1, y1, x2, y2 = box
    width, height = expected_size
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("SAM prompt box_xyxy must be nonempty and inside the image")
    coordinates = prompt.get("point_coords", [])
    labels = prompt.get("point_labels", [])
    if (
        not isinstance(coordinates, list)
        or not isinstance(labels, list)
        or len(coordinates) != len(labels)
    ):
        raise ValueError("SAM prompt point_coords and point_labels must be equally sized lists")
    if any(type(label) is not int or label not in (0, 1) for label in labels):
        raise ValueError(
            "SAM prompt point_labels must contain only 1 (foreground) or 0 (background)"
        )
    if coordinates:
        points = _coordinates(coordinates, (len(coordinates), 2), "point_coords")
        if not ((points >= box[:2]).all() and (points < box[2:]).all()):
            raise ValueError("SAM prompt points must lie inside box_xyxy")
        coordinates = points.tolist()
    return {
        "image_size": expected_size,
        "camera_serial": serial,
        "box_xyxy": box.tolist(),
        "point_coords": coordinates.copy(),
        "point_labels": labels.copy(),
    }


def load_prompt(path, image_size, camera_serial=None):
    path = Path(path).expanduser()
    if not path.is_absolute():
        raise ValueError("sam_prompt must be an absolute JSON file path")
    return validate_prompt(json.loads(path.read_text()), image_size, camera_serial)


class SamPromptDetector:
    """Use the official SamPredictor box/point interface and return one mask."""

    def __init__(self, config, camera_serial=None, *, prompt=None, stage="initial"):
        self.model = self.predictor = self.torch = None
        self.image_size = _size(config["receive_image_size"], "image_size")
        self.camera_serial = camera_serial
        self.stage = stage
        self.set_prompt(
            load_prompt(prompt_path(config, stage), self.image_size, camera_serial)
            if prompt is None
            else prompt
        )
        if not CHECKPOINT.is_file():
            raise FileNotFoundError(f"Missing local SAM-H checkpoint: {CHECKPOINT}")
        try:
            import torch
            from segment_anything import SamPredictor, sam_model_registry

            self.torch = torch
            if not torch.cuda.is_available():
                raise RuntimeError("SAM-H requires CUDA in this deployment")
            # FoundationPose can change torch's default device to CUDA. The
            # official builder must still construct/load on CPU before moving.
            with torch.device("cpu"):
                self.model = sam_model_registry["vit_h"](checkpoint=str(CHECKPOINT))
            self.model.to("cuda").eval()
            self.predictor = SamPredictor(self.model)
        except BaseException:
            self.close()
            raise

    def set_prompt(self, prompt):
        """Replace only this detector's in-memory prompt, never the saved file."""
        self.prompt = validate_prompt(prompt, self.image_size, self.camera_serial)
        box = self.prompt["box_xyxy"]
        self.roi = [
            int(np.floor(box[0])),
            int(np.floor(box[1])),
            int(np.ceil(box[2])),
            int(np.ceil(box[3])),
        ]

    def detect(self, frame):
        record = {
            "valid": False,
            "detector": "sam_prompt",
            "pipeline": "official_SamPredictor",
            "prompt_stage": self.stage,
            "output_instances": 0,
            "prompt_box_xyxy": self.prompt["box_xyxy"].copy(),
            "point_coords": self.prompt["point_coords"],
            "point_labels": self.prompt["point_labels"],
            "roi_xyxy": self.roi.copy(),
            "detected_in_roi": False,
            "quality_score_type": "SAM_predicted_IoU",
        }
        rgb = np.asarray(frame.rgb)
        if rgb.shape != (self.image_size[1], self.image_size[0], 3):
            return None, dict(record, reason="sam_prompt_image_size_mismatch")
        if rgb.dtype != np.uint8:
            raise ValueError("SamPredictor requires an RGB uint8 image")
        points = self.prompt["point_coords"]
        with self.torch.inference_mode():
            self.predictor.set_image(np.ascontiguousarray(rgb), image_format="RGB")
            masks, quality, _ = self.predictor.predict(
                box=np.asarray(self.prompt["box_xyxy"], np.float32),
                point_coords=np.asarray(points, np.float32) if points else None,
                point_labels=np.asarray(self.prompt["point_labels"], np.int32) if points else None,
                multimask_output=False,
            )
        masks = np.asarray(masks)
        quality = np.asarray(quality)
        if masks.shape != (1, self.image_size[1], self.image_size[0]) or quality.shape != (1,):
            raise RuntimeError(
                "SamPredictor did not return exactly one mask in original image coordinates"
            )
        if not np.isfinite(quality[0]):
            raise RuntimeError("SamPredictor returned a nonfinite mask quality score")
        # Keep all visible pieces of the same object, including disconnected
        # pieces on either side of an occluding hand. Restrict only to the box.
        mask = np.zeros(rgb.shape[:2], bool)
        x1, y1, x2, y2 = self.roi
        mask[y1:y2, x1:x2] = masks[0, y1:y2, x1:x2].astype(bool)
        record["quality_score"] = float(quality[0])
        rows, columns = np.nonzero(mask)
        if not len(rows):
            return None, dict(record, reason="sam_prompt_empty_mask")
        record.update(
            valid=True,
            output_instances=1,
            detected_in_roi=True,
            bbox_xyxy=[
                int(columns.min()),
                int(rows.min()),
                int(columns.max()) + 1,
                int(rows.max()) + 1,
            ],
            mask_pixels=int(len(rows)),
            reason="segmented",
        )
        return mask, record

    def close(self):
        self.predictor = None
        self.model = None
        gc.collect()
        if self.torch is not None:
            self.torch.cuda.empty_cache()
