"""Bridge to the unmodified official CNOS inference pipeline (SAM-H + DINOv2-L)."""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

from rail_preinsert.core import HERE, digest, write_json

ROOT = HERE.parents[1] / "third_party/cnos"
WEIGHTS = {
    "segment-anything/sam_vit_h_4b8939.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "dinov2_vitl14_pretrain.pth": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
}


def official_config():
    """Load upstream defaults; only resolve local asset paths and choose Pyrender."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="run_inference", overrides=[f"machine.root_dir={ROOT}"])
    OmegaConf.set_struct(cfg, False)
    cfg.model.onboarding_config.rendering_type = "pyrender"
    cfg.model.descriptor_model.model.repo_or_dir = str(HERE.parents[1] / "third_party/dinov2")
    cfg.model.descriptor_model.model.source = "local"
    cfg.model.descriptor_model.model.pretrained = False
    return cfg


def prepare_templates(config):
    """Official renderer and BOP view indexing; cache separately from the old adapter."""
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender
    import trimesh
    from src.poses.pyrender import render
    from src.poses.utils import get_obj_poses_from_template_level, load_index_level_in_level2
    spec = {"pipeline": "official_cnos_pyrender_v1", "mesh_sha256": digest(config["mesh"]),
            "radius_m": .8, "lighting_intensity": 1., "level": 0,
            "descriptor": "dinov2_vitl14", "background": "official_BOPTemplate"}
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    cache = ROOT / "pretrained/rail_official" / key
    template_dir = cache / "templates"
    ids = load_index_level_in_level2(0, "all")
    expected = [template_dir / "obj_000001" / f"{int(i):06d}.png" for i in ids]
    if all(path.is_file() for path in expected):
        return template_dir
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="render-", dir=cache) as temporary:
        temporary = Path(temporary)
        rendered = temporary / "rendered"; rendered.mkdir()
        poses = get_obj_poses_from_template_level(2, "all")[ids].copy()
        poses[:, :3, 3] *= spec["radius_m"] / 1000
        mesh = trimesh.load(config["mesh"], force="mesh")
        recenter = np.eye(4); recenter[:3, 3] = -mesh.bounding_box.centroid
        render(pyrender.Mesh.from_trimesh(mesh), str(rendered), poses, (480, 640),
               np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]]),
               light_itensity=spec["lighting_intensity"], re_center_transform=recenter)
        target = temporary / "templates/obj_000001"; target.mkdir(parents=True)
        for number, official_id in enumerate(ids):
            shutil.copyfile(rendered / f"{number:06d}.png", target / f"{int(official_id):06d}.png")
        if template_dir.exists():
            shutil.rmtree(template_dir)
        (temporary / "templates").replace(template_dir)
    write_json(cache / "template_spec.json", spec)
    return template_dir


class CnosDetector:
    """Return one instance AFTER the complete upstream multi-instance pipeline."""
    def __init__(self, config):
        import torch
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        sys.path.insert(0, str(ROOT))
        from src.dataloader.bop import BOPTemplate
        self.torch, self.config, self.model, self.runtime = torch, config, None, None
        self.roi = config["receive_roi_xyxy"]
        self.index = 0
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("Official CNOS requires CUDA in this deployment")
            for relative in WEIGHTS:
                if not (ROOT / "pretrained" / relative).is_file():
                    raise FileNotFoundError(f"Missing official CNOS weight: {relative}; run prepare_cnos.py")
            templates = prepare_templates(config)
            self.runtime = tempfile.TemporaryDirectory(prefix="rail-cnos-official-")
            cfg = official_config()
            if config["cnos_detection"]["confidence_threshold"] != cfg.model.matching_config.confidence_thresh:
                raise ValueError("CNOS threshold must match the upstream official configuration")
            cfg.model.log_dir = self.runtime.name
            # Construct on CPU, then move the complete official model to CUDA.
            with torch.device("cpu"):
                self.model = instantiate(cfg.model)
            self.model.descriptor_model.model.load_state_dict(torch.load(
                ROOT / "pretrained/dinov2_vitl14_pretrain.pth", map_location="cpu", weights_only=True))
            self.model.to("cuda").eval()
            self.model.dataset_name = "rail"
            self.model.name_prediction_file = "official"
            self.model.ref_dataset = BOPTemplate(template_dir=str(templates), obj_ids=[1],
                processing_config=OmegaConf.create({"image_size": cfg.model.descriptor_model.image_size}),
                level_templates=cfg.model.onboarding_config.level_templates, pose_distribution="all")
            self.model.set_reference_objects()
            self.model.move_to_device()
            self.settings = cfg.model
        except BaseException:
            self.close()
            raise

    def detect(self, frame):
        if list(frame.rgb.shape[1::-1]) != self.config["receive_image_size"]:
            return None, {"valid": False, "reason": "roi_image_size_mismatch"}
        # Input normalization is the same as BaseBOPTest. test_step performs
        # SAM, small-mask/box filtering, descriptors, avg_5 matching, threshold,
        # max_num_instances and per-object NMS, and writes all final instances.
        batch = {"image": self.model.descriptor_model.rgb_normalize(frame.rgb.copy())[None],
                 "scene_id": [0], "frame_id": [self.index]}
        with self.torch.inference_mode():
            self.model.test_step(batch, self.index)
        path = Path(self.runtime.name) / f"predictions/rail/official/scene0_frame{self.index}.npz"
        self.index += 1
        with np.load(path) as result:
            scores, boxes, masks = result["score"], result["bbox"], result["segmentation"]
            count = len(scores)
            record = {"valid": count > 0, "detector": "cnos", "pipeline": "official_CNOS.test_step",
                "instance_count": count, "output_instances": min(1, count),
                "confidence_threshold": float(self.settings.matching_config.confidence_thresh),
                "selection": "highest_score_after_official_filtering_and_nms",
                "instances": [{"score": float(score), "bbox_xywh": box.tolist()}
                              for score, box in zip(scores, boxes)]}
            if not count:
                record.update(reason="searching_rail", detected_in_roi=False)
                return None, record
            selected = int(np.argmax(scores))
            mask = masks[selected] > .5
            x, y, w, h = boxes[selected].tolist()
            record.update(score=float(scores[selected]), bbox_xyxy=[x, y, x+w, y+h],
                          selected_instance=selected, roi_xyxy=self.roi,
                          detected_in_roi=bool(mask[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]].any()))
            # This is solely the single-instance interface to FoundationPose;
            # it does not replace upstream proposal filtering or NMS.
            return mask.copy(), record

    def close(self):
        self.model = None
        gc.collect()
        if hasattr(self, "torch"):
            self.torch.cuda.empty_cache()
        if self.runtime is not None:
            self.runtime.cleanup()
            self.runtime = None
