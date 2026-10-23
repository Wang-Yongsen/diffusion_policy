"""Chunk pair processing while retaining Scorer's full candidate attention."""
from __future__ import annotations

import torch
from learning.training.predict_pose_refine import PoseRefinePredictor
from learning.training.predict_score import ScorePredictor, make_crop_data_batch


def rank_embeddings(model, embeddings):
    """Same second-level attention and projection as ScoreNetMultiPair.forward."""
    sequence = embeddings.unsqueeze(0)
    attended, _ = model.att_cross(sequence, sequence, sequence)
    return model.linear(attended).reshape(-1)


class ChunkedRefiner(PoseRefinePredictor):
    def __init__(self, batch_size=8):
        super().__init__()
        self.batch_size = batch_size
        self.cancel_event = None

    @torch.inference_mode()
    def predict(self, *args, **kwargs):
        if args:
            raise TypeError("workspace refiner uses the upstream keyword interface")
        if kwargs.get("get_vis"):
            raise ValueError("use workspace overlays instead of all-candidate debug images")
        poses = kwargs["ob_in_cams"]
        results = []
        for start in range(0, len(poses), self.batch_size):
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise RuntimeError("initialization cancelled")
            batch = dict(kwargs, ob_in_cams=poses[start:start + self.batch_size], get_vis=False)
            refined, _ = super().predict(**batch)
            results.append(refined)
        return torch.cat(results, dim=0), None


class ChunkedScorer(ScorePredictor):
    def __init__(self, batch_size=8):
        super().__init__()
        self.batch_size = batch_size
        self.cancel_event = None

    @torch.inference_mode()
    def predict(self, rgb, depth, K, ob_in_cams, normal_map=None, get_vis=False,
                mesh=None, mesh_tensors=None, glctx=None, mesh_diameter=None):
        if get_vis:
            raise ValueError("use workspace overlays instead of all-candidate debug images")
        ob_in_cams = torch.as_tensor(ob_in_cams, device="cuda", dtype=torch.float32)
        rgb = torch.as_tensor(rgb, device="cuda", dtype=torch.float32)
        depth = torch.as_tensor(depth, device="cuda", dtype=torch.float32)
        embeddings = []
        for start in range(0, len(ob_in_cams), self.batch_size):
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise RuntimeError("initialization cancelled")
            poses = ob_in_cams[start:start + self.batch_size]
            data = make_crop_data_batch(
                self.cfg.input_resize, poses, mesh, rgb, depth, K,
                crop_ratio=self.cfg.crop_ratio, glctx=glctx, mesh_tensors=mesh_tensors,
                normal_map=normal_map, mesh_diameter=mesh_diameter, dataset=self.dataset, cfg=self.cfg,
            )
            A = torch.cat([data.rgbAs, data.xyz_mapAs], dim=1).float()
            B = torch.cat([data.rgbBs, data.xyz_mapBs], dim=1).float()
            if data.normalAs is not None:
                A = torch.cat([A, data.normalAs.float()], dim=1)
                B = torch.cat([B, data.normalBs.float()], dim=1)
            with torch.amp.autocast("cuda", enabled=self.amp):
                embeddings.append(self.model.extract_feat(A, B))
            del data, A, B
        # Do not rank each chunk separately: candidate attention must see ALL K.
        with torch.amp.autocast("cuda", enabled=self.amp):
            scores = rank_embeddings(self.model, torch.cat(embeddings, dim=0))
        return scores.float() + 100, None  # Match the upstream score offset.
