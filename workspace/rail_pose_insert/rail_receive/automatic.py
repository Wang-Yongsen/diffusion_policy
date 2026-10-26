"""Automatic CAD-sized rail proposals around the measured palm; no learned detector."""
from __future__ import annotations

from dataclasses import dataclass, replace
import time

import cv2
import numpy as np

from rail_preinsert.core import transform


def xyz_image(frame):
    y, x = np.indices(frame.depth_m.shape, dtype=np.float32)
    z = frame.depth_m
    return np.stack(((x-frame.K[0, 2])*z/frame.K[0, 0],
                     (y-frame.K[1, 2])*z/frame.K[1, 1], z), axis=-1)


def checked_context(context, frame, settings, now_ns=None):
    """Reject stale feedback and mismatched RGB-D/robot acquisition times."""
    now_ns = time.monotonic_ns() if now_ns is None else now_ns
    if not context or not context.get("enabled"):
        return None
    limit = settings["context_max_age_s"] * 1e9
    if not 0 <= now_ns-context["timestamp_ns"] <= limit:
        return None
    if abs(frame.timestamp_ns-context["timestamp_ns"]) > limit:
        return None
    return transform(context["T_C_G"])


@dataclass
class Candidate:
    mask: np.ndarray
    metrics: dict


class RailProposer:
    """Parallel image edges + depth consensus + known CAD dimensions and palm prior."""
    def __init__(self, config, mesh_extents):
        self.settings = config["automatic_detection"]
        self.length = float(np.max(mesh_extents))
        self.width = float(np.sort(mesh_extents)[-2])
        self.palm_half = np.asarray(config["grasp_half_extents_m"])

    def _geometry(self, mask, xyz, T_C_G):
        pts = xyz[mask]
        if len(pts) < self.settings["min_pixels"]:
            return None
        pts = pts[::max(1, len(pts)//3000)]
        mean = pts.mean(axis=0)
        _, basis = np.linalg.eigh(np.cov(pts.T))
        basis = basis[:, ::-1]
        local = (pts-mean) @ basis
        low, high = np.quantile(local, [.01, .99], axis=0)
        dimensions = high-low
        center = mean + basis @ ((low+high)/2)
        c = self.settings
        if not c["min_visible_fraction"]*self.length <= dimensions[0] <= c["max_length_ratio"]*self.length:
            return None
        if not .004 <= dimensions[1] <= self.width*c["max_width_ratio"]:
            return None
        if dimensions[2] > self.width or dimensions[0]/max(dimensions[1], .001) < c["min_elongation"]:
            return None
        palm = np.linalg.inv(T_C_G) @ np.r_[center, 1.]
        if np.any(np.abs(palm[:3]) > self.palm_half+c["center_margin_m"]):
            return None
        score = float(dimensions[0]/self.length - abs(dimensions[0]/self.length-1)
                      - .4*np.linalg.norm(center-T_C_G[:3, 3])/self.length
                      + .15*min(dimensions[1]/self.width, 1))
        return Candidate(mask, {
            "length_m": float(dimensions[0]), "width_m": float(dimensions[1]),
            "thickness_m": float(dimensions[2]), "pixels": int(mask.sum()), "score": score,
            "center_in_palm_m": palm[:3].tolist()})

    def detect(self, frame, T_C_G):
        frame.validate()
        T_C_G = transform(T_C_G)
        center = T_C_G[:3, 3]
        if center[2] <= .05:
            return None, "palm_out_of_view"
        xyz = xyz_image(frame)
        radius = self.length/2 + self.settings["search_margin_m"]
        roi = (frame.depth_m > .001) & (np.linalg.norm(xyz-center, axis=-1) <= radius)
        if roi.sum() < self.settings["min_pixels"]:
            return None, "no_depth_near_palm"
        h, w = roi.shape
        projected = frame.K @ center
        palm_pixel = projected[:2]/projected[2]
        focal = (frame.K[0, 0]+frame.K[1, 1])/2
        lines = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(
            cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY))[0]
        if lines is None:
            return None, "searching_rail"
        min_length = max(25., self.length*focal/center[2]*self.settings["min_visible_fraction"])
        segments = []
        for values in lines[:, 0]:
            ends = values.reshape(2, 2).astype(float)
            size = np.linalg.norm(ends[1]-ends[0])
            if size < max(15., min_length*.3):
                continue
            axis = (ends[1]-ends[0])/size
            normal = np.array([-axis[1], axis[0]])
            if abs((ends.mean(axis=0)-palm_pixel) @ normal)*center[2]/focal > self.settings["center_margin_m"]+self.width:
                continue
            segments.append((ends, axis, size))
        # Join collinear fragments across a hand occluding the middle of the rail.
        joined = []
        for ends, axis, size in sorted(segments, key=lambda value: -value[2]):
            for index, (previous, direction, previous_size) in enumerate(joined):
                if abs(axis @ direction) < np.cos(np.radians(4)):
                    continue
                normal = np.array([-direction[1], direction[0]])
                if abs((ends.mean(axis=0)-previous.mean(axis=0)) @ normal) > 2.5:
                    continue
                old, new = previous @ direction, ends @ direction
                gap = max(old.min(), new.min())-min(old.max(), new.max())
                if gap > .45*self.length*focal/center[2]:
                    continue
                low, high = min(old.min(), new.min()), max(old.max(), new.max())
                if high-low > self.length*focal/center[2]*self.settings["max_length_ratio"]*1.2:
                    continue
                offset = (np.mean(previous @ normal)*previous_size+np.mean(ends @ normal)*size)/(previous_size+size)
                combined = np.array([direction*low+normal*offset, direction*high+normal*offset])
                joined[index] = combined, direction, high-low
                break
            else:
                joined.append((ends, axis, size))
        segments = [segment for segment in joined if segment[2] >= min_length]
        candidates = []
        rng = np.random.default_rng(0)
        y, x = np.indices(roi.shape)
        for index, (first, axis, _) in enumerate(segments):
            normal = np.array([-axis[1], axis[0]])
            for second, other_axis, _ in segments[index+1:]:
                if abs(axis @ other_axis) < np.cos(np.radians(8)):
                    continue
                a, b = first @ axis, second @ axis
                overlap = min(a.max(), b.max())-max(a.min(), b.min())
                if overlap < .45*min(np.ptp(a), np.ptp(b)):
                    continue
                offsets = np.array([np.mean(first @ normal), np.mean(second @ normal)])
                width = abs(np.diff(offsets)[0])*center[2]/focal
                if not .006 <= width <= self.width*self.settings["max_width_ratio"]:
                    continue
                low, high = min(a.min(), b.min()), max(a.max(), b.max())
                polygon = np.array([axis*s+normal*n for s, n in (
                    (low, offsets.min()), (high, offsets.min()),
                    (high, offsets.max()), (low, offsets.max()))])
                region = np.zeros((h, w), np.uint8)
                cv2.fillConvexPoly(region, np.round(polygon).astype(np.int32), 1)
                region = region.astype(bool) & roi
                if region.sum() < self.settings["min_pixels"]:
                    continue
                # Fit depth along the long edge. Multiple hypotheses avoid holes/background.
                s_map = (x-palm_pixel[0])*axis[0]+(y-palm_pixel[1])*axis[1]
                s, z = s_map[region], frame.depth_m[region]
                step = max(1, len(z)//700)
                ss, zz = s[::step], z[::step]
                best_mask, best_score = None, -np.inf
                for _ in range(48):
                    i, j = rng.integers(0, len(ss), size=2)
                    if abs(ss[i]-ss[j]) < min_length*.3:
                        continue
                    slope = (zz[i]-zz[j])/(ss[i]-ss[j])
                    intercept = zz[i]-slope*ss[i]
                    if abs(intercept-center[2]) > self.settings["center_margin_m"]:
                        continue
                    inliers = np.abs(z-(slope*s+intercept)) <= self.settings["proposal_depth_tolerance_m"]
                    score = inliers.sum()*(1-.5*abs(intercept-center[2])/self.settings["center_margin_m"])
                    if score > best_score:
                        best_mask, best_score = inliers, score
                if best_mask is None:
                    continue
                mask = np.zeros_like(region)
                mask[region] = best_mask
                candidate = self._geometry(mask, xyz, T_C_G)
                if candidate is not None:
                    candidates.append(candidate)
        if not candidates:
            return None, "searching_rail"
        best = max(candidates, key=lambda candidate: candidate.metrics["score"])
        return best, "rail_candidate"


class AutomaticTracker:
    """Initialize from the best palm candidate and restart after a tracking failure."""
    def __init__(self, engine, proposer, settings):
        self.engine, self.proposer, self.settings = engine, proposer, settings
        self.epoch = None
        self.on_initializing = None
        self.initial_mask = None
        self.next_search_ns = 0

    def reset(self):
        self.engine.reset()
        self.initial_mask = None
        self.next_search_ns = 0

    def step(self, frame, T_C_G, epoch):
        if epoch != self.epoch:
            self.reset()
            self.epoch = epoch
        self.initial_mask = None
        initialize = not self.engine.initialized
        if initialize:
            if frame.timestamp_ns < self.next_search_ns:
                return None, {"valid": False, "reason": "searching_rail"}
            self.next_search_ns = frame.timestamp_ns+round(self.settings["search_period_s"]*1e9)
            candidate, reason = self.proposer.detect(frame, T_C_G)
            if candidate is None:
                return None, {"valid": False, "reason": reason}
            frame = replace(frame, mask=candidate.mask)
            if self.on_initializing is not None:
                self.on_initializing(frame)
        try:
            pose, record = self.engine.infer(frame, initialize=initialize)
        except Exception as error:
            self.reset()
            self.next_search_ns = frame.timestamp_ns+round(self.settings["retry_delay_s"]*1e9)
            return None, {"valid": False, "reason": "pose_estimation_failed", "detail": str(error)}
        if initialize:
            self.initial_mask = candidate.mask
            record["proposal"] = candidate.metrics
        record["context_epoch"] = epoch
        return pose, record

