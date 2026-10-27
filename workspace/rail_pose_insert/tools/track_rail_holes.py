#!/usr/bin/env python3
"""Offline RGB-D rail-hole tracking from two numeric bounding boxes. No GPT/API/GPU.

Boxes are [xmin, ymin, xmax, ymax), in ORIGINAL RGB pixel coordinates. IDs follow
the input box order throughout tracking, not the current left/right ordering.
Dependencies: numpy, scipy, opencv-python. Run --help for the recording CLI.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

STAMP = np.dtype([('time_ns', '<i8'), ('sequence', '<i8'), ('source_time_ms', '<f8')])


@dataclass
class Settings:
    expected_spacing_m: float = .265
    spacing_tolerance_m: float = .015
    depth_min_m: float = .15
    depth_max_m: float = 2.
    max_sync_ms: float = 5.
    min_contrast: float = 15.
    max_fit_rmse: float = 15.
    max_fb_px: float = .8
    min_patch_ncc: float = .65
    max_center_correction_px: float = 2.5
    max_image_speed_px_s: float = 300.
    max_recovery_s: float = .5
    max_translation_speed_m_s: float = .5
    estimate_surface_orientation: bool = False
    min_plane_points: int = 30
    plane_inlier_m: float = .002
    max_plane_rmse_m: float = .0015
    strip_halfwidth_px: float = 2.5
    strip_extents_px: tuple = ((15., 27.), (39., 41.))
    hole_exclusion_px: float = 5.
    max_normal_disagreement_deg: float = 8.
    min_cutoff_hz: float = 1.5
    filter_beta: float = 8.
    max_filter_gap_s: float = .15
    stationary_window_s: float = .4
    stationary_spread_m: float = .003
    stationary_speed_m_s: float = .015

    def validate(self):
        values = asdict(self)
        for key, value in values.items():
            if key == 'estimate_surface_orientation':
                if not isinstance(value, bool):
                    raise ValueError('estimate_surface_orientation must be boolean')
            elif key == 'strip_extents_px':
                a = np.asarray(value, float)
                if a.shape != (2, 2) or not np.isfinite(a).all() or (a <= 0).any():
                    raise ValueError('strip_extents_px must be two positive [before,after] pairs')
            elif not np.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if self.depth_min_m >= self.depth_max_m:
            raise ValueError('depth_min_m must be smaller than depth_max_m')
        if self.min_patch_ncc > 1:
            raise ValueError('min_patch_ncc cannot exceed 1')


def intrinsic_matrix(info):
    return np.array([[info['fx'], 0, info['ppx']], [0, info['fy'], info['ppy']], [0, 0, 1.]], float)


class Calibration:
    """Depth points are transformed into the RGB optical frame, in metres."""
    def __init__(self, info):
        if info.get('image_rotation_deg', 0):
            raise ValueError('This tool requires unrotated RGB (camera_0)')
        rgb, depth = info['rgb_intrinsics'], info['depth_intrinsics']
        self.K, self.D = intrinsic_matrix(rgb), np.array(rgb['coeffs'], float)
        self.shape = (rgb['height'], rgb['width'])
        self.depth_shape = (depth['height'], depth['width'])
        if rgb['model'] not in ('distortion.none', 'distortion.brown_conrady',
                                'distortion.inverse_brown_conrady'):
            raise ValueError(f"Unsupported RGB distortion: {rgb['model']}")
        # On this D436/librealsense 2.59, OpenCV's ray differs from the SDK by
        # <0.004 px over the image. Do not generalize to arbitrary camera models.
        if not np.isfinite(depth['coeffs']).all() or np.any(np.abs(depth['coeffs']) > 1e-8):
            raise ValueError('Nonzero depth distortion is not supported')
        self.R = np.array(info['depth_to_rgb']['rotation'], float).reshape(3, 3, order='F')
        self.t = np.array(info['depth_to_rgb']['translation'], float)
        self.scale = float(info['depth_scale'])
        numeric = np.r_[self.K.ravel(), self.D, self.R.ravel(), self.t,
                        intrinsic_matrix(depth).ravel(), self.scale]
        if (not np.isfinite(numeric).all() or self.scale <= 0 or
                min(rgb['fx'], rgb['fy'], depth['fx'], depth['fy']) <= 0 or
                min(*self.shape, *self.depth_shape) <= 0 or
                not np.allclose(self.R.T @ self.R, np.eye(3), atol=1e-4) or
                not np.isclose(np.linalg.det(self.R), 1., atol=1e-4)):
            raise ValueError('Invalid calibration: expected finite parameters, positive focal lengths, and SO(3) rotation')
        y, x = np.indices(self.depth_shape)
        self.depth_rays = np.stack([(x-depth['ppx'])/depth['fx'],
                                    (y-depth['ppy'])/depth['fy'], np.ones_like(x)], -1)

    def rectify(self, uv):
        return cv2.undistortPoints(np.asarray(uv, float).reshape(-1, 1, 2),
                                  self.K, self.D, P=self.K).reshape(-1, 2)

    def ray(self, uv_rectified):
        return np.linalg.solve(self.K, [*uv_rectified, 1.])

    def cloud(self, raw, settings):
        if raw.shape != self.depth_shape:
            raise ValueError('Raw depth dimensions differ from calibration')
        z = raw.astype(float) * self.scale
        valid = np.isfinite(z) & (z >= settings.depth_min_m) & (z <= settings.depth_max_m)
        points = (self.depth_rays[valid] * z[valid, None]) @ self.R.T + self.t
        points = points[points[:, 2] > .001]
        homogeneous = points @ self.K.T
        return points, homogeneous[:, :2] / homogeneous[:, 2:]


def fit_plane(points, settings):
    """Robust narrow-strip fit; its residual is not an absolute accuracy bound."""
    points = np.asarray(points, float)
    if len(points) < settings.min_plane_points or not np.isfinite(points).all():
        raise ValueError('insufficient_depth_support')
    keep = np.ones(len(points), bool)
    for _ in range(8):
        if keep.sum() < settings.min_plane_points:
            raise ValueError('insufficient_plane_inliers')
        centre = points[keep].mean(0)
        _, singular, vh = np.linalg.svd(points[keep]-centre, full_matrices=False)
        normal = vh[-1]
        offset = -normal @ centre
        residual = points @ normal + offset
        new_keep = np.abs(residual) < settings.plane_inlier_m
        if np.array_equal(keep, new_keep):
            break
        keep = new_keep
    if keep.sum() < settings.min_plane_points or keep.mean() < .6:
        raise ValueError('depth_plane_outliers')
    centre = points[keep].mean(0)
    _, singular, vh = np.linalg.svd(points[keep]-centre, full_matrices=False)
    if singular[1] / math.sqrt(keep.sum()) < .001 or singular[1] / singular[0] < .025:
        raise ValueError('degenerate_plane_support')
    normal = vh[-1]
    if normal @ centre > 0:  # same camera-facing sign for both holes
        normal = -normal
    offset = float(-normal @ centre)
    rmse = float(np.sqrt(np.mean((points[keep] @ normal + offset)**2)))
    if rmse > settings.max_plane_rmse_m:
        raise ValueError('depth_plane_residual')
    return {'normal': normal.tolist(), 'offset_m': offset, 'inliers': int(keep.sum()),
            'support': len(points), 'rmse_m': rmse,
            'support_sigma_m': (singular / math.sqrt(keep.sum())).tolist()}


def intersect_plane(ray, plane, settings):
    denominator = np.dot(plane['normal'], ray)
    if abs(denominator) / np.linalg.norm(ray) < .1:
        raise ValueError('grazing_ray')
    xyz = -plane['offset_m'] / denominator * ray
    if not np.isfinite(xyz).all() or not settings.depth_min_m <= xyz[2] <= settings.depth_max_m:
        raise ValueError('intersection_outside_depth_range')
    return xyz


def measure_centres(raw_depth, uv_raw, calibration, settings):
    """Independent per-hole planes; nominal CAD spacing is checked, never imposed."""
    uv = calibration.rectify(uv_raw)
    along = uv[1] - uv[0]
    if np.linalg.norm(along) < 10:
        raise ValueError('holes_too_close_in_image')
    along /= np.linalg.norm(along)
    across = np.array([-along[1], along[0]])
    points, pixels = calibration.cloud(raw_depth, settings)
    xyz, planes = [], []
    for i, centre in enumerate(uv):
        delta = pixels - centre
        u, v = delta @ along, delta @ across
        before, after = settings.strip_extents_px[i]
        mask = ((u >= -before) & (u <= after) & (np.abs(v) <= settings.strip_halfwidth_px)
                & (np.abs(u) > settings.hole_exclusion_px))
        plane = fit_plane(points[mask], settings)
        xyz.append(intersect_plane(calibration.ray(centre), plane, settings))
        planes.append(plane)
    xyz = np.asarray(xyz)
    spacing = float(np.linalg.norm(xyz[1]-xyz[0]))
    if abs(spacing - settings.expected_spacing_m) > settings.spacing_tolerance_m:
        raise ValueError('cad_spacing_mismatch')
    return xyz, planes, spacing


def surface_rotation(xyz, planes, settings):
    """Pair frame: X from hole 0 to 1, Z camera-facing surface normal, Y=Z×X.

    This is an inferred surface frame, not the original CAD coordinate frame.
    Two centres alone cannot determine rotation about their connecting line.
    """
    normals = np.array([p['normal'] for p in planes])
    angle = float(np.degrees(np.arccos(np.clip(normals[0] @ normals[1], -1, 1))))
    if angle > settings.max_normal_disagreement_deg:
        return None, 'surface_normals_disagree'
    if any(p['support_sigma_m'][1] < .003 for p in planes):
        return None, 'surface_normal_weak_support'
    x = np.asarray(xyz[1])-xyz[0]
    x /= np.linalg.norm(x)
    z = normals.mean(0)
    z /= np.linalg.norm(z)
    if abs(z @ x) > .15:
        return None, 'normal_not_perpendicular_to_hole_baseline'
    z -= x * (z @ x)
    z /= np.linalg.norm(z)
    return np.column_stack([x, np.cross(z, x), z]), None


def fit_dark_hole(gray, guess, bounds=None):
    """Small elliptical dark spot with an independent linear background per row."""
    cx, cy = np.asarray(guess, float)
    x0, y0 = int(round(cx)), int(round(cy))
    if x0 < 8 or y0 < 5 or x0 >= gray.shape[1]-8 or y0 >= gray.shape[0]-5:
        raise ValueError('hole_near_image_boundary')
    xx, yy = np.arange(x0-7, x0+8), np.arange(y0-4, y0+5)
    X, Y = np.meshgrid(xx, yy)
    data = gray[Y, X].astype(float)
    low, high = np.array([cx-2., cy-2.]), np.array([cx+2., cy+2.])
    if bounds is not None:
        low = np.maximum(low, bounds[:2])
        high = np.minimum(high, np.asarray(bounds[2:]) - 1e-3)
    if (low >= high).any():
        raise ValueError('empty_fit_bounds')
    def model(p):
        a, b, sx, sy, amplitude = p[:5]
        return (p[5:14, None] + p[14:, None] * (X-x0)
                - amplitude * np.exp(-.5 * (((X-a)/sx)**2 + ((Y-b)/sy)**2)))
    initial = np.r_[np.clip([cx, cy], low+1e-5, high-1e-5), 1.2, 1., 40,
                    np.median(data, axis=1), np.zeros(9)]
    lo = np.r_[low, .5, .35, 0, np.zeros(9), np.full(9, -20)]
    hi = np.r_[high, 2.8, 2.8, 200, np.full(9, 255), np.full(9, 20)]
    solution = least_squares(lambda p: (model(p)-data).ravel(), initial,
                             bounds=(lo, hi), loss='soft_l1', f_scale=4, max_nfev=100)
    p = solution.x
    return p[:2], {'contrast': float(p[4]), 'rmse': float(np.sqrt(np.mean(solution.fun**2))),
                    'sigma_px': p[2:4].tolist(), 'converged': bool(solution.success),
                    'at_bound': bool(np.any(p[:2]-low < .08) or np.any(high-p[:2] < .08))}


def fit_is_good(quality, settings):
    return (quality['converged'] and not quality['at_bound']
            and max(quality['sigma_px']) < 2.75
            and quality['contrast'] >= settings.min_contrast and quality['rmse'] <= settings.max_fit_rmse)


def initialise_boxes(gray, boxes, settings):
    boxes = np.asarray(boxes, float)
    h, w = gray.shape
    if (boxes.shape != (2, 4) or not np.isfinite(boxes).all()
            or (boxes[:, :2] < 0).any() or (boxes[:, 2:] > [w, h]).any()
            or (boxes[:, 2:] - boxes[:, :2] < 3).any()):
        raise ValueError('Provide two valid [xmin,ymin,xmax,ymax) RGB boxes, at least 3 px wide/high')
    points, qualities = [], []
    response = (cv2.GaussianBlur(gray.astype(float), (0, 0), 3., sigmaY=.8)
                - cv2.GaussianBlur(gray.astype(float), (0, 0), .6))
    for i, box in enumerate(boxes):
        x0, y0 = np.ceil(box[:2]).astype(int)
        x1, y1 = np.ceil(box[2:]).astype(int)
        patch = response[y0:y1, x0:x1].copy()
        guesses = [(box[:2] + box[2:]) / 2]
        for _ in range(3):
            yy, xx = np.unravel_index(np.argmax(patch), patch.shape)
            guesses.append(np.array([xx+x0, yy+y0], float))
            patch[max(0, yy-2):yy+3, max(0, xx-2):xx+3] = -np.inf
        candidates = []
        for guess in guesses:
            try:
                point, quality = fit_dark_hole(gray, guess, box)
            except ValueError:
                continue
            if fit_is_good(quality, settings):
                score = quality['contrast'] / max(quality['rmse'], 3.)
                candidates.append((score, point, quality))
        if not candidates:
            raise ValueError(f'bbox_{i}: no reliable dark hole; choose a clearer seed frame or tighter box')
        candidates.sort(key=lambda c: c[0], reverse=True)
        score, point, quality = candidates[0]
        if any(np.linalg.norm(point-q) > 2.5 and s > .8*score for s, q, _ in candidates[1:]):
            raise ValueError(f'bbox_{i}: multiple dark spots; narrow the bbox to the intended hole')
        points.append(point)
        qualities.append(quality)
    if np.linalg.norm(points[0]-points[1]) < 10:
        raise ValueError('Both boxes selected the same or adjacent spot')
    return np.array(points), qualities


def patch_ncc(a, b, pa, pb):
    a = cv2.getRectSubPix(a, (25, 15), tuple(np.asarray(pa, np.float32))).astype(float)
    b = cv2.getRectSubPix(b, (25, 15), tuple(np.asarray(pb, np.float32))).astype(float)
    a, b = a-a.mean(), b-b.mean()
    return float(np.sum(a*b) / (np.linalg.norm(a)*np.linalg.norm(b)+1e-9))


class ImageTracker:
    """Short bounded image-only recovery; never publish predicted holes as measurements."""
    def __init__(self, gray, points, timestamp_s, settings):
        self.gray, self.points, self.time = gray.copy(), np.array(points, float), timestamp_s
        self.settings = settings
        self.lost = False
        self.last_verified_time = timestamp_s
        self.failures = 0

    def update(self, gray, timestamp_s):
        if self.lost:
            return None, {'reason': 'reinitialization_required'}
        if abs(timestamp_s-self.last_verified_time) > self.settings.max_recovery_s:
            self.lost = True
            return None, {'reason': 'reinitialization_required'}
        dt = abs(timestamp_s-self.time)
        if not 0 < dt <= .2:
            self.lost = True
            return None, {'reason': 'image_timestamp_gap'}
        p = self.points.astype(np.float32).reshape(-1, 1, 2)
        options = dict(winSize=(25, 15), maxLevel=2,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
        q, status, _ = cv2.calcOpticalFlowPyrLK(self.gray, gray, p, None, **options)
        if q is None or not np.isfinite(q).all():
            self.lost = True
            return None, {'reason': 'optical_flow_failed'}
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, q, None, **options)
        if back is None:
            self.lost = True
            return None, {'reason': 'backward_flow_failed'}
        fb = np.linalg.norm(back.reshape(2, 2)-self.points, axis=1)
        result, qualities = [], []
        reason = None
        for i, predicted in enumerate(q.reshape(2, 2)):
            if (not status[i, 0] or not back_status[i, 0] or fb[i] > self.settings.max_fb_px
                    or not np.isfinite(predicted).all()):
                reason = 'optical_flow_inconsistent'
                break
            try:
                point, quality = fit_dark_hole(gray, predicted)
            except ValueError as exc:
                reason = str(exc)
                break
            ncc = patch_ncc(self.gray, gray, self.points[i], point)
            quality.update(fb_px=float(fb[i]), ncc=ncc)
            if not fit_is_good(quality, self.settings):
                reason = 'hole_appearance_unreliable'
            elif ncc < self.settings.min_patch_ncc:
                reason = 'patch_appearance_changed'
            elif np.linalg.norm(point-predicted) > self.settings.max_center_correction_px:
                reason = 'hole_refinement_jump'
            elif np.linalg.norm(point-self.points[i]) > self.settings.max_image_speed_px_s*dt+1:
                reason = 'image_motion_too_fast'
            qualities.append(quality)
            if reason:
                break
            result.append(point)
        if reason:
            # Preserve only a bounded image-motion prediction through reflections.
            # Neither predicted centres nor old XYZ are emitted as valid output.
            proposed = q.reshape(2, 2)
            coarse_ok = (status.all() and back_status.all() and
                         np.all(fb <= self.settings.max_fb_px) and
                         np.all(np.linalg.norm(proposed-self.points, axis=1)
                                <= self.settings.max_image_speed_px_s*dt+1) and
                         all(patch_ncc(self.gray, gray, self.points[i], proposed[i]) >= .8
                             for i in range(2)))
            if coarse_ok:
                self.gray, self.points, self.time = gray.copy(), proposed.astype(float), timestamp_s
            self.failures += 1
            return None, {'reason': reason, 'holes': qualities}
        self.gray, self.points, self.time = gray.copy(), np.array(result), timestamp_s
        recovered = self.failures
        self.failures = 0
        self.last_verified_time = timestamp_s
        return self.points.copy(), {'holes': qualities, 'recovered_after_invalid_frames': recovered}


class OneEuro:
    """Causal adaptive low-pass filter, with explicit reset on missing measurements."""
    def __init__(self, settings):
        self.settings = settings
        self.reset()

    def reset(self):
        self.time = None
        self.value = self.previous_raw = self.derivative = None

    @staticmethod
    def alpha(cutoff, dt):
        return 1 / (1 + 1/(2*np.pi*cutoff*dt))

    def update(self, value, timestamp_s):
        value = np.asarray(value, float)
        if not np.isfinite(value).all():
            self.reset()
            raise ValueError('nonfinite_filter_input')
        if self.time is not None and timestamp_s <= self.time:
            raise ValueError('filter_timestamps_must_increase')
        if self.time is None or timestamp_s-self.time > self.settings.max_filter_gap_s:
            self.value, self.previous_raw = value.copy(), value.copy()
            self.derivative = np.zeros_like(value)
        else:
            dt = timestamp_s-self.time
            a = self.alpha(1., dt)
            self.derivative += a*((value-self.previous_raw)/dt-self.derivative)
            cutoff = self.settings.min_cutoff_hz + self.settings.filter_beta*np.abs(self.derivative)
            self.value += self.alpha(cutoff, dt)*(value-self.value)
            self.previous_raw = value.copy()
        self.time = timestamp_s
        return self.value.copy()


class Episode:
    def __init__(self, path, camera, settings):
        self.path, self.camera, self.settings = Path(path), camera, settings
        self.document = json.loads((self.path/'episode.json').read_text())
        info = self.document['metadata']['cameras'][camera]
        self.calibration = Calibration(info)
        self.fps = float(info['fps'])
        folder = self.path/'raw_spool/cameras'
        self.rgb_stamps = np.fromfile(folder/f'{camera}_rgb.bin', dtype=STAMP)
        self.depth_stamps = np.fromfile(folder/f'{camera}_depth.bin', dtype=STAMP)
        for stamps in (self.rgb_stamps, self.depth_stamps):
            if len(stamps) == 0 or np.any(np.diff(stamps['time_ns']) <= 0):
                raise ValueError('Empty or nonmonotonic camera timestamps')
        self.raw = np.memmap(folder/f'{camera}_depth.raw', '<u2', mode='r')
        expected = len(self.depth_stamps)*np.prod(self.calibration.depth_shape)
        if self.raw.size != expected:
            raise ValueError('Raw depth size does not match metadata')
        self.raw = self.raw.reshape(len(self.depth_stamps), *self.calibration.depth_shape)
        self.video = cv2.VideoCapture(str(self.path/f'{camera}.mp4'))
        if (not self.video.isOpened() or int(self.video.get(cv2.CAP_PROP_FRAME_COUNT)) != len(self.rgb_stamps)
                or abs(self.video.get(cv2.CAP_PROP_FPS)-self.fps) > .01):
            self.video.release()
            raise ValueError('Video frame count/fps does not match camera metadata')
        self.next_index = 0

    def read(self, index):
        if not 0 <= index < len(self.rgb_stamps):
            raise ValueError('Video frame index out of range')
        if self.next_index != index:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = self.video.read()
        if not ok or image.shape[:2] != self.calibration.shape:
            raise ValueError(f'Cannot decode video frame {index}')
        self.next_index = index+1
        stamp = int(self.rgb_stamps[index]['time_ns'])
        j = int(np.searchsorted(self.depth_stamps['time_ns'], stamp))
        candidates = [k for k in (j-1, j) if 0 <= k < len(self.depth_stamps)]
        j = min(candidates, key=lambda k: abs(int(self.depth_stamps[k]['time_ns'])-stamp))
        offset = int(self.depth_stamps[j]['time_ns'])-stamp
        depth = self.raw[j] if abs(offset) <= self.settings.max_sync_ms*1e6 else None
        return image, depth, stamp, j, offset

    def close(self):
        self.video.release()


def observe(index, points, image_quality, depth, stamp, depth_index, offset, episode):
    record = {'frame_index': index, 'video_time_s': index/episode.fps,
              'timestamp_ns': stamp, 'depth_index': depth_index, 'depth_minus_rgb_ns': offset,
              'valid': False, 'reason': None, 'uv_raw_px': None if points is None else points.tolist(),
              'xyz_raw_m': None, 'xyz_filtered_m': None, 'planes': None,
              'surface_orientation_valid': False, 'rotation_camera_pair': None,
              'T_camera_holes': None, 'stationary': False, 'stable_xyz_m': None,
              'image_quality': image_quality}
    if points is None:
        record['reason'] = image_quality['reason']
        return record
    if depth is None:
        record['reason'] = 'rgb_depth_timestamp_mismatch'
        return record
    try:
        xyz, planes, spacing = measure_centres(depth, points, episode.calibration, episode.settings)
    except ValueError as exc:
        record['reason'] = str(exc)
        return record
    rotation, orientation_reason = (surface_rotation(xyz, planes, episode.settings)
                                    if episode.settings.estimate_surface_orientation
                                    else (None, 'disabled_unverified_surface_orientation'))
    record.update(valid=True, xyz_raw_m=xyz.tolist(), planes=planes, spacing_m=spacing,
                  orientation_reason=orientation_reason, surface_orientation_valid=rotation is not None,
                  rotation_camera_pair=None if rotation is None else rotation.tolist())
    return record


def smooth_records(records, settings):
    filt, window = OneEuro(settings), deque()
    previous = None
    for record in records:
        if not record['valid']:
            filt.reset()
            window.clear()
            continue
        stamp = record['timestamp_ns']/1e9
        xyz = np.array(record['xyz_raw_m'])
        if previous is not None:
            dt = stamp-previous[0]
            if (0 < dt <= settings.max_filter_gap_s and
                    np.max(np.linalg.norm(xyz-previous[1], axis=1)) > settings.max_translation_speed_m_s*dt+.003):
                record.update(valid=False, reason='position_jump', xyz_rejected_m=record['xyz_raw_m'],
                              xyz_raw_m=None, xyz_filtered_m=None, surface_orientation_valid=False,
                              rotation_camera_pair=None, T_camera_holes=None, stationary=False, stable_xyz_m=None)
                filt.reset()
                window.clear()
                continue
        previous = stamp, xyz.copy()
        smooth = filt.update(xyz, stamp)
        record['xyz_filtered_m'] = smooth.tolist()
        if record['surface_orientation_valid']:
            transforms = np.repeat(np.eye(4)[None], 2, axis=0)
            transforms[:, :3, :3] = record['rotation_camera_pair']
            transforms[:, :3, 3] = smooth
            record['T_camera_holes'] = transforms.tolist()
        window.append((stamp, xyz))
        while window and stamp-window[0][0] > settings.stationary_window_s+.05:
            window.popleft()
        if len(window) < 6 or stamp-window[0][0] < settings.stationary_window_s:
            continue
        values = np.array([v for _, v in window])
        centre = np.median(values, axis=0)
        spread = float(np.max(np.linalg.norm(values-centre, axis=2)))
        speed = float(np.max(np.linalg.norm(values[-3:].mean(0)-values[:3].mean(0), axis=1))
                      / (stamp-window[0][0]))
        record['stationary'] = spread <= settings.stationary_spread_m and speed <= settings.stationary_speed_m_s
        if record['stationary']:
            record['stable_xyz_m'] = centre.tolist()


def summarize(records):
    good = [r for r in records if r['valid']]
    raw_steps, smooth_steps, correction = [], [], []
    for a, b in zip(records, records[1:]):
        if a['valid'] and b['valid'] and b['timestamp_ns']-a['timestamp_ns'] <= 150_000_000:
            raw_steps.append(np.array(b['xyz_raw_m'])-a['xyz_raw_m'])
            smooth_steps.append(np.array(b['xyz_filtered_m'])-a['xyz_filtered_m'])
    for r in good:
        correction.append(np.linalg.norm(np.array(r['xyz_filtered_m'])-r['xyz_raw_m'], axis=1))
    def rms(steps):
        return None if not steps else float(np.sqrt(np.mean(np.sum(np.array(steps)**2, axis=-1)))*1000)
    return {'frames': len(records), 'valid_frames': len(good),
            'valid_fraction': len(good)/len(records) if records else 0,
            'invalid_reasons': dict(Counter(r['reason'] for r in records if not r['valid'])),
            'surface_orientation_valid_frames': sum(r['surface_orientation_valid'] for r in records),
            'stationary_frames': sum(r['stationary'] for r in records),
            'raw_adjacent_step_rms_mm': rms(raw_steps), 'filtered_adjacent_step_rms_mm': rms(smooth_steps),
            'median_filter_displacement_mm': None if not correction else float(np.median(correction)*1000),
            'physical_accuracy_verified': False,
            'metric_note': 'Step RMS describes motion plus noise, not accuracy or stationary jitter; filtering introduces lag.'}


def save_outputs(output, records, episode, metadata, preview):
    output.mkdir(parents=True, exist_ok=False)
    with (output/'trajectory.jsonl').open('w') as f:
        for r in records:
            f.write(json.dumps(r, allow_nan=False)+'\n')
    with (output/'trajectory.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'video_s', 'timestamp_ns', 'hole_id', 'valid', 'reason',
                         'u_px', 'v_px', 'raw_x_m', 'raw_y_m', 'raw_z_m',
                         'filtered_x_m', 'filtered_y_m', 'filtered_z_m', 'stationary'])
        for r in records:
            for i in range(2):
                writer.writerow([r['frame_index'], r['video_time_s'], r['timestamp_ns'], i,
                                 r['valid'], r['reason'], *(r['uv_raw_px'][i] if r['uv_raw_px'] else ['', '']),
                                 *(r['xyz_raw_m'][i] if r['valid'] else ['', '', '']),
                                 *(r['xyz_filtered_m'][i] if r['valid'] else ['', '', '']), r['stationary']])
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    summary = summarize(records)
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    writer = None
    try:
        if preview:
            h, w = episode.calibration.shape
            writer = cv2.VideoWriter(str(output/'overlay.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), episode.fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError('Cannot create overlay video')
        for j, r in enumerate(records):
            if not preview and j not in (0, len(records)-1):
                continue
            image, *_ = episode.read(r['frame_index'])
            if r['uv_raw_px']:
                for i, uv in enumerate(r['uv_raw_px']):
                    point = tuple(np.round(uv).astype(int))
                    cv2.drawMarker(image, point, (0, 230, 255), cv2.MARKER_CROSS, 9, 1)
                    cv2.putText(image, f'H{i}', (point[0]+6, point[1]-6), 0, .4, (0, 230, 255), 1)
            status = 'VALID' if r['valid'] else 'INVALID: '+str(r['reason'])
            cv2.rectangle(image, (0, 0), (image.shape[1], 26), (0, 0, 0), -1)
            cv2.putText(image, f"{r['video_time_s']:.3f}s {status}", (8, 18), 0, .45, (255, 255, 255), 1)
            if writer:
                writer.write(image)
            if j in (0, len(records)-1):
                cv2.imwrite(str(output/f"frame_{r['frame_index']:06d}.png"), image)
    finally:
        if writer:
            writer.release()
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--episode', type=Path, required=True)
    p.add_argument('--camera', default='camera_0')
    init = p.add_mutually_exclusive_group(required=True)
    init.add_argument('--init-json', type=Path, help='JSON: seed_frame, bboxes_xyxy, optional settings/episode_sha256')
    init.add_argument('--bboxes', help='Two original RGB boxes: "xmin,ymin,xmax,ymax;xmin,ymin,xmax,ymax"')
    p.add_argument('--seed-time', type=float, help='Video seconds of the supplied bboxes; required with --bboxes')
    p.add_argument('--start-time', type=float, help='Inclusive start; defaults to seed time')
    p.add_argument('--end-time', type=float, help='Inclusive end; defaults to seed+2 seconds')
    p.add_argument('--output', type=Path, required=True, help='New output directory (never overwrite)')
    p.add_argument('--no-video', action='store_true', help='Skip annotated MP4; still save endpoint PNGs')
    p.add_argument('--surface-orientation', action='store_true',
                   help='Opt in to unverified surface-frame orientation; default is position only')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.output.exists():
        raise ValueError('Output already exists; choose a new directory')
    config = json.loads(args.init_json.read_text()) if args.init_json else {}
    settings = Settings(**config.get('settings', {}))
    if args.surface_orientation:
        settings.estimate_surface_orientation = True
    settings.validate()
    episode = Episode(args.episode, args.camera, settings)
    try:
        digest = hashlib.sha256((args.episode/'episode.json').read_bytes()).hexdigest()
        if config.get('episode_sha256') and config['episode_sha256'] != digest:
            raise ValueError('Preset belongs to a different recording; provide new bboxes')
        if args.init_json:
            if args.seed_time is not None:
                raise ValueError('--init-json already defines seed_frame; do not add --seed-time')
            seed = int(config['seed_frame'])
            boxes = config['bboxes_xyxy']
        else:
            if args.seed_time is None or not np.isfinite(args.seed_time) or args.seed_time < 0:
                raise ValueError('--bboxes requires a nonnegative --seed-time in video seconds')
            seed = round(args.seed_time*episode.fps)
            boxes = [[float(v) for v in box.split(',')] for box in args.bboxes.split(';')]
        start = seed if args.start_time is None else round(args.start_time*episode.fps)
        end = min(seed+round(2*episode.fps), len(episode.rgb_stamps)-1) if args.end_time is None else round(args.end_time*episode.fps)
        if not 0 <= start <= seed <= end < len(episode.rgb_stamps):
            raise ValueError('Require 0 <= start frame <= seed frame <= end frame < video frame count')
        image, depth, stamp, depth_index, offset = episode.read(seed)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        points, qualities = initialise_boxes(gray, boxes, settings)
        print(json.dumps({'seed_frame': seed, 'centres_px': points.tolist()}), flush=True)
        observations = {seed: observe(seed, points, {'holes': qualities}, depth, stamp, depth_index, offset, episode)}
        for indices in (range(seed-1, start-1, -1), range(seed+1, end+1)):
            tracker = ImageTracker(gray, points, stamp/1e9, settings)
            for index in indices:
                image, depth, time_ns, depth_index, offset = episode.read(index)
                uv, quality = tracker.update(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), time_ns/1e9)
                observations[index] = observe(index, uv, quality, depth, time_ns, depth_index, offset, episode)
                if abs(index-seed) % 30 == 0:
                    print(json.dumps({'frame': index, 'valid': observations[index]['valid'],
                                      'reason': observations[index]['reason']}), flush=True)
        records = [observations[i] for i in range(start, end+1)]
        smooth_records(records, settings)
        metadata = {'episode': str(args.episode.resolve()), 'episode_sha256': digest, 'camera': args.camera,
                    'coordinate_frame': 'RGB optical: X right, Y down, Z forward; metres',
                    'bbox_convention': '[xmin,ymin,xmax,ymax), original RGB pixels',
                    'hole_ids': 'Input bbox order is preserved, even if left/right order changes',
                    'seed_frame': seed, 'bboxes_xyxy': boxes, 'settings': asdict(settings),
                    'start_frame': start, 'end_frame': end,
                    'uses_gpt': False, 'uses_foundationpose': False,
                    'tracking': 'Local LK + dark-spot fit; bounded image-only recovery, prolonged loss requires new bboxes',
                    'filtering': 'Causal OneEuro in timestamp order; reset after invalid data',
                    'backward_tracking_used': start < seed,
                    'orientation': 'Optional inferred surface frame, unsmoothed; X H0->H1, Z toward camera; not CAD frame',
                    'limitations': 'Temporal smoothness is not absolute accuracy. Narrow-strip depth normals can be biased. No robot commands.'}
        summary = save_outputs(args.output, records, episode, metadata, not args.no_video)
        print(json.dumps({'output': str(args.output.resolve()), **summary}, indent=2))
        return 0 if summary['valid_frames'] else 2
    finally:
        episode.close()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError) as error:
        print(f'error: {error}', file=sys.stderr)
        raise SystemExit(2)
