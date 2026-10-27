"""Independent geometry, tracking-loss and causal-filter checks; no hardware/GPT."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from tools import track_rail_holes as tracker


def calibration_info():
    intrinsic = dict(width=320, height=200, fx=400., fy=405., ppx=160., ppy=100.,
                     model='distortion.brown_conrady', coeffs=[0.] * 5)
    rotation = cv2.Rodrigues(np.array([.01, .025, -.006]))[0]
    return dict(rgb_intrinsics={**intrinsic, 'coeffs': [-.04, .015, .001, -.002, 0.]},
                depth_intrinsics=intrinsic, depth_scale=.0001, fps=30,
                depth_to_rgb=dict(rotation=rotation.ravel(order='F').tolist(),
                                  translation=[.015, .001, .002]))


def synthetic_plane_scene():
    """Render depth by intersecting depth-camera rays with a known RGB-frame plane."""
    info = calibration_info()
    rotation = np.array(info['depth_to_rgb']['rotation']).reshape(3, 3, order='F')
    translation = np.array(info['depth_to_rgb']['translation'])
    depth = info['depth_intrinsics']
    yy, xx = np.indices((depth['height'], depth['width']))
    rays = np.stack([(xx-depth['ppx'])/depth['fx'],
                     (yy-depth['ppy'])/depth['fy'], np.ones_like(xx)], axis=-1)
    normal = np.array([-.08, .05, 1.])
    z = (.75-normal@translation) / ((rays@rotation.T)@normal)
    raw = np.round(z/info['depth_scale']).astype(np.uint16)
    expected = np.array([[-.1325, 0., .75-.08*.1325], [.1325, 0., .75+.08*.1325]])
    rgb = info['rgb_intrinsics']
    K = np.array([[rgb['fx'], 0, rgb['ppx']], [0, rgb['fy'], rgb['ppy']], [0, 0, 1.]])
    uv = cv2.projectPoints(expected, np.zeros(3), np.zeros(3), K, np.array(rgb['coeffs']))[0][:, 0]
    return info, raw, uv, expected


def hole_image(centres, occlude=False):
    """Pixel-integrated dark circular openings with row highlights and blur."""
    scale, width, height = 8, 260, 120
    yy = np.arange(height*scale)[:, None]/scale
    canvas = np.broadcast_to(155.+15*np.sin(yy*.45), (height*scale, width*scale)).copy()
    for x, y in centres:
        cv2.ellipse(canvas, (round(x*scale), round(y*scale)), (round(1.7*scale), round(1.25*scale)),
                    0, 0, 360, 45., -1)
    canvas = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)
    canvas = cv2.GaussianBlur(canvas, (0, 0), .5)
    if occlude:
        x, y = np.round(centres[0]).astype(int)
        canvas[y-12:y+13, x-15:x+16] = 160.
    return np.clip(canvas, 0, 255).astype(np.uint8)


def record_at(t, xyz, valid=True):
    return dict(valid=valid, timestamp_ns=round(t*1e9), xyz_raw_m=np.asarray(xyz).tolist() if valid else None,
                xyz_filtered_m=None, surface_orientation_valid=False, rotation_camera_pair=None,
                T_camera_holes=None, stationary=False, stable_xyz_m=None)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.settings = tracker.Settings()

    def test_invalid_calibration_never_enters_geometry(self):
        cases = [
            ('depth_to_rgb', 'translation', [np.nan, 0., 0.]),
            ('depth_to_rgb', 'rotation', np.diag([-1., 1., 1.]).ravel(order='F').tolist()),
            ('rgb_intrinsics', 'fx', 0.),
            ('depth_intrinsics', 'fy', -400.),
            ('rgb_intrinsics', 'coeffs', [np.nan, 0., 0., 0., 0.]),
            ('depth_intrinsics', 'coeffs', [np.nan, 0., 0., 0., 0.]),
        ]
        for section, key, value in cases:
            info = calibration_info()
            info[section][key] = value
            with self.subTest(section=section, key=key):
                with self.assertRaises(ValueError):
                    tracker.Calibration(info)

    def test_two_centres_on_tilted_plane_with_distortion_and_nonidentity_extrinsics(self):
        info, raw, uv, expected = synthetic_plane_scene()
        xyz, planes, spacing = tracker.measure_centres(raw, uv, tracker.Calibration(info), self.settings)
        np.testing.assert_allclose(xyz, expected, atol=2e-4, rtol=0)
        self.assertAlmostEqual(spacing, np.linalg.norm(expected[1]-expected[0]), delta=2e-4)
        for plane, point in zip(planes, xyz):
            self.assertLess(np.dot(plane['normal'], point), 0, 'normal must face the camera')
        # The check must not force the measured distance to the nominal 265 mm.
        self.assertGreater(abs(spacing-self.settings.expected_spacing_m), .0005)

    def test_zero_or_nonfinite_depth_does_not_create_a_centre(self):
        info, raw, uv, _ = synthetic_plane_scene()
        for invalid in (np.zeros_like(raw), np.full(raw.shape, np.nan)):
            with self.subTest(kind=str(invalid.dtype)):
                with self.assertRaisesRegex(ValueError, 'insufficient_depth_support'):
                    tracker.measure_centres(invalid, uv, tracker.Calibration(info), self.settings)

    def test_wrong_hole_pair_fails_metric_spacing_gate(self):
        info, raw, uv, _ = synthetic_plane_scene()
        wrong = uv.copy()
        wrong[1, 0] -= 30
        with self.assertRaisesRegex(ValueError, 'cad_spacing_mismatch'):
            tracker.measure_centres(raw, wrong, tracker.Calibration(info), self.settings)

    def test_plane_rejects_outliers_and_recovers_known_surface(self):
        x, y = np.meshgrid(np.linspace(-.08, .08, 25), np.linspace(-.012, .012, 15))
        z = .7+.08*x-.04*y
        points = np.c_[x.ravel(), y.ravel(), z.ravel()]
        points[::37, 2] += .012
        plane = tracker.fit_plane(points, self.settings)
        answer = tracker.intersect_plane(np.array([.05, -.01, 1.]), plane, self.settings)
        expected_z = .7/(1-.08*.05+.04*(-.01))
        np.testing.assert_allclose(answer, np.array([.05, -.01, 1.])*expected_z, atol=1e-9)
        self.assertLess(plane['inliers'], len(points))

    def test_degenerate_plane_and_grazing_ray_are_rejected(self):
        line = np.c_[np.linspace(-.1, .1, 100), np.zeros(100), np.full(100, .7)]
        with self.assertRaisesRegex(ValueError, 'degenerate_plane_support'):
            tracker.fit_plane(line, self.settings)
        with self.assertRaisesRegex(ValueError, 'grazing_ray'):
            tracker.intersect_plane(np.array([0., 0., 1.]),
                                    dict(normal=[1., 0., 0.], offset_m=-.1), self.settings)

    def test_surface_rotation_is_right_handed_and_conflicting_normals_return_none(self):
        xyz = np.array([[-.1325, 0, .7], [.1325, 0, .7]])
        plane = dict(normal=[0., 0., -1.], support_sigma_m=[.08, .008, 0.])
        rotation, reason = tracker.surface_rotation(xyz, [plane, plane], self.settings)
        self.assertIsNone(reason)
        np.testing.assert_allclose(rotation.T@rotation, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.)
        np.testing.assert_allclose(rotation[:, 0], [1, 0, 0])
        np.testing.assert_allclose(rotation[:, 2], [0, 0, -1])
        conflict = dict(plane, normal=[0., .5, -np.sqrt(.75)])
        rotation, reason = tracker.surface_rotation(xyz, [plane, conflict], self.settings)
        self.assertIsNone(rotation)
        self.assertEqual(reason, 'surface_normals_disagree')
        weak = dict(plane, support_sigma_m=[.08, .0015, 0.])
        rotation, reason = tracker.surface_rotation(xyz, [weak, weak], self.settings)
        self.assertIsNone(rotation)
        self.assertEqual(reason, 'surface_normal_weak_support')


class ImageTrackingTests(unittest.TestCase):
    def setUp(self):
        self.settings = tracker.Settings()
        self.centres = np.array([[70.7, 65.3], [190.4, 60.2]])
        self.gray = hole_image(self.centres)
        self.boxes = np.c_[self.centres-4, self.centres+4]

    def test_numeric_boxes_find_centres_and_preserve_supplied_identity_order(self):
        points, quality = tracker.initialise_boxes(self.gray, self.boxes[::-1], self.settings)
        # Half-resolution raster convention contributes up to ~0.5 pixel here.
        np.testing.assert_allclose(points, self.centres[::-1], atol=.7, rtol=0)
        self.assertTrue(all(tracker.fit_is_good(q, self.settings) for q in quality))

    def test_invalid_boxes_and_featureless_boxes_fail(self):
        for boxes in ([[-1, 2, 8, 9], [10, 10, 20, 20]], self.boxes[:1],
                      [[3, 3, 4, 4], [10, 10, 20, 20]]):
            with self.subTest(boxes=str(boxes)):
                with self.assertRaises(ValueError):
                    tracker.initialise_boxes(self.gray, boxes, self.settings)
        with self.assertRaisesRegex(ValueError, 'no reliable dark hole'):
            tracker.initialise_boxes(np.full_like(self.gray, 150), self.boxes, self.settings)

    def test_subpixel_motion_tracks_without_reordering_hole_ids(self):
        points, _ = tracker.initialise_boxes(self.gray, self.boxes[::-1], self.settings)
        flow = tracker.ImageTracker(self.gray, points, 0., self.settings)
        for i in range(1, 6):
            displacement = np.array([.35*i, -.2*i])
            image = hole_image(self.centres+displacement)
            measured, quality = flow.update(image, i/30)
            self.assertIsNotNone(measured, quality)
            np.testing.assert_allclose(measured, self.centres[::-1]+displacement, atol=.75, rtol=0)

    def test_occlusion_outputs_no_stale_centres_and_short_recovery_is_verified(self):
        points, _ = tracker.initialise_boxes(self.gray, self.boxes, self.settings)
        flow = tracker.ImageTracker(self.gray, points, 0., self.settings)
        measured, quality = flow.update(hole_image(self.centres, occlude=True), 1/30)
        self.assertIsNone(measured)
        measured, quality = flow.update(self.gray, 2/30)
        self.assertIsNotNone(measured, quality)
        np.testing.assert_allclose(measured, self.centres, atol=.7, rtol=0)
        self.assertEqual(quality['recovered_after_invalid_frames'], 1)

    def test_loss_cannot_be_recovered_after_the_allowed_time_window(self):
        points, _ = tracker.initialise_boxes(self.gray, self.boxes, self.settings)
        flow = tracker.ImageTracker(self.gray, points, 0., self.settings)
        measured, quality = flow.update(hole_image(self.centres, occlude=True), 1/30)
        self.assertIsNone(measured)
        measured, quality = flow.update(self.gray, self.settings.max_recovery_s+.01)
        self.assertIsNone(measured)
        self.assertEqual(quality['reason'], 'reinitialization_required')
        self.assertTrue(flow.lost)
        measured, quality = flow.update(self.gray, self.settings.max_recovery_s+.04)
        self.assertIsNone(measured)

    def test_timestamp_gap_invalidates_even_identical_images(self):
        points, _ = tracker.initialise_boxes(self.gray, self.boxes, self.settings)
        flow = tracker.ImageTracker(self.gray, points, 0., self.settings)
        measured, quality = flow.update(self.gray, .3)
        self.assertIsNone(measured)
        self.assertEqual(quality['reason'], 'image_timestamp_gap')


class FilterTests(unittest.TestCase):
    def setUp(self):
        self.settings = tracker.Settings()
        self.xyz = np.array([[-.13, .1, .7], [.135, .1, .7]])

    def test_smoothing_is_causal_and_reduces_stationary_measurement_noise(self):
        rng = np.random.default_rng(9)
        rows = [record_at(i/30, self.xyz+rng.normal(0, .001, (2, 3))) for i in range(50)]
        prefix, whole = copy.deepcopy(rows[:20]), copy.deepcopy(rows)
        tracker.smooth_records(prefix, self.settings)
        tracker.smooth_records(whole, self.settings)
        np.testing.assert_array_equal([r['xyz_filtered_m'] for r in prefix],
                                      [r['xyz_filtered_m'] for r in whole[:20]])
        raw = np.array([r['xyz_raw_m'] for r in whole[10:]])
        filtered = np.array([r['xyz_filtered_m'] for r in whole[10:]])
        self.assertLess(np.mean((filtered-self.xyz)**2), np.mean((raw-self.xyz)**2))

    def test_invalid_observation_resets_filter_and_stationary_window(self):
        rows = [record_at(i/30, self.xyz) for i in range(16)]
        rows += [record_at(16/30, self.xyz, valid=False), record_at(17/30, self.xyz+[.004, 0., 0.])]
        tracker.smooth_records(rows, self.settings)
        self.assertTrue(rows[15]['stationary'])
        self.assertIsNone(rows[16]['xyz_filtered_m'])
        self.assertIsNone(rows[16]['stable_xyz_m'])
        np.testing.assert_array_equal(rows[17]['xyz_filtered_m'], rows[17]['xyz_raw_m'])
        self.assertFalse(rows[17]['stationary'])

    def test_metric_jumps_do_not_contaminate_next_valid_position_or_filter(self):
        # Both holes translate together, so the CAD-spacing gate cannot catch this.
        wrong = self.xyz+[.1, 0., 0.]
        returned = self.xyz+[.001, 0., 0.]
        rows = [record_at(0., self.xyz), record_at(1/30, wrong),
                record_at(2/30, wrong), record_at(3/30, returned)]
        tracker.smooth_records(rows, self.settings)
        for row in rows[1:3]:
            self.assertFalse(row['valid'])
            self.assertEqual(row['reason'], 'position_jump')
            self.assertIsNone(row['xyz_raw_m'])
            self.assertIsNone(row['xyz_filtered_m'])
            self.assertIsNone(row['stable_xyz_m'])
            self.assertIsNone(row['T_camera_holes'])
            np.testing.assert_array_equal(row['xyz_rejected_m'], wrong)
        self.assertTrue(rows[3]['valid'])
        np.testing.assert_array_equal(rows[3]['xyz_filtered_m'], returned)
        self.assertFalse(rows[3]['stationary'])

    def test_one_euro_resets_after_long_gap_or_nonfinite_measurement(self):
        filt = tracker.OneEuro(self.settings)
        filt.update(self.xyz, 0.)
        smooth = filt.update(self.xyz+.01, .03)
        self.assertTrue(np.all(smooth < self.xyz+.01))
        np.testing.assert_array_equal(filt.update(self.xyz+.04, 1.), self.xyz+.04)
        with self.assertRaisesRegex(ValueError, 'nonfinite_filter_input'):
            filt.update(np.full((2, 3), np.nan), 1.03)
        np.testing.assert_array_equal(filt.update(self.xyz, 1.06), self.xyz)
        with self.assertRaisesRegex(ValueError, 'timestamps_must_increase'):
            filt.update(self.xyz, 1.06)


class EpisodeSynchronizationTests(unittest.TestCase):
    def test_nearest_depth_uses_its_timestamp_and_mismatch_returns_none(self):
        info = calibration_info()
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder/'episode.json').write_text(json.dumps({'metadata': {'cameras': {'camera_0': info}}}))
            spool = folder/'raw_spool/cameras'
            spool.mkdir(parents=True)
            for name, times in [('rgb', [1_000_000_000, 1_033_000_000, 1_066_000_000]),
                                ('depth', [1_000_010_000, 1_034_000_000, 1_080_000_000])]:
                stamps = np.zeros(3, tracker.STAMP)
                stamps['time_ns'] = times
                stamps['sequence'] = np.arange(3)
                stamps.tofile(spool/f'camera_0_{name}.bin')
            raw = np.stack([np.full((200, 320), z, dtype='<u2') for z in [7000, 7100, 7200]])
            raw.tofile(spool/'camera_0_depth.raw')
            video = Mock()
            video.isOpened.return_value = True
            video.get.side_effect = lambda key: {cv2.CAP_PROP_FRAME_COUNT: 3, cv2.CAP_PROP_FPS: 30}[key]
            video.read.return_value = True, np.zeros((200, 320, 3), np.uint8)
            with patch.object(tracker.cv2, 'VideoCapture', return_value=video):
                episode = tracker.Episode(folder, 'camera_0', tracker.Settings())
                try:
                    _, depth, _, index, offset = episode.read(1)
                    self.assertEqual(index, 1)
                    self.assertEqual(offset, 1_000_000)
                    self.assertTrue(np.all(depth == 7100))
                    _, depth, stamp, index, offset = episode.read(2)
                    self.assertIsNone(depth)
                    self.assertEqual(offset, 14_000_000)
                    observation = tracker.observe(2, np.array([[90., 100.], [230., 100.]]), {},
                                                  depth, stamp, index, offset, episode)
                    self.assertFalse(observation['valid'])
                    self.assertIsNone(observation['xyz_raw_m'])
                    self.assertEqual(observation['reason'], 'rgb_depth_timestamp_mismatch')
                finally:
                    episode.close()
                video.release.assert_called_once()


if __name__ == '__main__':
    unittest.main()
