"""Whole-robot directions, centered-CAD invariants and estimator state; no devices."""
import copy
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1]), str(HERE.parents[1]/"third_party/foundationpose/workspace/src")]
from rail_preinsert.body_orientation_prior import BodyOrientationPrior, PriorPoseEngine, PriorRefiner, PriorScorer
from rail_receive.logic import load_config


class PriorTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(HERE / "receive.yaml")
        self.settings = self.config["body_orientation_prior"]
        self.T_body_C = np.eye(4)
        self.T_body_C[:3, :3] = Rotation.from_euler("xyz", [31, -57, 83], degrees=True).as_matrix()
        self.T_body_C[:3, 3] = [.1, -.2, 1.3]
        self.prior = BodyOrientationPrior(self.T_body_C, self.settings)

    def pose(self, degrees=(0, 0, 0), branch=0):
        pose = np.eye(4)
        pose[:3, :3] = self.T_body_C[:3, :3].T @ Rotation.from_euler("xyz", degrees, degrees=True).as_matrix() @ self.prior.references[branch]
        pose[:3, 3] = [.1, .2, .8]
        return pose

    def test_axes_in_whole_body_and_both_end_branches(self):
        for branch in (0, 1):
            pose = self.pose(branch=branch)
            np.testing.assert_allclose(self.prior.angles(pose[:3, :3]), [0, 0], atol=2e-6)
            corrected, valid = self.prior.regularize_centered(pose[None])
            self.assertTrue(valid[0]); np.testing.assert_allclose(corrected[0], pose, atol=1e-12)
        # Camera Z and long-axis X are deliberately NOT the body-frame axes.
        camera_upright = self.prior.references[0]
        self.assertFalse(self.prior.admissible(camera_upright))

    def test_grid_covers_both_branches_within_body_limits(self):
        grid = self.prior.rotation_grid()
        tilt, heading = self.prior.angles(grid[:, :3, :3])
        self.assertLessEqual(tilt.max(), 15.000001)
        self.assertLessEqual(heading.max(), 30.000001)
        for reference in self.prior.references:
            distances = Rotation.from_matrix(self.T_body_C[:3, :3] @ grid[:, :3, :3] @ reference.T).magnitude()
            self.assertLess(distances.min(), 1e-12)
        np.testing.assert_allclose(np.linalg.det(grid[:, :3, :3]), 1, atol=1e-12)

    def test_soft_correction_preserves_center_and_changes_endpoint_consistently(self):
        raw = self.pose([8, -4, 12])
        fixed, valid = self.prior.regularize_centered(raw[None]); fixed = fixed[0]
        np.testing.assert_array_equal(raw[:3, 3], fixed[:3, 3])
        center = np.array([0., .24, -.004])
        original, corrected = raw.copy(), fixed.copy()
        original[:3, 3] -= raw[:3, :3] @ center
        corrected[:3, 3] -= fixed[:3, :3] @ center
        np.testing.assert_allclose(original[:3, :3] @ center + original[:3, 3],
                                   corrected[:3, :3] @ center + corrected[:3, 3], atol=1e-12)
        self.assertGreater(np.linalg.norm(original[:3, 3]-corrected[:3, 3]), .001)
        raw_error = Rotation.from_matrix(self.T_body_C[:3, :3] @ raw[:3, :3] @ self.prior.references[0].T).magnitude()
        fixed_error = Rotation.from_matrix(self.T_body_C[:3, :3] @ fixed[:3, :3] @ self.prior.references[0].T).magnitude()
        self.assertAlmostEqual(fixed_error/raw_error, .75)

    def test_correction_cap_and_zero_strength(self):
        raw = self.pose([0, 0, 40])
        fixed, _ = self.prior.regularize_centered(raw[None])
        change = Rotation.from_matrix(fixed[0, :3, :3] @ raw[:3, :3].T).magnitude()
        self.assertAlmostEqual(np.degrees(change), 5.)
        prior = BodyOrientationPrior(self.T_body_C, dict(self.settings, strength=0))
        np.testing.assert_allclose(prior.regularize_centered(raw[None])[0][0], raw, atol=1e-12)

    def test_contradictions_are_rejected_not_clamped(self):
        for degrees in ([180, 0, 0], [0, 31, 0], [0, 0, 46]):
            raw = self.pose(degrees)
            corrected, valid = self.prior.regularize_centered(raw[None])
            self.assertFalse(valid[0]); np.testing.assert_array_equal(corrected[0], raw)

    def test_invalid_settings_and_mesh_serial_fail_closed(self):
        for changes in ({"frame": "tianji_right_base"}, {"strength": 1}, {"strength": True},
                        {"strength": float("nan")}, {"max_correction_deg": 0},
                        {"search_heading_deg": 46}):
            with self.assertRaises(ValueError):
                BodyOrientationPrior(self.T_body_C, dict(self.settings, **changes))
        bad = copy.deepcopy(self.config); bad["body_orientation_prior"]["mesh_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "different CAD"):
            BodyOrientationPrior.from_config(bad)
        with patch("rail_preinsert.body_orientation_prior.camera_to_body", return_value=(self.T_body_C, {"camera_serial": "a"})):
            with self.assertRaisesRegex(ValueError, "camera"):
                BodyOrientationPrior.from_config(self.config, camera_serial="b")

    def test_disabled_prior_does_not_load_kinematics_or_calibration(self):
        with patch("rail_preinsert.body_orientation_prior.camera_to_body", side_effect=AssertionError("unexpected")):
            self.assertIsNone(BodyOrientationPrior.from_config({}))
            self.assertIsNone(BodyOrientationPrior.from_config({"body_orientation_prior": {"enabled": False}}))

    def test_real_calibration_removes_sim_world_yaw(self):
        prior = BodyOrientationPrior.from_config(self.config, camera_serial="405622076394")
        expected = [[-.999983, -.003737, .004487], [-.005759, .758116, -.652094], [-.000965, -.652109, -.758125]]
        np.testing.assert_allclose(prior.T_body_C[:3, :3], expected, atol=1e-6)
        goal = __import__("json").loads((HERE / "targets/rail_goal.json").read_text())["T_C_O_goal"]
        tilt, heading = prior.angles(np.asarray(goal)[:3, :3])
        self.assertLess(tilt, 5); self.assertLess(heading, 5)

    def test_scorer_rejects_invalid_high_score_and_batch_retry_is_forwarded(self):
        import torch
        raw = np.stack([self.pose([0, 0, 10]), self.pose([0, 0, 80])])
        delegate = Mock(); delegate.predict.return_value = torch.tensor(raw), None
        refiner = PriorRefiner(delegate, self.prior)
        refiner.predict()
        scorer_delegate = Mock(); scorer_delegate.predict.return_value = torch.tensor([1., 100.]), None
        scorer = PriorScorer(scorer_delegate, refiner)
        scores, _ = scorer.predict()
        self.assertEqual(int(scores.argmax()), 0); self.assertTrue(torch.isneginf(scores[1]))
        scorer.batch_size = 2; refiner.batch_size = 2
        self.assertEqual(delegate.batch_size, 2); self.assertEqual(scorer_delegate.batch_size, 2)

    def fake_engine(self, poses):
        import torch
        refiner = Mock(); refiner.predict.return_value = torch.tensor(poses), None
        scorer = Mock(); scorer.predict.return_value = torch.arange(len(poses), dtype=torch.float32), None
        estimator = NS(rot_grid=torch.eye(4)[None], refiner=refiner, scorer=scorer,
                       model_center=np.array([0., .24, -.004]), pose_last=None)
        engine = NS(estimator=estimator, initialized=False, reset=Mock())
        def infer(frame, initialize=False):
            result, _ = estimator.refiner.predict()
            if not engine.initialized or initialize:
                scores, _ = estimator.scorer.predict()
                estimator.best_id = scores.argmax()
                index = int(estimator.best_id)
            else:
                index = 0
            estimator.pose_last = result[index]
            original = result[index].numpy().copy()
            original[:3, 3] -= original[:3, :3] @ estimator.model_center
            engine.initialized = True
            return original, {"valid": True, "T_C_CAD": original.tolist()}
        engine.infer = infer
        return engine, refiner

    def test_register_and_tracking_state_match_returned_pose(self):
        raw = np.stack([self.pose([8, 0, 5]), self.pose([4, 0, 8], branch=1)])
        engine, delegate = self.fake_engine(raw)
        wrapped = PriorPoseEngine(engine, self.prior)
        for inputs in (raw, raw[0:1]):
            import torch
            delegate.predict.return_value = torch.tensor(inputs), None
            pose, record = wrapped.infer(None)
            state = engine.estimator.pose_last.numpy()
            np.testing.assert_allclose(pose[:3, :3], state[:3, :3], atol=1e-12)
            np.testing.assert_allclose(pose[:3, 3]+pose[:3, :3]@engine.estimator.model_center, state[:3, 3])
            visual = np.asarray(record["T_C_CAD_before_body_prior"])
            np.testing.assert_allclose(visual[:3, 3]+visual[:3, :3]@engine.estimator.model_center, state[:3, 3])
            self.assertGreater(record["orientation_prior"]["correction_deg"], 0.)

    def test_rejected_or_skipped_visual_refinement_resets_tracking(self):
        engine, _ = self.fake_engine(self.pose([180, 0, 0])[None])
        wrapped = PriorPoseEngine(engine, self.prior)
        with self.assertRaisesRegex(ValueError, "contradicts"):
            wrapped.infer(None)
        engine.reset.assert_called_once()
        engine.reset.reset_mock()
        engine.infer = lambda *a, **kw: (np.eye(4), {})
        with self.assertRaisesRegex(ValueError, "skipped"):
            wrapped.infer(None)
        engine.reset.assert_called_once()

    def test_live_worker_factory_installs_prior_and_disable_keeps_original_engine(self):
        import receive_perception_worker as worker
        engine, _ = self.fake_engine(self.pose()[None])
        args = NS(foundationpose_root=Path(self.config["foundationpose_root"]),
                  mesh=Path(self.config["mesh"]), camera_serial="405622076394")
        with patch("rail_pose.engine.PoseEngine", return_value=engine):
            self.assertIsInstance(worker.pose_engine(args, HERE, self.config), PriorPoseEngine)
            disabled = copy.deepcopy(self.config); disabled["body_orientation_prior"]["enabled"] = False
            self.assertIs(worker.pose_engine(args, HERE, disabled), engine)

    def test_preflight_reports_invalid_prior_before_device_start(self):
        import receive
        with patch("rail_preinsert.body_orientation_prior.BodyOrientationPrior.from_config",
                   side_effect=ValueError("wrong body frame")):
            report = receive.preflight(self.config, probe_gpu=False)
        self.assertFalse(report["ready"])
        self.assertTrue(any("wrong body frame" in value for value in report["problems"]))

    def test_jitter_metrics_do_not_bridge_rejected_frames(self):
        from tools.validate_body_prior import metrics
        records = []
        for index in range(7):
            pose = np.eye(4); pose[0, 3] = index * .001
            records.append({"index": index, "valid": index != 3, "T_C_CAD": pose.tolist()})
        result = metrics(records, np.zeros(3), burn_in=0)
        self.assertEqual(result["valid"], 6)
        self.assertEqual(result["three_frame_windows"], 2)
        self.assertEqual(result["three_frame_passed"], 2)
        self.assertAlmostEqual(result["endpoint_step_rms_mm"], 1.)


if __name__ == "__main__":
    unittest.main()
