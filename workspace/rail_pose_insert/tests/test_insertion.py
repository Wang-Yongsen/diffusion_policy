"""Deterministic insertion dynamics and safety contracts, without devices."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
from rail_receive.insertion import DEFAULT_INSERTION, InsertionController, InsertionFault, validate_settings


def synthetic_settings():
    """Synthetic test-only values: never a validated hardware profile."""
    s = deepcopy(DEFAULT_INSERTION)
    s.update(T_F_S=np.eye(4).tolist(), wrench_sign=1, hole_midpoint_cad_m=[0., 0., 0.],
        impedance=dict(stiffness=[100.]*6, damping=[.5]*6, nullspace_stiffness=10.,
                       nullspace_damping=.5, velocity_ratio=1, acceleration_ratio=1),
        baseline_duration_s=.04, baseline_min_samples=3, baseline_timeout_s=1.,
        state_max_age_s=.1, max_control_dt_s=.05, force_filter_tau_s=.001,
        contact_confirm_s=.04, candidate_confirm_s=.04, settle_s=.04,
        baseline_force_std_n=.2, baseline_torque_std_nm=.02,
        baseline_position_tolerance_m=.0001, orientation_tolerance_rad=.01,
        hand_shape_tolerance_rad=.02, max_reference_lead_m=.002, max_tracking_error_m=.005,
        approach_speed_m_s=.002, insert_speed_m_s=.001, retract_speed_m_s=.002,
        admittance_m_s_n=.0005, contact_force_n=1., target_force_n=2.,
        max_axial_force_n=5., max_lateral_force_n=4., max_tilt_torque_nm=.5,
        hard_force_n=10., hard_torque_nm=1., pitch_m=.0005, max_radius_m=.003,
        lateral_speed_m_s=.001, candidate_depth_m=.0004, candidate_force_drop_n=.7,
        candidate_confirm_depth_m=.0003, candidate_timeout_s=1., max_descent_m=.02,
        endpoint_tolerance_m=.0001, progress_epsilon_m=.00001,
        stall_timeout_s=1., phase_timeout_s=5., total_timeout_s=20., retract_timeout_s=15.,
        retract_unload_force_n=.5, retract_unload_torque_nm=.05)
    return s


class InsertionTests(unittest.TestCase):
    def fixture(self, settings=None, start_height=.01, grasp=None):
        settings = synthetic_settings() if settings is None else settings
        self.now, self.seq = 0., 0
        self.baseline = np.array([7., -12., 3., .1, .2, -.3])
        self.grasp = np.eye(4) if grasp is None else grasp
        self.goal = np.eye(4)
        self.start = self.goal @ np.linalg.inv(self.grasp)
        self.start[1, 3] += start_height
        self.controller = InsertionController(settings, self.grasp, self.goal, self.start, now=0.)
        self.actual = self.start.copy()
        self.raw = self.baseline.copy()
        self.hand = np.zeros(20)
        return self.controller

    def state(self, *, advance=True, delta=None, actual=None):
        if advance:
            self.now += .02
            self.seq = (self.seq+1) % 1_000_000
        if delta is not None:
            self.raw = self.baseline+np.asarray(delta)
        if actual is not None:
            self.actual = actual.copy()
        return dict(timestamp_ns=round(self.now*1e9), right_sequence=self.seq,
                    T_B_F=self.actual.copy(), wrench=self.raw.copy(), hand_position_rad=self.hand.copy(),
                    joints_rad=np.zeros(7))

    def tick(self, *, delta=None, follow=True, **extra):
        st = self.state(delta=delta)
        st.update(extra)
        target = self.controller.step(st, self.now)
        self.controller.accepted(target)
        if follow:
            self.actual = target.copy()
        return target

    def baseline_ready(self):
        for _ in range(10):
            self.tick()
            if self.controller.status == "approaching":
                return
        self.fail("Baseline did not finish")

    def contact(self):
        self.baseline_ready()
        self.actual[1, 3] -= .001
        self.controller.accepted(self.actual)
        for _ in range(10):
            self.tick(delta=[0., 2., 0., 0., 0., 0.])
            if self.controller.status == "searching":
                return
        self.fail("Contact did not enter search")

    def test_configuration_requires_calibration_limits_and_impedance(self):
        missing = validate_settings(deepcopy(DEFAULT_INSERTION))
        self.assertIn('T_F_S', missing)
        self.assertIn('hard_force_n', missing)
        self.assertIn('impedance', missing)
        self.assertNotIn('raw_force_limit_n', missing)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            validate_settings(DEFAULT_INSERTION, require_ready=True)
        self.assertEqual(validate_settings(synthetic_settings(), require_ready=True), [])
        for key, value in [('wrench_sign', 0), ('max_radius_m', float('nan')),
                           ('baseline_min_samples', 2.2), ('contact_force_n', 3.)]:
            s = synthetic_settings(); s[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_settings(s)
        s = synthetic_settings(); s['impedance']['damping'][0] = 2.
        with self.assertRaises(ValueError):
            validate_settings(s)

    def test_baseline_uses_unique_stable_samples_and_retains_loaded_bias(self):
        c = self.fixture()
        st = self.state()
        c.step(st, self.now)
        for _ in range(5):
            c.step(st, self.now)
        self.assertEqual(len(c.baseline_samples), 1)
        self.baseline_ready()
        np.testing.assert_allclose(c.baseline, self.baseline)
        self.assertFalse(c.summary()['assembly_verified'])
        self.assertEqual(c.status, 'approaching')
        self.assertIsNone(c.last_wrench)

    def test_unstable_baseline_and_hand_or_attitude_change_are_hard_faults(self):
        for case in ('noise', 'hand', 'attitude', 'translation'):
            with self.subTest(case=case):
                c = self.fixture()
                self.tick()
                if case == 'noise':
                    self.raw[0] += 3.
                    self.tick()
                    self.raw[0] -= 3.
                elif case == 'hand':
                    self.hand[0] = .03
                elif case == 'attitude':
                    self.actual[:3, :3] = Rotation.from_rotvec([.02, 0., 0.]).as_matrix()
                else:
                    self.actual[0, 3] += .0002
                with self.assertRaises(InsertionFault):
                    for _ in range(3):
                        self.tick()

    def test_transform_rotates_force_applies_sign_and_translates_moment(self):
        s = synthetic_settings()
        sensor = np.eye(4)
        sensor[:3, :3] = Rotation.from_rotvec([0., 0., np.pi/2]).as_matrix()
        sensor[0, 3] = .2
        s.update(T_F_S=sensor.tolist(), wrench_sign=-1, hole_midpoint_cad_m=[.05, 0., 0.])
        c = self.fixture(s)
        c.baseline = np.zeros(6)
        # Sensor +X -> base +Y, sign flips to -Y; (sensor - point) x F = -.15 Z Nm.
        result = c.compensated_wrench(np.eye(4), np.array([1., 0., 0., 0., 0., .1]))
        np.testing.assert_allclose(result, [0., -1., 0., 0., 0., -.25], atol=1e-12)

    def test_raw_hard_guard_is_optional_loaded_wrench_is_not_contact(self):
        c = self.fixture()
        c.hard_guard(self.state(), self.now)  # Loaded norm exceeds compensated limit.
        s = synthetic_settings(); s['raw_force_limit_n'] = 10.
        c = self.fixture(s)
        with self.assertRaisesRegex(InsertionFault, 'absolute raw'):
            c.hard_guard(self.state(), self.now)

    def test_unfiltered_force_and_point_moment_limits_precede_lowpass(self):
        for delta in ([0., 10., 0., 0., 0., 0.], [0., 0., 0., 0., 1.1, 0.]):
            with self.subTest(delta=delta):
                c = self.fixture(); self.baseline_ready()
                with self.assertRaisesRegex(InsertionFault, 'hard limit'):
                    self.tick(delta=delta)
                self.assertNotEqual(c.status, 'retracting')
        s = synthetic_settings(); s['hole_midpoint_cad_m'] = [1., 0., 0.]
        c = self.fixture(s); self.baseline_ready()
        with self.assertRaisesRegex(InsertionFault, 'hard limit'):
            self.tick(delta=[0., 1.1, 0., 0., 0., 0.])

    def test_feedback_freshness_checks_right_sequence_even_with_new_packet_timestamp(self):
        c = self.fixture()
        c.hard_guard(self.state(), self.now)
        self.now += .11
        st = self.state(advance=False)
        with self.assertRaisesRegex(InsertionFault, 'source counter is stale'):
            c.hard_guard(st, self.now)
        c = self.fixture()
        st = self.state()
        with self.assertRaisesRegex(InsertionFault, 'stale'):
            c.hard_guard(st, self.now+.11)
        c = self.fixture()
        st = self.state(); c.hard_guard(st, self.now)
        st['right_sequence'] = 0
        with self.assertRaisesRegex(InsertionFault, 'reset or reordered'):
            c.hard_guard(st, self.now)
        c = self.fixture(); self.seq = 999_998
        c.hard_guard(self.state(), self.now)
        c.hard_guard(self.state(), self.now)  # Legal vendor wrap.

    def test_missing_force_or_timing_overrun_never_proposes_motion(self):
        for case in ('missing', 'nan', 'loop'):
            with self.subTest(case=case):
                c = self.fixture(); self.tick()
                st = self.state()
                if case == 'missing': st['wrench'] = None
                elif case == 'nan': st['wrench'][0] = np.nan
                else:
                    self.now += .08
                    st['timestamp_ns'] = round(self.now*1e9)
                with self.assertRaises(InsertionFault):
                    c.step(st, self.now)

    def test_spiral_follows_arc_speed_keeps_attitude_and_does_not_force_descent(self):
        c = self.fixture(); self.contact()
        previous = self.actual.copy()
        for _ in range(20):
            target = self.tick(delta=[0., 2., 0., 0., 0., 0.])
            self.assertLessEqual(np.linalg.norm((target-previous)[[0, 2], 3]), .001*.02+1e-10)
            self.assertLess(abs(target[1, 3]-previous[1, 3]), 1e-7)
            np.testing.assert_allclose(target[:3, :3], self.start[:3, :3])
            previous = target
        self.assertGreater(c.theta, 0.)
        self.assertLess(c.summary()['search_radius_m'], c.settings['max_radius_m'])

    def test_pending_servo_reference_freezes_spiral_until_applied_and_measured_catch_up(self):
        c = self.fixture(); self.contact()
        st = self.state(delta=[0., 2., 0., 0., 0., 0.])
        pending = c.step(st, self.now)
        theta = c.theta
        for _ in range(4):
            st = self.state(delta=[0., 2., 0., 0., 0., 0.]); st['servo_limited'] = True
            target = c.step(st, self.now)
            self.assertEqual(c.theta, theta)
            np.testing.assert_allclose(target, pending)
        c.accepted(pending); self.actual = pending.copy()
        self.tick(delta=[0., 2., 0., 0., 0., 0.], servo_limited=True)
        self.assertGreater(c.theta, theta)  # Flag alone does not deadlock an arrived reference.

    def test_depth_or_force_drop_freezes_lateral_search_without_claiming_success(self):
        for trigger in ('depth', 'unload'):
            with self.subTest(trigger=trigger):
                c = self.fixture(); self.contact()
                self.tick(delta=[0., 2., 0., 0., 0., 0.])
                if trigger == 'depth': self.actual[1, 3] -= .0005
                actual_xz = self.actual[[0, 2], 3].copy()
                theta = c.theta
                self.tick(delta=[0., .5 if trigger == 'unload' else 2., 0., 0., 0., 0.])
                self.assertEqual(c.status, 'capture')
                self.assertEqual(c.theta, theta)
                np.testing.assert_allclose(c.target[[0, 2], 3], actual_xz)
                for _ in range(5):
                    self.tick(delta=[0., 2., 0., 0., 0., 0.])
                    self.assertEqual(c.theta, theta)
                self.assertEqual(c.status, 'capture')
                self.assertFalse(c.summary()['assembly_verified'])

    def test_confirmed_depth_reaches_terminal_hold_without_releasing_or_recentering(self):
        c = self.fixture(start_height=.003); self.contact()
        self.actual[0, 3] += .0002
        self.actual[1, 3] -= .0005
        self.tick(delta=[0., .5, 0., 0., 0., 0.])
        self.assertEqual(c.status, 'capture')
        for _ in range(200):
            self.tick(delta=[0., .5, 0., 0., 0., 0.])
            if c.done:
                break
        self.assertEqual(c.status, 'awaiting_confirmation')
        hold = c.target.copy()
        self.assertAlmostEqual(hold[0, 3], .0002)
        self.actual[1, 3] += .00002
        self.tick(delta=[0., .5, 0., 0., 0., 0.], follow=False)
        np.testing.assert_allclose(c.target, hold)
        np.testing.assert_allclose(c.summary()['T_B_F_measured'], self.actual)
        self.assertFalse(c.summary()['assembly_verified'])
        self.assertFalse(c.summary()['automatic_release'])

    def test_soft_limits_retract_vertical_and_require_unload_before_retracted(self):
        c = self.fixture(start_height=.003); self.contact()
        self.actual[0, 3] = .0003
        old_y = self.actual[1, 3]
        self.tick(delta=[4.5, 2., 0., 0., 0., 0.])
        self.assertEqual(c.status, 'retracting')
        self.assertGreaterEqual(c.target[1, 3], old_y)
        xz = c.target[[0, 2], 3].copy()
        for _ in range(100):
            self.tick(delta=[0., 1., 0., 0., 0., 0.])
            np.testing.assert_allclose(c.target[[0, 2], 3], xz)
        self.assertEqual(c.status, 'retracting')  # At start height but still loaded.
        for _ in range(10):
            self.tick(delta=[0., 0., 0., 0., 0., 0.])
            if c.done: break
        self.assertEqual(c.status, 'retracted')
        self.assertTrue(c.failed)

    def test_search_radius_timeout_and_unconfirmed_capture_soft_abort(self):
        for cause in ('radius', 'timeout', 'capture'):
            with self.subTest(cause=cause):
                s = synthetic_settings()
                if cause == 'radius': s.update(max_radius_m=.00003, pitch_m=.0005)
                if cause == 'timeout': s['phase_timeout_s'] = .08
                if cause == 'capture': s['candidate_timeout_s'] = .1
                c = self.fixture(s); self.contact()
                if cause == 'capture':
                    self.tick(delta=[0., .5, 0., 0., 0., 0.])
                    self.assertEqual(c.status, 'capture')
                for _ in range(30):
                    self.tick(delta=[0., 2., 0., 0., 0., 0.])
                    if c.status == 'retracting': break
                self.assertEqual(c.status, 'retracting')
                self.assertIn(c.reason, ('search_radius_exhausted', 'phase_timeout', 'unconfirmed_capture'))

    def test_measured_goal_overshoot_or_excess_tracking_is_hard_fault(self):
        for cause in ('goal', 'tracking', 'lateral'):
            with self.subTest(cause=cause):
                c = self.fixture(); self.baseline_ready()
                if cause == 'goal': self.actual[1, 3] = -.0002
                elif cause == 'tracking': self.actual[1, 3] -= .006
                else: self.actual[0, 3] += .009
                with self.assertRaises(InsertionFault):
                    self.tick()

    def test_upward_unloading_is_allowed_while_downward_reference_is_limited(self):
        c = self.fixture(); self.contact()
        self.tick(delta=[0., 1.8, 0., 0., 0., 0.])
        c.target[1, 3] -= .001
        before = self.actual[1, 3]
        self.tick(delta=[0., 3., 0., 0., 0., 0.], servo_limited=True)
        self.assertGreater(c.target[1, 3], before)

    def test_no_contact_endpoint_and_stalled_insertion_request_retreat(self):
        c = self.fixture(start_height=.001)
        self.baseline_ready()
        for _ in range(40):
            self.tick(delta=np.zeros(6))
            if c.status == 'retracting': break
        self.assertEqual(c.reason, 'goal_reached_without_contact')
        self.assertFalse(c.contact_verified)
        c = self.fixture(); self.contact()
        self.actual[1, 3] -= .0005
        self.tick(delta=[0., .5, 0., 0., 0., 0.])
        for _ in range(25):
            self.tick(delta=[0., .5, 0., 0., 0., 0.])
            if c.status == 'inserting': break
        self.assertEqual(c.status, 'inserting')
        for _ in range(60):
            self.tick(delta=[0., 2., 0., 0., 0., 0.])
            if c.status == 'retracting': break
        self.assertEqual(c.reason, 'insertion_stalled')

    def test_baseline_timeout_and_failed_retreat_are_hard_faults(self):
        s = synthetic_settings(); s.update(baseline_duration_s=.06, baseline_timeout_s=.08,
                                           baseline_min_samples=100)
        self.fixture(s)
        with self.assertRaisesRegex(InsertionFault, 'baseline acquisition timed out'):
            for _ in range(6): self.tick()
        s = synthetic_settings(); s['retract_timeout_s'] = .1
        c = self.fixture(s); self.contact()
        self.tick(delta=[4.5, 2., 0., 0., 0., 0.])
        with self.assertRaisesRegex(InsertionFault, 'retreat timed out'):
            for _ in range(10): self.tick(delta=np.zeros(6), follow=False)

    def test_hard_guard_remains_active_after_terminal_arrival(self):
        c = self.fixture(start_height=.003); self.contact()
        c.abort(self.state(), 'test_abort', self.now)
        for _ in range(100):
            self.tick(delta=np.zeros(6))
            if c.done: break
        self.assertEqual(c.status, 'retracted')
        with self.assertRaisesRegex(InsertionFault, 'hard limit'):
            self.tick(delta=[0., 11., 0., 0., 0., 0.])

    def test_terminal_wrench_telemetry_updates_while_pose_and_stage_remain_latched(self):
        for status in ('awaiting_confirmation', 'retracted'):
            with self.subTest(status=status):
                c = self.fixture(); self.baseline_ready()
                c.status = status  # Isolate the common terminal monitoring contract.
                hold = c.target.copy()
                old_sequence = c.sequence
                self.tick(delta=[.1, .4, .2, .01, -.02, .03], follow=False)
                report = c.summary()
                np.testing.assert_allclose(report['wrench_at_hole_midpoint_base'], [.1, .4, .2, .01, -.02, .03])
                self.assertGreater(np.linalg.norm(report['filtered_wrench_at_hole_midpoint_base']), 0.)
                self.assertEqual(report['sequence'], old_sequence+1)
                np.testing.assert_allclose(c.target, hold)
                self.assertEqual(c.status, status)
                self.tick(delta=np.zeros(6), follow=False)
                np.testing.assert_allclose(c.summary()['wrench_at_hole_midpoint_base'], np.zeros(6))
                self.assertLess(np.linalg.norm(c.filtered), .05)

    def test_terminal_persistent_soft_limit_faults_without_restarting_retreat(self):
        c = self.fixture(); self.baseline_ready()
        c.status = 'awaiting_confirmation'
        hold = c.target.copy()
        self.tick(delta=[0., 6., 0., 0., 0., 0.], follow=False)
        self.assertEqual(c.status, 'awaiting_confirmation')
        # A single excursion followed by safe force resets the duration.
        self.tick(delta=np.zeros(6), follow=False)
        self.assertIsNone(c.terminal_contact_since)
        with self.assertRaisesRegex(InsertionFault, 'terminal contact limit persisted'):
            for _ in range(5):
                self.tick(delta=[0., 6., 0., 0., 0., 0.], follow=False)
        self.assertEqual(c.status, 'awaiting_confirmation')
        np.testing.assert_allclose(c.target, hold)
        np.testing.assert_allclose(c.last_wrench, [0., 6., 0., 0., 0., 0.])

    def test_capture_after_actual_drop_holds_measured_pose_until_pending_reference_is_applied(self):
        c = self.fixture(); self.contact()
        old_applied = c.applied_target.copy()
        self.actual[1, 3] -= .0005
        dropped = self.actual.copy()
        for _ in range(3):
            st = self.state(delta=[0., .5, 0., 0., 0., 0.])
            st.update(applied_target=old_applied, servo_limited=True)
            target = c.step(st, self.now)
            self.assertEqual(c.status, 'capture')
            np.testing.assert_allclose(target, dropped)
            self.assertEqual(c.theta, 0.)
            self.assertLessEqual(np.linalg.norm(target[:3, 3]-self.actual[:3, 3]), c.settings['max_reference_lead_m'])
        c.accepted(dropped)
        target = self.tick(delta=[0., .5, 0., 0., 0., 0.])
        self.assertGreaterEqual(target[1, 3], dropped[1, 3]-c.settings['insert_speed_m_s']*.02-1e-12)

    def test_soft_retreat_after_actual_drop_preserves_lateral_pose_and_reference_bound(self):
        c = self.fixture(); self.contact()
        self.actual[1, 3] -= .003
        actual = self.actual.copy()
        target = self.tick(delta=[0., 6., 0., 0., 0., 0.], follow=False)
        self.assertEqual(c.status, 'retracting')
        self.assertGreaterEqual(target[1, 3], actual[1, 3])
        np.testing.assert_allclose(target[[0, 2], 3], actual[[0, 2], 3])
        self.assertLessEqual(np.linalg.norm(target[:3, 3]-actual[:3, 3]), c.settings['max_reference_lead_m']+1e-12)


if __name__ == '__main__':
    unittest.main()
