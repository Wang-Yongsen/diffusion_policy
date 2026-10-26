"""Receiving startup translation, without connecting or moving hardware."""
from pathlib import Path
import sys
import threading
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from rail_receive.hardware import RightHardware, initial_receiving_pose
from rail_receive.logic import load_config
from rail_preinsert.core import matrix
from test_above import ToyKinematics
import receive


class InitialOffsetTests(unittest.TestCase):
    def hardware(self):
        hw = RightHardware.__new__(RightHardware)
        hw.config = load_config(HERE/'receive.yaml')
        hw.config['move_above'].update(settle_s=.001, reach_timeout_s=2000.)
        hw.kinematics = ToyKinematics()
        hw.arm_profile = NS(parameters={'arms': {'right': {'velocity_ratio': 3, 'acceleration_ratio': 3}}})
        hw.motion, hw.arm_enabled = True, True
        hw.lock, hw.stop = threading.RLock(), threading.Event()
        hw.emit, hw.raise_if_fault = Mock(), Mock()
        q = np.array([.2, -.02, .5, 0., .4, 0., 0.])
        state = {'joints_rad': q, 'T_B_F': matrix(hw.kinematics.fk('right', q))}
        initial = state['T_B_F'].copy()
        hw.snapshot = lambda: state
        hw.jogger, hw.arm_path = NS(target=initial.copy()), None
        def follow(path):
            hw.arm_path = (path, 0.)
            state['joints_rad'] = path.points[-1]
            state['T_B_F'] = matrix(hw.kinematics.fk('right', path.points[-1]))
        hw.follow_path = Mock(side_effect=follow)
        return hw, initial

    def test_left_15cm_is_native_base_translation_with_orientation_preserved(self):
        hw, initial = self.hardware()
        reference = matrix(hw.kinematics.fk('right', np.radians(hw.config['initial_reference_joints_deg'])))
        with patch('rail_receive.hardware.time.monotonic', side_effect=[0., 1000., 1000.01]):
            hw.move_to_initial_pose()
        actual = hw.snapshot()['T_B_F']
        np.testing.assert_allclose(actual[:3, 3]-reference[:3, 3], [0., 0., -.15], atol=1e-10)
        np.testing.assert_allclose(actual[:3, :3], reference[:3, :3], atol=1e-10)
        self.assertIsNone(hw.arm_path)
        self.assertFalse(hw.initial_move_active)
        np.testing.assert_allclose(hw.jogger.target, actual)
        hw.follow_path.assert_called_once()

    def test_read_only_does_not_start_an_initial_motion(self):
        hw, _ = self.hardware()
        hw.motion = False
        with self.assertRaisesRegex(RuntimeError, 'engaged'):
            hw.move_to_initial_pose()
        hw.follow_path.assert_not_called()

    def test_unreached_startup_target_still_times_out(self):
        hw, _ = self.hardware()
        hw.follow_path.side_effect = None  # Accepted command, but no measured movement.
        with patch('rail_receive.hardware.time.monotonic', side_effect=[0., 100000.]):
            with self.assertRaisesRegex(TimeoutError, 'fixed initial receiving pose'):
                hw.move_to_initial_pose()
        self.assertFalse(hw.initial_move_active)

    def test_startup_controller_fault_is_not_ignored(self):
        hw, _ = self.hardware()
        hw.raise_if_fault.side_effect = [None, RuntimeError('controller fault')]
        with patch('rail_receive.hardware.time.monotonic', return_value=0.):
            with self.assertRaisesRegex(RuntimeError, 'controller fault'):
                hw.move_to_initial_pose()
        self.assertFalse(hw.initial_move_active)

    def test_receiving_tolerance_accepts_observed_impedance_residual_without_waiting_full_path(self):
        hw, _ = self.hardware()
        target = initial_receiving_pose(hw.config, hw.kinematics)
        actual = target.copy()
        actual[0, 3] += .0026
        actual[:3, :3] = Rotation.from_euler('x', 1.59, degrees=True).as_matrix() @ target[:3, :3]
        state = hw.snapshot()
        state['T_B_F'] = actual
        hw.follow_path.side_effect = None
        # 0.01 s elapsed is much shorter than the nominal trajectory. The
        # measured flange is already stable within the receiving tolerance.
        with patch('rail_receive.hardware.time.monotonic', side_effect=[0., 0., .01]):
            hw.move_to_initial_pose()
        self.assertIsNone(hw.arm_path)
        np.testing.assert_allclose(hw.jogger.target, target)
        self.assertEqual(hw.config['move_above']['rotation_tolerance_rad'], .01745329252)

    def test_restarting_from_the_reached_target_does_not_add_another_offset(self):
        hw, _ = self.hardware()
        targets = []
        for _ in range(3):
            with patch('rail_receive.hardware.time.monotonic', side_effect=[0., 1000., 1000.01]):
                hw.move_to_initial_pose()
            targets.append(hw.snapshot()['T_B_F'].copy())
        for target in targets:
            np.testing.assert_allclose(target, initial_receiving_pose(hw.config, hw.kinematics), atol=1e-10)
        np.testing.assert_allclose(targets[0], targets[-1], atol=1e-10)

    def test_different_measured_starting_poses_reach_the_same_fixed_target(self):
        hw, _ = self.hardware()
        expected = initial_receiving_pose(hw.config, hw.kinematics)
        state = hw.snapshot()
        q = np.array([1., .5, -.3, .1, 0., 0., 0.])
        state.update(joints_rad=q, T_B_F=matrix(hw.kinematics.fk('right', q)))
        with patch('rail_receive.hardware.time.monotonic', side_effect=[0., 1000., 1000.01]):
            hw.move_to_initial_pose()
        np.testing.assert_allclose(hw.snapshot()['T_B_F'], expected, atol=1e-10)

    def test_setup_skips_translation_when_already_holding_a_rail(self):
        hw = Mock(paused=False)
        receive.start_with_preview(hw, Mock(), Path('unused/preview.png'), {}, threading.Event(), move_initial=False)
        hw.engage.assert_called_once()
        hw.move_to_initial_pose.assert_not_called()

    def test_setup_moves_after_engagement_and_before_return(self):
        hw = Mock(paused=False)
        calls = []
        hw.start.side_effect = lambda: calls.append('start')
        hw.engage.side_effect = lambda: calls.append('engage')
        hw.move_to_initial_pose.side_effect = lambda: calls.append('offset')
        receive.start_with_preview(hw, Mock(), Path('unused/preview.png'), {}, threading.Event())
        self.assertEqual(calls, ['start', 'engage', 'offset'])


if __name__ == '__main__':
    unittest.main()
