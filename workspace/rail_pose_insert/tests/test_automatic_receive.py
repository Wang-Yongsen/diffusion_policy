"""Automatic receiving recognition and rejection/recovery without devices or a GPU."""
from dataclasses import dataclass
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from rail_receive.automatic import Candidate, RailProposer, checked_context, AutomaticTracker
from rail_receive.logic import load_config


def config():
    return load_config(HERE / 'receive.yaml')


@dataclass
class TestFrame:
    rgb: np.ndarray
    depth_m: np.ndarray
    K: np.ndarray
    timestamp_ns: int
    mask: np.ndarray

    def validate(self):
        pass


def frame(seconds=1., rails=1):
    rgb = np.full((240, 320, 3), 25, np.uint8)
    depth = np.full((240, 320), .95, np.float32)
    if rails:
        for y in ([120] if rails == 1 else [98, 142]):
            cv2.rectangle(rgb, (57, y-8), (263, y+8), (180, 180, 180), -1)
            depth[y-8:y+9, 57:264] = .7
    return TestFrame(rgb=rgb, depth_m=depth, K=np.array([[300., 0, 160], [0, 300, 120], [0, 0, 1]]),
                     timestamp_ns=round(seconds*1e9), mask=np.ones(depth.shape, bool))


def palm():
    T = np.eye(4); T[2, 3] = .7
    return T


class ProposerTests(unittest.TestCase):
    def proposer(self):
        return RailProposer(config(), [.0458, .48, .0257])

    def test_rail_is_found_from_rgbd_without_using_stored_mask(self):
        f = frame()
        result, reason = self.proposer().detect(f, palm())
        self.assertIsNotNone(result, reason)
        self.assertLess(abs(result.metrics['length_m']-.48), .03)
        self.assertLess(result.mask.sum(), f.mask.sum()/3)
        f.mask[:] = False
        other, _ = self.proposer().detect(f, palm())
        np.testing.assert_array_equal(result.mask, other.mask)

    def test_empty_palm_missing_depth_and_wrong_location_are_rejected(self):
        p = self.proposer()
        self.assertIsNone(p.detect(frame(rails=0), palm())[0])
        f = frame(); f.depth_m[:] = 0
        self.assertIsNone(p.detect(f, palm())[0])
        moved = palm(); moved[0, 3] = .4
        self.assertIsNone(p.detect(frame(), moved)[0])

    def test_multiple_rails_use_the_best_candidate(self):
        candidate, reason = self.proposer().detect(frame(rails=2), palm())
        self.assertIsNotNone(candidate)
        self.assertEqual(reason, 'rail_candidate')

    def test_large_board_and_hand_sized_patch_do_not_pass_as_a_rail(self):
        for rectangle in (((60, 50), (260, 190)), ((125, 90), (195, 150))):
            f = frame(rails=0)
            low, high = rectangle
            cv2.rectangle(f.rgb, low, high, (180, 180, 180), -1)
            f.depth_m[low[1]:high[1]+1, low[0]:high[0]+1] = .7
            self.assertIsNone(self.proposer().detect(f, palm())[0])

    def test_hand_occluding_middle_does_not_require_a_continuous_long_edge(self):
        f = frame()
        f.rgb[100:141, 130:191] = 35
        f.depth_m[100:141, 130:191] = .6
        candidate, reason = self.proposer().detect(f, palm())
        self.assertIsNotNone(candidate, reason)
        self.assertLess(abs(candidate.metrics['length_m']-.48), .04)
        self.assertFalse(candidate.mask[100:141, 130:191].any())


class ContextTests(unittest.TestCase):
    def test_enabled_and_fresh_feedback_defines_the_search_region(self):
        s, f = config()['automatic_detection'], frame()
        c = dict(enabled=True, epoch=0, timestamp_ns=1_000_000_000, T_C_G=palm().tolist())
        np.testing.assert_equal(checked_context(c, f, s, 1_100_000_000), palm())
        for change in (dict(enabled=False), dict(timestamp_ns=0), dict(timestamp_ns=2_000_000_000)):
            self.assertIsNone(checked_context(dict(c, **change), f, s, 1_100_000_000))


class RecoveryTests(unittest.TestCase):
    def tracker(self):
        engine = Mock(); engine.initialized = False
        engine.reset.side_effect = lambda: setattr(engine, 'initialized', False)
        def infer(f, initialize):
            engine.initialized = True
            return np.eye(4), dict(valid=True, mode='register' if initialize else 'track')
        engine.infer.side_effect = infer
        candidate = Candidate(np.ones((240, 320), bool), {})
        proposer = Mock(); proposer.detect.return_value = candidate, 'rail_candidate'
        return AutomaticTracker(engine, proposer, config()['automatic_detection'])

    def test_first_candidate_initializes_without_second_confirmation(self):
        t = self.tracker()
        self.assertTrue(t.step(frame(1), palm(), 0)[1]['valid'])
        self.assertIsNotNone(t.initial_mask)
        self.assertEqual(t.engine.infer.call_count, 1)

    def test_no_candidate_does_not_start_pose_estimation(self):
        t = self.tracker(); t.proposer.detect.return_value = None, 'searching_rail'
        self.assertFalse(t.step(frame(1), palm(), 0)[1]['valid'])
        t.engine.infer.assert_not_called()

    def test_jog_reinitializes_and_lost_tracking_restarts_search(self):
        t = self.tracker()
        self.assertTrue(t.step(frame(1), palm(), 0)[1]['valid'])
        self.assertEqual(t.step(frame(1.4), palm(), 1)[1]['mode'], 'register')
        t.engine.infer.side_effect = RuntimeError('lost track')
        self.assertEqual(t.step(frame(1.8), palm(), 1)[1]['reason'], 'pose_estimation_failed')
        self.assertFalse(t.engine.initialized)
        self.assertFalse(t.step(frame(2), palm(), 1)[1]['valid'])
        t.engine.infer.side_effect = lambda f, initialize: (np.eye(4), dict(valid=True))
        self.assertTrue(t.step(frame(3), palm(), 1)[1]['valid'])


if __name__ == '__main__':
    unittest.main()
