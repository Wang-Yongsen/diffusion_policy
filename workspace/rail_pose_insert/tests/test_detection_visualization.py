"""The visualization must display the real gate timer, including resets and G bypass."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
from rail_receive.logic import load_config
from rail_receive.visualization import PreviewWindow, receiving_overlay, save_pose_snapshot, save_mask_snapshot
import view_detection


class OverlayTests(unittest.TestCase):
    def status(self, **changes):
        config = load_config(HERE / "receive.yaml")
        result = {"config": config, "stage": "waiting_rail",
                  "decision": {"reason": "detecting", "stable_s": .45},
                  "record": {"valid": True, "detected_in_roi": True, "score": .8,
                             "timestamp_ns": 10_000_000_000, "bbox_xyxy": [490, 160, 610, 240]}}
        result.update(changes)
        return result

    def render(self, status, now=10_500_000_000):
        with patch("rail_receive.visualization.cv2.putText", wraps=cv2.putText) as text:
            image = receiving_overlay(np.zeros((480, 640, 3), np.uint8), status, now)
        return image, [call.args[1] for call in text.call_args_list]

    def test_roi_box_accepted_box_and_true_timer_are_visible(self):
        image, text = self.render(self.status())
        self.assertEqual(image.shape, (646, 640, 3))
        self.assertIn("Continuous detection: 0.45 / 1.00 s", text)
        self.assertIn("RAIL DETECTED - COUNTING", text)
        x1, y1, _, _ = self.status()["config"]["receive_roi_xyxy"]
        np.testing.assert_array_equal(image[y1, x1], [0, 220, 255])
        np.testing.assert_array_equal(image[160, 490], [70, 230, 70])

    def test_timed_grasp_shows_countdown_without_fixed_roi_or_detection_timer(self):
        image, text = self.render(self.status(timed_grasp=True, record=None, decision=None,
                                             grasp_remaining_s=3.2))
        self.assertIn("TIMED GRASP - NO VISION CHECK", text)
        self.assertIn("Auto close in: 3.2 s", text)
        self.assertNotIn("Fixed ROI", text)
        x1, y1, _, _ = self.status()["config"]["receive_roi_xyxy"]
        self.assertFalse(image[y1, x1].any())
        _, text = self.render(self.status(timed_grasp=True, stage="closing", grasp_source="timer"))
        self.assertIn("CLOSING HAND (TIMER)", text)

    def test_pose_progress_distinguishes_camera_detector_and_pose_initialization(self):
        for phase, expected in (("opening_rgbd", "Opening RGB-D camera (first frame timeout: 10 s)"),
                                ("loading_sam", "Loading SAM (saved box / points)"),
                                ("initializing_pose", "New mask ready; initializing FoundationPose")):
            _, text = self.render(self.status(timed_grasp=True, stage="estimating_pose", pose_phase=phase))
            self.assertIn(expected, text)

    def test_opening_and_postgrasp_delays_have_separate_countdowns(self):
        _, text = self.render(self.status(timed_grasp=True, stage="opening"))
        self.assertIn("OPENING HAND - GRASP TIMER NOT STARTED", text)
        self.assertIn("Grasp countdown starts when the hand is open and ready", text)
        _, text = self.render(self.status(timed_grasp=True, stage="waiting_pose", pose_remaining_s=4.2))
        self.assertIn("GRASP COMPLETE - WAITING BEFORE POSE", text)
        self.assertIn("Pose recognition starts in: 4.2 s", text)

    def test_transfer_stages_show_motion_status_instead_of_grasp_countdown(self):
        cases = {"estimating_held_pose": "ESTIMATING HELD POSE - WILL WAIT FOR M",
                 "waiting_pose_override": "WAITING FOR M - USE SAVED MEAN TO MOVE",
                 "waiting_move_key": "POSE MEAN SAVED - PRESS M TO MOVE",
                 "planning_above": "PLANNING MOVE ABOVE TARGET",
                 "moving_above": "MOVING ABOVE TARGET - KEEPING GRASP",
                 "verifying_above": "VERIFYING RAIL POSE ABOVE TARGET",
                 "above_target": "ABOVE TARGET VERIFIED - N: LOWER",
                 "above_target_unverified": "VISION NOT VERIFIED - N: RECHECK ONLY"}
        for stage, expected in cases.items():
            with self.subTest(stage=stage):
                _, text = self.render(self.status(stage=stage, timed_grasp=True,
                                                  live_pose_preview=True, prompt_editable=False))
                self.assertIn(expected, text)
                self.assertIn("Above taught goal: 15 mm | hand remains closed", text)
                self.assertFalse(any("Auto close in:" in value for value in text))

    def test_position_spiral_disabled_alignment_offers_n_after_mechanical_arrival(self):
        status = self.status(stage="above_target", timed_grasp=True, live_pose_preview=True,
                             prompt_editable=False, record=None)
        status["config"]["lowering"]["mode"] = "position_spiral"
        status["config"]["preplace_alignment"]["enabled"] = False
        _, text = self.render(status)
        self.assertIn("ABOVE TARGET REACHED - N: SPIRAL INSERTION", text)
        self.assertTrue(any("No visual recheck" in value for value in text))
        self.assertTrue(any("N: position spiral" in value for value in text))
        self.assertFalse(any("VERIFIED" in value or "N: recheck" in value for value in text))

    def test_position_spiral_aligned_arrival_shows_last_visual_error_and_convergence(self):
        summary = dict(corrections=2, converged=True, final_pose_lock=True, measurements=[{
            "position_error_m": .0012, "position_tolerance_m": .002,
            "rotation_error_rad": np.radians(.9), "rotation_tolerance_rad": np.radians(1.)}])
        status = self.status(stage="above_target", timed_grasp=True, live_pose_preview=True,
                             prompt_editable=False, record=None, preplace_alignment=summary)
        status["config"]["lowering"]["mode"] = "position_spiral"
        _, text = self.render(status)
        self.assertIn("PREPLACE ALIGNED - N: SPIRAL INSERTION", text)
        self.assertIn("Last vision: 1.20 / 2.00 mm | 0.90 / 1.00 deg (error / limit)", text)
        self.assertIn("Visual corrections: 2 / 3 | converged; SDK / FK arrival confirmed", text)
        self.assertTrue(any("N: position spiral" in value for value in text))
        self.assertFalse(any("No visual recheck" in value for value in text))

    def test_preplace_stages_show_progress_and_do_not_offer_n_until_finished(self):
        cases = {"reestimating_preplace": "REESTIMATING PREPLACE POSE - N WAITS",
                 "planning_preplace": "PLANNING PREPLACE ALIGNMENT",
                 "aligning_preplace": "ALIGNING PREPLACE - KEEPING GRASP"}
        for stage, expected in cases.items():
            with self.subTest(stage=stage):
                status = self.status(stage=stage, timed_grasp=True, live_pose_preview=True,
                    prompt_editable=False, record=None,
                    preplace_alignment=dict(corrections=1, converged=False, measurements=[]))
                status["config"]["lowering"]["mode"] = "position_spiral"
                _, text = self.render(status)
                self.assertIn(expected, text)
                self.assertIn("Visual corrections: 1 / 3 | not converged; N waits", text)
                self.assertFalse(any("N: position spiral" in value or "No visual recheck" in value for value in text))
                if stage == "reestimating_preplace":
                    self.assertIn("Fresh SAM / FoundationPose | Projected rail box | N waits", text)

    def test_final_pose_lock_still_waits_for_measured_arrival(self):
        for stage, expected in (("planning_preplace", "PLANNING FINAL POSE LOCK"),
                                ("aligning_preplace", "FINAL POSE LOCK - KEEPING GRASP")):
            with self.subTest(stage=stage):
                status = self.status(stage=stage, timed_grasp=True, live_pose_preview=True,
                    prompt_editable=False, record=None,
                    preplace_alignment=dict(corrections=3, converged=False, final_pose_lock=True,
                                            measurements=[]))
                status["config"]["lowering"]["mode"] = "position_spiral"
                _, text = self.render(status)
                self.assertIn(expected, text)
                self.assertIn("Visual corrections: 3 / 3 | final pose lock; awaiting SDK / FK arrival", text)
                self.assertFalse(any("N: position spiral" in value for value in text))

    def test_transfer_keeps_pose_overlay_and_disables_prompt_editor_but_allows_pause(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", DISPLAY=":test"), \
                patch("rail_receive.visualization.cv2.imshow") as show, \
                patch("rail_receive.visualization.cv2.waitKey", side_effect=[ord("e"), ord("q")]), \
                patch("rail_receive.visualization.cv2.destroyWindow"):
            directory = Path(directory)
            cv2.imwrite(str(directory / "camera.png"), np.zeros((480, 640, 3), np.uint8))
            overlay = np.full((480, 640, 3), 100, np.uint8)
            cv2.imwrite(str(directory / "preview.png"), overlay)
            window = PreviewWindow()
            status = self.status(stage="moving_above", timed_grasp=True,
                                 live_pose_preview=True, prompt_editable=False)
            status["config"] = dict(status["config"], sam_prompt=str(directory / "unset_prompt.json"))
            try:
                window.update(directory / "preview.png", receiving=status)
                self.assertIsNone(window.editor)
                np.testing.assert_array_equal(show.call_args.args[1][:480], overlay)
                self.assertEqual(window.update(directory / "preview.png", receiving=status), "q")
            finally:
                window.close()

    def test_lowering_panel_shows_placement_goal_instead_of_above_goal(self):
        for stage in ("planning_lowering", "lowering", "placement_reached"):
            with self.subTest(stage=stage):
                _, text = self.render(self.status(stage=stage, timed_grasp=True,
                                                  live_pose_preview=True, prompt_editable=False))
                self.assertIn("Lower to taught goal | hand remains closed", text)
                self.assertNotIn("Above taught goal: 15 mm | hand remains closed", text)

    def test_preview_m_always_reaches_main_loop_for_feedback_including_paused_state(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", DISPLAY=":test"), \
                patch("rail_receive.visualization.cv2.imshow"), \
                patch("rail_receive.visualization.cv2.waitKey", return_value=ord("M")), \
                patch("rail_receive.visualization.cv2.destroyWindow"):
            directory = Path(directory)
            cv2.imwrite(str(directory / "preview.png"), np.zeros((480, 640, 3), np.uint8))
            window = PreviewWindow()
            try:
                for stage in ("estimating_held_pose", "waiting_move_key", "waiting_pose_override", "planning_above", "moving_above", "paused"):
                    status = self.status(stage=stage, timed_grasp=True,
                                         live_pose_preview=True, prompt_editable=False)
                    status["config"] = dict(status["config"], sam_prompt=str(directory / "unset_prompt.json"))
                    key = window.update(directory / "preview.png", receiving=status)
                    self.assertEqual(key, "m")
            finally:
                window.close()

    def test_preview_n_always_reaches_main_loop_for_feedback_including_paused_state(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", DISPLAY=":test"), \
                patch("rail_receive.visualization.cv2.imshow"), \
                patch("rail_receive.visualization.cv2.waitKey", return_value=ord("N")), \
                patch("rail_receive.visualization.cv2.destroyWindow"):
            directory = Path(directory)
            cv2.imwrite(str(directory / "preview.png"), np.zeros((480, 640, 3), np.uint8))
            window = PreviewWindow()
            try:
                for stage in ("estimating_held_pose", "waiting_move_key", "waiting_pose_override", "above_target", "lowering", "placement_reached", "paused"):
                    status = self.status(stage=stage, timed_grasp=True,
                                         live_pose_preview=True, prompt_editable=False)
                    status["config"] = dict(status["config"], sam_prompt=str(directory / "unset_prompt.json"))
                    key = window.update(directory / "preview.png", receiving=status)
                    self.assertEqual(key, "n")
            finally:
                window.close()

    def test_no_wall_clock_interpolation_and_stale_result_resets_timer(self):
        _, first = self.render(self.status(), now=10_600_000_000)
        _, later = self.render(self.status(), now=11_100_000_000)
        self.assertIn("Continuous detection: 0.45 / 1.00 s", first)
        self.assertIn("Continuous detection: 0.45 / 1.00 s", later)
        image, stale = self.render(self.status(), now=14_000_000_000)
        self.assertIn("RESULT STALE - TIMER RESET", stale)
        self.assertIn("Continuous detection: 0.00 / 1.00 s", stale)
        self.assertFalse(image[160, 490].any())

    def test_rejected_candidate_and_negative_frame_show_reset(self):
        status = self.status()
        status["record"].update(valid=False, detected_in_roi=False, score=.44)
        status["decision"] = {"reason": "invalid_observation", "stable_s": 0.}
        image, text = self.render(status)
        self.assertIn("rejected 0.44", text)
        self.assertIn("Continuous detection: 0.00 / 1.00 s", text)
        np.testing.assert_array_equal(image[160, 490], [155, 155, 155])

    def test_auto_closing_g_closing_pause_and_preview_only_are_distinct(self):
        cases = [(dict(stage="closing", grasp_source="cnos_detection"), "CLOSING HAND (CNOS)"),
                 (dict(stage="closing", grasp_source="keyboard"), "CLOSING HAND (G)"),
                 (dict(stage="paused"), "PAUSED - M CANNOT RESUME; REASON BELOW"),
                 (dict(stage="monitor_ready", monitor_only=True), "READY - PREVIEW ONLY, NO HAND COMMAND")]
        for changes, expected in cases:
            _, text = self.render(self.status(**changes))
            self.assertIn(expected, text)
            if changes.get("grasp_source") == "keyboard":
                self.assertIn("Continuous detection: bypassed by G", text)

    def test_saved_mean_remains_available_in_window_after_camera_failure(self):
        _, text = self.render(self.status(stage="waiting_move_key", timed_grasp=True,
            live_pose_preview=True, prompt_editable=False, saved_pose_age_s=12.5,
            vision_error="camera disconnected"))
        self.assertIn("POSE MEAN SAVED - PRESS M TO MOVE", text)
        self.assertIn("Saved mean: 12.5 s ago | M uses this mean; no resampling", text)

    def test_paused_window_displays_tracking_error_and_m_cannot_resume(self):
        reason = "Right arm departed from commanded trajectory: joint error 0.0352 rad > 0.0350 rad"
        _, text = self.render(self.status(stage="paused", live_pose_preview=True,
                                         prompt_editable=False, pause_reason=reason))
        self.assertIn("PAUSED - M CANNOT RESUME; REASON BELOW", text)
        self.assertIn(reason, " ".join(text))

    def test_window_refreshes_status_on_the_same_camera_frame(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", DISPLAY=":test"), \
                patch("rail_receive.visualization.cv2.imshow") as show, \
                patch("rail_receive.visualization.cv2.waitKey", return_value=-1), \
                patch("rail_receive.visualization.cv2.destroyWindow"):
            directory = Path(directory)
            cv2.imwrite(str(directory / "camera.png"), np.zeros((480, 640, 3), np.uint8))
            window = PreviewWindow()
            try:
                window.update(directory / "preview.png", receiving=self.status())
                first = show.call_args.args[1].copy()
                window.last_draw = 0.
                window.update(directory / "preview.png", receiving=self.status(stage="paused"))
                self.assertEqual(show.call_count, 2)
                self.assertFalse(np.array_equal(first, show.call_args.args[1]))
            finally:
                window.close()

    def test_finished_pose_is_labelled_as_a_snapshot_while_window_keeps_camera_image(self):
        pose = {"T_C_CAD": np.eye(4).tolist(), "timestamp_ns": 10_000_000_000,
                "quaternion_xyzw": [0., 0., 0., 1.]}
        image, text = self.render(self.status(stage="pose_done", pose_record=pose))
        self.assertEqual(image.shape, (720, 640, 3))
        self.assertIn("POSE READY (T: REFRESH)", text)
        self.assertTrue(any("snapshot age" in line and "not live" in line for line in text))
        self.assertTrue(any("CAD xyz [m]" in line for line in text))
        self.assertFalse(image[160, 490].any())  # No old detection box on the current image.

    def test_pose_snapshot_draws_projected_cad_and_colored_axes_on_its_own_frame(self):
        import trimesh
        from types import SimpleNamespace
        mesh = trimesh.creation.box(extents=[.04, .16, .02])
        pose = np.eye(4); pose[:3, 3] = [.04, .02, .6]
        frame = SimpleNamespace(rgb=np.full((480, 640, 3), 40, np.uint8),
                                K=np.array([[400., 0, 320], [0, 400., 240], [0, 0, 1]]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"pose.png"
            save_pose_snapshot(path, frame, mesh, pose)
            image = cv2.imread(str(path))
        self.assertEqual(image.shape, (480, 640, 3))
        for color in ((70, 230, 70), (0, 0, 255), (0, 255, 0), (255, 0, 0)):
            self.assertTrue(np.any(np.all(image[80:]==color, axis=-1)), color)
        # Surface tint inside the projected CAD, beyond just its box edges/axes.
        self.assertGreater(image[220, 345, 1], 40)

    def test_live_image_refreshes_while_pose_snapshot_stays_paired_with_original_frame(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", DISPLAY=":test"), \
                patch("rail_receive.visualization.cv2.imshow") as show, \
                patch("rail_receive.visualization.cv2.waitKey", return_value=-1), \
                patch("rail_receive.visualization.cv2.destroyWindow"):
            directory = Path(directory)
            snapshot = directory/"first"; snapshot.mkdir()
            cv2.imwrite(str(snapshot/"pose.png"), np.full((480, 640, 3), 60, np.uint8))
            cv2.imwrite(str(directory/"camera.png"), np.full((480, 640, 3), 20, np.uint8))
            record = {"T_C_CAD": np.eye(4).tolist(), "timestamp_ns": 10_000_000_000,
                      "pose_snapshot": "first/pose.png"}
            status = self.status(stage="pose_done", timed_grasp=True, pose_record=record)
            window = PreviewWindow()
            try:
                window.update(directory/"preview.png", receiving=status)
                first = show.call_args.args[1].copy()
                self.assertEqual(first.shape, (720, 1280, 3))
                cv2.imwrite(str(directory/"camera.png"), np.full((480, 640, 3), 90, np.uint8))
                window.last_draw = 0.
                window.update(directory/"preview.png", receiving=status)
                second = show.call_args.args[1].copy()
                np.testing.assert_array_equal(second[100, 100], [90]*3)
                np.testing.assert_array_equal(first[:480, 640:], second[:480, 640:])
                snapshot = directory/"second"; snapshot.mkdir()
                cv2.imwrite(str(snapshot/"pose.png"), np.full((480, 640, 3), 120, np.uint8))
                record["pose_snapshot"] = "second/pose.png"
                window.last_draw = 0.
                window.update(directory/"preview.png", receiving=status)
                np.testing.assert_array_equal(show.call_args.args[1][100, 740], [120]*3)
            finally:
                window.close()

    def test_sam_mask_snapshot_keeps_both_disconnected_parts_and_excludes_hand(self):
        from types import SimpleNamespace
        frame = SimpleNamespace(rgb=np.full((480, 640, 3), 40, np.uint8))
        mask = np.zeros((480, 640), bool)
        mask[210:230, 250:330] = True
        mask[210:230, 440:510] = True
        record = {"prompt_box_xyxy": [230, 190, 530, 260], "point_coords": [], "point_labels": []}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"mask.png"
            save_mask_snapshot(path, frame, mask, record)
            image = cv2.imread(str(path))
        self.assertGreater(image[220, 280, 1], 40)
        self.assertGreater(image[220, 480, 1], 40)
        np.testing.assert_array_equal(image[220, 380], [40]*3)

    def test_e_edits_the_existing_camera_frame_and_saves_original_pixel_coordinates(self):
        from rail_receive.sam_prompt import load_prompt
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", DISPLAY=":test"), \
                patch("rail_receive.visualization.cv2.imshow") as show, \
                patch("rail_receive.visualization.cv2.waitKey", side_effect=[ord('e'), -1, 13]), \
                patch("rail_receive.visualization.cv2.setMouseCallback") as callback, \
                patch("rail_receive.visualization.cv2.destroyWindow"), contextlib.redirect_stdout(io.StringIO()):
            directory = Path(directory)
            cv2.imwrite(str(directory/"camera.png"), np.full((480, 640, 3), 20, np.uint8))
            status = self.status(timed_grasp=True)
            status["config"]["sam_prompt"] = str(directory/"prompt.json")
            from rail_preinsert.core import write_json
            write_json(status["config"]["sam_prompt"], {
                "image_size": [640, 480], "camera_serial": "old-camera",
                "box_xyxy": [100, 100, 600, 400],
                "point_coords": [[300, 200]], "point_labels": [1]})
            window = PreviewWindow()
            try:
                window.update(directory/"preview.png", receiving=status)
                self.assertIsNotNone(window.editor)
                self.assertIsNone(window.editor.box)  # E begins drawing a new box directly.
                self.assertFalse(window.editor.points)  # No stale points in the new frame.
                callback.assert_called_once()
                window._mouse(cv2.EVENT_LBUTTONDOWN, 200, 180+34, 0, None)
                window._mouse(cv2.EVENT_MOUSEMOVE, 540, 270+34, 0, None)
                window._mouse(cv2.EVENT_LBUTTONUP, 540, 270+34, 0, None)
                window._mouse(cv2.EVENT_LBUTTONDOWN, 250, 220+34, 0, None)
                window._mouse(cv2.EVENT_RBUTTONDOWN, 410, 230+34, 0, None)
                cv2.imwrite(str(directory/"camera.png"), np.full((480, 640, 3), 90, np.uint8))
                window.update(directory/"preview.png", receiving=status)
                np.testing.assert_array_equal(show.call_args.args[1][134, 100], [20]*3)
                # The transfer view may stop supplying status while editing.
                # Saving must still use the configuration captured by E.
                action = window.update(directory/"preview.png", receiving=None)
                self.assertEqual(action, "prompt_saved")  # Controller requests a new pose.
                prompt = load_prompt(directory/"prompt.json", [640, 480])
                self.assertEqual(prompt["box_xyxy"], [200, 180, 540, 270])
                self.assertEqual(prompt["point_coords"], [[250, 220], [410, 230]])
                self.assertEqual(prompt["point_labels"], [1, 0])
                self.assertIsNone(window.editor)
            finally:
                window.close()

    def test_mask_is_displayed_on_its_snapshot_without_painting_it_on_live_camera(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", DISPLAY=":test"), \
                patch("rail_receive.visualization.cv2.imshow") as show, \
                patch("rail_receive.visualization.cv2.waitKey", return_value=-1), \
                patch("rail_receive.visualization.cv2.destroyWindow"):
            directory = Path(directory)
            (directory/"first").mkdir()
            cv2.imwrite(str(directory/"camera.png"), np.full((480, 640, 3), 20, np.uint8))
            cv2.imwrite(str(directory/"first/mask.png"), np.full((480, 640, 3), 80, np.uint8))
            status = self.status(timed_grasp=True, stage="estimating_pose",
                                 mask_record={"mask_snapshot": "first/mask.png"})
            window = PreviewWindow()
            try:
                window.update(directory/"preview.png", receiving=status)
                image = show.call_args.args[1]
                self.assertEqual(image.shape[1], 1280)
                np.testing.assert_array_equal(image[100, 100], [20]*3)
                np.testing.assert_array_equal(image[100, 740], [80]*3)
            finally:
                window.close()


class ViewerTests(unittest.TestCase):
    def test_viewer_reads_only_perception_and_exits_without_robot_or_hand(self):
        window, perception = Mock(enabled=True), Mock()
        window.update.return_value = "q"
        perception.latest.return_value = None
        with patch.object(view_detection, "PreviewWindow", return_value=window), \
                patch.object(view_detection, "ReceivePerception", return_value=perception), \
                contextlib.redirect_stdout(io.StringIO()):
            view_detection.main()
        perception.publish_context.assert_called_once()
        perception.close.assert_called_once()
        window.close.assert_called_once()

    def test_camera_failure_stays_visible_and_closes_resources_on_exit(self):
        window, perception = Mock(enabled=True), Mock()
        window.update.return_value = "q"
        perception.latest.side_effect = RuntimeError("Frame did not arrive")
        with patch.object(view_detection, "PreviewWindow", return_value=window), \
                patch.object(view_detection, "ReceivePerception", return_value=perception), \
                contextlib.redirect_stdout(io.StringIO()):
            view_detection.main()
        status = window.update.call_args.kwargs["receiving"]
        self.assertEqual(status["vision_error"], "Frame did not arrive")
        self.assertEqual(status["decision"]["stable_s"], 0.)
        perception.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
