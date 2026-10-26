"""Editing and saving fixed prompts never require cameras, CUDA, or robot SDKs."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from set_sam_prompt import PromptEditor, _save_prompt


class PromptEditorTests(unittest.TestCase):
    def frozen_editor(self):
        editor = PromptEditor([640, 480])
        editor.freeze()
        return editor

    def test_editing_requires_frozen_frame(self):
        editor = PromptEditor([640, 480])
        self.assertFalse(editor.press("left", 100, 100))
        self.assertIsNone(editor.drag_start)
        with self.assertRaises(ValueError):
            editor.record("camera")

    def test_reverse_drag_and_clipped_image_boundaries(self):
        editor = self.frozen_editor()
        editor.press("left", 700, 500)
        editor.move(200, 100)
        self.assertEqual(editor.display_box, [200, 100, 640, 480])
        self.assertTrue(editor.release(-5, -20))
        self.assertEqual(editor.box, [0, 0, 640, 480])
        self.assertEqual(editor.record("main")["point_coords"], [])

    def test_degenerate_drag_is_not_a_box(self):
        editor = self.frozen_editor()
        editor.press("left", 100, 100)
        self.assertFalse(editor.release(100, 120))
        self.assertIsNone(editor.box)
        with self.assertRaises(ValueError):
            editor.record("main")

    def test_points_have_labels_and_stay_inside_box(self):
        editor = self.frozen_editor()
        editor.press("left", 100, 100)
        editor.release(500, 300)
        self.assertTrue(editor.press("left", 120, 150))
        self.assertTrue(editor.press("right", 300, 200))
        self.assertFalse(editor.press("left", 500, 200))
        self.assertFalse(editor.press("right", -10, 200))
        self.assertEqual(editor.points, [[120, 150], [300, 200]])
        self.assertEqual(editor.labels, [1, 0])
        editor.undo()
        self.assertEqual(editor.labels, [1])
        editor.clear_points()
        self.assertEqual(editor.points, [])
        self.assertEqual(editor.box, [100, 100, 500, 300])

    def test_redraw_and_fresh_capture_remove_stale_prompts(self):
        previous = {"box_xyxy": [10, 20, 30, 40], "point_coords": [[15, 25]], "point_labels": [1]}
        editor = PromptEditor([640, 480], previous)
        self.assertEqual(editor.box, previous["box_xyxy"])
        editor.freeze()
        self.assertIsNone(editor.box)
        self.assertEqual(editor.points, [])
        editor.press("left", 20, 30)
        editor.release(200, 300)
        editor.press("right", 50, 50)
        editor.redraw_box()
        self.assertIsNone(editor.box)
        self.assertEqual(editor.labels, [])
        self.assertEqual(previous["point_coords"], [[15, 25]])

    def test_save_uses_rgb_coordinates_and_does_not_alias_editor_state(self):
        editor = self.frozen_editor()
        editor.press("left", 100, 100)
        editor.release(500, 300)
        editor.press("left", 130, 160)
        result = editor.record("405622076394")
        editor.points[0][0] = 140
        self.assertEqual(result, {"image_size": [640, 480], "camera_serial": "405622076394",
                                  "box_xyxy": [100, 100, 500, 300],
                                  "point_coords": [[130, 160]], "point_labels": [1]})
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "nested/prompt.json"
            _save_prompt(target, result)
            self.assertEqual(json.loads(target.read_text()), result)
            self.assertEqual(list(target.parent.iterdir()), [target])


if __name__ == "__main__":
    unittest.main()
