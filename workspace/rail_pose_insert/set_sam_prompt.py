"""Set a fixed SAM box and optional points using only the rectified main camera."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile


class PromptEditor:
    """Image-coordinate editing state, independent of OpenCV and camera hardware."""

    def __init__(self, image_size, prompt=None):
        self.width, self.height = map(int, image_size)
        self.frozen = False
        self.box = None
        self.points, self.labels = [], []
        self.drag_start = self.drag_end = None
        self.status = "SPACE: freeze a fresh frame before editing"
        if prompt is not None:
            self.box = list(prompt["box_xyxy"])
            self.points = [list(p) for p in prompt.get("point_coords", [])]
            self.labels = list(prompt.get("point_labels", []))

    def freeze(self):
        self.frozen = True
        self.box = None
        self.points.clear()
        self.labels.clear()
        self.drag_start = self.drag_end = None
        self.status = "Drag left mouse around the whole visible rail"

    def redraw_box(self):
        if not self.frozen:
            return
        self.box = None
        self.points.clear()
        self.labels.clear()
        self.drag_start = self.drag_end = None
        self.status = "Drag a new box; previous box and points cleared"

    def clear_points(self):
        if self.frozen:
            self.points.clear()
            self.labels.clear()
            self.status = "Points cleared; box retained"

    def undo(self):
        if self.frozen and self.points:
            self.points.pop()
            self.labels.pop()
            self.status = "Last point removed"

    def _box_position(self, x, y):
        return [min(self.width, max(0, int(x))), min(self.height, max(0, int(y)))]

    def press(self, button, x, y):
        if not self.frozen:
            self.status = "Press SPACE to freeze a fresh frame first"
            return False
        if button == "left" and self.box is None:
            self.drag_start = self.drag_end = self._box_position(x, y)
            return True
        if self.box is None or button not in ("left", "right"):
            return False
        if not (0 <= x < self.width and 0 <= y < self.height):
            self.status = "Point rejected: click inside the image and box"
            return False
        x, y = int(x), int(y)
        x1, y1, x2, y2 = self.box
        if not (x1 <= x < x2 and y1 <= y < y2):
            self.status = "Point rejected: click inside the box"
            return False
        self.points.append([x, y])
        self.labels.append(1 if button == "left" else 0)
        self.status = "Foreground point added" if button == "left" else "Background point added"
        return True

    def move(self, x, y):
        if self.drag_start is not None:
            self.drag_end = self._box_position(x, y)

    def release(self, x, y):
        if self.drag_start is None:
            return False
        self.move(x, y)
        (ax, ay), (bx, by) = self.drag_start, self.drag_end
        box = [min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)]
        self.drag_start = self.drag_end = None
        if box[0] == box[2] or box[1] == box[3]:
            self.status = "Box rejected: drag a box with positive width and height"
            return False
        self.box = box
        self.status = "Box ready; ENTER saves, optional left/right clicks add +/- points"
        return True

    @property
    def display_box(self):
        if self.drag_start is None:
            return self.box
        (ax, ay), (bx, by) = self.drag_start, self.drag_end
        return [min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)]

    def record(self, camera_serial):
        if not self.frozen or self.box is None or self.drag_start is not None:
            raise ValueError("请先按空格冻结画面，并拖出一个有效的导轨范围框。")
        return {"image_size": [self.width, self.height], "camera_serial": str(camera_serial),
                "box_xyxy": list(self.box), "point_coords": [list(p) for p in self.points],
                "point_labels": list(self.labels)}


def _draw(cv2, rgb, editor):
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    box = editor.display_box
    if box is not None:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(canvas, (x1, y1), (min(x2, editor.width-1), min(y2, editor.height-1)),
                      (0, 220, 255), 2)
    for point, label in zip(editor.points, editor.labels):
        position = tuple(int(v) for v in point)
        color = (60, 235, 60) if label else (50, 50, 245)
        cv2.circle(canvas, position, 6, (0, 0, 0), 3)
        cv2.circle(canvas, position, 4, color, -1)
        cv2.putText(canvas, "+" if label else "-", (position[0]+8, position[1]+4),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
    canvas = cv2.copyMakeBorder(canvas, 34, 114, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
    lines = [("FROZEN - edit prompt" if editor.frozen else "LIVE - existing prompt is preview only", 23),
             ("SPACE: fresh frame + clear all | drag left: box (when no box)", editor.height+57),
             ("After box: left +rail | right -hand | Z undo | C clear points", editor.height+81),
             ("B redraw box | ENTER save | Q / ESC cancel", editor.height+105),
             (editor.status, editor.height+132)]
    for text, y in lines:
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .47, (235, 235, 235), 1,
                    cv2.LINE_AA)
    return canvas


def _save_prompt(path, record):
    """Replace only after a complete JSON has been written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name+".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(record, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    from rail_preinsert.core import read_json
    from rail_receive.logic import load_config
    from rail_receive.sam_prompt import load_prompt

    config = load_config(Path(__file__).with_name("receive.yaml"))
    serial = read_json(config["calibration"])["camera_serial"]
    size = config["receive_image_size"]
    prompt_path = Path(config["sam_prompt"])
    prompt = None
    if prompt_path.is_file():
        try:
            prompt = load_prompt(prompt_path, size, camera_serial=serial)
            print(f"已加载提示：{prompt_path}；初始只预览，空格重拍会清空旧框和点。", flush=True)
        except (ValueError, KeyError, OSError) as error:
            print(f"已有提示不能使用：{error}；请重新框选，保存前不会覆盖原文件。", flush=True)
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError("需要在有图形桌面的本机终端运行 SAM 提示设置窗口。")
    root = Path(config["foundationpose_root"])
    sys.path[:0] = [str(root), str(root / "workspace/src")]
    from perception_worker import Camera
    import cv2

    print("只读取主视相机，不连接机械臂或灵巧手。\n"
          "空格：冻结/重拍并清空全部提示；左键拖框：大致包围整根可见导轨。\n"
          "选框后左键点击加正点（导轨），右键点击加负点（手）；点必须在框内。\n"
          "Z：撤销最后一个点；C：清空点；B：重画框；Enter：保存；Q/Esc：取消。\n"
          "框提示可以单独保存，不必添加点。固定提示只适用于导轨仍位于该范围的画面。", flush=True)
    editor, camera, frozen = PromptEditor(size, prompt), None, None
    title = "Rail SAM prompt - main camera only"
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)

    def mouse(event, x, y, flags, userdata):
        # The 34-pixel header is display-only; saved coordinates refer to RGB.
        y -= 34
        if event == cv2.EVENT_LBUTTONUP:
            editor.release(x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            editor.move(x, y)
        elif 0 <= y < editor.height:
            if event == cv2.EVENT_LBUTTONDOWN:
                editor.press("left", x, y)
            elif event == cv2.EVENT_RBUTTONDOWN:
                editor.press("right", x, y)

    cv2.setMouseCallback(title, mouse)
    try:
        camera = Camera(str(serial), rgb_only=True)
        while True:
            rgb = camera.frame().rgb
            if list(rgb.shape[1::-1]) != list(size):
                raise ValueError("主视相机图像尺寸与 receive_image_size 不一致。")
            cv2.imshow(title, _draw(cv2, rgb if frozen is None else frozen, editor))
            key = cv2.waitKey(1) & 0xff
            if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                print("已取消，原提示文件未修改。", flush=True)
                return 0
            if key == ord(" "):
                frozen = rgb.copy()
                editor.freeze()
            elif key in (ord("b"), ord("B")):
                editor.redraw_box()
            elif key in (ord("c"), ord("C")):
                editor.clear_points()
            elif key in (ord("z"), ord("Z")):
                editor.undo()
            elif key in (10, 13):
                try:
                    record = editor.record(serial)
                except ValueError as error:
                    editor.status = "Freeze with SPACE, then drag a valid box before saving"
                    print(str(error), flush=True)
                    continue
                _save_prompt(prompt_path, record)
                print(f"已保存 SAM 提示：{prompt_path}\n"
                      f"范围框：{record['box_xyxy']}；正点 {record['point_labels'].count(1)} 个，"
                      f"负点 {record['point_labels'].count(0)} 个。", flush=True)
                return 0
    finally:
        if camera is not None:
            camera.close()
        cv2.destroyWindow(title)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"SAM 提示设置失败：{error}", file=sys.stderr)
        raise SystemExit(1)
