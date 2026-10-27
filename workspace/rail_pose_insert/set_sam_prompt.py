"""Draw separate initial and above-table SAM prompts using only the main camera."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import tempfile
from zoneinfo import ZoneInfo

from rail_receive.sam_prompt import PROMPT_KEYS, PROMPT_NAMES, load_prompt, prompt_path


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


class DualPromptEditor:
    """Two independent boxes on one frozen image."""

    def __init__(self, image_size, prompts=None):
        self.editors = {stage: PromptEditor(image_size, (prompts or {}).get(stage))
                        for stage in PROMPT_KEYS}
        self.stage = "initial"

    @property
    def current(self):
        return self.editors[self.stage]

    def select(self, stage):
        if stage not in self.editors:
            raise ValueError(f"Unknown SAM prompt stage: {stage}")
        self.current.drag_start = self.current.drag_end = None
        self.stage = stage

    def freeze(self):
        for editor in self.editors.values():
            editor.freeze()

    def records(self, serial):
        records = {}
        for stage, editor in self.editors.items():
            try:
                records[stage] = editor.record(serial)
            except ValueError:
                raise ValueError(f"请先画好{PROMPT_NAMES[stage]}，按 1/2 切换；两个框都完成后再保存。") from None
        return records


def clear_saved_prompts(config):
    """Remove both active prompts, retaining their files in a dated backup."""
    stamp = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d_%H%M%S_%f")
    backups = {}
    for stage in PROMPT_KEYS:
        path = prompt_path(config, stage)
        if path.is_file():
            backup = path.parent / "sam_prompt_backups" / stamp / path.name
            backup.parent.mkdir(parents=True, exist_ok=True)
            path.replace(backup)
            backups[stage] = backup
    return backups


def _draw(cv2, rgb, editor, *, stage=None, companion=None, dual=False):
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if companion is not None and companion.box is not None:
        x1, y1, x2, y2 = map(int, companion.box)
        cv2.rectangle(canvas, (x1, y1), (min(x2, editor.width-1), min(y2, editor.height-1)),
                      (230, 180, 70), 1)
        other = "ABOVE TABLE" if stage == "initial" else "INITIAL"
        cv2.putText(canvas, other, (x1+4, max(16, y1-7)), cv2.FONT_HERSHEY_SIMPLEX,
                    .45, (230, 180, 70), 1, cv2.LINE_AA)
    box = editor.display_box
    if box is not None:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(canvas, (x1, y1), (min(x2, editor.width-1), min(y2, editor.height-1)),
                      (0, 220, 255), 2)
        if stage:
            cv2.putText(canvas, "INITIAL" if stage == "initial" else "ABOVE TABLE",
                        (x1+4, max(16, y1-7)), cv2.FONT_HERSHEY_SIMPLEX, .45,
                        (0, 220, 255), 1, cv2.LINE_AA)
    for point, label in zip(editor.points, editor.labels):
        position = tuple(int(v) for v in point)
        color = (60, 235, 60) if label else (50, 50, 245)
        cv2.circle(canvas, position, 6, (0, 0, 0), 3)
        cv2.circle(canvas, position, 4, color, -1)
        cv2.putText(canvas, "+" if label else "-", (position[0]+8, position[1]+4),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
    canvas = cv2.copyMakeBorder(canvas, 34, 114, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
    title = "FROZEN - edit prompt" if editor.frozen else "LIVE - existing prompt is preview only"
    if stage:
        title = ("FROZEN" if editor.frozen else "LIVE") + " | editing: " + (
            "1 INITIAL" if stage == "initial" else "2 ABOVE TABLE")
    lines = [(title, 23),
             (("SPACE: fresh frame + clear current | 1 initial | 2 above table" if dual else
               "SPACE: fresh frame + clear all | drag left: box (when no box)"), editor.height+57),
             ("After box: left +rail | right -hand | Z undo | C clear points", editor.height+81),
             (("B redraw current | R clear BOTH | ENTER save BOTH | Q/ESC cancel" if dual else
               "B redraw box | ENTER save | Q / ESC cancel"), editor.height+105),
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true", help="备份并清除两个已保存的旧框，再重新绘制")
    args = parser.parse_args(argv)
    from rail_common.core import read_json
    from rail_receive.logic import load_config

    config = load_config(Path(__file__).with_name("receive.yaml"))
    serial = read_json(config["calibration"])["camera_serial"]
    size = config["receive_image_size"]
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError("需要在有图形桌面的本机终端运行 SAM 提示设置窗口。")
    if args.reset:
        for stage, backup in clear_saved_prompts(config).items():
            print(f"已清除{PROMPT_NAMES[stage]}；旧文件备份到：{backup}", flush=True)
    prompts = {}
    for stage in PROMPT_KEYS:
        path = prompt_path(config, stage)
        if not path.is_file():
            continue
        try:
            prompts[stage] = load_prompt(path, size, camera_serial=serial)
            print(f"已加载{PROMPT_NAMES[stage]}：{path}；空格重拍会清空两个框和所有点。", flush=True)
        except (ValueError, KeyError, OSError) as error:
            print(f"{PROMPT_NAMES[stage]}不能使用：{error}；请重新框选。", flush=True)
    root = Path(config["foundationpose_root"])
    sys.path[:0] = [str(root), str(root / "workspace/src")]
    from rail_common.camera import Camera
    import cv2

    print("只读取主视相机，不连接机械臂或灵巧手。\n"
          "空格：冻结/重拍并清空当前框；首次冻结时清空两个旧框；左键拖框。\n"
          "1：初始分割框；2：导轨台上方分割框。画完第一个框后按 2 画第二个框。\n"
          "选框后左键点击加正点（导轨），右键点击加负点（手）；点必须在框内。\n"
          "Z：撤销最后一点；C：清当前框的点；B：重画当前框；R：清空两个框。\n"
          "Enter：两个框都画好后一起保存；Q/Esc：取消本次编辑。\n"
          "两个框都可不添加点；固定提示不跟随导轨移动。", flush=True)
    pair, camera, frozen = DualPromptEditor(size, prompts), None, None
    title = "Rail SAM prompts - 1 INITIAL / 2 ABOVE TABLE"
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)

    def mouse(event, x, y, flags, userdata):
        editor = pair.current
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
            editor = pair.current
            other = pair.editors["preplace" if pair.stage == "initial" else "initial"]
            cv2.imshow(title, _draw(cv2, rgb if frozen is None else frozen, editor,
                                   stage=pair.stage, companion=other, dual=True))
            key = cv2.waitKey(1) & 0xff
            if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                print("已取消，未保存新框。" + ("旧框已备份并清除。" if args.reset else "原提示文件未修改。"), flush=True)
                return 0
            if key == ord(" "):
                if frozen is None:
                    pair.freeze()
                else:
                    editor.freeze()
                frozen = rgb.copy()
            elif key in (ord("1"), ord("2")):
                pair.select("initial" if key == ord("1") else "preplace")
            elif key in (ord("r"), ord("R")) and frozen is not None:
                pair.freeze()
            elif key in (ord("b"), ord("B")):
                editor.redraw_box()
            elif key in (ord("c"), ord("C")):
                editor.clear_points()
            elif key in (ord("z"), ord("Z")):
                editor.undo()
            elif key in (10, 13):
                try:
                    records = pair.records(serial)
                except ValueError as error:
                    editor.status = "Draw BOTH boxes first: 1 initial / 2 above table"
                    print(str(error), flush=True)
                    continue
                for stage, record in records.items():
                    path = prompt_path(config, stage)
                    _save_prompt(path, record)
                    print(f"已保存{PROMPT_NAMES[stage]}：{path}\n"
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
