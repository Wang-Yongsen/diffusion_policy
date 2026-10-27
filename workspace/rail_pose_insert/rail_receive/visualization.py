"""RGB overlays of the measured CAD pose, taught goal, and above-goal pose."""

from __future__ import annotations

import os
from pathlib import Path
import time

import cv2
import numpy as np


def draw_sam_prompt(image, prompt):
    box = prompt.get("box_xyxy", prompt.get("prompt_box_xyxy"))
    if box is None:
        return
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), (0, 220, 255), 2)
    cv2.putText(
        image,
        "SAM prompt box",
        (x1 + 4, max(16, y1 - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 220, 255),
        1,
        cv2.LINE_AA,
    )
    for point, label in zip(prompt.get("point_coords", []), prompt.get("point_labels", [])):
        p = tuple(map(int, point))
        color = (70, 230, 70) if label else (80, 80, 255)
        cv2.circle(image, p, 5, (0, 0, 0), 2)
        cv2.circle(image, p, 3, color, -1)
        cv2.putText(
            image,
            "+" if label else "-",
            (p[0] + 7, p[1] + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )


def receiving_overlay(image, status):
    """Show the current insertion stage and the next operator action."""
    config = status["config"]
    stage = status.get("stage", "starting")
    labels = {
        "starting": "INITIALIZING DEVICES",
        "left_initial_position": "RETURNING LEFT ARM TO INITIAL POSE",
        "initial_position": "RETURNING RIGHT ARM TO RECEIVING START",
        "opening": "OPENING HAND",
        "waiting_rail": "WAITING FOR RAIL - G: GRASP NOW",
        "closing": "CLOSING HAND",
        "holding": "HOLDING RAIL",
        "waiting_pose": "WAITING BEFORE INITIAL POSE",
        "estimating_held_pose": "ESTIMATING HELD POSE - WILL WAIT FOR M",
        "waiting_move_key": "POSE MEAN SAVED - M: MOVE ABOVE GOAL",
        "waiting_initial_bbox": "INITIAL BBOX - E: DRAW / ENTER: CONFIRM",
        "waiting_preplace_bbox": "ABOVE TABLE BBOX - E: DRAW / ENTER: CONFIRM",
        "planning_above": "PLANNING MOVE ABOVE GOAL",
        "moving_above": "MOVING ABOVE GOAL",
        "reestimating_preplace": "REESTIMATING ABOVE TABLE - KEEPING GRASP",
        "planning_preplace": "PLANNING SINGLE CORRECTION",
        "aligning_preplace": "EXECUTING SINGLE CORRECTION",
        "above_target": "ALIGNMENT COMPLETE - NEW N: SPIRAL INSERTION",
        "planning_position_spiral": "PLANNING POSITION SPIRAL",
        "position_spiral": "SPIRAL INSERTION - KEEPING GRASP",
        "insertion_reached": "INSERTION REACHED - AUTO RELEASE NEXT",
        "planning_return": "PLANNING LIFT AND RETURN - KEEPING GRASP",
        "releasing_hand": "OPENING HAND - ARM HOLDS STILL",
        "lifting_after_release": "LIFTING WITH HAND OPEN",
        "returning_initial": "RETURNING RIGHT ARM TO INITIAL POSE",
        "returned_initial": "COMPLETE - HAND OPEN / X: EXIT",
        "paused": "PAUSED - X: EXIT",
    }
    rows = [labels.get(stage, stage.upper())]
    if stage == "waiting_rail":
        rows.append(f"Grasp in {status.get('grasp_remaining_s', config['grasp_delay_s']):.1f} s")
    elif stage == "waiting_pose":
        rows.append(f"Pose in {status.get('pose_remaining_s', 0.):.1f} s")
    elif stage in (
        "planning_return",
        "releasing_hand",
        "lifting_after_release",
        "returning_initial",
        "returned_initial",
    ):
        rows.append("Open command + wait | Lift +Y 15 cm | Return to fixed initial pose")
    else:
        rows.append("G: grasp | M: transfer | N: insert | Q/SPACE: pause | X: exit")
    alignment = status.get("preplace_alignment") or {}
    measurements = alignment.get("measurements") or []
    if measurements:
        last = measurements[-1]
        rows.append(
            f"Above-table error: {last['position_error_m']*1000:.2f} mm / "
            f"{np.degrees(last['rotation_error_rad']):.2f} deg"
        )
    reason = status.get("pause_reason") or status.get("vision_error")
    if reason:
        rows.append(str(reason))
    rows.append("Position arrival only - assembly is not verified")
    if status.get("sam_prompt"):
        image = image.copy()
        draw_sam_prompt(image, status["sam_prompt"])
    panel = np.full((len(rows) * 24 + 12, image.shape[1], 3), 25, np.uint8)
    for index, row in enumerate(rows):
        color = (80, 80, 255) if stage == "paused" else (0, 220, 255)
        cv2.putText(
            panel, row, (10, 23 + index * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
        )
    return np.concatenate((image, panel), axis=0)


def pose_overlay(
    rgb, mesh, pose, K, references=None, reason="", *, show_model=False, show_info=True
):
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if show_model and mesh is not None and pose is not None:
        vertices = np.asarray(mesh.vertices) @ pose[:3, :3].T + pose[:3, 3]
        faces = np.asarray(mesh.faces)
        visible = faces[np.all(vertices[faces, 2] > 0.001, axis=1)]
        projected = vertices @ K.T
        uv = np.clip(projected[:, :2] / np.maximum(projected[:, 2:3], 0.001), -10000, 10000).astype(
            np.int32
        )
        silhouette = np.zeros(image.shape[:2], np.uint8)
        for face in visible:
            cv2.fillConvexPoly(silhouette, uv[face], 255)
        selected = silhouette > 0
        image[selected] = (0.78 * image[selected] + 0.22 * np.array([70, 230, 70])).astype(np.uint8)
        contours, _ = cv2.findContours(silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (70, 230, 70), 1, cv2.LINE_AA)
    low, high = mesh.bounds if mesh is not None else (np.zeros(3), np.zeros(3))
    corners = np.array(
        [[x, y, z] for x in [low[0], high[0]] for y in [low[1], high[1]] for z in [low[2], high[2]]]
    )

    def project(points, T):
        camera = points @ T[:3, :3].T + T[:3, 3]
        if np.any(camera[:, 2] <= 0.001):
            return None
        p = camera @ K.T
        return np.clip(p[:, :2] / p[:, 2, None], -10000, 10000).astype(int)

    references = references or {}
    for value, color in [
        (references.get("T_C_O_goal"), (0, 150, 255)),
        (references.get("T_C_O_above"), (255, 220, 0)),
        (pose, (70, 230, 70)),
    ]:
        if value is None:
            continue
        T = np.asarray(value)
        pixels = project(corners, T)
        if pixels is not None:
            for i in range(8):
                for bit in (1, 2, 4):
                    if i < (i ^ bit):
                        cv2.line(
                            image, tuple(pixels[i]), tuple(pixels[i ^ bit]), color, 2, cv2.LINE_AA
                        )
    if pose is not None:
        axes = project(np.vstack([np.zeros(3), np.eye(3) * 0.06]), pose)
        if axes is not None:
            for index, color in enumerate(((0, 0, 255), (0, 255, 0), (255, 0, 0)), 1):
                cv2.arrowedLine(image, tuple(axes[0]), tuple(axes[index]), color, 2, cv2.LINE_AA)
                cv2.putText(
                    image,
                    "XYZ"[index - 1],
                    tuple(axes[index] + [4, -4]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        from scipy.spatial.transform import Rotation

        xyz = pose[:3, 3]
        quat = Rotation.from_matrix(pose[:3, :3]).as_quat()
        legend = (
            "GREEN observed | ORANGE goal | CYAN above | axes X red Y green Z blue"
            if references
            else "GREEN rail | axes X red Y green Z blue"
        )
        text = [
            legend,
            "camera CAD xyz [m]: " + " ".join(f"{v:.3f}" for v in xyz),
            "camera CAD quat xyzw: " + " ".join(f"{v:.3f}" for v in quat),
        ]
        if reason:
            text.append(reason)
    else:
        text = [
            ("FoundationPose: " if mesh is not None else "Camera: ")
            + (reason or "waiting for rail"),
            (
                "ORANGE goal | CYAN above | controls: terminal"
                if mesh is not None
                else "YELLOW receiving region | GREEN matched rail | controls: terminal"
            ),
        ]
    if not show_info:
        return image
    cv2.rectangle(image, (0, 0), (image.shape[1], len(text) * 22 + 8), (25, 25, 25), -1)
    for index, line in enumerate(text):
        cv2.putText(
            image,
            line,
            (8, 22 + 22 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
    return image


def save_preview(directory, frame, mesh, pose, references=None, reason=""):
    if mesh is None:
        # The parent draws boxes and the authoritative timer on an unpainted image.
        raw_path = Path(directory) / "camera.png"
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("Could not encode main-camera preview")
        temporary = raw_path.with_suffix(".tmp")
        temporary.write_bytes(encoded.tobytes())
        temporary.replace(raw_path)
        from rail_common.core import write_json

        timestamp = getattr(frame, "timestamp_ns", None)
        metadata = getattr(frame, "metadata", None) or {}
        write_json(
            raw_path.with_name("camera_frame.json"),
            {
                "timestamp_ns": int(timestamp) if timestamp is not None else None,
                "timestamp_source": metadata.get("timestamp_source"),
                "image_mtime_ns": raw_path.stat().st_mtime_ns,
            },
        )
    image = pose_overlay(frame.rgb, mesh, pose, frame.K, references, reason)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Could not encode FoundationPose preview")
    path = Path(directory) / "preview.png"
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(encoded.tobytes())
    temporary.replace(path)


class PreviewWindow:
    title = "Rail - main camera"

    def __init__(self, enabled=True):
        self.enabled = bool(
            enabled and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        )
        self.stamp, self.opened = None, False
        self.image = None
        self.prompt = self.prompt_stamp = None
        self.editor = self.edit_rgb = self.edit_config = None
        self.edit_stage = "initial"
        self.edit_stage_locked = False
        self.last_draw = 0.0

    def _edit_image(self, camera, receiving):
        """Required-stage editing must use a raw frame captured after arrival."""
        required = receiving.get("prompt_required", False)
        try:
            stamp = camera.stat().st_mtime_ns
            if required and stamp < receiving.get("prompt_frame_not_before_ns", 0):
                raise ValueError("正在等待到位后的新相机画面，请稍后再按 E。")
            if required:
                from rail_common.core import read_json

                metadata = read_json(camera.with_name("camera_frame.json"))
                cutoff = receiving.get("prompt_capture_not_before_ns")
                if (
                    not isinstance(metadata, dict)
                    or metadata.get("image_mtime_ns") != stamp
                    or metadata.get("timestamp_source")
                    != "sdk_global_timestamp_mapped_to_host_monotonic"
                    or type(metadata.get("timestamp_ns")) is not int
                    or type(cutoff) is not int
                    or metadata["timestamp_ns"] <= cutoff
                ):
                    raise ValueError("正在等待到位后采集且时间戳匹配的新画面，请稍后再按 E。")
            image = cv2.imdecode(
                np.frombuffer(camera.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if required and camera.stat().st_mtime_ns != stamp:
                raise ValueError("相机画面正在更新，请再按 E 冻结新画面。")
            if image is None or list(image.shape[1::-1]) != list(
                receiving["config"]["receive_image_size"]
            ):
                raise ValueError("相机原图尚未就绪或尺寸不匹配，请稍后再按 E。")
            return image
        except (OSError, ValueError) as error:
            if required:
                print(f"暂不能绘制 bbox：{error}", flush=True)
                return None
            return self.image if self.stamp is not None else None

    def _mouse(self, event, x, y, flags, userdata):
        if self.editor is None:
            return
        y -= 34  # PromptEditor displays a header above the unchanged RGB image.
        if event == cv2.EVENT_LBUTTONUP:
            self.editor.release(x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            self.editor.move(x, y)
        elif 0 <= y < self.editor.height:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.editor.press("left", x, y)
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.editor.press("right", x, y)

    def update(self, path, receiving=None):
        path = Path(path)
        if not self.enabled:
            return None
        if receiving is not None:
            receiving = dict(receiving)
            if not receiving.get("prompt_editable", True):
                self.editor = self.edit_rgb = self.edit_config = None
            stage = receiving.get("sam_prompt_stage", "initial")
            key = "sam_prompt_preplace" if stage == "preplace" else "sam_prompt"
            prompt_path = Path(receiving["config"].get(key, ""))
            prompt_stamp = (
                str(prompt_path),
                prompt_path.stat().st_mtime_ns if prompt_path.is_file() else None,
            )
            if prompt_stamp != self.prompt_stamp:
                from rail_receive.sam_prompt import load_prompt

                try:
                    self.prompt = load_prompt(
                        prompt_path, receiving["config"]["receive_image_size"]
                    )
                except (OSError, ValueError):
                    self.prompt = None
                self.prompt_stamp = prompt_stamp
            receiving["sam_prompt"] = self.prompt
        camera = path.parent / "camera.png"
        if (
            receiving is not None
            and camera.is_file()
            and not (receiving.get("live_pose_preview") and path.is_file())
        ):
            path = camera
        stamp = (str(path), path.stat().st_mtime_ns) if path.is_file() else None
        if stamp is not None and stamp != self.stamp:
            image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                self.image, self.stamp = image, stamp
        if self.image is None and receiving is not None:
            w, h = receiving["config"]["receive_image_size"]
            self.image = np.zeros((h, w, 3), np.uint8)
        now = time.monotonic()
        if self.image is not None and now - self.last_draw >= 0.05:
            image = self.image
            image = receiving_overlay(image, receiving) if receiving is not None else image
            if self.editor is not None:
                from set_sam_prompt import _draw

                image = _draw(cv2, self.edit_rgb, self.editor, stage=self.edit_stage)
                h = self.editor.height
                cv2.rectangle(image, (0, h + 39), (image.shape[1], h + 65), (24, 24, 24), -1)
                edit_help = (
                    "Current stage locked | drag left: new box"
                    if self.edit_stage_locked
                    else "1 initial / 2 above table | drag left: new box"
                )
                cv2.putText(
                    image,
                    edit_help,
                    (10, h + 57),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (235, 235, 235),
                    1,
                    cv2.LINE_AA,
                )
                cv2.rectangle(image, (0, h + 89), (image.shape[1], h + 114), (24, 24, 24), -1)
                cv2.putText(
                    image,
                    "ENTER save current | E cancel | Q/SPACE pause | X exit",
                    (10, h + 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (235, 235, 235),
                    1,
                    cv2.LINE_AA,
                )
            cv2.imshow(self.title, image)
            self.opened, self.last_draw = True, now
        if not self.opened:
            return None
        key = cv2.waitKey(1) & 0xFF
        if self.editor is not None:
            if key in (ord("e"), ord("E"), 27):
                self.editor = self.edit_rgb = self.edit_config = None
                self.last_draw = 0.0
                return None
            elif key in (10, 13):
                from set_sam_prompt import _save_prompt
                from rail_common.core import read_json
                from rail_receive.sam_prompt import prompt_path, PROMPT_NAMES

                try:
                    serial = read_json(self.edit_config["calibration"])["camera_serial"]
                    _save_prompt(
                        prompt_path(self.edit_config, self.edit_stage), self.editor.record(serial)
                    )
                    self.editor = self.edit_rgb = self.edit_config = None
                    self.prompt_stamp = None
                    print(f"{PROMPT_NAMES[self.edit_stage]}已保存。", flush=True)
                    self.last_draw = 0.0
                    return (
                        "prompt_saved" if self.edit_stage == "initial" else "preplace_prompt_saved"
                    )
                except (ValueError, KeyError, OSError) as error:
                    print(f"SAM 提示未保存：{error}", flush=True)
            elif key in (ord("b"), ord("B")):
                self.editor.redraw_box()
            elif key in (ord("c"), ord("C")):
                self.editor.clear_points()
            elif key in (ord("z"), ord("Z")):
                self.editor.undo()
            elif key in (ord("1"), ord("2")) and not self.edit_stage_locked:
                from set_sam_prompt import PromptEditor

                self.edit_stage = "initial" if key == ord("1") else "preplace"
                self.editor = PromptEditor(self.edit_config["receive_image_size"])
                self.editor.freeze()
        elif (
            key in (ord("e"), ord("E"))
            and receiving is not None
            and receiving.get("prompt_editable", True)
        ):
            from set_sam_prompt import PromptEditor

            edit_image = self._edit_image(camera, receiving)
            if edit_image is None:
                return None
            self.editor = PromptEditor(receiving["config"]["receive_image_size"])
            self.edit_stage = receiving.get("sam_prompt_stage", "initial")
            self.edit_stage_locked = receiving.get("prompt_stage_locked", False)
            self.edit_config = dict(receiving["config"])
            self.editor.frozen = True
            self.editor.status = "Drag a new rail box; ENTER saves and requests a new pose"
            self.edit_rgb = cv2.cvtColor(edit_image, cv2.COLOR_BGR2RGB)
            cv2.setMouseCallback(self.title, self._mouse)
        elif (
            key in (10, 13)
            and receiving is not None
            and receiving.get("prompt_required")
            and receiving.get("prompt_editable", True)
        ):
            from rail_common.core import read_json
            from rail_receive.sam_prompt import load_prompt, prompt_path, PROMPT_NAMES

            stage = receiving.get("sam_prompt_stage", "initial")
            try:
                config = receiving["config"]
                serial = read_json(config["calibration"])["camera_serial"]
                load_prompt(
                    prompt_path(config, stage), config["receive_image_size"], camera_serial=serial
                )
                print(f"复用已保存的{PROMPT_NAMES[stage]}。", flush=True)
                return "prompt_reuse_" + stage
            except (ValueError, KeyError, OSError) as error:
                print(f"已保存的 bbox 不可复用：{error}；请按 E 重新绘制。", flush=True)
        self.last_draw = 0.0 if self.editor is not None else self.last_draw
        if (
            key in (ord("m"), ord("M"), ord("n"), ord("N"))
            and self.editor is None
            and receiving is not None
        ):
            return chr(key).lower()
        return chr(key).lower() if key in (ord("q"), ord("Q"), ord("x"), ord("X"), 32, 27) else None

    def close(self):
        if self.opened:
            cv2.destroyWindow(self.title)
