"""RGB overlays of the measured CAD pose, taught goal, and above-goal pose."""
from __future__ import annotations

import os
from pathlib import Path
import textwrap
import time

import cv2
import numpy as np


def draw_sam_prompt(image, prompt):
    box = prompt.get("box_xyxy", prompt.get("prompt_box_xyxy"))
    if box is None:
        return
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2-1, y2-1), (0, 220, 255), 2)
    cv2.putText(image, "SAM prompt box", (x1+4, max(16, y1-7)),
                cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 220, 255), 1, cv2.LINE_AA)
    for point, label in zip(prompt.get("point_coords", []), prompt.get("point_labels", [])):
        p = tuple(map(int, point))
        color = (70, 230, 70) if label else (80, 80, 255)
        cv2.circle(image, p, 5, (0, 0, 0), 2)
        cv2.circle(image, p, 3, color, -1)
        cv2.putText(image, "+" if label else "-", (p[0]+7, p[1]+4),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)


def receiving_overlay(image, status, now_ns=None):
    """Draw the controller's actual detection decision; never invent timer progress."""
    now_ns = time.monotonic_ns() if now_ns is None else now_ns
    image = image.copy()
    config = status["config"]
    decision, record = status.get("decision") or {}, status.get("record") or {}
    stage = status.get("stage", "starting")
    position_spiral = config.get("lowering", {}).get("mode") == "position_spiral"
    preplace_enabled = position_spiral and config.get("preplace_alignment", {}).get("enabled", True)
    preplace = status.get("preplace_alignment") or {}
    preplace_stage = stage in ("reestimating_preplace", "planning_preplace", "aligning_preplace")
    target = config["detection_duration_s"]
    elapsed = float(decision.get("stable_s", 0.))
    stamp = record.get("timestamp_ns")
    age = (now_ns-stamp)/1e9 if stamp is not None else None
    fresh = age is not None and 0 <= age <= config["detection_max_age_s"]
    accepted = (fresh and record.get("valid") and record.get("detected_in_roi")
                and not status.get("vision_error")
                and decision.get("reason") not in ("invalid_observation", "stale_observation"))
    yellow, green, gray, red = (0, 220, 255), (70, 230, 70), (155, 155, 155), (80, 80, 255)
    timed_grasp = status.get("timed_grasp", False)
    if timed_grasp and status.get("sam_prompt"):
        draw_sam_prompt(image, status["sam_prompt"])
    x1, y1, x2, y2 = config["receive_roi_xyxy"]
    if not timed_grasp:
        cv2.rectangle(image, (x1, y1), (x2, y2), yellow, 2)
        cv2.putText(image, "Fixed ROI", (x1, max(15, y1-7)), cv2.FONT_HERSHEY_SIMPLEX, .5, yellow, 1, cv2.LINE_AA)
    if fresh and "bbox_xyxy" in record and stage in ("waiting_rail", "monitor_ready", "closing"):
        bx1, by1, bx2, by2 = map(int, record["bbox_xyxy"])
        color = green if accepted else gray
        cv2.rectangle(image, (bx1, by1), (bx2, by2), color, 2)
        label = f"{'rail' if accepted else 'rejected'} {record.get('score', 0.):.2f}"
        cv2.putText(image, label, (max(0, min(bx1, image.shape[1]-150)), max(16, by1-7)),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
    color = yellow
    message = {"starting": "STARTING", "opening": "OPENING HAND",
               "initial_position": "MOVING RIGHT ARM TO RECEIVING START",
               "holding": "HOLDING - DETECTION STOPPED",
               "waiting_pose": "GRASP COMPLETE - WAITING BEFORE POSE",
               "estimating_pose": "ESTIMATING HELD POSE - ONE SHOT",
               "pose_done": "POSE READY (T: REFRESH)",
               "pose_error": "POSE UNAVAILABLE - CAMERA ONLY (T: RETRY)",
               "estimating_held_pose": "ESTIMATING HELD POSE - AUTO TRANSFER NEXT",
               "waiting_pose_override": "WAITING FOR M - USE SAVED MEAN TO MOVE",
               "waiting_move_key": "POSE MEAN SAVED - PRESS M TO MOVE",
               "planning_above": "PLANNING MOVE ABOVE TARGET",
               "moving_above": "MOVING ABOVE TARGET - KEEPING GRASP",
               "reestimating_preplace": "REESTIMATING PREPLACE POSE - N WAITS",
               "planning_preplace": "PLANNING PREPLACE ALIGNMENT",
               "aligning_preplace": "ALIGNING PREPLACE - KEEPING GRASP",
               "verifying_above": "VERIFYING RAIL POSE ABOVE TARGET",
               "above_target": "ABOVE TARGET VERIFIED - N: LOWER",
               "above_target_unverified": "VISION NOT VERIFIED - N: RECHECK ONLY",
               "planning_lowering": "PLANNING SLOW LOWERING - KEEPING GRASP",
               "lowering": "LOWERING SLOWLY - KEEPING GRASP",
               "planning_position_spiral": "PLANNING POSITION SPIRAL - KEEPING GRASP",
               "position_spiral": "POSITION SPIRAL - NO FORCE FEEDBACK",
               "placement_reached": "PLACEMENT TARGET REACHED - KEEPING GRASP",
               "switching_to_insertion": "PREPARING FORCE-CONTROLLED INSERTION",
               "baseline": "SAMPLING SUSPENDED FORCE BASELINE",
               "approaching": "APPROACHING CONTACT - KEEPING GRASP",
               "searching": "SPIRAL SEARCH - FORCE LIMITED",
               "capture": "ENTRY CANDIDATE - LATERAL SEARCH STOPPED",
               "inserting": "INSERTING - FORCE LIMITED",
               "awaiting_confirmation": "DEPTH REACHED - MANUAL CONFIRMATION REQUIRED",
               "retracting": "UNLOADING AND RETRACTING - KEEPING GRASP",
               "retracted": "RETRACTED - ATTEMPT ENDED; KEEPING GRASP",
               "paused": "PAUSED - M CANNOT RESUME; REASON BELOW",
               "monitor_ready": "READY - PREVIEW ONLY, NO HAND COMMAND"}.get(stage)
    if stage == "estimating_held_pose" and config.get("move_above", {}).get("start_mode", "manual") == "manual":
        message = "ESTIMATING HELD POSE - WILL WAIT FOR M"
    if stage == "above_target" and position_spiral:
        message = ("PREPLACE ALIGNED - N: SPIRAL INSERTION" if preplace_enabled
                   else "ABOVE TARGET REACHED - N: SPIRAL INSERTION")
    if preplace.get("final_pose_lock") and stage in ("planning_preplace", "aligning_preplace"):
        message = ("PLANNING FINAL POSE LOCK" if stage == "planning_preplace"
                   else "FINAL POSE LOCK - KEEPING GRASP")
    if stage == "paused":
        color = red
    elif stage in ("pose_error", "above_target_unverified"):
        color = red
    elif stage in ("pose_done", "above_target", "placement_reached"):
        color = green
    elif stage == "closing":
        manual = status.get("grasp_source") == "keyboard"
        message, color = ("CLOSING HAND (G)" if manual else "CLOSING HAND (CNOS)"), green
        if not manual:
            elapsed = target
    elif stage in ("holding", "monitor_ready"):
        color = green
        if status.get("grasp_source") != "keyboard":
            elapsed = target
    elif status.get("vision_error") and not status.get("live_pose_preview"):
        message = "VISION UNAVAILABLE - CHECK CAMERA" if status.get("monitor_only") else "VISION UNAVAILABLE - G STILL AVAILABLE"
        color = red
        elapsed = 0.
    elif message is None:
        if not record:
            message = "LOADING CNOS / WAITING FOR FIRST RESULT"
        elif not fresh or decision.get("reason") == "stale_observation":
            message, elapsed = "RESULT STALE - TIMER RESET", 0.
        elif accepted:
            message, color = "RAIL DETECTED - COUNTING", green
        else:
            message, elapsed = "WAITING FOR RAIL - TIMER RESET", 0.
    elapsed = min(target, max(0., elapsed))
    if timed_grasp:
        target = config["grasp_delay_s"]
        elapsed = target-max(0., min(target, status.get("grasp_remaining_s", target)))
        if stage in ("starting", "opening", "waiting_rail"):
            manual = config["grasp_mode"] == "manual"
            message = "WAITING FOR G" if manual else "TIMED GRASP - NO VISION CHECK"
            if manual:
                elapsed = 0.
            if stage == "starting":
                message = "INITIALIZING DEVICES - WAITING FOR CAMERA"
            elif stage == "opening":
                message = "OPENING HAND - GRASP TIMER NOT STARTED"
        elif stage == "closing":
            message = "CLOSING HAND (G)" if status.get("grasp_source") == "keyboard" else "CLOSING HAND (TIMER)"
        elif stage == "waiting_pose":
            target = config["post_grasp_pose_delay_s"]
            elapsed = target-max(0., min(target, status.get("pose_remaining_s", target)))
    h, w = image.shape[:2]
    pose_record = status.get("pose_record")
    panel = np.full((240 if pose_record else 166, w, 3), 25, np.uint8)
    cv2.putText(panel, message, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .57, color, 1, cv2.LINE_AA)
    timer = f"Continuous detection: {elapsed:.2f} / {target:.2f} s"
    if timed_grasp:
        timer = f"Auto close in: {status.get('grasp_remaining_s', target):.1f} s"
        if config["grasp_mode"] == "manual":
            timer = "Manual grasp: G to close"
    if stage in ("closing", "holding") and status.get("grasp_source") == "keyboard":
        timer = "Continuous detection: bypassed by G"
    if status.get("live_pose_preview"):
        timer = f"Above taught goal: {config['move_above']['height_m']*1000:.0f} mm | hand remains closed"
        if stage in ("planning_lowering", "lowering", "placement_reached"):
            timer = "Lower to taught goal | hand remains closed"
        if stage in ("switching_to_insertion", "baseline", "approaching", "searching", "capture",
                     "inserting", "awaiting_confirmation", "retracting", "retracted"):
            timer = "Force-guided insertion | hand remains closed | no automatic release"
        if (stage in ("planning_position_spiral", "position_spiral", "awaiting_confirmation")
                and config.get("lowering", {}).get("mode") == "position_spiral"):
            timer = "Position spiral | no force feedback | hand remains closed"
        elapsed = 0.
    elif stage in ("estimating_pose", "pose_done", "pose_error"):
        timer = "SAM prompted mask -> FoundationPose - hand is closed"
    elif timed_grasp and stage in ("starting", "opening", "initial_position"):
        timer = "Grasp countdown starts when the hand is open and ready"
    elif stage == "waiting_pose":
        timer = f"Pose recognition starts in: {status.get('pose_remaining_s', target):.1f} s"
    cv2.putText(panel, timer, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, .62, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.rectangle(panel, (12, 64), (w-12, 78), (65, 65, 65), -1)
    if elapsed > 0:
        cv2.rectangle(panel, (12, 64), (12+int((w-24)*elapsed/target), 78), color, -1)
    score = f"{record['score']:.2f}" if "score" in record else "--"
    age_text = f"{max(0., age):.2f}s" if age is not None else "--"
    detail = ("Error: "+str(status["vision_error"])[:90] if status.get("vision_error")
              else f"CNOS score: {score} | threshold: {config['cnos_detection']['confidence_threshold']:.2f} | result age: {age_text}")
    if timed_grasp and not status.get("vision_error"):
        detail = "No detection before grasp | After grasp: SAM prompt -> FoundationPose"
        phases = {"opening_rgbd": "Opening RGB-D camera (first frame timeout: 10 s)",
                  "acquiring_rgbd": "Acquiring RGB-D image", "loading_sam": "Loading SAM (saved box / points)",
                  "segmenting_held_rail": "SAM: segmenting one rail with box / points",
                  "initializing_pose": "New mask ready; initializing FoundationPose"}
        if stage in ("estimating_pose", "estimating_held_pose"):
            detail = phases.get(status.get("pose_phase"), "Waiting for postgrasp recognition")
        elif status.get("live_pose_preview"):
            detail = "SDK position control | One endpoint above assembly target"
    if pose_record and pose_record.get("pose_snapshot"):
        detail = "Green: projected CAD / box | axes X: red, Y: green, Z: blue"
    if stage in ("waiting_move_key", "waiting_pose_override"):
        age = status.get("saved_pose_age_s")
        detail = f"Saved mean: {age:.1f} s ago | M uses this mean; no resampling" if age is not None else "M uses the saved mean; no resampling"
    if stage in ("planning_lowering", "lowering", "placement_reached"):
        detail = "Position control | Short vertical steps with measured arrival | No auto release"
    verification = status.get("visual_verification")
    if stage in ("above_target", "above_target_unverified") and verification:
        detail = (f"Vision: {verification['position_error_m']*1000:.2f} / "
                  f"{verification['position_tolerance_m']*1000:.2f} mm | "
                  f"{np.degrees(verification['rotation_error_rad']):.2f} / "
                  f"{np.degrees(verification['rotation_tolerance_rad']):.2f} deg (error / limit)")
    if stage == "above_target" and position_spiral:
        detail = ("Alignment confirmed | New grasp fixed | N: spiral down" if preplace_enabled
                  else "Position arrival confirmed | N: spiral down | No visual recheck")
    if preplace_stage:
        detail = {"reestimating_preplace": "Fresh SAM / FoundationPose | Projected rail box | N waits",
                  "planning_preplace": "Planning XYZ / attitude correction | N waits",
                  "aligning_preplace": "XYZ / attitude correction | Waiting for measured arrival"}[stage]
        if preplace.get("final_pose_lock"):
            detail = "Final small pose correction | New grasp fixed | N waits"
    if preplace_enabled and (preplace_stage or stage == "above_target") and preplace.get("measurements"):
        measurement = preplace["measurements"][-1]
        detail = (f"Last vision: {measurement['position_error_m']*1000:.2f} / "
                  f"{measurement['position_tolerance_m']*1000:.2f} mm | "
                  f"{np.degrees(measurement['rotation_error_rad']):.2f} / "
                  f"{np.degrees(measurement['rotation_tolerance_rad']):.2f} deg (error / limit)")
    if (stage in ("planning_position_spiral", "position_spiral", "awaiting_confirmation")
            and config.get("lowering", {}).get("mode") == "position_spiral"):
        detail = "XZ spiral / -Y descent | Position arrival only; assembly unverified"
    if stage == "paused" and status.get("pause_reason"):
        detail = textwrap.wrap("Paused: " + str(status["pause_reason"]), width=78)[0]
    cv2.putText(panel, detail,
                (12, 102), cv2.FONT_HERSHEY_SIMPLEX, .45, (220, 220, 220), 1, cv2.LINE_AA)
    prompt_help = ("Yellow: SAM box | E: draw new box | ENTER: save + recognize" if timed_grasp
                   else "Yellow: fixed ROI | Green: accepted rail | Gray: rejected candidate")
    if not status.get("prompt_editable", True):
        prompt_help = "SAM prompt editing disabled during transfer / pause"
    if preplace_enabled and (preplace_stage or stage == "above_target"):
        count = preplace.get("corrections", 0)
        limit = config.get("preplace_alignment", {}).get("max_corrections", 3)
        progress = "converged; SDK / FK arrival confirmed" if preplace.get("converged") else "not converged; N waits"
        if preplace.get("final_pose_lock") and not preplace.get("converged"):
            progress = "final pose lock; awaiting SDK / FK arrival"
        prompt_help = f"Visual corrections: {count} / {limit} | {progress}"
    if stage == "paused" and status.get("pause_reason"):
        lines = textwrap.wrap("Paused: " + str(status["pause_reason"]), width=78)
        prompt_help = lines[1] if len(lines) > 1 else "M cannot restart a paused arm; inspect the reason above"
    cv2.putText(panel, prompt_help,
                (12, 126), cv2.FONT_HERSHEY_SIMPLEX, .43, (210, 210, 210), 1, cv2.LINE_AA)
    controls = "Q / SPACE / ESC: exit preview" if status.get("monitor_only") else "Q / SPACE / ESC: pause | X: exit | G / jog: terminal"
    if status.get("live_pose_preview"):
        controls = "Q / SPACE / ESC: pause | X: exit"
        if stage in ("estimating_held_pose", "waiting_move_key", "waiting_pose_override"):
            controls += " | M: move using saved mean"
        if stage == "above_target":
            controls += {"force_spiral": " | N: force-guided insertion",
                         "position_spiral": " | N: position spiral",
                         "position_steps": " | N: lower slowly"}[
                             config.get("lowering", {}).get("mode", "force_spiral")]
        if stage == "above_target_unverified":
            controls += " | N: recheck vision (no motion)"
        if stage in ("above_target", "above_target_unverified", "placement_reached"):
            controls += " | jog: terminal"
    cv2.putText(panel, controls, (12, 150), cv2.FONT_HERSHEY_SIMPLEX, .43, (210, 210, 210), 1, cv2.LINE_AA)
    if pose_record:
        xyz = np.asarray(pose_record["T_C_CAD"])[:3, 3]
        pose_age = max(0., (now_ns-pose_record["timestamp_ns"])/1e9)
        lines = [f"Last measured pose in camera frame | snapshot age {pose_age:.1f}s (not live)",
                 "CAD xyz [m]: "+" ".join(f"{v:.4f}" for v in xyz)]
        quat = pose_record.get("quaternion_xyzw")
        alignment = pose_record.get("pose_alignment")
        if alignment is not None:
            lines.append("Aligned | goal-relative XYZ [deg]: " +
                         " / ".join(f"{v:.2f}" for v in alignment["relative_xyz_deg"]))
        elif quat is not None:
            lines.append("CAD quat xyzw: "+" ".join(f"{v:.4f}" for v in quat))
        for index, line in enumerate(lines):
            cv2.putText(panel, line, (12, 177+23*index), cv2.FONT_HERSHEY_SIMPLEX, .43, (220, 240, 220), 1, cv2.LINE_AA)
    return np.concatenate((image, panel), axis=0)


def pose_overlay(rgb, mesh, pose, K, references=None, reason="", *, show_model=False, show_info=True):
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if show_model and mesh is not None and pose is not None:
        vertices = np.asarray(mesh.vertices) @ pose[:3, :3].T + pose[:3, 3]
        faces = np.asarray(mesh.faces)
        visible = faces[np.all(vertices[faces, 2] > .001, axis=1)]
        projected = vertices @ K.T
        uv = np.clip(projected[:, :2]/np.maximum(projected[:, 2:3], .001), -10000, 10000).astype(np.int32)
        silhouette = np.zeros(image.shape[:2], np.uint8)
        for face in visible:
            cv2.fillConvexPoly(silhouette, uv[face], 255)
        selected = silhouette > 0
        image[selected] = (.78*image[selected] + .22*np.array([70, 230, 70])).astype(np.uint8)
        contours, _ = cv2.findContours(silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (70, 230, 70), 1, cv2.LINE_AA)
    low, high = mesh.bounds if mesh is not None else (np.zeros(3), np.zeros(3))
    corners = np.array([[x, y, z] for x in [low[0], high[0]]
                        for y in [low[1], high[1]] for z in [low[2], high[2]]])

    def project(points, T):
        camera = points @ T[:3, :3].T+T[:3, 3]
        if np.any(camera[:, 2] <= .001):
            return None
        p = camera @ K.T
        return np.clip(p[:, :2]/p[:, 2, None], -10000, 10000).astype(int)

    references = references or {}
    for value, color in [(references.get("T_C_O_goal"), (0, 150, 255)),
                         (references.get("T_C_O_above"), (255, 220, 0)),
                         (pose, (70, 230, 70))]:
        if value is None:
            continue
        T = np.asarray(value)
        pixels = project(corners, T)
        if pixels is not None:
            for i in range(8):
                for bit in (1, 2, 4):
                    if i < (i ^ bit):
                        cv2.line(image, tuple(pixels[i]), tuple(pixels[i ^ bit]), color, 2, cv2.LINE_AA)
    if pose is not None:
        axes = project(np.vstack([np.zeros(3), np.eye(3)*.06]), pose)
        if axes is not None:
            for index, color in enumerate(((0, 0, 255), (0, 255, 0), (255, 0, 0)), 1):
                cv2.arrowedLine(image, tuple(axes[0]), tuple(axes[index]), color, 2, cv2.LINE_AA)
                cv2.putText(image, "XYZ"[index-1], tuple(axes[index]+[4, -4]),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
        from scipy.spatial.transform import Rotation
        xyz = pose[:3, 3]
        quat = Rotation.from_matrix(pose[:3, :3]).as_quat()
        legend = ("GREEN observed | ORANGE goal | CYAN above | axes X red Y green Z blue" if references else
                  "GREEN rail | axes X red Y green Z blue")
        text = [legend,
                "camera CAD xyz [m]: "+" ".join(f"{v:.3f}" for v in xyz),
                "camera CAD quat xyzw: "+" ".join(f"{v:.3f}" for v in quat)]
        if reason:
            text.append(reason)
    else:
        text = [("FoundationPose: " if mesh is not None else "CNOS: ")+(reason or "waiting for rail"),
                "ORANGE goal | CYAN above | controls: terminal" if mesh is not None
                else "YELLOW receiving region | GREEN matched rail | controls: terminal"]
    if not show_info:
        return image
    cv2.rectangle(image, (0, 0), (image.shape[1], len(text)*22+8), (25, 25, 25), -1)
    for index, line in enumerate(text):
        cv2.putText(image, line, (8, 22+22*index), cv2.FONT_HERSHEY_SIMPLEX, .43,
                    (240, 240, 240), 1, cv2.LINE_AA)
    return image


def save_pose_snapshot(path, frame, mesh, pose, reason=""):
    """Persist the overlay on the exact RGB frame used for the reported pose."""
    image = pose_overlay(frame.rgb, mesh, pose, frame.K, reason=reason, show_model=True, show_info=False)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Could not encode pose snapshot")
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(encoded.tobytes())
    temporary.replace(path)


def save_mask_snapshot(path, frame, mask, record):
    """Show the one mask on the exact frame SAM segmented, not a newer frame."""
    image = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    image[mask] = (.55*image[mask]+.45*np.array([70, 230, 70])).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (70, 230, 70), 1)
    draw_sam_prompt(image, record)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Could not encode SAM mask snapshot")
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(encoded.tobytes())
    temporary.replace(path)


def save_preview(directory, frame, mesh, pose, references=None, reason="", roi=None, detection=None):
    if mesh is None:
        # The parent draws boxes and the authoritative timer on an unpainted image.
        raw_path = Path(directory) / "camera.png"
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("Could not encode main-camera preview")
        temporary = raw_path.with_suffix(".tmp")
        temporary.write_bytes(encoded.tobytes())
        temporary.replace(raw_path)
    image = pose_overlay(frame.rgb, mesh, pose, frame.K, references, reason)
    if roi is not None:
        x1, y1, x2, y2 = roi
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 255), 2)
        cv2.putText(image, "CNOS receiving region", (x1-25, y1-8), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 220, 255), 1)
    if detection and detection.get("valid"):
        x1, y1, x2, y2 = map(int, detection["bbox_xyxy"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (50, 230, 50), 2)
        cv2.putText(image, f"CNOS {detection['score']:.2f}", (x1, max(90, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (50, 230, 50), 1)
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
        self.enabled = bool(enabled and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")))
        self.stamp, self.opened = None, False
        self.image = None
        self.snapshot = self.snapshot_stamp = None
        self.prompt = self.prompt_stamp = None
        self.editor = self.edit_rgb = self.edit_config = None
        self.last_draw = 0.

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
            prompt_path = Path(receiving["config"].get("sam_prompt", ""))
            prompt_stamp = (str(prompt_path), prompt_path.stat().st_mtime_ns) if prompt_path.is_file() else None
            if prompt_stamp != self.prompt_stamp:
                from rail_receive.sam_prompt import load_prompt
                try:
                    self.prompt = load_prompt(prompt_path, receiving["config"]["receive_image_size"])
                except (OSError, ValueError):
                    self.prompt = None
                self.prompt_stamp = prompt_stamp
            receiving["sam_prompt"] = self.prompt
        camera = path.parent / "camera.png"
        if (receiving is not None and camera.is_file()
                and not (receiving.get("live_pose_preview") and path.is_file())):
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
        if self.image is not None and now-self.last_draw >= .05:
            image = self.image
            record = (receiving or {}).get("pose_record") or (receiving or {}).get("mask_record") or {}
            snapshot_name = record.get("pose_snapshot") or record.get("mask_snapshot")
            if snapshot_name:
                snapshot_path = path.parent / snapshot_name
                snapshot_stamp = ((str(snapshot_path), snapshot_path.stat().st_mtime_ns)
                                  if snapshot_path.is_file() else None)
                if snapshot_stamp is not None and snapshot_stamp != self.snapshot_stamp:
                    snapshot = cv2.imdecode(np.frombuffer(snapshot_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
                    if snapshot is not None:
                        self.snapshot, self.snapshot_stamp = snapshot, snapshot_stamp
                if snapshot_stamp is not None and self.snapshot is not None:
                    h, w = image.shape[:2]
                    snapshot = cv2.resize(self.snapshot, (w, h))
                    image = np.concatenate((image, snapshot), axis=1)
                    label = "LAST POSE SNAPSHOT" if record.get("pose_snapshot") else "LAST SAM MASK"
                    for x, label in ((0, "LIVE CAMERA"), (w, label+" (NOT LIVE) | T: refresh")):
                        cv2.rectangle(image, (x, 0), (x+w, 24), (25, 25, 25), -1)
                        cv2.putText(image, label, (x+8, 17), cv2.FONT_HERSHEY_SIMPLEX, .45,
                                    (240, 240, 240), 1, cv2.LINE_AA)
            image = receiving_overlay(image, receiving) if receiving is not None else image
            if self.editor is not None:
                from set_sam_prompt import _draw
                image = _draw(cv2, self.edit_rgb, self.editor)
                h = self.editor.height
                cv2.rectangle(image, (0, h+39), (image.shape[1], h+65), (24, 24, 24), -1)
                cv2.putText(image, "Frozen current frame | drag left: box (when no box)",
                            (10, h+57), cv2.FONT_HERSHEY_SIMPLEX, .45, (235, 235, 235), 1, cv2.LINE_AA)
                cv2.rectangle(image, (0, h+89), (image.shape[1], h+114), (24, 24, 24), -1)
                cv2.putText(image, "ENTER save + pose | E cancel | Q/SPACE pause | X exit",
                            (10, h+105), cv2.FONT_HERSHEY_SIMPLEX, .45, (235, 235, 235), 1, cv2.LINE_AA)
            cv2.imshow(self.title, image)
            self.opened, self.last_draw = True, now
        if not self.opened:
            return None
        key = cv2.waitKey(1) & 0xff
        if self.editor is not None:
            if key in (ord("e"), ord("E"), 27):
                self.editor = self.edit_rgb = self.edit_config = None
            elif key in (10, 13):
                from set_sam_prompt import _save_prompt
                from rail_preinsert.core import read_json
                try:
                    serial = read_json(self.edit_config["calibration"])["camera_serial"]
                    _save_prompt(self.edit_config["sam_prompt"], self.editor.record(serial))
                    self.editor = self.edit_rgb = self.edit_config = None
                    self.prompt_stamp = None
                    print("新 SAM bbox 已保存。", flush=True)
                    self.last_draw = 0.
                    return "prompt_saved"
                except (ValueError, OSError) as error:
                    print(f"SAM 提示未保存：{error}", flush=True)
            elif key in (ord("b"), ord("B")):
                self.editor.redraw_box()
            elif key in (ord("c"), ord("C")):
                self.editor.clear_points()
            elif key in (ord("z"), ord("Z")):
                self.editor.undo()
        elif (key in (ord("e"), ord("E")) and receiving is not None and self.stamp is not None
              and receiving.get("prompt_editable", True)):
            from set_sam_prompt import PromptEditor
            self.editor = PromptEditor(receiving["config"]["receive_image_size"])
            self.edit_config = dict(receiving["config"])
            self.editor.frozen = True
            self.editor.status = "Drag a new rail box; ENTER saves and requests a new pose"
            self.edit_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            cv2.setMouseCallback(self.title, self._mouse)
        self.last_draw = 0. if self.editor is not None else self.last_draw
        if (key in (ord("m"), ord("M"), ord("n"), ord("N")) and self.editor is None and receiving is not None
                and not receiving.get("monitor_only")):
            return chr(key).lower()
        return chr(key).lower() if key in (ord("q"), ord("Q"), ord("x"), ord("X"), 32, 27) else None

    def close(self):
        if self.opened:
            cv2.destroyWindow(self.title)
