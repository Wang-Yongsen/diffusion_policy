from __future__ import annotations

from dataclasses import replace
from queue import Empty, Queue
from threading import Event, Thread
from pathlib import Path
import json
import os
import time
import numpy as np

from .data import SceneWriter
from .engine import failure_record
from .geometry import overlay
from .common import save_json


def require_display():
    if os.name != "nt" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError("Open this command in the remote/local desktop terminal, or use a headless command")


class OfflinePreview:
    """Display the exact processed frame and keep the final result visible."""

    title = "FoundationPose demo | q/Esc: quit | Space: pause"

    def __enter__(self):
        import cv2
        require_display()
        self.last_image = None
        cv2.namedWindow(self.title, cv2.WINDOW_AUTOSIZE)
        return self

    def __exit__(self, *exc):
        import cv2
        try:
            cv2.destroyWindow(self.title)
        except cv2.error:
            # Some GUI backends already destroy a window when its X is clicked.
            pass

    def _poll(self, delay_ms):
        import cv2
        key = cv2.waitKey(delay_ms) & 255
        try:
            visible = cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            visible = False
        if key in (ord("q"), 27) or not visible:
            return False
        if key == ord(" "):
            while True:
                key = cv2.waitKey(20) & 255
                try:
                    visible = cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) >= 1
                except cv2.error:
                    visible = False
                if key in (ord("q"), 27) or not visible:
                    return False
                if key == ord(" "):
                    break
        return True

    def show(self, image):
        import cv2
        self.last_image = image.copy()
        cv2.imshow(self.title, self.last_image)
        return self._poll(100)

    def hold(self):
        import cv2
        canvas = self.last_image.copy()
        cv2.rectangle(canvas, (0, 32), (canvas.shape[1], 60), (0, 0, 0), -1)
        cv2.putText(canvas, "Finished | q/Esc or close window to exit", (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        cv2.imshow(self.title, canvas)
        while self._poll(20):
            pass


def annotate(rgb, title="Rail mask: click boundary; Enter accept; u undo; q cancel"):
    import cv2
    require_display()
    points = []
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mask = np.zeros(rgb.shape[:2], np.uint8)

    def mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, mouse)
    try:
        while True:
            canvas = image.copy()
            canvas[mask > 0] = (canvas[mask > 0] * .6 + np.array([0, 150, 0]) * .4).astype(np.uint8)
            if points:
                cv2.polylines(canvas, [np.asarray(points, np.int32)], len(points) > 2, (0, 255, 255), 1)
                for point in points:
                    cv2.circle(canvas, point, 3, (0, 255, 255), -1)
            cv2.putText(canvas, "a: add polygon | h: exclude hole/hand | Enter: accept | u: undo | q: cancel",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
            cv2.imshow(title, canvas)
            key = cv2.waitKey(20) & 255
            if key in (ord("q"), 27) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                return None
            if key == ord("u") and points:
                points.pop()
            if key in (ord("a"), ord("h"), 13, 10) and len(points) >= 3:
                cv2.fillPoly(mask, [np.asarray(points, np.int32)], 0 if key == ord("h") else 255)
                points.clear()
            if key in (13, 10) and np.any(mask):
                return mask.astype(bool)
    finally:
        cv2.destroyWindow(title)


def run_live(camera, engine, output: Path, max_age_s: float):
    import cv2
    require_display()
    stop = Event()
    commands = Queue(maxsize=1)
    results = Queue(maxsize=1)
    record_file = (output / "poses.jsonl").open("w", encoding="utf-8")
    writer = SceneWriter(output / "scene", camera.metadata, engine.config["max_depth_m"])
    durations = []
    initializations = []
    failures = []
    reset_done = Event()
    title = "Rail 6D | i: freeze & mask | r: reset | s: save | q: quit"

    def publish(value):
        try:
            results.get_nowait()
        except Empty:
            pass
        results.put_nowait(value)

    def worker():
        last_id = None
        try:
            while not stop.is_set():
                try:
                    command, chosen = commands.get_nowait()
                    if command == "reset":
                        engine.reset()
                        reset_done.set()
                        publish((None, None, {"valid": False, "reason": "WAITING FOR MASK"}))
                        continue
                    frame, initialize = chosen, True
                except Empty:
                    if not engine.initialized:
                        stop.wait(.03)
                        continue
                    frame, initialize = camera.next(after_id=last_id), False
                last_id = frame.frame_id
                if not initialize and (time.monotonic_ns() - frame.timestamp_ns) / 1e9 > max_age_s:
                    engine.reset()
                    publish((frame, None, failure_record(frame, "stale RGB-D input; reinitialize")))
                    continue
                try:
                    pose, record = engine.infer(frame, initialize=initialize)
                    record["output_age_s"] = (time.monotonic_ns() - frame.timestamp_ns) / 1e9
                    if not initialize and record["output_age_s"] > max_age_s:
                        record["valid"] = False
                        record["reason"] = "output is older than configured age limit"
                    (initializations if initialize else durations).append(record["processing_s"])
                except Exception as exc:
                    engine.reset()
                    pose, record = None, failure_record(frame, exc)
                    failures.append(record)
                if frame.frame_id not in writer.seen:
                    writer.write(frame)
                elif initialize:
                    writer.attach_mask(frame)
                if initialize:
                    from PIL import Image
                    mask_name = f"initialization-{frame.frame_id}-{time.monotonic_ns()}.png"
                    Image.fromarray(frame.mask.astype(np.uint8) * 255).save(output / mask_name)
                    record["initialization_mask"] = mask_name
                record_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                record_file.flush()
                publish((frame, pose, record))
        except Exception as exc:
            publish((None, None, {"valid": False, "reason": str(exc), "fatal": True}))

    thread = Thread(target=worker, name="rail-pose", daemon=True)
    thread.start()
    displayed = None
    latest_result = None
    started = time.monotonic()
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    try:
        while not stop.is_set():
            frame = camera.next()
            try:
                latest_result = results.get_nowait()
            except Empty:
                pass
            if latest_result and latest_result[0] is not None:
                chosen, pose, record = latest_result
                age = (time.monotonic_ns() - chosen.timestamp_ns) / 1e9
                valid = record["valid"] and age <= max_age_s
                label = (f"{record['mode']} {record['processing_s']*1000:.0f} ms | age {age:.2f} s"
                         if valid else "STALE / LOST: press i to reinitialize")
                # Show the actual inference frame, never an old pose on a newer image.
                canvas = overlay(chosen.rgb, engine.mesh, pose if valid else None, chosen.K, label)
                displayed = (chosen, canvas)
            else:
                if latest_result and latest_result[2].get("fatal"):
                    raise RuntimeError(latest_result[2]["reason"])
                reason = latest_result[2].get("reason", "WAITING FOR MASK") if latest_result else "WAITING FOR MASK"
                canvas = overlay(frame.rgb, engine.mesh, None, frame.K, reason)
                displayed = (frame, canvas)
            cv2.imshow(title, canvas)
            key = cv2.waitKey(15) & 255
            if key in (ord("q"), 27) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("i"), 13):
                # Stop tracking before asking for a new frozen-frame annotation.
                if not commands.empty():
                    continue
                reset_done.clear()
                engine.cancel_pending()
                commands.put_nowait(("reset", None))
                # Wait for the worker to finish the current GPU call and reset.
                if not reset_done.wait(timeout=30):
                    raise RuntimeError("Timed out waiting for the inference worker to reset")
                frozen = camera.next()
                mask = annotate(frozen.rgb)
                if mask is not None and commands.empty():
                    commands.put_nowait(("initialize", replace(frozen, mask=mask)))
                    latest_result = None
            if key == ord("r") and commands.empty():
                engine.cancel_pending()
                commands.put_nowait(("reset", None))
                latest_result = None
            if key == ord("s") and displayed:
                chosen, canvas = displayed
                cv2.imwrite(str(output / f"check-{chosen.frame_id}.png"), canvas)
    finally:
        stop.set()
        engine.cancel_pending()
        thread.join(timeout=30)
        if thread.is_alive():
            raise RuntimeError("GPU worker did not finish shutdown within 30 seconds")
        record_file.close()
        cv2.destroyWindow(title)
        records = [json.loads(line) for line in (output / "poses.jsonl").read_text().splitlines()]
        save_json(output / "summary.json", {
            "duration_s": time.monotonic() - started, "initializations_s": initializations,
            "tracked_frames": len(durations), "mean_tracking_s": float(np.mean(durations)) if durations else None,
            "valid_frames": sum(record["valid"] for record in records),
            "invalid_frames": sum(not record["valid"] for record in records),
            "mean_tracking_fps": 1 / float(np.mean(durations)) if durations else None,
            "failures": len(failures), "hardware_validation": "local camera run",
        })
