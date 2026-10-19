"""Frozen visual/stability factors and the shared initialization check.

Nothing in this module constructs a device, opens a camera, or starts motion.
The visual specification is applied only when a single geometry-preserving
candidate has actually been frozen. The prepared batch does not have one.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


SIDES = ("left", "right")

STABILITY_PARAMETERS = {
    "window_s": 0.2,
    "max_eef_translation_excursion_mm": 0.5,
    "max_eef_rotation_excursion_deg": 0.2,
    "max_hand_joint_excursion_deg": 0.5,
    "max_feedback_gap_s": 0.05,
    "max_feedback_age_s": 0.08,
    "post_inference_max_eef_translation_mm": 1.0,
    "post_inference_max_eef_rotation_deg": 0.5,
    "post_inference_max_hand_joint_deg": 0.5,
    "max_wait_s": 2.0,
    "max_resamples": 2,
}

INITIALIZATION_TOLERANCES = {
    "max_eef_position_error_mm_per_arm": 3.0,
    "max_eef_orientation_error_deg_per_arm": 1.0,
    "max_hand_joint_error_deg": 3.0,
    "max_arm_joint_error_deg": 2.0,
}


class VisualCandidateError(RuntimeError):
    """The frozen batch has no deployable visual correction."""


def visual_is_deployable(spec):
    """A deployable V is one frozen formula with parameters, not a brightness target."""
    return bool(spec.get("frozen") and spec.get("identifiable") is True
                and spec.get("formula") and spec.get("parameters"))


def apply_visual(observation, spec):
    """Return a copy of the observation with the frozen correction applied.

    The current batch spec is intentionally not identifiable. Calling this with
    that spec raises instead of inventing a brightness match.
    """
    if not visual_is_deployable(spec):
        raise VisualCandidateError(
            "没有可重复施加、且保留场景几何的单一视觉修正；不能用示范画面或目标亮度代替")
    if spec["formula"] != "identity":
        raise VisualCandidateError("未实现的视觉公式：" + str(spec["formula"]))
    corrected = {key: np.array(value, copy=True) for key, value in observation.items()}
    return corrected


def rotation_delta_deg(left_rotvec, right_rotvec):
    """Geodesic angle between two rotation vectors, in degrees."""
    delta = Rotation.from_rotvec(np.asarray(right_rotvec, dtype=np.float64))
    delta = delta * Rotation.from_rotvec(np.asarray(left_rotvec, dtype=np.float64)).inv()
    return float(np.rad2deg(delta.magnitude()))


def _window(series, now_ns, params):
    if not series:
        return None, "no feedback"
    latest = int(series[-1][0])
    age_s = (int(now_ns) - latest) / 1e9
    if age_s < -1e-4:
        return None, "feedback timestamp is ahead of the clock"
    if age_s > params["max_feedback_age_s"]:
        return None, "feedback too old"
    start = latest - int(round(params["window_s"] * 1e9))
    chosen = [(int(stamp), np.asarray(value, dtype=np.float64))
              for stamp, value in series if int(stamp) >= start]
    if len(chosen) < 2:
        return None, "stable window has too few samples"
    span_s = (chosen[-1][0] - chosen[0][0]) / 1e9
    if span_s + 1e-6 < params["window_s"]:
        return None, "stable window is shorter than the frozen duration"
    gaps = np.diff([stamp for stamp, _ in chosen]) / 1e9
    if np.any(gaps <= 0) or np.any(gaps > params["max_feedback_gap_s"] + 1e-9):
        return None, "feedback gap exceeds the frozen limit"
    return chosen, None


def _arm_excursion(chosen):
    origin, origin_rotation = chosen[0][1][:3], chosen[0][1][3:6]
    translation_mm = max(float(np.linalg.norm(value[:3] - origin)) * 1000 for _, value in chosen)
    rotation_deg = max(rotation_delta_deg(origin_rotation, value[3:6]) for _, value in chosen)
    return translation_mm, rotation_deg


def _hand_excursion_deg(chosen):
    origin = chosen[0][1]
    return float(np.rad2deg(max(np.max(np.abs(value - origin)) for _, value in chosen)))


def _plain_verdict(stable, reason, measurements, snapshot=None, timed_out=False):
    return {"stable": bool(stable), "timed_out": bool(timed_out), "reason": reason,
            "measurements": measurements, "snapshot": snapshot}


def latest_snapshot(histories, max_gap_s=None):
    """Latest simultaneous flange pose and all 40 hand joints, or None."""
    try:
        arms = histories["arms"]
        hands = histories["hands"]
        if any(not arms[side] or not hands[side] for side in SIDES):
            return None
    except (KeyError, TypeError):
        return None
    gap_s = STABILITY_PARAMETERS["max_feedback_gap_s"] if max_gap_s is None else max_gap_s
    stamps = [int(arms[side][-1][0]) for side in SIDES] + [int(hands[side][-1][0]) for side in SIDES]
    if max(stamps) - min(stamps) > int(round(gap_s * 1e9)):
        return None
    snapshot = {"stamp_ns": int(max(stamps))}
    for side in SIDES:
        value = np.asarray(arms[side][-1][1], dtype=np.float64)
        snapshot[f"{side}_xyz"] = value[:3].copy()
        snapshot[f"{side}_rotvec"] = value[3:6].copy()
        snapshot[f"{side}_hand"] = np.asarray(hands[side][-1][1], dtype=np.float64).copy()
    return snapshot


def assess_feedback(histories, now_ns, params=None):
    """Return whether the newest actual-feedback window is inside the frozen gate."""
    params = STABILITY_PARAMETERS if params is None else params
    measurements = {}
    windows = {}
    for side in SIDES:
        for kind, source in (("arm", histories.get("arms", {}).get(side, ())),
                             ("hand", histories.get("hands", {}).get(side, ()))):
            chosen, reason = _window(source, now_ns, params)
            measurements[f"{side}_{kind}"] = reason or "ok"
            if chosen is None:
                return _plain_verdict(False, f"{side} {kind}: {reason}", measurements)
            windows[(side, kind)] = chosen
    for side in SIDES:
        translation_mm, rotation_deg = _arm_excursion(windows[(side, "arm")])
        hand_deg = _hand_excursion_deg(windows[(side, "hand")])
        measurements[f"{side}_translation_mm"] = translation_mm
        measurements[f"{side}_rotation_deg"] = rotation_deg
        measurements[f"{side}_hand_deg"] = hand_deg
        if translation_mm > params["max_eef_translation_excursion_mm"]:
            return _plain_verdict(False, f"{side} translation excursion {translation_mm:.3f} mm", measurements)
        if rotation_deg > params["max_eef_rotation_excursion_deg"]:
            return _plain_verdict(False, f"{side} rotation excursion {rotation_deg:.3f} deg", measurements)
        if hand_deg > params["max_hand_joint_excursion_deg"]:
            return _plain_verdict(False, f"{side} hand excursion {hand_deg:.3f} deg", measurements)
    snapshot = latest_snapshot(histories, params["max_feedback_gap_s"])
    if snapshot is None:
        return _plain_verdict(False, "feedback streams are not simultaneous", measurements)
    return _plain_verdict(True, None, measurements, snapshot)


def snapshot_delta(before, after, params=None):
    """Translation, rotation and hand change between two actual-feedback snapshots."""
    params = STABILITY_PARAMETERS if params is None else params
    if before is None or after is None:
        return {"ok": False, "reason": "missing actual feedback", "measurements": {}}
    measurements = {}
    ok = True
    reason = None
    for side in SIDES:
        translation_mm = float(np.linalg.norm(after[f"{side}_xyz"] - before[f"{side}_xyz"]) * 1000)
        rotation_deg = rotation_delta_deg(before[f"{side}_rotvec"], after[f"{side}_rotvec"])
        hand_deg = float(np.rad2deg(np.max(np.abs(after[f"{side}_hand"] - before[f"{side}_hand"]))))
        measurements[f"{side}_translation_mm"] = translation_mm
        measurements[f"{side}_rotation_deg"] = rotation_deg
        measurements[f"{side}_hand_deg"] = hand_deg
        if translation_mm > params["post_inference_max_eef_translation_mm"]:
            ok, reason = False, f"{side} moved {translation_mm:.3f} mm during inference"
        elif rotation_deg > params["post_inference_max_eef_rotation_deg"]:
            ok, reason = False, f"{side} rotated {rotation_deg:.3f} deg during inference"
        elif hand_deg > params["post_inference_max_hand_joint_deg"]:
            ok, reason = False, f"{side} hand moved {hand_deg:.3f} deg during inference"
    return {"ok": ok, "reason": reason, "measurements": measurements}


def submission_decision(post_ok, resamples, max_resamples):
    """Initial attempt plus at most ``max_resamples`` retries, then stop."""
    if post_ok:
        return "submit"
    if resamples >= max_resamples:
        return "stop"
    return "resample"


class StabilityGate:
    """Bounded wait and post-inference check. The servo target is never reset here."""

    def __init__(self, params=None):
        self.params = dict(STABILITY_PARAMETERS if params is None else params)
        self.resamples = 0

    def wait_stable(self, read_since, clock_ns, sleep, should_continue):
        start = int(clock_ns())
        deadline = start + int(round(self.params["max_wait_s"] * 1e9))
        last = _plain_verdict(False, "stability wait did not run", {})
        while should_continue():
            now = int(clock_ns())
            since = now - int(round((self.params["window_s"] + self.params["max_feedback_gap_s"]) * 1e9))
            last = assess_feedback(read_since(since), now, self.params)
            if last["stable"]:
                return last
            if now >= deadline:
                last = {**last, "stable": False, "timed_out": True, "reason": "stability wait timeout"}
                return last
            sleep(0.01)
            if int(clock_ns()) == now:
                # A real sleep that returns immediately must not spin forever.
                break
        if int(clock_ns()) >= deadline:
            return {**last, "stable": False, "timed_out": True, "reason": "stability wait timeout"}
        return {**last, "stable": False, "timed_out": False, "reason": last.get("reason") or "stopped"}

    def postcheck(self, before, after):
        return snapshot_delta(before, after, self.params)

    def note_accept(self):
        self.resamples = 0

    def note_reject(self):
        """Record one discarded prediction. True means another resample is still allowed."""
        decision = submission_decision(False, self.resamples, self.params["max_resamples"])
        if decision == "stop":
            return False
        self.resamples += 1
        return True


def _as_vector(values, width, name):
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (width,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite vector of length {width}")
    return vector


def compare_initialization(measured, target, tolerances=None):
    """Compare actual joints, flange pose and all 40 hand joints with the frozen start."""
    tolerances = INITIALIZATION_TOLERANCES if tolerances is None else tolerances
    deltas = {}
    comparable = True

    def fail(name, value, limit):
        nonlocal comparable
        deltas[name] = float(value)
        if value > limit:
            comparable = False

    for side in SIDES:
        arm = _as_vector(measured["arm_q"][side], 7, side + " arm")
        goal_arm = _as_vector(target["arm_q"][side], 7, side + " arm target")
        fail(f"{side}_arm_joint_deg", float(np.rad2deg(np.max(np.abs(arm - goal_arm)))),
             tolerances["max_arm_joint_error_deg"])
        pose = _as_vector(measured["eef"][side], 6, side + " eef")
        goal_pose = _as_vector(target["eef"][side], 6, side + " eef target")
        fail(f"{side}_eef_mm", float(np.linalg.norm(pose[:3] - goal_pose[:3]) * 1000),
             tolerances["max_eef_position_error_mm_per_arm"])
        fail(f"{side}_eef_deg", rotation_delta_deg(pose[3:], goal_pose[3:]),
             tolerances["max_eef_orientation_error_deg_per_arm"])
        hand = _as_vector(measured["hand"][side], 20, side + " hand")
        goal_hand = _as_vector(target["hand"][side], 20, side + " hand target")
        fail(f"{side}_hand_deg", float(np.rad2deg(np.max(np.abs(hand - goal_hand)))),
             tolerances["max_hand_joint_error_deg"])
    return {"comparable": comparable, "deltas": deltas, "tolerances": dict(tolerances)}


def target_from_reference(reference, frame=None):
    """Frame 1 of the training episode is the comparison target, not a control observation."""
    frame = reference["preferred_latest_observation_frame"] if frame is None else int(frame)
    joints = reference["robot_joint"][frame]
    pose = reference["robot_eef_pose"][frame]
    hands = reference["hand_joint"][frame]
    return {
        "frame": frame,
        "arm_q": {"left": list(joints[:7]), "right": list(joints[7:])},
        "eef": {"left": list(pose[:6]), "right": list(pose[6:])},
        "hand": {"left": list(hands[:20]), "right": list(hands[20:])},
    }


def _constant_histories(start_ns, end_ns, step_ns, *, arm_offset_m=0.0, hand_offset_rad=0.0):
    arms, hands = {}, {}
    for index, side in enumerate(SIDES):
        arm_series, hand_series = [], []
        stamp = int(start_ns)
        while stamp <= end_ns:
            moved = arm_offset_m if stamp == end_ns else 0.0
            arm = np.zeros(6, np.float64)
            arm[0] = 0.6 + index * 0.01 + moved
            arm[4] = 1.62
            hand = np.full(20, 0.2 + index * 0.01, np.float64)
            if stamp == end_ns:
                hand[0] += hand_offset_rad
            arm_series.append((stamp, arm))
            hand_series.append((stamp, hand))
            stamp += int(step_ns)
        arms[side] = arm_series
        hands[side] = hand_series
    return {"arms": arms, "hands": hands}


def _matching_state(target):
    return {"arm_q": {side: np.asarray(target["arm_q"][side]) for side in SIDES},
            "eef": {side: np.asarray(target["eef"][side]) for side in SIDES},
            "hand": {side: np.asarray(target["hand"][side]) for side in SIDES}}


def run_fake_preflight(reference_target):
    """Fake clock, record-free gate checks and the stop path. Does not load a policy."""
    params = STABILITY_PARAMETERS
    now = 2_000_000_000
    step = 10_000_000
    quiet = _constant_histories(now - int(params["window_s"] * 1e9), now, step)
    stable = assess_feedback(quiet, now, params)
    if not stable["stable"] or stable["snapshot"] is None:
        raise AssertionError("quiet feedback was not accepted: " + str(stable["reason"]))
    moving = _constant_histories(now - int(params["window_s"] * 1e9), now, step, arm_offset_m=0.002)
    moved = assess_feedback(moving, now, params)
    if moved["stable"]:
        raise AssertionError("a 2 mm excursion was accepted")
    gapped = _constant_histories(now - int(params["window_s"] * 1e9), now, step)
    for side in SIDES:
        gapped["arms"][side] = [item for item in gapped["arms"][side]
                                if item[0] <= now - 100_000_000 or item[0] >= now - 40_000_000]
    if assess_feedback(gapped, now, params)["stable"]:
        raise AssertionError("a 60 ms feedback gap was accepted")
    stale = assess_feedback(quiet, now + int(0.2e9), params)
    if stale["stable"]:
        raise AssertionError("stale feedback was accepted")
    clock = {"now": 0}

    def clock_ns():
        return clock["now"]

    def sleep(seconds):
        clock["now"] += int(seconds * 1e9)

    def read_since(since_ns):
        return _constant_histories(clock["now"] - int(0.05e9), clock["now"], step, arm_offset_m=0.01)

    timed_out = StabilityGate(params).wait_stable(read_since, clock_ns, sleep, lambda: True)
    if not timed_out["timed_out"] or timed_out["stable"] or clock["now"] < int(params["max_wait_s"] * 1e9):
        raise AssertionError("fake clock did not stop at the frozen wait")
    gate = StabilityGate(params)
    if submission_decision(True, 2, params["max_resamples"]) != "submit":
        raise AssertionError("an inside-tolerance prediction was not submitted")
    if not gate.note_reject() or not gate.note_reject() or gate.note_reject():
        raise AssertionError("resample limit did not stop on the third rejection")
    same = snapshot_delta(stable["snapshot"], stable["snapshot"], params)
    if not same["ok"]:
        raise AssertionError("identical feedback failed the post-inference check")
    shifted = {key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value
               for key, value in stable["snapshot"].items()}
    shifted["left_xyz"] = shifted["left_xyz"] + np.array([0.002, 0, 0])
    if snapshot_delta(stable["snapshot"], shifted, params)["ok"]:
        raise AssertionError("2 mm of inference motion was submitted")
    measured = _matching_state(reference_target)
    if not compare_initialization(measured, reference_target)["comparable"]:
        raise AssertionError("the frozen start did not match itself")
    measured["arm_q"]["left"] = measured["arm_q"]["left"] + np.deg2rad(3)
    if compare_initialization(measured, reference_target)["comparable"]:
        raise AssertionError("a 3 degree arm error was accepted")
    try:
        apply_visual({}, {"frozen": True, "identifiable": False, "formula": None, "parameters": None})
    except VisualCandidateError:
        pass
    else:
        raise AssertionError("an unidentified visual candidate was applied")
    _check_trace_keeps_rejected_decision_and_faults()
    return {"fake_clock": True, "stability_stop": True, "initialization_check": True,
            "visual_candidate_refused": True, "record_integrity": True, "hardware_started": False}


def _check_trace_keeps_rejected_decision_and_faults():
    """Source-gap floods must not evict a fault, and a rejected decision keeps its gate record."""
    import tempfile
    from types import SimpleNamespace
    from diffusion_policy.real_world.decision_trace import SessionTrace

    trace = SessionTrace(decision_capacity=2, feedback_capacity=2, command_capacity=2,
                         hand_capacity=2, status_capacity=2)
    example = {name: np.zeros((2, 2, 2, 3), np.uint8) for name in ("camera_0", "camera_1", "camera_2")}
    example["robot_eef_pose"] = np.zeros((2, 12), np.float32)
    example["hand_joint"] = np.zeros((2, 40), np.float32)
    trace.allocate_decisions(example, 2)
    trace.begin_trial()
    for stamp in range(1000):
        trace.record_status(SimpleNamespace(kind="tianji.source_gap", observed_monotonic_ns=stamp, details={}))
    trace.record_status(SimpleNamespace(kind="motor.fault", observed_monotonic_ns=1000, details={"error": 1}))
    stored = trace.store_decision(
        obs=example, alignment={"query_ns": [1, 2]}, anchor_ns=10,
        actions=np.zeros((2, 52)), started_ns=11, observed_ns=12, preprocessed_ns=13,
        inputs_ready_ns=14, returned_ns=15, stability={"accepted": False, "resamples": 1})
    if stored != 0 or not trace.valid:
        raise AssertionError("recording a rejected decision or a source-gap flood invalidated the trace")
    with tempfile.TemporaryDirectory() as folder:
        meta = trace.flush(folder)
    if not meta["valid"] or meta["critical_status_dropped"] != 0:
        raise AssertionError("trace flush dropped a fault or marked the record invalid")
    if meta["status_summary"]["tianji.source_gap"]["count"] != 1000:
        raise AssertionError("source_gap aggregation did not keep the full count")
    if not any(item["kind"] == "motor.fault" for item in meta["status"]):
        raise AssertionError("the fault was evicted by source_gap events")
    if meta["decisions"][0]["stability"]["accepted"] is not False:
        raise AssertionError("the rejected decision lost its stability record")
