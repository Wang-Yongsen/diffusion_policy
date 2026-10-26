"""Read-only right-arm wrench diagnostics, or replay without any device imports."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from rail_preinsert.core import matrix, transform, vector, write_json


def load_settings(path):
    """Read only diagnostic fields; hand, vision and insertion limits may be unset."""
    import yaml
    path = Path(path).resolve()
    source = yaml.safe_load(path.read_text())
    if not isinstance(source, dict) or not isinstance(source.get("insertion", {}), dict):
        raise ValueError("Configuration and insertion settings must be mappings")
    supplied = source.get("insertion", {})
    settings = dict(state_max_age_s=.1, baseline_duration_s=1., baseline_min_samples=30,
                    T_F_S=None, wrench_sign=None, baseline_position_tolerance_m=None,
                    orientation_tolerance_rad=None)
    settings.update({key: supplied[key] for key in settings if key in supplied})
    for key in ("state_max_age_s", "baseline_duration_s"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"insertion.{key} must be finite and positive")
    if type(settings["baseline_min_samples"]) is not int or settings["baseline_min_samples"] < 2:
        raise ValueError("insertion.baseline_min_samples must be an integer >= 2")
    for key in ("baseline_position_tolerance_m", "orientation_tolerance_rad"):
        value = settings[key]
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not np.isfinite(value) or value <= 0):
            raise ValueError(f"insertion.{key} must be null or finite and positive")
    if settings["T_F_S"] is not None:
        settings["T_F_S"] = transform(settings["T_F_S"], "insertion.T_F_S")
    sign = settings["wrench_sign"]
    if sign is not None and (isinstance(sign, bool) or sign not in (-1, 1)):
        raise ValueError("insertion.wrench_sign must be null, +1 or -1")
    arm_path = source.get("arm_config")
    settings["arm_config"] = str((path.parent/arm_path).resolve()) if arm_path else None
    return settings


def replay_records(path):
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield {"_malformed_json": True}


def live_records(settings, duration, *, driver_factory=None, kinematics_factory=None,
                 clock_ns=time.monotonic_ns, sleep=time.sleep):
    """Open feedback only; do not configure, engage, move, or load a hand SDK."""
    if not settings.get("arm_config"):
        raise ValueError("Live diagnostics require arm_config")
    from diffusion_policy.third_party.bimanual_teleop.devices.tianji.config import load_config
    if driver_factory is None:
        from diffusion_policy.third_party.bimanual_teleop.devices.tianji.driver import TianjiDriver
        driver_factory = TianjiDriver
    if kinematics_factory is None:
        from diffusion_policy.third_party.bimanual_teleop.devices.tianji.model import TianjiKinematics
        kinematics_factory = TianjiKinematics
    arms = load_config(settings["arm_config"])
    kinematics = kinematics_factory()
    driver = driver_factory(arms["controller_ip"], record_force=True)
    try:
        driver.start()
        deadline = clock_ns() + int(duration*1e9)
        while clock_ns() < deadline:
            sample = driver.get_latest()
            observed = clock_ns()
            if sample is None:
                yield {"_unavailable": "no_feedback"}
            elif not driver.health(sides=("right",)).ready:
                yield {"_unavailable": "unhealthy_feedback"}
            else:
                arm = sample.payload.arms["right"]
                yield {"event": "right_state", "timestamp_ns": sample.header.received_monotonic_ns,
                       "host_monotonic_ns": observed, "right_sequence": arm.source_sequence,
                       "wrench": arm.wrench,
                       "T_B_F": matrix(kinematics.fk("right", arm.joints.position_rad)).tolist()}
            sleep(.005)
    finally:
        # No motion was started, so TianjiDriver.close has no arms to disable.
        driver.close()


def _integer(value):
    return type(value) is int and value >= 0


def normalize_record(row):
    if not isinstance(row, dict):
        raise ValueError("invalid_record")
    if row.get("_malformed_json"):
        raise ValueError("malformed_json")
    if "_unavailable" in row:
        raise ValueError(row["_unavailable"])
    is_trace = "raw_wrench" in row
    if row.get("event") != "right_state" and not is_trace:
        return None
    stamp = row.get("timestamp_ns")
    observed = row.get("host_monotonic_ns")
    freshness_known = _integer(stamp) and _integer(observed)
    if stamp is None:
        stamp = observed
    if observed is None:
        observed = stamp
    if not _integer(stamp) or not _integer(observed):
        raise ValueError("invalid_timestamp")
    sequence = row.get("right_sequence", row.get("source_sequence"))
    if sequence is not None and (not _integer(sequence) or sequence >= 1_000_000):
        raise ValueError("invalid_source_sequence")
    try:
        wrench = vector(row.get("raw_wrench") if is_trace else row.get("wrench"), 6)
        pose = transform(row.get("measured_T_B_F") if is_trace else row.get("T_B_F"))
    except (ValueError, TypeError):
        raise ValueError("invalid_wrench_or_pose") from None
    return dict(timestamp_ns=stamp, host_monotonic_ns=observed, sequence=sequence,
                freshness_known=freshness_known, wrench=wrench, pose=pose)


def six_axis_statistics(values):
    values = np.asarray(values, dtype=float)
    return {"order": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
            "units": ["N", "N", "N", "Nm", "Nm", "Nm"],
            "mean": values.mean(axis=0).tolist(), "std": values.std(axis=0).tolist(),
            "peak_abs": np.abs(values).max(axis=0).tolist(),
            "min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist(),
            "force_norm_peak_n": float(np.linalg.norm(values[:, :3], axis=1).max()),
            "torque_norm_peak_nm": float(np.linalg.norm(values[:, 3:], axis=1).max())}


def analyze_records(records, settings, duration=3.):
    counts, samples = Counter(), []
    first_host = last_stamp = last_sequence = None
    for row in records:
        counts["records_seen"] += 1
        try:
            sample = normalize_record(row)
        except ValueError as error:
            counts[str(error)] += 1
            continue
        if sample is None:
            counts["unrelated_records"] += 1
            continue
        stamp, observed, seq = sample["timestamp_ns"], sample["host_monotonic_ns"], sample["sequence"]
        first_host = observed if first_host is None else first_host
        if observed-first_host > duration*1e9:
            break
        if not 0 <= observed-stamp <= settings["state_max_age_s"]*1e9:
            counts["stale_or_future"] += 1
            continue
        if last_sequence is not None and seq is not None:
            delta = (seq-last_sequence) % 1_000_000
            if delta == 0:
                counts["duplicate_source_sequence"] += 1
                continue
            if delta >= 500_000:
                counts["reordered_source_sequence"] += 1
                continue
        if last_stamp is not None and stamp <= last_stamp:
            counts["duplicate_or_reordered_timestamp"] += 1
            continue
        if seq is None:
            counts["source_sequence_unavailable"] += 1
        else:
            last_sequence = seq
        counts["freshness_verified" if sample["freshness_known"] else "freshness_unverifiable"] += 1
        samples.append(sample)
        last_stamp = stamp
    counts["valid_samples"] = len(samples)
    limitations = ["Read-only diagnostics; no control-mode changes or motion commands.",
        "Static baseline subtraction assumes no contact, unchanged load/grasp and fixed attitude; it is not gravity compensation.",
        "These statistics do not select safe contact-force limits or verify insertion."]
    if counts["source_sequence_unavailable"]:
        limitations.append("Some replay records lack the device source sequence; those records are deduplicated by timestamp only.")
    if counts["freshness_unverifiable"]:
        limitations.append("Some replay records lack separate feedback and observation timestamps; feedback freshness cannot be verified for them.")
    calibrated = settings["T_F_S"] is not None and settings["wrench_sign"] is not None
    if not calibrated:
        limitations.append("T_F_S or wrench_sign is unset; only raw sensor-axis statistics are available.")
    result = {"schema_version": 1, "status": "insufficient_samples", "motion_enabled": False,
              "counts": dict(counts), "required_baseline_samples": settings["baseline_min_samples"],
              "required_baseline_span_s": settings["baseline_duration_s"],
              "raw_sensor_wrench": None, "baseline": None, "base_frame_residual": None,
              "limitations": limitations}
    if not samples:
        return result
    stamps = np.asarray([s["timestamp_ns"] for s in samples], dtype=np.int64)
    gaps = np.diff(stamps)/1e6
    result["observed_span_s"] = float((stamps[-1]-stamps[0])/1e9)
    result["sample_interval_ms"] = ({"min": float(gaps.min()), "mean": float(gaps.mean()),
        "std": float(gaps.std()), "max": float(gaps.max())} if len(gaps) else None)
    values = np.asarray([s["wrench"] for s in samples])
    result["raw_sensor_wrench"] = six_axis_statistics(values)
    baseline_end = next((i+1 for i in range(len(samples)) if i+1 >= settings["baseline_min_samples"]
                         and stamps[i]-stamps[0] >= settings["baseline_duration_s"]*1e9), None)
    if baseline_end is None:
        return result
    result["status"] = "sampled"
    baseline = values[:baseline_end].mean(axis=0)
    result["baseline"] = {"sample_count": baseline_end, "raw_sensor_mean": baseline.tolist(),
                          "span_s": float((stamps[baseline_end-1]-stamps[0])/1e9)}
    origin = samples[0]["pose"]
    positions = [np.linalg.norm(s["pose"][:3, 3]-origin[:3, 3]) for s in samples]
    angles = [np.arccos(np.clip((np.trace(s["pose"][:3, :3] @ origin[:3, :3].T)-1)/2, -1, 1))
              for s in samples]
    p_limit, r_limit = settings["baseline_position_tolerance_m"], settings["orientation_tolerance_rad"]
    stationary = (bool(max(positions) <= p_limit and max(angles) <= r_limit)
                  if p_limit is not None and r_limit is not None else None)
    result["stationarity"] = {"max_position_excursion_m": float(max(positions)),
        "max_orientation_excursion_rad": float(max(angles)), "within_configured_limits": stationary}
    if stationary is None:
        limitations.append("Stationarity limits are unset; static-baseline validity has not been established.")
    elif not stationary:
        result["status"] = "motion_detected"
        limitations.append("Pose changes exceed the configured static limits; compensated residual is withheld.")
    if calibrated and stationary is not False:
        residual = []
        for sample in samples:
            R_B_S = sample["pose"][:3, :3] @ settings["T_F_S"][:3, :3]
            delta = settings["wrench_sign"] * (sample["wrench"]-baseline)
            residual.append(np.r_[R_B_S @ delta[:3], R_B_S @ delta[3:]])
        result["base_frame_residual"] = {"frame": "tianji_right_base", "moment_origin": "force_sensor_origin",
            "moment_shift_applied": False, "statistics": six_axis_statistics(residual),
            "last_wrench": residual[-1].tolist()}
        limitations.append("Torque is rotated about the force-sensor origin, not shifted to the flange or the two-hole midpoint; no grasp transform is available.")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE/"receive.yaml")
    parser.add_argument("--duration", type=float, default=3., help="Sampling/replay window in seconds (default: 3)")
    parser.add_argument("--output", type=Path, help="Optional JSON report file")
    parser.add_argument("--replay", type=Path, help="Read events.jsonl or insertion_trace JSONL without opening devices")
    args = parser.parse_args(argv)
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be finite and positive")
    try:
        settings = load_settings(args.config)
        records = replay_records(args.replay) if args.replay else live_records(settings, args.duration)
        try:
            report = analyze_records(records, settings, args.duration)
        finally:
            records.close()
        report["source"] = str(args.replay.resolve()) if args.replay else "live_right_arm_feedback"
        if args.output:
            write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if report["status"] == "sampled" else 1
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Force diagnostics failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
