"""Shared Tianji device, motion, ready-pose and Quest settings."""

from __future__ import annotations

from dataclasses import replace
import ipaddress
from pathlib import Path

from diffusion_policy.third_party.bimanual_teleop.paths import PROJECT_ROOT
from diffusion_policy.third_party.bimanual_teleop.common.config import load_yaml_config


DEFAULT_CONFIG = PROJECT_ROOT / "config/tianji_teleop.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG, override: str | None = None) -> dict:
    source = load_yaml_config(path)
    address = source.get("controller_ip") if override is None else override
    if not isinstance(address, str):
        raise ValueError("Tianji controller_ip must be an IPv4 address")
    try:
        source["controller_ip"] = str(ipaddress.IPv4Address(address))
    except ipaddress.AddressValueError as error:
        raise ValueError(f"invalid Tianji controller IPv4 address: {address!r}") from error
    quest = source.setdefault("quest", {})
    if quest.setdefault("coordinate_frame", "headset") not in ("headset", "world"):
        raise ValueError("quest.coordinate_frame must be headset or world")
    controls = source.setdefault("controls", {})
    if not isinstance(controls, dict):
        raise ValueError("Tianji controls must be an object")
    unknown = controls.keys() - {"toggle_engagement_key", "ready_pose_key", "gesture_engagement_enabled"}
    if unknown:
        raise ValueError(f"Unknown Tianji controls: {', '.join(map(str, unknown))}")
    for name, default in (("toggle_engagement_key", "enter"), ("ready_pose_key", "h")):
        key = controls.setdefault(name, default)
        if name == "toggle_engagement_key" and isinstance(key, str) and key.lower() == "enter":
            controls[name] = "enter"
            continue
        if (not isinstance(key, str) or len(key) != 1
                or key.lower() not in "abcdefghijklmnopqrstuvwxyz0123456789"
                or key.lower() in "qcsx"):
            choices = "enter or one letter or digit" if name == "toggle_engagement_key" else "one letter or digit"
            raise ValueError(f"controls.{name} must be {choices}, excluding Q/C/S/X")
        controls[name] = key.lower()
    if controls["toggle_engagement_key"] == controls["ready_pose_key"]:
        raise ValueError("controls toggle_engagement_key and ready_pose_key must differ")
    if not isinstance(controls.setdefault("gesture_engagement_enabled", True), bool):
        raise ValueError("controls.gesture_engagement_enabled must be true or false")
    if "profile" in source:
        profile = source["profile"]
        if not isinstance(profile, dict):
            raise ValueError("Tianji profile must be an object")
        profile.setdefault("profile_id", "tianji-teleop")
        profile.setdefault("mode", "cartesian_impedance")
    return source


def select_profile_side(profile, side):
    """Make the selected arm the only native motion target."""
    if side == "both":
        return profile
    if side not in ("left", "right"):
        raise ValueError("side must be left, right, or both")
    return replace(profile, profile_id=f"{profile.profile_id}-{side}",
                   parameters={**profile.parameters, "active_arms": [side],
                               "arms": {side: profile.parameters["arms"][side]}})
