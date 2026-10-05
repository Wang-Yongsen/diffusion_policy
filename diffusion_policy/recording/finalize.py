"""Reuse the validated raw-spool finalizer; keep one implementation."""

from diffusion_policy.third_party.bimanual_teleop.recording.finalize import finalize_recordings

__all__ = ["finalize_recordings"]
