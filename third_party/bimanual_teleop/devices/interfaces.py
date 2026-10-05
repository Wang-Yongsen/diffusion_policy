"""Vendor SDKs belong behind these contracts, never in shared types."""

from typing import Protocol

from diffusion_policy.third_party.bimanual_teleop.types import (
    CommandEvent,
    Event,
    Sample,
)

class RealtimeObserver(Protocol):
    """Nonblocking in-memory callback for live samples and status events.

    Returning False or raising means the observer cannot safely process live
    state. Sources latch that failure, and an engaged driver requests hold.
    """

    def try_publish(self, sample: Sample[object]) -> bool:
        ...

    def try_event(self, event: Event | CommandEvent) -> bool:
        ...
