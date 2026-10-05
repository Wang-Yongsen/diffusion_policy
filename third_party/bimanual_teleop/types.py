"""Shared in-memory contracts, not a device protocol or storage schema.

Dataclasses describe data; they do not validate motion or implement safety checks.
Arrays use named joint order and SI units. Frozen records are shallow: producers
must transfer ownership of payloads or snapshot them before asynchronous use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, Literal, Mapping, TypeVar

Side = Literal["left", "right"]
Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]
JointValues = tuple[float | None, ...]
PayloadT = TypeVar("PayloadT", covariant=True)


@dataclass(frozen=True)
class SourceTime:
    """Unconverted source timestamp; meaning may be capture, query, or send.

    Clock domain and epoch identify the timeline. Receive time must not be
    substituted for a missing source time, nor XR query time called capture time.
    """

    value: int | float
    unit: Literal["s", "ms", "us", "ns"]
    clock_domain: str
    meaning: str


@dataclass(frozen=True)
class SampleRef:
    """Session-unique stream/epoch plus application sequence, not device seq.

    An epoch changes when a source clock or reference frame is reset. Application
    sequences remain unique even when the device sequence wraps or repeats.
    """

    stream: str
    epoch: str
    sequence: int


@dataclass(frozen=True)
class SampleHeader:
    ref: SampleRef
    received_monotonic_ns: int
    valid: bool
    source_time: SourceTime | None = None
    source_sequence: int | None = None


@dataclass(frozen=True)
class Sample(Generic[PayloadT]):
    header: SampleHeader
    payload: PayloadT


@dataclass(frozen=True)
class Pose:
    """Pose of child_frame in parent_frame: meters and quaternion x, y, z, w."""

    parent_frame: str
    child_frame: str
    position_m: Vector3
    orientation_xyzw: Quaternion


@dataclass(frozen=True)
class TrackedPose:
    """Tracking flags preserve per-component validity; no identity fill-in."""

    pose: Pose | None
    position_valid: bool
    orientation_valid: bool
    position_tracked: bool
    orientation_tracked: bool


@dataclass(frozen=True)
class JointState:
    """Actual feedback in joint_names order; None means unavailable, never zero.

    A whole optional channel may be absent, or individual joints may be missing.
    Device-reported estimates are distinct from measured torque and motor current.
    """

    joint_names: tuple[str, ...]
    position_rad: JointValues
    velocity_rad_s: JointValues | None = None
    motor_current_a: JointValues | None = None
    measured_torque_nm: JointValues | None = None
    estimated_external_torque_nm: JointValues | None = None


@dataclass(frozen=True)
class HandSkeleton:
    """Runtime hand shape with references to the source frames used by the SDK."""

    frame: str
    joint_names: tuple[str, ...]
    positions_m: tuple[Vector3, ...]
    confidences: tuple[float, ...]
    source_refs: tuple[SampleRef, ...]


@dataclass(frozen=True)
class OperatorInput:
    """Runtime aggregate; each component retains its original sample header."""

    wrists: Mapping[Side, Sample[TrackedPose]]


@dataclass(frozen=True)
class RobotState:
    """Runtime feedback aggregate."""

    tool_poses: Mapping[Side, Sample[Pose]]


@dataclass(frozen=True)
class JointTarget:
    joint_names: tuple[str, ...]
    position_rad: tuple[float, ...]


@dataclass(frozen=True)
class ControlProfile:
    """Versioned effective configuration; parameter schemas belong to adapters.

    Adapters validate units, mode-specific parameters and device capabilities.
    An intended profile is not proof it is active on hardware.
    """

    profile_id: str
    mode: str
    parameters: Mapping[str, object]


@dataclass(frozen=True)
class RobotTarget:
    """Mapped arm goals; the executor applies kinematics and motion constraints.

    This boundary does not fix a vendor control mode or command payload. All
    monotonic times refer to the acquisition host's session clock.
    """

    command_id: str
    tool_poses: Mapping[Side, Pose]
    source_refs: tuple[SampleRef, ...]
    created_monotonic_ns: int
    expires_monotonic_ns: int
    control_profile_id: str


@dataclass(frozen=True)
class DeviceCommand(Generic[PayloadT]):
    """Complete final payload at a device boundary, including dynamic terms.

    Events reference device/command_id without copying the payload. Submitting
    a target does not establish that this payload was sent.
    """

    device_id: str
    command_id: str
    payload: PayloadT
    source_refs: tuple[SampleRef, ...]
    created_monotonic_ns: int
    expires_monotonic_ns: int
    control_profile_id: str


@dataclass(frozen=True)
class Submission:
    """Local acceptance only; not a transport or physical execution receipt."""

    command_id: str
    accepted: bool
    reason: str | None = None


class CommandStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SDK_SUBMITTED = "sdk_submitted"
    SENT = "sent"
    SEND_FAILED = "send_failed"


@dataclass(frozen=True)
class CommandEvent:
    """Submission and transport status; neither asserts physical execution."""

    device_id: str
    command_id: str
    status: CommandStatus
    observed_monotonic_ns: int
    detail: str | None = None


@dataclass(frozen=True)
class Event:
    kind: str
    observed_monotonic_ns: int
    source: str
    details: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Health:
    """Observed component status; ready does not mean motion is enabled."""

    ready: bool
    observed_monotonic_ns: int
    detail: str | None = None
