"""Pinned M6 model, motion configuration and local SDK kinematics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import threading
from typing import Mapping

from diffusion_policy.third_party.bimanual_teleop.types import ControlProfile, Pose, Side
from diffusion_policy.third_party.bimanual_teleop.devices.tianji.sdk import DEFAULT_MODEL, SDK_COMMIT, load_sdk, sdk_root as resolve_sdk_root


MODEL_SHA256 = "ce20b1a80974c3ed58cdac4973152625096c8ad93a779e5fd4108c26ce19bd52"
SIDES: tuple[Side, ...] = ("left", "right")


class ProfileError(ValueError):
    """A motion parameter or its verification is missing or inconsistent."""


class KinematicsError(RuntimeError):
    """A local SDK kinematics operation failed or has no usable solution."""

    diagnostic: dict | None = None

    def __str__(self):
        message = super().__str__()
        if self.diagnostic is not None:
            label = "控制诊断" if self.diagnostic.get("schema") == "tianji_cartesian_servo_failure_v1" else "IK诊断"
            message += f"\n[{label}] " + json.dumps(self.diagnostic, ensure_ascii=False, allow_nan=False,
                                               separators=(",", ":"))
        return message


class IKTargetError(KinematicsError):
    """SDK-reported target infeasibility, distinct from an invalid model or input."""


def _number(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ProfileError(f"{name} must be a finite number")
    if minimum is not None and value < minimum:
        raise ProfileError(f"{name} must be >= {minimum}")
    return float(value)


def _vector(value: object, length: int, name: str, *, minimum: float | None = None) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ProfileError(f"{name} must contain {length} numbers")
    return tuple(_number(item, name, minimum=minimum) for item in value)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProfileError(f"{name} must be an object")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProfileError(f"{name} must be specified")
    return value


@dataclass(frozen=True)
class ArmModel:
    controller_type: int
    gravity_m_s2: tuple[float, ...]
    dh_native: tuple[tuple[float, ...], ...]  # alpha deg, a mm, d mm, theta deg; flange last
    limits_native: tuple[tuple[float, ...], ...]  # positive/negative deg, velocity deg/s, acceleration deg/s²
    bd67_native: tuple[tuple[float, ...], ...]  # PP, NP, NN, PN, following the SDK file
    dof: int = 7

    @property
    def lower_rad(self) -> tuple[float, ...]:
        return tuple(math.radians(row[1]) for row in self.limits_native)

    @property
    def upper_rad(self) -> tuple[float, ...]:
        return tuple(math.radians(row[0]) for row in self.limits_native)

    @property
    def max_joint_velocity_rad_s(self) -> tuple[float, ...]:
        return tuple(math.radians(row[2]) for row in self.limits_native)


@dataclass(frozen=True)
class M6Model:
    path: Path
    digest: str
    arms: Mapping[Side, ArmModel]
    sdk_commit: str = SDK_COMMIT

    @classmethod
    def from_file(cls, path: str | Path = DEFAULT_MODEL) -> M6Model:
        path = Path(path).resolve()
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != MODEL_SHA256:
            raise ProfileError("This version requires the pinned M6 4.0 model; a path override only relocates that file")
        try:
            rows = [tuple(float(item) for item in line.split(",") if item.strip())
                    for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
        except (ValueError, UnicodeError) as error:
            raise ProfileError(f"Invalid M6 model file: {path}") from error
        if len(rows) != 26:
            raise ProfileError("M6 model must contain two 13-row arm records")
        arms = {}
        for side, start in zip(SIDES, (0, 13)):
            block = rows[start:start + 13]
            if ([len(row) for row in block] != [4] + [18] * 7 + [4] + [3] * 4
                    or block[0][0] != 1017 or not all(math.isfinite(x) for row in block for x in row)):
                raise ProfileError(f"Invalid M6 4.0 geometry for {side}")
            limits = tuple((row[5], row[4], row[6], row[7]) for row in block[1:8])
            if any(positive <= negative or velocity <= 0 or acceleration <= 0
                   for positive, negative, velocity, acceleration in limits):
                raise ProfileError(f"Invalid joint limits for {side}")
            arms[side] = ArmModel(1017, block[0][1:], tuple(row[:4] for row in block[1:9]),
                                 limits, tuple(block[9:13]))
        return cls(path, digest, arms)

    def arm(self, side: Side) -> ArmModel:
        if side not in SIDES:
            raise ProfileError(f"Unknown arm: {side}")
        return self.arms[side]

@dataclass(frozen=True)
class ArmMotionProfile:
    stiffness: tuple[float, ...]
    damping: tuple[float, ...]
    nullspace_stiffness: float
    nullspace_damping: float
    tool_dyn10: tuple[float, ...]
    velocity_ratio: int
    acceleration_ratio: int


@dataclass(frozen=True)
class MotionProfile:
    profile_id: str
    active_arms: tuple[Side, ...]
    arms: Mapping[Side, ArmMotionProfile]
    model: M6Model

    @classmethod
    def from_control_profile(cls, profile: ControlProfile, model_path: str | Path | None = None) -> MotionProfile:
        if profile.mode != "cartesian_impedance":
            raise ProfileError("M6 motion requires mode='cartesian_impedance'")
        _text(profile.profile_id, "profile_id")
        parameters = _mapping(profile.parameters, "parameters")
        active = parameters.get("active_arms")
        if (not isinstance(active, (list, tuple)) or not active or
                any(side not in SIDES for side in active) or len(set(active)) != len(active)):
            raise ProfileError("active_arms must explicitly name left, right, or both without duplicates")
        model = M6Model.from_file(model_path or DEFAULT_MODEL)
        arm_parameters = _mapping(parameters.get("arms"), "arms")
        arms = {}
        for side in active:
            config = _mapping(arm_parameters.get(side), f"arms.{side}")
            ratios = []
            for key in ("velocity_ratio", "acceleration_ratio"):
                ratio = config.get(key)
                if isinstance(ratio, bool) or not isinstance(ratio, int) or not 1 <= ratio <= 100:
                    raise ProfileError(f"{side}.{key} must be an integer from 1 to 100")
                ratios.append(ratio)
            dynamics = _vector(config.get("tool_dyn10"), 10, f"{side}.tool_dyn10")
            if dynamics[0] < 0:
                raise ProfileError(f"{side} tool load mass must be nonnegative")
            if dynamics[0] == 0 and any(value != 0 for value in dynamics[1:]):
                raise ProfileError(f"{side} unloaded tool dynamics must contain ten zeros")
            damping = _vector(config.get("damping"), 6, f"{side}.damping", minimum=0)
            nullspace_damping = _number(config.get("nullspace_damping"), f"{side}.nullspace_damping", minimum=0)
            if any(value > 1 for value in (*damping, nullspace_damping)):
                raise ProfileError(f"{side} damping and nullspace_damping must be within [0, 1]")
            arms[side] = ArmMotionProfile(
                _vector(config.get("stiffness"), 6, f"{side}.stiffness", minimum=0),
                damping,
                _number(config.get("nullspace_stiffness"), f"{side}.nullspace_stiffness", minimum=0),
                nullspace_damping,
                dynamics, *ratios,
            )
        return cls(profile.profile_id, tuple(active), arms, model)


_KINE_LOCK = threading.RLock()


def _matrix_from_pose(pose: Pose, side: Side) -> tuple[float, ...]:
    if pose.parent_frame != f"tianji_{side}_base" or pose.child_frame != f"tianji_{side}_flange":
        raise KinematicsError("Pose must express this arm's flange in its base frame")
    try:
        position = _vector(pose.position_m, 3, "position_m")
        x, y, z, w = _vector(pose.orientation_xyzw, 4, "orientation_xyzw")
    except ProfileError as error:
        raise KinematicsError(str(error)) from error
    if not math.isclose(x*x + y*y + z*z + w*w, 1, abs_tol=1e-6):
        raise KinematicsError("Pose quaternion must have unit norm")
    return (1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w), position[0]*1000,
            2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w), position[1]*1000,
            2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y), position[2]*1000,
            0, 0, 0, 1)


def _pose_from_matrix(matrix, side: Side) -> Pose:
    m = matrix
    if not all(math.isfinite(value) for value in m):
        raise KinematicsError("FK returned a nonfinite transform")
    trace = m[0] + m[5] + m[10]
    if trace > 0:
        scale = math.sqrt(trace + 1) * 2
        q = ((m[9]-m[6])/scale, (m[2]-m[8])/scale, (m[4]-m[1])/scale, scale/4)
    else:
        i = max(range(3), key=lambda index: m[index*5])
        j, k = (i+1) % 3, (i+2) % 3
        scale = math.sqrt(1 + m[i*5] - m[j*5] - m[k*5]) * 2
        q = [0.0]*4
        q[i], q[j], q[k] = scale/4, (m[j*4+i]+m[i*4+j])/scale, (m[k*4+i]+m[i*4+k])/scale
        q[3] = (m[k*4+j]-m[j*4+k])/scale
    norm = math.sqrt(sum(value*value for value in q))
    return Pose(f"tianji_{side}_base", f"tianji_{side}_flange",
                (m[3]/1000, m[7]/1000, m[11]/1000), tuple(value/norm for value in q))


class TianjiKinematics:
    """Official offline kinematics, with SI units at the project boundary."""

    def __init__(self, sdk_root: str | Path | None = None, model_path: str | Path | None = None):
        self.model = M6Model.from_file(model_path or DEFAULT_MODEL)
        self.sdk_root = resolve_sdk_root(sdk_root)
        self.library_path = self.sdk_root / "SDK_PYTHON/libKine.so"
        self._module = load_sdk("kine", self.sdk_root)
        self._kines = {}

    @staticmethod
    def _check(operation, result):
        if result is False or result is None:
            raise KinematicsError(f"Official SDK {operation} failed")

    def _initialize(self, side):
        if side not in SIDES:
            raise KinematicsError(f"Unknown arm: {side}")
        if side not in self._kines:
            kine = self._module.Marvin_Kine()
            kine.log_switch(0)
            config = kine.load_config(SIDES.index(side), str(self.model.path))
            self._check("load model", config)
            arm = self.model.arm(side)
            self._check("initialize", kine.initial_kine(arm.controller_type, arm.dh_native,
                                                        arm.limits_native, arm.bd67_native))
            self._check("remove tool frame", kine.remove_tool_kine())
            self._check("remove user frame", kine.remove_user_frame_kine())
            self._kines[side] = kine
        return self._kines[side]

    @staticmethod
    def _joints(joints_rad) -> tuple[float, ...]:
        try:
            return _vector(joints_rad, 7, "joints_rad")
        except ProfileError as error:
            raise KinematicsError(str(error)) from error

    def fk(self, side: Side, joints_rad: tuple[float, ...]) -> Pose:
        joints = [math.degrees(q) for q in self._joints(joints_rad)]
        with _KINE_LOCK:
            matrix = self._initialize(side).fk(joints)
            self._check("FK", matrix)
        return _pose_from_matrix(tuple(value for row in matrix for value in row), side)

    def jacobian(self, side: Side, joints_rad: tuple[float, ...]) -> tuple[tuple[float, ...], ...]:
        """Base-frame geometric Jacobian: linear m/rad, angular rad/rad.

        The official library already returns SI Jacobian units; unlike FK
        translation, its first three rows must not be divided by 1000.
        """
        joints = [math.degrees(q) for q in self._joints(joints_rad)]
        with _KINE_LOCK:
            matrix = self._initialize(side).joints2JacobMatrix(joints)
            self._check("Jacobian", matrix)
        if len(matrix) != 6 or any(len(row) != 7 or any(not math.isfinite(x) for x in row) for row in matrix):
            raise KinematicsError("Jacobian returned invalid or nonfinite values")
        return tuple(tuple(row) for row in matrix)

    def solve(self, side, pose, reference_rad, *, direction=None, angle=None):
        """Return the official IK flags even on failure; also used by offline replay."""
        matrix = _matrix_from_pose(pose, side)
        reference = [math.degrees(q) for q in self._joints(reference_rad)]
        solve = self._module.FX_InvKineSolvePara()
        solve.set_input_ik_target_tcp(matrix)
        solve.set_input_ik_ref_joint(reference)
        if direction is not None:
            solve.set_input_ik_zsp_type(1)
            solve.set_input_ik_zsp_para([*direction, 0., 0., 0.])
            solve.m_DGR1 = solve.m_DGR2 = solve.m_DGR3 = 5
        with _KINE_LOCK:
            kine = self._initialize(side)
            success = kine.ik(solve) is not False
            if angle is not None and success and not solve.m_Output_IsOutRange:
                solve.set_input_zsp_angle(angle)
                solve.m_Output_IsJntExd, solve.m_Output_JntExdABS = False, 0.
                success = kine.ik_nsp(solve) is not False
        return success, solve

    def ik(self, side: Side, pose: Pose, reference_rad: tuple[float, ...]) -> tuple[float, ...]:
        success, result = self.solve(side, pose, reference_rad)
        q = tuple(result.m_Output_RetJoint.to_list())
        count, outside = result.m_OutPut_Result_Num, result.m_Output_IsOutRange
        singular = sum(bool(flag) << i for i, flag in enumerate(result.m_Output_IsDeg))
        limits = sum(bool(flag) << i for i, flag in enumerate(result.m_Output_JntExdTags))
        try:
            if not success and not (outside or singular or limits):
                raise KinematicsError("Official SDK IK failed")
            if any(not math.isfinite(value) for value in q):
                raise KinematicsError("IK returned nonfinite joints")
            if outside or count <= 0:
                raise IKTargetError(f"{side}: IK target is unreachable")
            if limits:
                exceeded = ", ".join(f"J{i+1}={value:.2f}deg" for i, value in enumerate(q) if limits & (1 << i))
                raise IKTargetError(f"{side}: IK violates joint/coupled limits ({exceeded})")
            if singular:
                raise IKTargetError(f"{side}: IK is singular (mask={singular})")
            joints = tuple(math.radians(value) for value in q)
            arm = self.model.arm(side)
            if any(value < low-1e-9 or value > high+1e-9
                   for value, low, high in zip(joints, arm.lower_rad, arm.upper_rad)):
                raise KinematicsError("IK returned joints outside model limits")
            return joints
        except KinematicsError as error:
            error.diagnostic = {
                "schema": "tianji_ik_failure_v1", "sdk_commit": self.model.sdk_commit,
                "model_sha256": self.model.digest, "side": side, "target": asdict(pose),
                "reference_deg": [math.degrees(q) for q in reference_rad],
                "result_deg": [value if math.isfinite(value) else None for value in q],
                "status": 0 if success else -1, "solution_count": count,
                "out_of_range": int(outside), "singular_mask": singular, "limit_mask": limits,
            }
            raise
