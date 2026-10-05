"""Constrained Cartesian velocity servo for the pinned M6 model.

The optimization chooses task progress and redundant joint motion together.
An unreachable pose slows/stops progress; it is not an IK exception. Position,
velocity and acceleration bounds remain hard, including the coupled wrist.
No transport is used here. A proposal advances state only after accept().
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math

from diffusion_policy.third_party.bimanual_teleop.devices.tianji.model import KinematicsError
from diffusion_policy.third_party.bimanual_teleop.types import Pose


@dataclass(frozen=True)
class ServoStep:
    joints: tuple[float, ...]
    pose: Pose
    velocity: tuple[float, ...]
    position_error_m: float
    orientation_error_rad: float
    limited: bool
    progress: float
    _owner: object = field(repr=False, compare=False)
    _generation: int = field(repr=False, compare=False)


class CartesianServo:
    """Direction-preserving task scaling with a feasible braking direction.

    For C q <= b, k = min(amax/vmax), enforce
        (1 + k dt) C v <= k (b - C q).
    After q' = q + dt v this implies C v <= k (b - C q').
    Thus v_brake = v_prev/(1+k dt) is feasible next cycle, including the
    acceleration bounds. This holds for varying dt and all wrist faces together.

    The Cartesian equality permits the requested twist and that braking twist.
    It prevents a blocked translation being traded for sideways/wrist motion,
    while permitting deceleration when a moving operator changes direction.
    The two direction columns are normalized to avoid ill-conditioning as the
    requested velocity tends to zero. Joint constraints are never relaxed.
    """

    def __init__(self, kinematics, side, model, *, velocity_ratio=100, acceleration_ratio=100):
        import numpy as np
        # Load numerical libraries during preflight, before the driver's
        # first-command watchdog starts. Module import itself remains passive.
        import quadprog  # noqa: F401
        from scipy.spatial.transform import Rotation  # noqa: F401

        self.kinematics, self.side = kinematics, side
        self._model = model
        self._ratios = (velocity_ratio, acceleration_ratio)
        arm = model.arm(side)
        if not (0 < velocity_ratio <= 100 and 0 < acceleration_ratio <= 100):
            raise ValueError("servo velocity/acceleration ratios must be in (0, 100]")
        self._lo, self._hi = np.array(arm.lower_rad), np.array(arm.upper_rad)
        self._vmax = np.array(arm.max_joint_velocity_rad_s) * velocity_ratio / 100
        self._amax = np.radians([row[3] for row in arm.limits_native]) * acceleration_ratio / 100
        # The pinned model's wrist envelope is a diamond, in addition to J6/J7
        # individual bounds. Refuse an incompatible model rather than silently
        # approximating its nonlinear envelope.
        expected = ((0., -1.025, 110.5), (0., 1.025, 110.5),
                    (0., -1.025, -110.5), (0., 1.025, -110.5))
        if not np.allclose(arm.bd67_native, expected, rtol=0, atol=1e-10):
            raise ValueError("Cartesian servo requires the pinned linear M6 wrist envelope")
        wrist = [[0., 0., 0., 0., 0., a, b] for a in (-1.025, 1.025) for b in (-1., 1.)]
        self._C = np.vstack((np.eye(7), -np.eye(7), wrist))
        self._b = np.r_[self._hi, -self._lo, np.full(4, math.radians(110.5))]
        self._physical_b = self._b.copy()
        # The controller receives each angle as float32 degrees. Independent
        # rounding can increase 1.025*abs(J6)+abs(J7) by at most 5.77e-6 deg.
        # Reserve 1e-5 deg inside the coupled boundary before optimization.
        self._b[-4:] -= math.radians(1e-5)
        self._k = float(np.min(self._amax / self._vmax))
        self._weight = np.array([4., 4., 4., 1., 1., 1.])
        self._q = self._v = self._anchor = None
        self._generation = 0

    @property
    def q(self):
        return tuple(self._q) if self._q is not None else ()

    @property
    def velocity(self):
        return tuple(self._v) if self._v is not None else ()

    def reset(self, q_rad):
        import numpy as np

        q = np.asarray(q_rad, dtype=float)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise ValueError(f"{self.side}: servo requires seven finite actual joints")
        if np.max((self._C @ q - self._b)[:14]) > 1e-9:
            raise ValueError(f"{self.side}: actual joints are outside the M6 joint limits")
        if np.max((self._C @ q - self._b)[14:]) > 1e-9:
            margin = 110.5 - math.degrees(1.025 * abs(q[5]) + abs(q[6]))
            raise ValueError(f"{self.side}: actual coupled wrist margin is {margin:.8g} deg; "
                             "float32-safe engagement requires at least 0.00001 deg")
        self._q, self._anchor, self._v = q.copy(), q.copy(), np.zeros(7)
        self._generation += 1

    @staticmethod
    def _error(goal, current):
        import numpy as np
        from scipy.spatial.transform import Rotation

        if (goal.parent_frame, goal.child_frame) != (current.parent_frame, current.child_frame):
            raise ValueError("Cartesian goal and flange must use the same coordinate frames")
        p = np.asarray(goal.position_m, dtype=float)
        quat = np.asarray(goal.orientation_xyzw, dtype=float)
        if (p.shape != (3,) or quat.shape != (4,) or not np.all(np.isfinite(p))
                or not np.all(np.isfinite(quat)) or abs(np.linalg.norm(quat) - 1) > 1e-3):
            raise ValueError("Cartesian goal must contain a finite position and unit quaternion")
        return np.r_[p - current.position_m,
                     (Rotation.from_quat(quat) * Rotation.from_quat(current.orientation_xyzw).inv()).as_rotvec()]

    def propose(self, pose, dt_s, actual_rad=None):
        import numpy as np

        try:
            return self._propose(pose, dt_s, actual_rad)
        except (KinematicsError, np.linalg.LinAlgError) as fault:
            error = fault if isinstance(fault, KinematicsError) else KinematicsError(
                f"{self.side}: Cartesian servo linear algebra failure ({fault})")
            if error.diagnostic is None:
                error.diagnostic = {
                    "schema": "tianji_cartesian_servo_failure_v1", "side": self.side,
                    "target": asdict(pose), "q_rad": list(self.q),
                    "velocity_rad_s": list(self.velocity),
                    "anchor_rad": self._anchor.tolist() if self._anchor is not None else None,
                    "dt_s": dt_s, "velocity_ratio": self._ratios[0], "acceleration_ratio": self._ratios[1],
                    "model_sha256": self._model.digest, "sdk_commit": self._model.sdk_commit,
                    "message": str(error),
                }
            if error is fault:
                raise
            raise error from fault

    def _propose(self, pose, dt_s, actual_rad):
        import numpy as np
        import quadprog

        if self._q is None:
            raise RuntimeError("Reset the Cartesian servo from actual joints before use")
        if not math.isfinite(dt_s) or dt_s < 0:
            raise ValueError("Cartesian servo dt must be finite and nonnegative")
        if actual_rad is not None:
            actual = np.asarray(actual_rad, dtype=float)
            if actual.shape != (7,) or not np.all(np.isfinite(actual)):
                raise ValueError(f"{self.side}: invalid actual joints during Cartesian servo")
        current = self.kinematics.fk(self.side, self.q)
        error = self._error(pose, current)
        if dt_s == 0:
            return ServoStep(self.q, current, self.velocity, float(np.linalg.norm(error[:3])),
                             float(np.linalg.norm(error[3:])), False, 0., self, self._generation)
        J = np.asarray(self.kinematics.jacobian(self.side, self.q), dtype=float)
        if J.shape != (6, 7) or not np.all(np.isfinite(J)):
            raise KinematicsError(f"{self.side}: invalid Cartesian Jacobian")
        H = self._weight[:, None] * J
        # Approach the target with a finite feedback gain. Asking for the
        # entire error/dt while limiting acceleration causes overshoot when
        # the operator stops: it requests braking only after crossing the goal.
        weighted_error = self._weight * error
        task = weighted_error * 16.
        brake = self._v / (1 + self._k * dt_s)
        braking_task = H @ brake
        task_norm, brake_norm = np.linalg.norm(task), np.linalg.norm(braking_task)
        task_dir = task / task_norm if task_norm > 1e-12 else np.zeros(6)
        brake_dir = braking_task / brake_norm if brake_norm > 1e-12 else np.zeros(6)

        # Prefer small joint motion around the measured engagement posture;
        # discourage squeezing against a joint limit. This preference acts only
        # in the Jacobian nullspace and fades quadratically for tracking noise.
        reserve = math.radians(15.)
        inward = (np.maximum(0., 1 - (self._q - self._lo) / reserve) ** 2
                  - np.maximum(0., 1 - (self._hi - self._q) / reserve) ** 2)
        preference = .4 * (self._anchor - self._q) + .3 * inward
        null = np.eye(7) - np.linalg.pinv(H, rcond=1e-6) @ H
        preference = null @ preference * min(1., (np.linalg.norm(self._weight * error) / .01) ** 2)
        regularization = 1e-5
        P = np.eye(9) * regularization
        P[:7, :7] += H.T @ H
        linear = np.r_[-H.T @ task - regularization * preference, 0., 0.]
        position_A = np.c_[(1 + self._k * dt_s) * self._C, np.zeros((18, 2))]
        A = np.vstack((position_A, np.c_[np.eye(7), np.zeros((7, 2))],
                       np.c_[H, -task_dir, -brake_dir], np.c_[np.zeros((2, 7)), np.eye(2)]))
        lower = np.r_[np.full(18, -np.inf), np.maximum(-self._vmax, self._v - self._amax * dt_s),
                      np.zeros(6), 0., 0.]
        upper = np.r_[self._k * (self._b - self._C @ self._q),
                      np.minimum(self._vmax, self._v + self._amax * dt_s), np.zeros(6), task_norm, brake_norm]
        # Eliminate the homogeneous Cartesian equalities exactly, then whiten
        # the quadratic objective and normalize constraint rows. Near parallel
        # task/braking directions otherwise make multipliers ill-conditioned;
        # changing solvers or reusing their dual state alone does not fix it.
        eq = slice(25, 31)
        inequalities = np.r_[np.arange(25), 31, 32]
        _, singular, vh = np.linalg.svd(A[eq], full_matrices=True)
        rank = int(np.count_nonzero(singular > max(1e-12, singular[0] * 1e-12)))
        basis = vh[rank:].T
        cholesky = np.linalg.cholesky(basis.T @ P @ basis)
        transform = basis @ np.linalg.solve(cholesky.T, np.eye(9 - rank))
        projected = A[inequalities] @ transform
        row_scale = np.maximum(np.linalg.norm(projected, axis=1), 1e-12)
        constraints = projected / row_scale[:, None]
        low, high = lower[inequalities] / row_scale, upper[inequalities] / row_scale
        finite = np.isfinite(low)
        # Goldfarb-Idnani solves the strictly convex reduced QP by an active
        # set. It has no approximate-iteration infeasibility test; those tests
        # falsely rejected feasible near-boundary cases with proximal/ADMM
        # solvers. All outputs are still checked in the original coordinates.
        try:
            solution = quadprog.solve_qp(np.eye(9 - rank), -transform.T @ linear,
                np.vstack((-constraints, constraints[finite])).T, np.r_[-high, low[finite]])[0]
        except ValueError as error:
            raise KinematicsError(f"{self.side}: Cartesian servo numerical failure ({error})") from error
        x = transform @ solution
        if not np.all(np.isfinite(x)):
            raise KinematicsError(f"{self.side}: nonfinite Cartesian servo solution")
        residual = max(float(np.max(A @ x - upper)), float(np.max(lower - A @ x)))
        if residual > 2e-7:
            raise KinematicsError(f"{self.side}: Cartesian servo constraint residual {residual:.3g}")
        v = x[:7]
        q = self._q + dt_s * v
        # Check the actual protocol representation as well as the double QP.
        # This is validation, not independent joint clipping or a second IK.
        encoded = np.radians(np.degrees(q).astype(np.float32).astype(float))
        if np.max(self._C @ encoded - self._physical_b) > 1e-12:
            raise KinematicsError(f"{self.side}: float32 Cartesian command exceeds M6 joint/wrist limits")
        applied = self.kinematics.fk(self.side, tuple(q))
        remaining = self._error(pose, applied)
        progress = float(np.clip(x[7] / task_norm, 0., 1.)) if task_norm > 1e-8 else 1.
        limited = bool(task_norm > 1e-5 and np.linalg.norm(H @ v - task) > max(1e-5, .05 * task_norm))
        return ServoStep(tuple(q), applied, tuple(v), float(np.linalg.norm(remaining[:3])),
                         float(np.linalg.norm(remaining[3:])), limited, progress, self, self._generation)

    def accept(self, step):
        import numpy as np

        if step._owner is not self or step._generation != self._generation:
            raise ValueError("Cartesian servo proposal is stale or belongs to another arm")
        self._q, self._v = np.asarray(step.joints), np.asarray(step.velocity)
        self._generation += 1
