from __future__ import annotations

import numpy as np

from .core import difference, matrix, pose, vector


class JointPath:
    """Quintic segments with zero speed/acceleration at each waypoint."""
    def __init__(self, waypoints, speed, acceleration):
        self.points = np.asarray(waypoints, dtype=float)
        if self.points.ndim != 2 or self.points.shape[1] != 7 or not np.isfinite(self.points).all():
            raise ValueError("Expected finite Nx7 joint waypoints")
        if len(self.points) < 2 or not np.isfinite([speed, acceleration]).all() or speed <= 0 or acceleration <= 0:
            raise ValueError("Need at least two waypoints and positive dynamic limits")
        delta = np.abs(np.diff(self.points, axis=0)).max(axis=1)
        self.durations = np.maximum.reduce([np.full(len(delta), .25),
                                            1.875 * delta / speed,
                                            np.sqrt(5.773503 * delta / acceleration)])
        self.times = np.r_[0., np.cumsum(self.durations)]
        self.duration = float(self.times[-1])

    def at(self, elapsed):
        if elapsed <= 0:
            return self.points[0].copy()
        if elapsed >= self.duration:
            return self.points[-1].copy()
        i = min(len(self.durations) - 1, int(np.searchsorted(self.times, elapsed, side="right")) - 1)
        u = (elapsed - self.times[i]) / self.durations[i]
        fraction = 10*u**3 - 15*u**4 + 6*u**5
        return self.points[i] + fraction * (self.points[i + 1] - self.points[i])


class Planner:
    """Solve a joint target and interpolate directly, checking joint limits only."""
    def __init__(self, config, kinematics):
        self.config, self.kinematics = config, kinematics
        arm = kinematics.model.arm("left")
        self.lower = np.asarray(arm.lower_rad) + .001
        self.upper = np.asarray(arm.upper_rad) - .001

    def valid(self, q):
        q = np.asarray(q, dtype=float)
        return (q.shape == (7,) and np.isfinite(q).all()
                and np.all(q >= self.lower) and np.all(q <= self.upper)
                and 1.025 * abs(q[5]) + abs(q[6]) < np.radians(110.5) - .001)

    def validate_path(self, points):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 7 or len(points) < 2:
            raise ValueError("Expected at least two Nx7 joint waypoints")
        if not all(self.valid(q) for q in points):
            raise ValueError("Joint path exceeds joint or coupled wrist limits")
        # Joint boxes and the coupled absolute-value wrist bound are convex.
        # Quintic interpolation stays on each segment between its endpoints.

    def plan(self, initial, target_flange):
        initial = vector(initial, 7)
        if not self.valid(initial):
            raise ValueError("Initial joint state exceeds joint or coupled wrist limits")
        for seed in [initial, (initial + (self.lower + self.upper)/2)/2,
                     np.radians([30, -60, -34, -52, 30, 12, 4])]:
            try:
                candidate = np.asarray(self.kinematics.ik("left", pose(target_flange), tuple(seed)))
                if not self.valid(candidate):
                    continue
                error = difference(matrix(self.kinematics.fk("left", tuple(candidate))), target_flange)
                if error[0] < .0001 and error[1] < .0002:
                    points = [initial, candidate]
                    self.validate_path(points)
                    return JointPath(points, self.config["joint_speed_rad_s"],
                                     self.config["joint_acceleration_rad_s2"])
            except RuntimeError:
                continue
        raise ValueError("No preinsert IK solution within joint and coupled wrist limits")
