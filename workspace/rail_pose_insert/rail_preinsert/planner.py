from __future__ import annotations

import time
import numpy as np

from .core import difference, matrix, pose, vector


class JointPath:
    """Quintic segments with zero speed/acceleration at each checked waypoint."""
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
    def __init__(self, config, kinematics, world):
        self.config, self.kinematics, self.world = config, kinematics, world
        arm = kinematics.model.arm("left")
        self.lower = np.asarray(arm.lower_rad) + .001
        self.upper = np.asarray(arm.upper_rad) - .001
        self.rng = np.random.default_rng(config["rrt_seed"])
        self.resolution = config["edge_resolution_rad"]
        self.deadline = float("inf")

    def valid(self, q):
        return (np.all(q >= self.lower) and np.all(q <= self.upper)
                and 1.025 * abs(q[5]) + abs(q[6]) < np.radians(110.5) - .001
                and self.world.check(q))

    def edge(self, start, end):
        count = max(1, int(np.ceil(np.max(np.abs(end - start)) / self.resolution)))
        for fraction in np.linspace(0, 1, count + 1):
            if time.monotonic() > self.deadline:
                raise TimeoutError("Collision planning exceeded its time budget")
            if not self.valid(start + fraction * (end - start)):
                return False
        return True

    def _extend(self, tree, parents, target):
        nearest = int(np.argmin(np.linalg.norm(np.asarray(tree) - target, axis=1)))
        delta = target - tree[nearest]
        distance = np.linalg.norm(delta)
        q = target if distance <= self.config["rrt_step_rad"] else tree[nearest] + delta / distance * self.config["rrt_step_rad"]
        if not self.edge(tree[nearest], q):
            return None, False
        tree.append(q.copy()); parents.append(nearest)
        return len(tree) - 1, distance <= self.config["rrt_step_rad"]

    @staticmethod
    def _trace(tree, parents, index):
        path = []
        while index >= 0:
            path.append(tree[index]); index = parents[index]
        return list(reversed(path))

    def plan(self, initial, target_flange):
        initial = vector(initial, 7)
        self.deadline = time.monotonic() + self.config["planning_timeout_s"]
        if not self.valid(initial):
            raise ValueError(f"Initial grasp/model is in collision or outside limits: {self.world.last_failure}")
        # Try multiple redundant IK seeds while retaining an exact checked joint goal.
        goal = None
        for seed in [initial, (initial + (self.lower + self.upper)/2)/2,
                     np.radians([30, -60, -34, -52, 30, 12, 4])]:
            try:
                candidate = np.asarray(self.kinematics.ik("left", pose(target_flange), tuple(seed)))
                error = difference(matrix(self.kinematics.fk("left", tuple(candidate))), target_flange)
                if error[0] < .0001 and error[1] < .0002 and self.valid(candidate):
                    goal = candidate; break
            except RuntimeError:
                continue
        if goal is None:
            raise ValueError(f"No reachable collision-free preinsert IK: {self.world.last_failure}")
        if self.edge(initial, goal):
            path = [initial, goal]
        else:
            a, b, pa, pb = [initial.copy()], [goal.copy()], [-1], [-1]
            swapped, path = False, None
            while time.monotonic() < self.deadline:
                sample = b[0] if self.rng.random() < .15 else self.rng.uniform(self.lower, self.upper)
                ia, _ = self._extend(a, pa, sample)
                if ia is not None:
                    while True:
                        ib, reached = self._extend(b, pb, a[ia])
                        if ib is None:
                            break
                        if reached:
                            path = self._trace(a, pa, ia) + list(reversed(self._trace(b, pb, ib)))[1:]
                            if swapped:
                                path.reverse()
                            break
                    if path is not None:
                        break
                a, b, pa, pb, swapped = b, a, pb, pa, not swapped
            if path is None:
                raise TimeoutError("No collision-free path found within planning budget")
            for _ in range(20):
                if len(path) < 3 or time.monotonic() > self.deadline:
                    break
                i, j = sorted(self.rng.choice(len(path), 2, replace=False))
                if j > i + 1 and self.edge(path[i], path[j]):
                    path = path[:i + 1] + path[j:]
        # Every edge, including shortcuts, has been checked at resolution_rad.
        trajectory = JointPath(path, self.config["joint_speed_rad_s"], self.config["joint_acceleration_rad_s2"])
        return trajectory
