"""Preplan bounded, position-only SE(3) corrections above the taught rail goal.

The optional lift protects the nominal rail geometry during the attitude change.
It is a geometric clearance bound, not robot/environment collision checking.
"""
from itertools import product
from numbers import Real
import time

import numpy as np
from scipy.spatial.transform import Rotation

from rail_preinsert.core import difference, matrix, transform, vector
from rail_preinsert.planner import Planner


DEFAULT_PREPLACE_ALIGNMENT = dict(
    enabled=True, samples=3, max_corrections=3,
    visual_position_tolerance_m=.002, visual_rotation_tolerance_rad=np.radians(1.),
    consistency_position_m=.003, consistency_rotation_rad=np.radians(1.),
    max_correction_m=.04, max_correction_rad=np.radians(10.),
    max_step_m=.002, max_rotation_step_rad=np.radians(.5), max_joint_step_rad=.05,
    velocity_ratio=1, acceleration_ratio=1,
    position_tolerance_m=.0005, rotation_tolerance_rad=.00349,
    path_tolerance_m=.003, path_rotation_tolerance_rad=np.radians(1.),
    settle_s=.1, step_timeout_s=20., planning_timeout_s=60.,
    perception_timeout_s=180., total_timeout_s=900., max_lift_m=.06)
MAX_STEPS = 4096


def validate_settings(settings):
    if type(settings.get("enabled")) is not bool:
        raise ValueError("preplace_alignment.enabled must be boolean")
    for name, limit in (("samples", 10), ("max_corrections", 10)):
        value = settings.get(name)
        minimum = 2 if name == "samples" else 1
        if type(value) is not int or not minimum <= value <= limit:
            raise ValueError(f"preplace_alignment.{name} must be an integer from {minimum} to {limit}")
    for name in DEFAULT_PREPLACE_ALIGNMENT:
        if name in ("enabled", "samples", "max_corrections"):
            continue
        value = settings.get(name)
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid preplace_alignment setting: {name}; require a finite positive number")
    for name in ("velocity_ratio", "acceleration_ratio"):
        if type(settings[name]) is not int or not 1 <= settings[name] <= 100:
            raise ValueError(f"preplace_alignment.{name} must be an integer from 1 to 100")
    limits = dict(max_correction_m=.1, max_correction_rad=np.radians(20.),
        max_step_m=.005, max_rotation_step_rad=np.radians(2.), max_joint_step_rad=.1,
        max_lift_m=.1, path_tolerance_m=.01, path_rotation_tolerance_rad=np.radians(5.),
        settle_s=5., step_timeout_s=120., planning_timeout_s=300.,
        perception_timeout_s=600., total_timeout_s=3600.)
    for name, limit in limits.items():
        if settings[name] > limit:
            raise ValueError(f"preplace_alignment.{name} must be at most {limit:g}")
    if settings["position_tolerance_m"] >= settings["max_step_m"] / 2:
        raise ValueError("Preplacement arrival tolerance must be smaller than half max_step_m")
    if settings["rotation_tolerance_rad"] >= settings["max_rotation_step_rad"]:
        raise ValueError("Preplacement angular arrival tolerance must be smaller than max_rotation_step_rad")
    if (settings["path_tolerance_m"] < settings["position_tolerance_m"]
            or settings["path_rotation_tolerance_rad"] < settings["rotation_tolerance_rad"]):
        raise ValueError("Preplacement path tolerances must cover arrival tolerances")
    if not settings["settle_s"] < settings["step_timeout_s"] <= settings["total_timeout_s"]:
        raise ValueError("Preplacement requires settle_s < step_timeout_s <= total_timeout_s")
    if max(settings["planning_timeout_s"], settings["perception_timeout_s"]) > settings["total_timeout_s"]:
        raise ValueError("Preplacement planning/perception deadlines must not exceed total_timeout_s")


def _mesh_corners(path):
    import trimesh
    mesh = trimesh.load(path, force="mesh", process=False)
    bounds = np.asarray(mesh.bounds, dtype=float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
        raise ValueError("Preplacement requires a nonempty finite 3D mesh in metres")
    return np.array(list(product(*zip(bounds[0], bounds[1]))), dtype=float)


def _segment(start, finish, count):
    relative = Rotation.from_matrix(start[:3, :3].T @ finish[:3, :3]).as_rotvec()
    values = []
    for index in range(1, count+1):
        fraction = index/count
        value = start.copy()
        value[:3, 3] = (1-fraction)*start[:3, 3]+fraction*finish[:3, 3]
        value[:3, :3] = start[:3, :3] @ Rotation.from_rotvec(relative*fraction).as_matrix()
        values.append(value)
    values[-1] = finish.copy()
    return values


def _count(start, finish, settings):
    distance, angle = difference(start, finish)
    requested = max(distance/settings["max_step_m"], angle/settings["max_rotation_step_rad"])
    if not np.isfinite(requested) or requested > MAX_STEPS:
        raise ValueError(f"Preplacement exceeds the {MAX_STEPS}-step planning budget")
    return max(1, int(np.ceil(requested)))


def _corner_positions(poses, corners):
    return np.einsum("nij,kj->nki", poses[:, :3, :3], corners)+poses[:, None, :3, 3]


def plan_preplace_alignment(kinematics, config, initial_joints, grasp, goal, above):
    """Return all IK endpoints before the caller dispatches any correction.

    During the central correction, every mesh-bound corner stays at least the
    preplacement height above the *same corner* at the taught goal. A second-
    derivative bound covers the SO(3) arc between sampled endpoints. Initial
    lifting and final lowering retain their fixed attitudes. The SDK executes
    joint interpolation, so the caller must still monitor measured path error.
    """
    settings = dict(DEFAULT_PREPLACE_ALIGNMENT, **config.get("preplace_alignment", {}))
    validate_settings(settings)
    started = time.monotonic()
    deadline = min(settings["planning_timeout_s"], settings["total_timeout_s"])
    q = vector(initial_joints, 7)
    grasp, goal, above = [transform(value).copy() for value in (grasp, goal, above)]
    initial_flange = matrix(kinematics.fk("right", tuple(q)))
    actual = initial_flange @ grasp
    height = float(above[1, 3]-goal[1, 3])
    if (height <= 0 or not np.allclose(above[[0, 2], 3], goal[[0, 2], 3], atol=1e-9, rtol=0)
            or difference(above, goal)[1] > 1e-8):
        raise ValueError("Preplacement target must be vertically above the taught goal with the same orientation")
    distance, angle = difference(actual, above)
    if distance > settings["max_correction_m"] or angle > settings["max_correction_rad"]:
        raise ValueError(f"Preplacement correction exceeds limits: {distance*1000:.2f} mm / "
                         f"{np.degrees(angle):.2f} deg")
    corners = _mesh_corners(config["mesh"])
    middle_count = _count(actual, above, settings)
    middle = np.asarray([actual, *_segment(actual, above, middle_count)])
    goal_corners = corners @ goal[:3, :3].T+goal[:3, 3]
    raw_clearance = _corner_positions(middle, corners)[:, :, 1]-goal_corners[None, :, 1]
    # For y(u)=a*cos(theta*u)+b*sin(theta*u)+c+linear(u), interpolation
    # error is bounded by max|y''|/8. This is zero for rotation about base Y.
    margin_by_corner = np.zeros(len(corners))
    if angle > 1e-12:
        axis = Rotation.from_matrix(actual[:3, :3].T @ above[:3, :3]).as_rotvec()/angle
        perpendicular = corners-np.outer(corners @ axis, axis)
        a = (perpendicular @ actual[:3, :3].T)[:, 1]
        b = (np.cross(axis, corners) @ actual[:3, :3].T)[:, 1]
        margin_by_corner = np.hypot(a, b)*(angle/middle_count)**2/8
    lower_bound = float(np.min(raw_clearance-margin_by_corner[None, :]))
    lift = max(0., height-lower_bound)
    if lift < 1e-12:
        lift = 0.
    if lift > settings["max_lift_m"]:
        raise ValueError(f"Preplacement geometric lift {lift*1000:.2f} mm exceeds max_lift_m")
    lift_count = int(np.ceil(lift/settings["max_step_m"])) if lift else 0
    count = 2*lift_count+middle_count
    if count > MAX_STEPS:
        raise ValueError(f"Preplacement exceeds the {MAX_STEPS}-step planning budget")
    if count*settings["settle_s"] >= settings["total_timeout_s"]:
        raise ValueError("Preplacement settling alone would exceed total_timeout_s")
    raised_actual, raised_above = actual.copy(), above.copy()
    raised_actual[1, 3] += lift
    raised_above[1, 3] += lift
    rails, phases = [actual.copy()], []
    if lift:
        rails.extend(_segment(actual, raised_actual, lift_count))
        phases.extend(["lift"]*lift_count)
    rails.extend(_segment(raised_actual, raised_above, middle_count))
    phases.extend(["align"]*middle_count)
    if lift:
        rails.extend(_segment(raised_above, above, lift_count))
        phases.extend(["descend"]*lift_count)
    rails[-1] = above.copy()
    planner = Planner(config["move_above"], kinematics, side="right")
    inverse_grasp = np.linalg.inv(grasp)
    points, flanges = [q.copy()], [initial_flange]
    translation_steps, rotation_steps, flange_steps, joint_steps = [], [], [], []
    for previous, target in zip(rails, rails[1:]):
        if time.monotonic()-started > deadline:
            raise TimeoutError("Preplacement alignment IK planning timed out")
        step, rotation_step = difference(previous, target)
        if (step > settings["max_step_m"]+1e-12
                or rotation_step > settings["max_rotation_step_rad"]+1e-12):
            raise ValueError("Preplacement Cartesian segment exceeds a translation/rotation step limit")
        flange = target @ inverse_grasp
        endpoint = planner.plan(q, flange).points[-1]
        joint_step = float(np.max(np.abs(endpoint-q)))
        if joint_step > settings["max_joint_step_rad"]:
            raise ValueError("Preplacement IK changed joint branch or exceeded the per-step joint limit")
        translation_steps.append(step)
        rotation_steps.append(rotation_step)
        flange_steps.append(difference(flanges[-1], flange)[0])
        joint_steps.append(joint_step)
        points.append(endpoint.copy())
        flanges.append(flange)
        q = endpoint
    if time.monotonic()-started > deadline:
        raise TimeoutError("Preplacement alignment IK planning timed out")
    midpoint = vector(config.get("rail_midpoint_cad_m", np.mean(corners, axis=0)), 3)
    midpoint_error = actual[:3, :3] @ midpoint+actual[:3, 3]-(above[:3, :3] @ midpoint+above[:3, 3])
    return dict(schema_version=1, side="right", mode="preplace_alignment", control_mode="position",
        parent_frame="tianji_right_base", unit="m", angle_unit="rad", settings=settings,
        steps=count, joint_waypoints_rad=np.asarray(points).tolist(),
        T_R_O_waypoints=np.asarray(rails).tolist(), T_R_F_waypoints=np.asarray(flanges).tolist(),
        T_F_O=grasp.tolist(), T_R_F_above=flanges[-1].tolist(),
        T_R_O_initial=actual.tolist(), T_R_O_goal=goal.tolist(), T_R_O_above=above.tolist(),
        phase_by_step=phases, height_m=height, lift_m=float(lift),
        position_error_m=distance, rotation_error_rad=angle,
        translation_error_base_m=(actual[:3, 3]-above[:3, 3]).tolist(),
        midpoint_cad_m=midpoint.tolist(), midpoint_position_error_m=float(np.linalg.norm(midpoint_error)),
        midpoint_translation_error_base_m=midpoint_error.tolist(),
        mesh_bound_corners_cad_m=corners.tolist(),
        clearance_reference="same_mesh_bound_corner_at_goal",
        initial_min_corner_clearance_m=float(raw_clearance[0].min()),
        alignment_min_corner_clearance_bound_m=lower_bound+lift,
        rotation_chord_margin_m=float(margin_by_corner.max()),
        max_translation_step_m=max(translation_steps), max_rotation_step_rad=max(rotation_steps),
        max_flange_translation_step_m=max(flange_steps), max_joint_step_rad=max(joint_steps),
        interpolation_owner="controller", collision_checked=False, force_sensor_used=False,
        force_controlled=False, assembly_verified=False, automatic_release=False,
        hand_action="keep_grasp")
