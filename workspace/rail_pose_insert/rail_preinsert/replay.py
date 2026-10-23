"""Replay joint trajectories and render link poses without connecting devices."""
from pathlib import Path

import numpy as np
import trimesh

from .core import digest, matrix, read_json, write_json
from .geometry import RobotGeometry
from .planner import JointPath, Planner


def replay(config, kinematics, plan_path, output):
    data = read_json(plan_path)
    if data.get("schema_version") != 2 or data.get("collision_checked") is not False:
        raise ValueError("Regenerate the plan using the current direct interpolation workflow")
    if data.get("end_stage") != "preinsert" or data.get("model_sha256") != kinematics.model.digest:
        raise ValueError("Plan stage or kinematic model mismatch")
    for key, plan_key in [("mesh", "mesh_sha256"),
                          ("calibration", "calibration_sha256"), ("target", "target_sha256"), ("urdf", "urdf_sha256"),
                          ("manifest", "manifest_sha256")]:
        if digest(config[key]) != data[plan_key]:
            raise ValueError(f"{key} changed since planning")
    frames = RobotGeometry(config, kinematics)
    planner = Planner(config, kinematics)
    points = np.asarray(data["joint_waypoints_rad"])
    planner.validate_path(points)
    path = JointPath(points, data["joint_speed_rad_s"], data["joint_acceleration_rad_s2"])
    directory = Path(output); directory.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    fig = plt.figure(figsize=(9, 7)); ax = fig.add_subplot(projection="3d")
    mesh = trimesh.load(config["mesh"], force="mesh")
    corners = trimesh.bounds.corners(mesh.bounds)
    edges = [[i, j] for i in range(8) for j in range(i + 1, 8)
             if np.count_nonzero(np.abs(corners[i] - corners[j]) > 1e-12) == 1]
    centres = []
    for elapsed in np.linspace(0, path.duration, 15):
        q = path.at(elapsed)
        T = matrix(kinematics.fk("left", tuple(q))) @ np.asarray(data["targets"]["T_F_O"])
        xyz = corners @ T[:3, :3].T + T[:3, 3]
        centres.append(xyz.mean(axis=0))
        ax.add_collection3d(Line3DCollection(xyz[edges], colors="tab:blue", alpha=.2))
    centres = np.asarray(centres)
    ax.plot(*centres.T, color="tab:blue", label="rail centre")
    all_xyz = list(centres)
    for side, indices, suffix in [("left", slice(0, 7), "L"), ("right", slice(7, 14), "R")]:
        for q7, alpha in [(points[0], .35), (points[-1], 1.)]:
            q = np.array(data["initial_full_joints"]); q[:7] = q7
            poses = frames.poses(q)
            xyz = np.array([poses[f"Link{i}_{suffix}"][:3, 3] for i in range(1, 8)]
                           + [poses[f"TCP_Link_{suffix}"][:3, 3]])
            ax.plot(*xyz.T, color="tab:green" if side == "left" else "gray", alpha=alpha)
            all_xyz.extend(xyz)
    all_xyz = np.asarray(all_xyz); centre = (all_xyz.max(axis=0) + all_xyz.min(axis=0))/2
    radius = max(.2, np.ptp(all_xyz, axis=0).max()/2 + .1)
    ax.set(xlim=(centre[0]-radius, centre[0]+radius), ylim=(centre[1]-radius, centre[1]+radius),
           zlim=(centre[2]-radius, centre[2]+radius), xlabel="Left base X [m]", ylabel="Left base Y [m]",
           zlabel="Left base Z [m]", title="Offline path to preinsert" + (" (SYNTHETIC INPUTS)" if data["test_data"] else ""))
    ax.set_box_aspect((1, 1, 1)); ax.legend(); fig.tight_layout()
    fig.savefig(directory / "path.png", dpi=130); plt.close(fig)
    report = {"status": "offline_path_verified", "end_stage": "preinsert", "test_data": data["test_data"],
              "duration_s": path.duration, "collision_checked": False,
              "verification": "joint limits and trajectory replay only",
              "hardware_connected": False, "path_image": str(directory / "path.png")}
    write_json(directory / "result.json", report)
    return report
