"""Conservative convex collision geometry in the official left-arm base frame."""
from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import fcl
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from .core import PROJECT, digest, matrix, read_json, transform, vector


@dataclass
class Part:
    name: str
    geometry: object
    vertices: np.ndarray
    local: np.ndarray
    link: str | None = None


def origin(element):
    T = np.eye(4)
    if element is not None:
        T[:3, 3] = [float(v) for v in element.get("xyz", "0 0 0").split()]
        T[:3, :3] = Rotation.from_euler("xyz", [float(v) for v in element.get("rpy", "0 0 0").split()]).as_matrix()
    return T


def part(name, mesh, local=None, link=None):
    hull = mesh.convex_hull
    vertices = np.ascontiguousarray(hull.vertices, dtype=np.float64)
    faces = np.column_stack([np.full(len(hull.faces), 3), hull.faces]).ravel()
    geometry = fcl.Convex(vertices, len(hull.faces), faces)
    return Part(name, geometry, vertices, np.eye(4) if local is None else local, link)


def grasp_link(link):
    return bool(link and (link == "l_wrist" or link.startswith(
        ("l_thumb_", "l_index_finger_", "l_middle_finger_", "l_ring_finger_", "l_pinky_"))))


class RobotGeometry:
    def __init__(self, config, kinematics):
        sys.path.insert(0, str(PROJECT / "bimanual-teleop"))
        from bimanual_teleop.sim.kinematics import RobotModel
        self.robot = RobotModel(Path(config["urdf"]))
        self.kinematics = kinematics
        zero = np.zeros(54)
        # The SDK base includes its first DH offset. The URDF first joint frame
        # is not that base: align through flange FK, then verify nonzero poses.
        self.T_B_W = matrix(kinematics.fk("left", tuple(zero[:7]))) @ np.linalg.inv(
            self.robot.flange("left", zero, world=True))
        self.registration_error_m = 0.
        for q7 in [np.radians([30, -60, -34, -52, 30, 12, 4]),
                   np.radians([25, -55, -30, -45, 20, 10, 2])]:
            q = zero.copy(); q[:7] = q7
            predicted = self.T_B_W @ self.robot.flange("left", q, world=True)
            actual = matrix(kinematics.fk("left", tuple(q7)))
            error = np.linalg.norm(predicted[:3, 3] - actual[:3, 3])
            angle = Rotation.from_matrix(predicted[:3, :3] @ actual[:3, :3].T).magnitude()
            self.registration_error_m = max(self.registration_error_m, float(error))
            if error > .0001 or angle > .0002:
                raise ValueError("URDF does not agree with official FK after fixed-base registration")
        self.parts = []
        path = Path(config["urdf"])
        manifest = read_json(config["manifest"])
        if digest(path) != manifest["robot_urdf_sha256"]:
            raise ValueError("URDF differs from its generated geometry manifest")
        for element in ET.parse(path).getroot().findall("link"):
            for index, collision in enumerate(element.findall("collision")):
                g = collision.find("geometry")
                mesh_element = g.find("mesh")
                if mesh_element is None:
                    raise ValueError("Expected the project's mesh-based collision URDF")
                filename = path.parent / mesh_element.get("filename")
                mesh = trimesh.load(filename, force="mesh")
                mesh.apply_scale([float(v) for v in mesh_element.get("scale", "1 1 1").split()])
                self.parts.append(part(f"robot:{element.get('name')}:{index}", mesh,
                                       origin(collision.find("origin")), element.get("name")))
        self.ignore = set()
        for descendant in self.robot.by_child:
            moving = 0
            for joint in reversed(self.robot.chain(descendant)):
                moving += joint.kind == "revolute"
                if moving > 2:
                    break
                self.ignore.add(frozenset([descendant, joint.parent]))
        self.active_links = {link for link in self.robot.collidable_links
                             if any(j.name in self.robot.arm_names["left"] for j in self.robot.chain(link))}

    def poses(self, q):
        values = dict(zip(self.robot.names, q))
        cache = {}

        def visit(link):
            if link in cache:
                return cache[link]
            if link not in self.robot.by_child:
                T = self.T_B_W @ self.robot.world
            else:
                joint = self.robot.by_child[link]
                T = visit(joint.parent) @ joint.origin
                if joint.kind == "revolute":
                    R = np.eye(4)
                    R[:3, :3] = Rotation.from_rotvec(joint.axis * values[joint.name]).as_matrix()
                    T = T @ R
            cache[link] = T
            return T
        return {link: visit(link) for link in {p.link for p in self.parts} | {"TCP_Link_L", "TCP_Link_R"}}


class CollisionWorld:
    def __init__(self, config, kinematics, full_joints, T_F_O, T_B_O_goal, *, allow_test_data=False):
        self.config, self.kinematics = config, kinematics
        self.model = RobotGeometry(config, kinematics)
        self.full_joints = vector(full_joints, 54, "full measured joints").copy()
        self.T_F_O = transform(T_F_O)
        self.margin = float(config["collision_margin_m"])
        scene = read_json(config["scene"])
        if scene.get("test_data", False) and not allow_test_data:
            raise ValueError("Synthetic environment is forbidden for live use")
        if scene.get("parent_frame") != "tianji_left_base" or scene.get("unit") != "m":
            raise ValueError("Scene obstacles must use the left base and metres")
        if not scene.get("obstacles") and not allow_test_data:
            raise ValueError("Provide measured table/bin obstacles in scene.json")
        self.parts = list(self.model.parts)
        self.rail_index = len(self.parts)
        self.parts.append(part("rail", trimesh.load(config["mesh"], force="mesh")))
        manifest = read_json(config["manifest"])
        rail = manifest["rail"]
        geometry = manifest["fixture_geometry"]
        T_sim_cad = np.eye(4)
        T_sim_cad[:3, :3] = rail["cad_rotation"]
        T_sim_cad[:3, 3] = -np.asarray(rail["origin_in_rotated_cad_m"])
        T_S_sim = np.eye(4)
        T_S_sim[:3, :3] = Rotation.from_euler("xyz", geometry["seated_rpy_deg"], degrees=True).as_matrix()
        T_S_sim[:3, 3] = geometry["seated_center_in_fixture_m"]
        # Remove the simulation's 0.4 mm support gap: target means fully seated.
        T_S_sim[2, 3] -= .0004
        T_B_S = transform(T_B_O_goal) @ np.linalg.inv(T_S_sim @ T_sim_cad)
        fixture = Path(config["manifest"]).parent / manifest["cad"]["models"]["fixture"]["mesh"]
        self.parts.append(part("fixture", trimesh.load(fixture, force="mesh"), T_B_S))
        names = {p.name for p in self.parts}
        for item in scene.get("obstacles", []):
            name = item["name"]
            if not name or name in names:
                raise ValueError("Obstacle names must be nonempty and unique")
            names.add(name)
            if item["type"] == "box":
                size = vector(item["size_m"], 3, "obstacle size_m")
                if np.any(size <= 0) or np.max(size) > 20:
                    raise ValueError("Obstacle dimensions must be positive metres")
                mesh = trimesh.creation.box(extents=size)
            elif item["type"] == "mesh":
                filename = Path(config["scene"]).parent / item["mesh"]
                mesh = trimesh.load(filename, force="mesh")
            else:
                raise ValueError("Obstacle type must be box or metre mesh")
            self.parts.append(part(name, mesh, transform(item["T_B_obstacle"])))
        self.pairs = []
        for i, a in enumerate(self.parts):
            for j in range(i + 1, len(self.parts)):
                b = self.parts[j]
                if a.link and b.link:
                    if a.link == b.link:
                        continue
                    # Measured, fixed grasp is a rigid compound tool. Soft finger
                    # contacts inside it are permitted; all external pairs remain.
                    if grasp_link(a.link) and grasp_link(b.link):
                        continue
                    if frozenset([a.link, b.link]) in self.model.ignore:
                        continue
                    if a.link not in self.model.active_links and b.link not in self.model.active_links:
                        continue
                elif a.name == "rail" or b.name == "rail":
                    other = b if a.name == "rail" else a
                    # Expected grasp contacts only; sensor/bracket/mount still collide.
                    if grasp_link(other.link):
                        continue
                elif not a.link and not b.link:
                    continue
                elif (a.link or b.link) not in self.model.active_links:
                    continue
                self.pairs.append((i, j))
        self.pairs = np.asarray(self.pairs, dtype=int)
        self.last_failure = None
        self.check_count = 0

    def check(self, q_left):
        q = self.full_joints.copy(); q[:7] = vector(q_left, 7)
        poses = self.model.poses(q)
        flange = matrix(self.kinematics.fk("left", tuple(q[:7])))
        transforms, bounds = [], []
        for p in self.parts:
            T = (poses[p.link] @ p.local if p.link else
                 flange @ self.T_F_O if p.name == "rail" else p.local)
            transforms.append(T)
            vertices = p.vertices @ T[:3, :3].T + T[:3, 3]
            bounds.append([vertices.min(axis=0), vertices.max(axis=0)])
        bounds = np.asarray(bounds)
        a, b = self.pairs.T
        separation = np.maximum(0, np.maximum(bounds[a, 0] - bounds[b, 1], bounds[b, 0] - bounds[a, 1]))
        near = np.linalg.norm(separation, axis=1) <= self.margin
        self.check_count += 1
        objects = {}
        def obj(index):
            if index not in objects:
                T = transforms[index]
                objects[index] = fcl.CollisionObject(self.parts[index].geometry, fcl.Transform(T[:3, :3], T[:3, 3]))
            return objects[index]
        for i, j in self.pairs[near]:
            distance = fcl.distance(obj(i), obj(j), fcl.DistanceRequest(), fcl.DistanceResult())
            a, b = self.parts[i], self.parts[j]
            same_hand = (a.link and b.link and
                         any(a.link.startswith(side) and b.link.startswith(side) for side in ("l_", "r_")))
            # Fingers may legitimately be separated by less than the external
            # obstacle margin. Still reject intersecting non-adjacent links.
            margin = 0. if same_hand else self.margin
            if distance < margin:
                self.last_failure = {"parts": [self.parts[i].name, self.parts[j].name], "distance_m": float(distance)}
                return False
        return True
