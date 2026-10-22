"""Audit the existing task CAD; output nominal geometry, not physical measurements."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from OCP.Bnd import Bnd_Box
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBndLib import BRepBndLib
from OCP.GeomAbs import GeomAbs_Cylinder
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    sim = root / "bimanual-teleop/bimanual_teleop/sim"
    manifest_path = sim / "assets/generated/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    fixture = sim / "assets/source/rail_task/导轨台长.step"
    digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
    if digest != manifest["cad"]["sources"]["fixture"]["sha256"]:
        raise ValueError("Fixture source does not match generated geometry manifest")
    pose_model = json.loads((root / "FoundationPose/workspace/input/model/rail.json").read_text())
    if pose_model["source_sha256"] != manifest["cad"]["sources"]["rail"]["sha256"]:
        raise ValueError("Perception and task CAD use different source rails")
    # The existing CAD conversion identifies this support plane in source millimetres.
    support_y_mm = 139.87
    reader = STEPControl_Reader()
    if reader.ReadFile(str(fixture)) != IFSelect_RetDone:
        raise ValueError("Cannot load fixture STEP")
    reader.TransferRoots()
    explorer = TopExp_Explorer(reader.OneShape(), TopAbs_FACE)
    pins = []
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        surface = BRepAdaptor_Surface(face)
        if surface.GetType() == GeomAbs_Cylinder:
            cylinder = surface.Cylinder()
            if 3.0 < cylinder.Radius() < 3.5:
                axis = cylinder.Axis().Direction()
                direction = np.array([axis.X(), axis.Y(), axis.Z()])
                if direction[1] < 0:
                    direction = -direction
                bounds = Bnd_Box()
                BRepBndLib.Add_s(face, bounds)
                limits = bounds.Get()
                pins.append({
                    "diameter_mm": 2 * cylinder.Radius(),
                    "axis_source_cad": direction.tolist(),
                    "tilt_from_support_normal_deg": float(np.degrees(np.arccos(np.clip(direction[1], -1, 1)))),
                    "maximum_exposed_height_mm": limits[4] - support_y_mm,
                    "bounds_source_cad_mm": list(limits),
                })
        explorer.Next()
    if len(pins) != 2:
        raise ValueError(f"Expected 2 pin cylinders, found {len(pins)}")
    hole_mm = [1000 * d for d in manifest["rail"]["hole_diameters_m"]]
    # Pin/hole pairing follows the existing generated seated pose; report a
    # pairing-independent lower bound as the conservative clearance.
    clearance = (min(hole_mm) - max(p["diameter_mm"] for p in pins)) / 2
    tallest = max(p["maximum_exposed_height_mm"] for p in pins)
    result = {
        "source": str(fixture), "source_sha256": digest,
        "scope": "Nominal CAD geometry; physical dimensions and alignment not verified",
        "perception_rail_matches_task_cad_source": True,
        "support_plane_source_cad_y_mm": support_y_mm,
        "pins": pins, "hole_diameters_mm": hole_mm,
        "conservative_radial_clearance_mm": clearance,
        "preinsert_offset_mm": 20.0,
        "nominal_min_vertical_clearance_at_preinsert_mm": 20.0 - tallest,
        "clearance_note": "Uses an ideal seated rail bottom at the support plane and a 20 mm support-normal offset; pose errors, actual tilt and manufacturing tolerances must be included separately.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "tallest_pin_mm": tallest,
                      "radial_clearance_mm": clearance, "preinsert_clearance_mm": 20.0 - tallest}))


if __name__ == "__main__":
    main()
