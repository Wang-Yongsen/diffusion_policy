from __future__ import annotations

from pathlib import Path
import subprocess
import numpy as np
import trimesh

from .common import WORKSPACE, save_json, sha256


def step_mesh(source: Path, output: Path) -> None:
    import cadquery as cq
    shape = cq.importers.importStep(str(source)).val()
    # OpenCascade's STEP import is in millimetres; export remains in those units.
    cq.exporters.export(shape, str(output), tolerance=.05, angularTolerance=.1)


def prepare_model(source: Path, output: Path, source_unit: str | None = None) -> dict:
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.resolve() == output.resolve():
        raise ValueError("output must not overwrite the original CAD")
    source_digest = sha256(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower()
    cad_audit = None
    if suffix == ".sldprt":
        intermediate = output.with_suffix(".stl")
        audit_path = output.with_name(output.stem + "-cad-audit.json")
        subprocess.run(["node", str(WORKSPACE / "scripts/extract_sldprt.cjs"),
                        str(source), str(intermediate), str(audit_path)], check=True)
        import json
        cad_audit = json.loads(audit_path.read_text())
        source_unit = "m"  # Documented SolidWorks DisplayLists coordinate convention.
        mesh_source = intermediate
    elif suffix in (".step", ".stp"):
        intermediate = output.with_suffix(".stl")
        step_mesh(source, intermediate)
        source_unit = "mm"
        mesh_source = intermediate
    elif suffix in (".obj", ".stl", ".ply"):
        if source_unit not in ("m", "mm"):
            raise ValueError("mesh files require explicit --source-unit m|mm")
        mesh_source = source
    else:
        raise ValueError("supported CAD: SLDPRT, STEP, OBJ, STL, PLY")
    mesh = trimesh.load(mesh_source, force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError("CAD contains no usable triangular mesh")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("CAD vertices are not finite")
    if source_unit == "mm":
        mesh.apply_scale(.001)
    if np.any(mesh.extents <= 0) or mesh.extents.max() > 10:
        raise ValueError("implausible metric dimensions; verify the CAD source units")
    # Preserve the original CAD origin and axes; do not auto-center or fill holes.
    mesh.visual.vertex_colors = np.tile([160, 160, 160, 255], (len(mesh.vertices), 1))
    mesh.export(output)
    result = {
        "source": str(source.resolve()), "source_sha256": source_digest,
        "mesh": str(output.resolve()), "mesh_sha256": sha256(output),
        "unit": "m", "source_unit": source_unit,
        "frame": "original CAD axes and origin", "T_mesh_CAD": np.eye(4).tolist(),
        "bounds_m": mesh.bounds.tolist(), "dimensions_m": mesh.extents.tolist(),
        "vertices": len(mesh.vertices), "triangles": len(mesh.faces),
        "watertight": bool(mesh.is_watertight), "volume_m3": float(mesh.volume),
        "cad_audit": cad_audit,
        "physical_dimensions_verified": False,
    }
    save_json(output.with_suffix(".json"), result)
    return result
