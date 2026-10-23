from pathlib import Path
import json
import shutil
import numpy as np
import pytest
import trimesh
from rail_pose.model import prepare_model
from rail_pose.common import WORKSPACE


def test_mesh_units_are_explicit_and_origin_is_preserved(tmp_path):
    mesh = trimesh.creation.box(extents=[100, 20, 10])
    mesh.apply_translation([500, 40, 30])
    source = tmp_path / "part.stl"
    mesh.export(source)
    original = source.read_bytes()
    with pytest.raises(ValueError):
        prepare_model(source, source, "mm")
    assert source.read_bytes() == original
    with pytest.raises(ValueError):
        prepare_model(source, tmp_path / "metric.obj")
    report = prepare_model(source, tmp_path / "metric.obj", "mm")
    np.testing.assert_allclose(report["dimensions_m"], [.1, .02, .01])
    np.testing.assert_allclose(np.mean(report["bounds_m"], axis=0), [.5, .04, .03])


@pytest.mark.skipif(not shutil.which("node"), reason="Node.js is required for native SLDPRT")
def test_attached_real_cad_extraction_is_complete_and_matches_dimensions(tmp_path):
    source = WORKSPACE / "input/model/source/导轨.SLDPRT"
    if not source.exists():
        pytest.skip("real CAD attachment is not present")
    report = prepare_model(source, tmp_path / "rail.obj")
    assert report["cad_audit"]["partial"] is False
    assert report["cad_audit"]["errors"] == []
    assert report["cad_audit"]["stats"]["triangles"] == 1896
    np.testing.assert_allclose(report["dimensions_m"], [.0458216276, .48, .0257090088], atol=1e-7)
    assert report["source_sha256"]
