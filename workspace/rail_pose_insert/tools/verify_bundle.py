"""Verify bundled source/weights/assets without importing drivers or using hardware."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def verify(root=ROOT, *, source_only=False):
    root = Path(root).resolve()
    third = root / "third_party"
    manifest = json.loads((third / "rail_insert_manifest.json").read_text())
    problems = []
    local_assets = manifest.get("local_assets", [])
    indexed = {item["path"] for item in manifest["files"] + local_assets}
    complete_source = manifest["foundationpose_complete_source"]["source_files"]
    for item in complete_source:
        if f"foundationpose/{item['vendored_path']}" not in indexed:
            problems.append(f"Source inventory missing from snapshot: {item['source_path']}")
    selected = manifest["files"] + ([] if source_only else [item for item in local_assets
                                                           if item.get("required_for_runtime")])
    for item in selected:
        path = third / item["path"]
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(third):
            problems.append(f"Missing or external snapshot file: {item['path']}")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != item["vendored_sha256"]:
            problems.append(f"Snapshot digest mismatch: {item['path']}")
    assets = third / "rail_assets/generated"
    for element in ET.parse(assets / "robot.urdf").iter("mesh"):
        path = assets / element.attrib["filename"]
        if not path.is_file() or not path.resolve().is_relative_to(third):
            problems.append(f"Missing or external URDF mesh: {path}")
    geometry = json.loads((assets / "manifest.json").read_text())
    for name, model in geometry["cad"]["models"].items():
        for item in (model, model.get("collision", {})):
            if "mesh" in item and not (assets / item["mesh"]).is_file():
                problems.append(f"Missing CAD mesh for {name}: {item['mesh']}")
    published = {item["path"] for item in manifest["files"]}
    return {"ready": not problems, "source_only": source_only, "files_checked": len(selected),
            "foundationpose_source_files_checked": sum(
                f"foundationpose/{item['vendored_path']}" in published for item in complete_source),
            "local_assets_not_versioned": len(local_assets), "problems": problems}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source-only", action="store_true", help="Verify published source/assets without local weights")
    args = parser.parse_args()
    report = verify(args.root, source_only=args.source_only)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
