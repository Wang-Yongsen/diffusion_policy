from pathlib import Path
import json

from setuptools import find_namespace_packages, find_packages, setup

bundle_manifest = json.loads(Path("third_party/rail_insert_manifest.json").read_text())
bundle_files = [item["path"] for item in bundle_manifest["files"]
    if item["path"].split("/", 1)[0] in {"foundationpose", "rail_assets", "camera_calibration"}
] + ["rail_insert_manifest.json", "README.md"]

vendor_packages = ["diffusion_policy.third_party"] + [
    f"diffusion_policy.third_party.{name}" for name in find_packages("third_party")
]

setup(
    name="diffusion_policy",
    packages=find_namespace_packages(
        include=["diffusion_policy", "diffusion_policy.*"],
        exclude=["*.__pycache__", "*.__pycache__.*"],
    ) + vendor_packages,
    package_dir={"diffusion_policy.third_party": "third_party"},
    package_data={
        "diffusion_policy.third_party": bundle_files,
        "diffusion_policy": ["config/*.yaml", "config/task/*.yaml"],
        "diffusion_policy.third_party.bimanual_teleop": ["manifest.json"],
        "diffusion_policy.third_party.bimanual_teleop.vendor.tianji": [
            "LICENSE", "manifest.json", "README.md", "SDK_PYTHON/*.py", "SDK_PYTHON/*.so",
            "CommonConfig/*.MvKDCfg",
        ],
    },
)
