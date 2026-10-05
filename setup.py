from setuptools import setup, find_packages, find_namespace_packages

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
        "diffusion_policy": ["config/*.yaml", "config/task/*.yaml"],
        "diffusion_policy.third_party.bimanual_teleop": ["manifest.json"],
        "diffusion_policy.third_party.bimanual_teleop.vendor.tianji": [
            "LICENSE", "manifest.json", "README.md", "SDK_PYTHON/*.py", "SDK_PYTHON/*.so",
            "CommonConfig/*.MvKDCfg",
        ],
    },
)
