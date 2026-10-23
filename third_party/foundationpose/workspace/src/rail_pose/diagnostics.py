from __future__ import annotations

import importlib
import importlib.metadata
import platform
import shutil
import subprocess
import sys

from .common import REPOSITORY, WORKSPACE


def check_environment():
    report = {"python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "errors": [], "versions": {}}
    for name in ["torch", "torchvision", "pytorch3d", "nvdiffrast", "trimesh", "open3d",
                 "cv2", "warp", "kornia", "pyrealsense2", "cadquery"]:
        try:
            module = importlib.import_module(name)
            report["versions"][name] = str(getattr(module, "__version__", "import-ok"))
        except Exception as exc:
            report["errors"].append(f"{name}: {exc}")
    try:
        import torch
        report["cuda_available"] = torch.cuda.is_available()
        report["torch_cuda_build"] = torch.version.cuda
        if torch.cuda.is_available():
            device = torch.cuda.get_device_properties(0)
            report["gpu"] = {"name": device.name, "memory_mib": device.total_memory / 2 ** 20,
                             "compute_capability": list(torch.cuda.get_device_capability(0))}
            torch.ones(1, device="cuda").add_(1)
            torch.cuda.synchronize()
        else:
            report["errors"].append("CUDA is unavailable")
    except Exception as exc:
        report["errors"].append(f"CUDA test: {exc}")
    try:
        import estimater
        if estimater.mycpp is None or not callable(getattr(estimater.mycpp, "cluster_poses", None)):
            raise RuntimeError("mycpp.cluster_poses is missing")
        if estimater.wp is None:
            raise RuntimeError("Warp is missing; depth filtering cannot run")
        report["estimater_import"] = True
    except Exception as exc:
        report["errors"].append(f"estimator import/native extension: {exc}")
    for stamp in ["2023-10-28-18-33-37", "2024-01-11-20-02-45"]:
        for name in ["config.yml", "model_best.pth"]:
            path = REPOSITORY / "weights" / stamp / name
            if not path.is_file():
                report["errors"].append(f"missing weight: {path}")
    try:
        import pyrealsense2 as rs
        report["camera_count"] = len(rs.context().query_devices())
    except Exception as exc:
        report["camera_error"] = str(exc)
    nvcc = shutil.which("nvcc")
    report["nvcc"] = nvcc
    if nvcc:
        report["nvcc_version"] = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout.strip()
    report["ready"] = not report["errors"]
    return report
