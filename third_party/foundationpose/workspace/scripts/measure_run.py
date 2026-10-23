#!/usr/bin/env python3
"""Run a command and sample its complete NVML process memory, not just Torch."""
import argparse
import json
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("--report", required=True)
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command and args.command[0] == "--" else args.command
if not command:
    parser.error("provide a command after --")
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
started = time.monotonic()
process = subprocess.Popen(command)
samples = []
sampling_errors = []
try:
    while process.poll() is None:
        try:
            for item in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                memory = item.usedGpuMemory
                if item.pid == process.pid and isinstance(memory, int) and 0 <= memory < 2 ** 63:
                    samples.append({"elapsed_s": time.monotonic() - started, "memory_mib": memory / 2 ** 20})
        except pynvml.NVMLError as exc:
            sampling_errors.append(str(exc))
        time.sleep(.1)
except BaseException:
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    raise
finally:
    pynvml.nvmlShutdown()
report = {"pid": process.pid, "returncode": process.wait(), "duration_s": time.monotonic() - started,
          "nvml_peak_process_mib": max((s["memory_mib"] for s in samples), default=None),
          "sampling_errors": sampling_errors, "samples": samples}
path = Path(args.report)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))
raise SystemExit(report["returncode"])
