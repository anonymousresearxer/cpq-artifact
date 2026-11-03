import io
import os
import platform
import resource
import subprocess
import sys
import time
from contextlib import redirect_stdout
from importlib.metadata import version
from pathlib import Path

import numpy as np

PACKAGES = ["numpy", "scipy", "pandas", "pyarrow", "scikit-learn", "mapie", "cotengra", "duckdb"]


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()


def _cpu() -> str:
    info = Path("/proc/cpuinfo")
    for line in info.read_text().splitlines() if info.exists() else []:
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def snapshot(params: dict, seed) -> dict:
    buf = io.StringIO()
    with redirect_stdout(buf):
        np.show_runtime()
    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "versions": {p: version(p) for p in PACKAGES},
        "python": platform.python_version(),
        "numpy_runtime": buf.getvalue(),
        "cpu": _cpu(),
        "cores": os.process_cpu_count(),
        "seed": seed,
        "params": params,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def peak_mb() -> float:
    # ru_maxrss is in KiB on Linux and in bytes on macOS
    scale = 2**20 if sys.platform == "darwin" else 2**10
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale
