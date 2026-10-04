#!/usr/bin/env python3
"""Run the local YAML inputs using PyTimeloop and the adjacent native build."""
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BUILD = HERE.parent / "build"

# Use matching native binaries and libraries; the loader reads this at startup.
os.environ["PATH"] = str(BUILD) + os.pathsep + os.environ.get("PATH", "")
if os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)[0] != str(BUILD):
    os.environ["LD_LIBRARY_PATH"] = (
        str(BUILD) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    )
    os.execv(sys.executable, [sys.executable, str(HERE / "run.py"), *sys.argv[1:]])

import pytimeloop.timeloopfe.v4 as tl

if __name__ == "__main__":
    spec = tl.Specification.from_yaml_files(
        str(HERE / "arch.yaml"),
        str(HERE / "problem.yaml"),
        str(HERE / "mapping.yaml"),
    )
    tl.call_model(spec, output_dir=str(HERE / "outputs"))
