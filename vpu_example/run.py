#!/usr/bin/env python3
"""Run the fixed mapping using the PyTimeloop v4 frontend."""
import os
import sys

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
ARCH_PATH = os.path.join(CURR_DIR, "arch.yaml")
MAP_PATH = os.path.join(CURR_DIR, "mapping.yaml")
PROB_PATH = os.path.join(CURR_DIR, "problem.yaml")
OUT_DIR = os.path.join(CURR_DIR, "outputs")

# Use the rebuilt executable and its matching library, including inside .venv.
ROOT = os.path.abspath(os.path.join(CURR_DIR, "../../../.."))
BUILD_DIR = os.path.join(ROOT, "accelergy-timeloop-infrastructure/src/timeloop/build")
os.environ["PATH"] = BUILD_DIR + os.pathsep + os.environ.get("PATH", "")
# The loader reads LD_LIBRARY_PATH at process startup; restart once before import.
if os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)[0] != BUILD_DIR:
    os.environ["LD_LIBRARY_PATH"] = BUILD_DIR + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    os.execv(sys.executable, [sys.executable, os.path.abspath(__file__), *sys.argv[1:]])

import pytimeloop.timeloopfe.v4 as tl

if __name__ == "__main__":
    spec = tl.Specification.from_yaml_files(ARCH_PATH, PROB_PATH, MAP_PATH)
    tl.call_model(spec, output_dir=OUT_DIR)
