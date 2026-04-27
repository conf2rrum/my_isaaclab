from __future__ import annotations

import os
import runpy
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

import g1_microwave_task  # noqa: F401  # registers Isaac-G1-Open-Microwave-*-v0

runpy.run_path(os.path.join(THIS_DIR, "record_demos.py"), run_name="__main__")
