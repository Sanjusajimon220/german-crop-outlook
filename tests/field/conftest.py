"""Tests import the pipeline code from this checkout's 06_field and read data from D:\vista (data paths are relative)."""
import os
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[2] / "06_field"
sys.path.insert(0, str(CODE))
for extra in ("02_district_model",):
    p = Path(__file__).resolve().parents[2] / extra
    if p.exists():
        sys.path.append(str(p))
os.chdir("D:/vista")
