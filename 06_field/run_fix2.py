"""Detached runner for fix 2 (pythonw): output to data/processed/field_v2/fix2_out.txt, 'FIX2 DONE' / 'FIX2 FAILED' at the end."""
import os
import sys
import traceback

os.chdir(r"D:\vista")
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), r"D:\vista"]
out = open(r"D:\vista\data\processed\field_v2\fix2_out.txt", "a", buffering=1, encoding="utf-8")
sys.stdout = sys.stderr = out
try:
    import phase15_fix2_dips
    phase15_fix2_dips.run()
    print("FIX2 DONE")
except Exception:
    traceback.print_exc()
    print("FIX2 FAILED")
