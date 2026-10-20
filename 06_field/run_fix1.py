"""Step 4 fix 1 (winter radar filter): area-test rows and test with the filter, then the pre-registered comparison.
Run from D:\vista (detached). Log: data/processed/field_v2/fix1_log.txt."""
import os
import subprocess
import time

PY = r"C:\Users\Asus\AppData\Local\Programs\Python\Python312\pythonw.exe"
HERE = os.path.dirname(os.path.abspath(__file__))
L = os.path.join("data", "processed", "field_v2")
NO_CONSOLE = 0x08000000 | 0x00000200


def log(m):
    with open(os.path.join(L, "fix1_log.txt"), "a") as f:
        f.write(f"{time.strftime('%H:%M')} {m}\n")


for step in ("rows", "test", "summary_rf"):
    with open(os.path.join(L, f"fix1_{step}_out.txt"), "a") as out:
        r = subprocess.run([PY, "-W", "ignore", os.path.join(HERE, "phase12_area_test.py"), step, "rf"], stdout=out,
                           stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=NO_CONSOLE,
                           env={**os.environ, "PYTHONPATH": "D:/vista"})          # district-model modules live in D:\vista
    log(f"{step} exit {r.returncode}")
    if r.returncode:
        raise SystemExit(r.returncode)
log("FIX1 DONE")
