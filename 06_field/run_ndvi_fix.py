"""NDVI fix re-run (project log 2026-10-17 15:44): retrieval -> NDVI part of the area test (same rule) -> NDVI model, weekly
series, export. Run from D:\vista. Steps log to data/processed/field_v2/ndvi_fix_log.txt."""
import glob
import os
import shutil
import subprocess
import time

PY = r"C:\Users\Asus\AppData\Local\Programs\Python\Python312\python.exe"
HERE = os.path.dirname(os.path.abspath(__file__))
L = os.path.join("data", "processed", "field_v2")


def log(m):
    with open(os.path.join(L, "ndvi_fix_log.txt"), "a") as f:
        f.write(f"{time.strftime('%H:%M')} {m}\n")


def run(name, args):
    with open(os.path.join(L, f"{name}_log.txt"), "a") as out:
        r = subprocess.run([PY, "-W", "ignore", *args], stdout=out, stderr=subprocess.STDOUT)
    log(f"{name} exit {r.returncode}")
    if r.returncode:
        raise SystemExit(r.returncode)


def rm(pattern):
    for p in glob.glob(pattern):
        shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)


if __name__ == "__main__":
    shutil.copy(os.path.join(HERE, "fv2.py"), "fv2.py")                     # the run copy used from D:\vista
    areas = [x for a in ("pilot", "selhausen", "north", "south") for y in range(2021, 2026) for x in (a, str(y))]
    if os.path.exists(os.path.join(L, "area_test_summary_log_contaminated.txt")):   # restarted: retrieval already done
        log("resumed after the retrieval")
    else:
        run("retrieval_fix", ["phase11_retrieval.py", *areas])
        rm(os.path.join(L, "area_rows", "ndvi_*")); rm(os.path.join(L, "area_parts", "ndvi_*"))
        shutil.move(os.path.join(L, "area_test_summary_log.txt"), os.path.join(L, "area_test_summary_log_contaminated.txt"))
    for s in ("rows", "test", "summary"):
        run(f"area_test_{s}_fix", ["phase12_area_test.py", s])
    shutil.copy(os.path.join(L, "area_test_summary_fix_log.txt"), os.path.join(L, "area_test_summary_log.txt"))
    rm(os.path.join(L, "models", "ndvi_*")); rm(os.path.join(L, "uncertainty_V2.json"))
    rm(os.path.join(L, "*_20*", "weekly_pixels_V2.npz")); rm("D:/german-crop-outlook-field/docs/field/*")
    run("platform_data_fix", [os.path.join(HERE, "run_platform_data.py")])
    log("NDVI FIX DONE")
