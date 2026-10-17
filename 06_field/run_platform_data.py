"""Chain (plan 2026-10-17, Tasks 1-3 runs): wait for the area test result -> uncertainty table + production models ->
weekly pixel series for 20 area-seasons -> platform export. Run from D:\\vista; uses this checkout's 06_field code."""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append("D:/vista")                                     # district-model modules (phase2_growth, ...)
LOG = os.path.join("data", "processed", "field_v2", "platform_data_log.txt")


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%H:%M')} {msg}\n")


if __name__ == "__main__":
    summary = os.path.join("data", "processed", "field_v2", "area_test_summary_log.txt")
    while not (os.path.exists(summary) and open(summary, errors="replace").read().count("RESULT") >= 2):
        time.sleep(60)
    import fv2_produce
    log(f"area test done; uncertainty {fv2_produce.uncertainty_table()}")
    log(f"design adopted: lai {fv2_produce.design_adopted('lai')}, ndvi {fv2_produce.design_adopted('ndvi')}")
    for t in ("lai", "ndvi"):
        if not os.path.exists(os.path.join("data", "processed", "field_v2", "models", f"{t}_re_V2.joblib")):
            fv2_produce.train_production(t)
        log(f"models {t} ready")
    import fv2_weekly
    from areas import AREAS
    for a in AREAS:
        for y in range(2021, 2026):
            fv2_weekly.weekly_pixels(a, y)
            log(f"weekly {a} {y} done")
    import fv2_platform
    for a in AREAS:
        for y in range(2021, 2026):
            fv2_platform.export(a, y)
            log(f"export {a} {y} done")
    fv2_platform.write_index()
    fv2_platform.write_districts()
    log("ALL DONE")
