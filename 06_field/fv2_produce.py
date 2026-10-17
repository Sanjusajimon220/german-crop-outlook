"""Production fusion models and the uncertainty table for the field platform (plan 2026-10-17, Task 1).

train_production(target)  V2 live + re-analysis models on all clear values of all 20 area-seasons (nothing hidden),
                          <= 250,000 rows per area-season and model -> data/processed/field_v2/models/
uncertainty_table()       pooled RMSE from the area test (whole S2 dates hidden, 20 area-seasons) per target and gap
                          class -> data/processed/field_v2/uncertainty_V2.json
design_adopted(target)    reads the area test's RESULT line (DESIGN ADOPTED / NOT adopted)
Run `python fv2_produce.py train` from D:\\vista after the area test has finished.
"""
import glob
import json
import os
import re
import sys

import joblib
import numpy as np
import pandas as pd

import phase11_fusion_test as F
from areas import AREAS

OUT = os.path.join("data", "processed", "field_v2")
PARTS = os.path.join(OUT, "area_parts")
SUMMARY_LOG = os.path.join(OUT, "area_test_summary_log.txt")
YEARS = range(2021, 2026)
NDVI_MEASURED_SD = 0.02                      # assumed sensor noise of a clear-day NDVI value (labelled as assumed)
KEYS = ("line_le15", "line_16_30", "radar_gt30", "live_le30", "live_gt30")


def _rmse(v):
    return float(np.sqrt(np.mean(np.square(v))))


def uncertainty_table():
    out = {}
    for t in ("lai", "ndvi"):
        files = sorted(glob.glob(os.path.join(PARTS, f"{t}_*.csv.gz")))
        if len(files) < 20:
            raise RuntimeError(f"area test incomplete for {t}: {len(files)} of 20 area-seasons")
        e = pd.concat([pd.read_csv(f, usecols=["obs", "line", "live", "reanalysis", "gap_s2"]) for f in files])
        g = e["gap_s2"]
        out[t] = {"line_le15": _rmse((e.line - e.obs)[g <= 15]), "line_16_30": _rmse((e.line - e.obs)[(g > 15) & (g <= 30)]),
                  "radar_gt30": _rmse((e.reanalysis - e.obs)[g > 30]), "live_le30": _rmse((e.live - e.obs)[g <= 30]),
                  "live_gt30": _rmse((e.live - e.obs)[g > 30])}
    json.dump(out, open(os.path.join(OUT, "uncertainty_V2.json"), "w"), indent=1)
    return out


def design_adopted(target, log=SUMMARY_LOG):
    for line in open(log, encoding="utf-8", errors="replace"):
        if line.startswith(f"RESULT {target}:"):
            return "DESIGN ADOPTED" in line
    raise RuntimeError(f"no RESULT line for {target} in {log}")


def train_production(target):
    rng = np.random.default_rng(7)
    F.N_SEASONS = 13                                           # 3 M / 12 -> 250,000 rows per area-season and model
    L, R = [], []
    for a in AREAS:
        for y in YEARS:
            S = F.Season(a, y, target, landsat=False, date_hide=True, hide_share=0.0)
            XL, yL, _, XR, yR, _ = S.training(rng)
            L.append((XL, yL)); R.append((XR, yR))
            del S
    models = []
    for name, parts in (("live", L), ("re", R)):
        X = pd.concat([p[0] for p in parts], ignore_index=True)
        y = np.concatenate([p[1] for p in parts])
        F.CAP = len(X)                                         # all rows (<= 250,000 per area-season already)
        m = F.fit(X, y, rng, "V2")
        os.makedirs(os.path.join(OUT, "models"), exist_ok=True)
        joblib.dump(m, os.path.join(OUT, "models", f"{target}_{name}_V2.joblib"))
        models.append(m)
    return tuple(models)


if __name__ == "__main__" and sys.argv[1:] == ["train"]:
    print(uncertainty_table())
    for t in ("lai", "ndvi"):
        train_production(t)
        print(t, "models saved", flush=True)
