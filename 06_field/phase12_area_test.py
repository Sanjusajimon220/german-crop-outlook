"""AREA test (pre-registered docs/project_log.md 2026-10-16 20:46): a new area in a new season.

Step rows: per area-season and target, V2 training rows (live pairs, re-analysis triples) from its kept values (whole-date
hiding), <= 250,000 rows per model -> data/processed/field_v2/area_rows/.
Step test: per test area-season, V2 models trained on the 12 area-seasons of the other 3 areas in the other 4 seasons;
hidden whole S2 dates of the test area-season scored (line, V2 re-analysis, design = line <= 30 d / V2 > 30 d).
Run `python phase12_area_test.py rows | test | summary` (resumable: finished files are skipped).
"""
import os
import sys
import time

import numpy as np
import pandas as pd

import phase11_fusion_test as F
from areas import AREAS

OUT = os.path.join("data", "processed", "field_v2")
ROWS = os.path.join(OUT, "area_rows")
PARTS = os.path.join(OUT, "area_parts")
YEARS = range(2021, 2026)
F.N_SEASONS = 13                                                   # 3 M / 12 = 250,000 rows per area-season and model
DROP = F.VARIANTS["V2"]


def rows_path(t, a, y, m):
    return os.path.join(ROWS, f"{t}_{a}_{y}_{m}.pkl")


def build_rows():
    os.makedirs(ROWS, exist_ok=True)
    for t in ("lai", "ndvi"):
        for a in AREAS:
            for y in YEARS:
                if os.path.exists(rows_path(t, a, y, "re")):
                    continue
                S = F.Season(a, y, t, landsat=False, date_hide=True)
                XL, yL, _, XR, yR, _ = S.training(np.random.default_rng(7))
                for m, X, yy in (("live", XL, yL), ("re", XR, yR)):
                    X = X.drop(columns=[c for c in DROP if c in X])
                    tmp = rows_path(t, a, y, m) + ".part"
                    pd.to_pickle((X, yy), tmp)
                    os.replace(tmp, rows_path(t, a, y, m))
                print(f"{time.strftime('%H:%M')} rows {t} {a} {y}: live {len(XL)}, re-analysis {len(XR)}", flush=True)
                del S


def test():
    os.makedirs(PARTS, exist_ok=True)
    rng = np.random.default_rng(7)
    for t in ("lai", "ndvi"):
        for a in AREAS:
            for y in YEARS:
                out = os.path.join(PARTS, f"{t}_{a}_{y}.csv.gz")
                if os.path.exists(out):
                    continue
                train = [(a2, y2) for a2 in AREAS for y2 in YEARS if a2 != a and y2 != y]
                models = {}
                for m in ("live", "re"):
                    parts = [pd.read_pickle(rows_path(t, a2, y2, m)) for a2, y2 in train]
                    X = pd.concat([p[0] for p in parts], ignore_index=True)
                    yy = np.concatenate([p[1] for p in parts])
                    models[m] = F.fit(X, yy, rng, "V2")
                    del parts, X, yy
                S = F.Season(a, y, t, landsat=False, date_hide=True)
                e = F.evaluate(S, models["live"], models["re"])
                tmp = out + ".part"
                e.assign(target=t, area=a, year=y).to_csv(tmp, index=False, compression="gzip")
                os.replace(tmp, out)
                print(f"{time.strftime('%H:%M')} test {t} {a} {y}: {len(e)} hidden values", flush=True)
                del S


def summary():
    rm = lambda v: float(np.sqrt(np.mean(np.square(v))))
    rows = []
    for t in ("lai", "ndvi"):
        for a in AREAS:
            for y in YEARS:
                f = os.path.join(PARTS, f"{t}_{a}_{y}.csv.gz")
                if not os.path.exists(f):
                    continue
                e = pd.read_csv(f)
                e["design"] = np.where(e.gap_s2 <= 30, e.line, e.reanalysis)
                e["gap"] = pd.cut(e.gap_s2, [0, 15, 30, 10000], labels=["<=15 d", "16-30 d", ">30 d"])
                for g, h in e.groupby("gap", observed=True):
                    rows.append(dict(target=t, area=a, year=y, gap=g, n=len(h), line=rm(h.line - h.obs),
                                     reanalysis=rm(h.reanalysis - h.obs), design=rm(h.design - h.obs)))
    s = pd.DataFrame(rows).round(3)
    s.to_csv(os.path.join(OUT, "area_test_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    for t in ("lai", "ndvi"):
        q = s[s.target == t]
        print(t.upper(), "by area (mean over seasons):")
        print(q.groupby(["area", "gap"], observed=True)[["line", "reanalysis", "design"]].mean().round(3).unstack().to_string())
        long = q[q.gap == ">30 d"]
        by_area = long.groupby("area")[["line", "reanalysis"]].mean()
        n_win = int((long.reanalysis < long.line).sum())
        ok = bool((by_area.reanalysis < by_area.line).all()) and n_win >= 16
        print(f"RESULT {t}: V2 beats the line > 30 d in {int((by_area.reanalysis < by_area.line).sum())}/4 areas and "
              f"{n_win}/{len(long)} area-seasons -> DESIGN {'ADOPTED' if ok else 'NOT adopted'}")


if __name__ == "__main__":
    {"rows": build_rows, "test": test, "summary": summary}[sys.argv[1]]()
