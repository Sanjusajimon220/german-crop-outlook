"""Per-field classes for the study-area map (owner decision 2026-10-22; definitions fixed before computing).

1 Productivity vs neighbours: pixel season sum of green LAI / median of the same crop group in the area that season; field
  value = mean over its non-headland pixels and arable seasons (>= 3); thirds per area = low / average / high.
2 Within-field variability from Module A (zones.json): uniform / variable, reliable zones / variable, less stable zones /
  not assessed.
Run from D:\\vista after phase18 / fv2_zones_export: python phase20_field_classes.py -> docs/field/<area>_2025/classes.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DOCS = r"D:\german-crop-outlook-field\docs\field"
OUT = r"D:\vista\data\processed\field_v2\zones"
SEASONS = list(range(2021, 2026))
MIN_SEASONS, GRID = 3, 800


def area_relative(s, groups, arable):
    out = np.full(len(s), np.nan)
    for g in set(groups) & set(arable):
        m = (groups == g) & np.isfinite(s)
        if m.any():
            out[m] = s[m] / np.median(s[m])
    return out


def tercile_class(values):
    ids = sorted(values, key=lambda k: values[k])
    n = len(ids)
    return {k: ("low", "average", "high")[min(2, (3 * i) // n)] for i, k in enumerate(ids)}


def variability_class(m):
    if not m or not m.get("tests") or m.get("spread") is None:
        return "not assessed"
    if m["spread"] < 0.05:
        return "uniform"
    return "variable, reliable zones" if m["held"] / m["tests"] >= 0.8 else "variable, less stable zones"


def pixel_variability(m):
    """m: arable seasons x field pixels (relative growth). Field spread = median of the seasonal pixel standard deviations;
    stability = median over seasons of the Pearson correlation with the mean map of the other seasons. Needs >= 3 seasons."""
    if m.shape[0] < MIN_SEASONS:
        return None, None
    spread = float(np.median([np.nanstd(r) for r in m]))
    cors = []
    for j in range(m.shape[0]):
        other = np.nanmean(np.delete(m, j, axis=0), axis=0)
        ok = np.isfinite(m[j]) & np.isfinite(other)
        if ok.sum() >= 3 and np.std(m[j][ok]) > 0 and np.std(other[ok]) > 0:
            cors.append(np.corrcoef(m[j][ok], other[ok])[0, 1])
    return spread, (float(np.median(cors)) if cors else None)


def pixel_class(spread, stability, uniform=0.025, stable=0.5):
    if spread is None or stability is None:
        return "not assessed"
    if spread < uniform:
        return "uniform"
    return "variable, stable pattern" if stability >= stable else "variable, less stable pattern"


def loso_rank_check(mat):
    """mat: fields x seasons (NaN allowed). Spearman across fields of the held-out season vs the mean of the others."""
    out = {}
    for j in range(mat.shape[1]):
        other = np.nanmean(np.delete(mat, j, axis=1), axis=1)
        ok = np.isfinite(mat[:, j]) & np.isfinite(other)
        if ok.sum() >= 3:
            out[j] = float(pd.Series(other[ok]).corr(pd.Series(mat[ok, j]), method="spearman"))
    return out


def run():
    from areas import out_dir
    from fv2_platform import ARABLE
    from phase18_zones import _season_grid
    checks = []
    for area in ["pilot", "selhausen", "north", "south"]:
        S = {y: _season_grid(area, y) for y in SEASONS}
        last = SEASONS[-1]
        px = pd.read_csv(os.path.join(out_dir(area, last), "pixels_nrw.csv.gz"))
        head = set((px.row[px.headland] * GRID + px.col[px.headland]).tolist())
        F = S[last]["field"]
        rel = {y: area_relative(S[y]["sum"], S[y]["pgroup"], ARABLE) for y in SEASONS}
        zmeta = json.load(open(os.path.join(DOCS, f"{area}_{last}", "zones.json"), encoding="utf-8"))["fields"]
        geo = json.load(open(os.path.join(DOCS, f"{area}_{last}", "fields.geojson"), encoding="utf-8"))
        prod, mat, ids, pv = {}, [], [], {}
        for ft in geo["features"]:
            f, p = int(ft["properties"]["id"]), ft["properties"]
            if p.get("partial"):
                continue
            P = np.array([k for k in np.flatnonzero(F == f) if k not in head], np.int64)
            if len(P) < 10:
                continue
            per = [np.nanmean(rel[y][P]) if np.isfinite(rel[y][P]).mean() >= 0.5 else np.nan for y in SEASONS]
            if (p.get("ha") or 0) >= 2.0 and len(P) >= 30:
                counted = [y for y in SEASONS if S[y]["arable"][P].mean() >= 0.5]
                if len(counted) >= MIN_SEASONS:
                    pv[f] = pixel_variability(np.vstack([S[y]["rel"][P] for y in counted]))
            if np.isfinite(per).sum() >= MIN_SEASONS:
                prod[f] = float(np.nanmean(per)); mat.append(per); ids.append(f)
        cls = tercile_class(prod)
        out = {}
        for ft in geo["features"]:
            f = int(ft["properties"]["id"])
            sp, st = pv.get(f, (None, None))
            out[str(f)] = {"prod": round(prod[f], 3) if f in prod else None, "prod_class": cls.get(f, "not assessed"),
                           "var_class": variability_class(zmeta.get(str(f))),          # zone-based (owner: option A, 2026-10-22)
                           "var_pixel": pixel_class(sp, st), "spread": None if sp is None else round(sp, 4),
                           "stability": None if st is None else round(st, 3)}
        with open(os.path.join(DOCS, f"{area}_{last}", "classes.json"), "w", encoding="utf-8") as fh:
            json.dump({"season": last, "seasons": "2021-2025", "fields": out}, fh, separators=(",", ":"), allow_nan=False)
        for j, r in loso_rank_check(np.array(mat)).items():
            checks.append(dict(area=area, held_out=SEASONS[j], spearman=round(r, 3), fields=len(ids)))
        vc = pd.Series([v["var_class"] for v in out.values()]).value_counts().to_dict()
        both = pd.DataFrame([(v["var_pixel"], v["var_class"]) for v in out.values() if v["var_pixel"] != "not assessed" and v["var_class"] != "not assessed"], columns=["pixel", "zone"])
        agree = ((both.pixel == "uniform") == (both.zone == "uniform")) & ((both.pixel == "variable, stable pattern") == (both.zone == "variable, reliable zones"))
        print(area, "agreement pixel vs zone class:", f"{agree.mean():.1%}", "of", len(both), "fields")
        print(pd.crosstab(both.pixel, both.zone).to_string(), flush=True)
        print(area, "productivity classes for", len(prod), "fields;", vc, flush=True)
    c = pd.DataFrame(checks)
    c.to_csv(os.path.join(OUT, "field_productivity_check.csv"), index=False)
    print("Ranking check (Spearman across fields, held-out season vs the other seasons):\n", c.pivot(index="area", columns="held_out", values="spearman").to_string())
    print("CLASSES DONE")


if __name__ == "__main__":
    run()
