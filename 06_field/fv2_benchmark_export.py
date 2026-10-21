"""Field benchmark results for the platform's Accuracy page (owner decision 2026-10-21): docs/field/benchmark.json.

Collects the logged, one-look results (area test, fixes 1-2, HR-VPP items 1-4, the fix-2 rebuild) from
data/processed/field_v2 into one small file; nothing is recomputed. Run from D:\\vista: python fv2_benchmark_export.py
"""
import json
import os
import re
import time

import numpy as np
import pandas as pd

FV2 = r"D:\vista\data\processed\field_v2"
BENCH = os.path.join(FV2, "benchmark")
OUT = r"D:\german-crop-outlook-field\docs\field\benchmark.json"


def _r(x, d=3):
    return None if x is None or not np.isfinite(x) else round(float(x), d)


def area_test_rows(summary, target):
    """Gaps > 30 days: line vs adopted design per area-season (wins) and the mean per area."""
    s = summary[(summary.target == target) & (summary.gap == ">30 d")]
    won = int((s.design < s.line).sum())
    areas = [{"area": a, "line": _r(g.line.mean()), "design": _r(g.design.mean())} for a, g in s.groupby("area", sort=True)]
    return {"won": won, "of": int(len(s)), "areas": areas}


def item2_points(t):
    pts = [{"season": int(d.split("|")[1]), "n": int(n), "ours": _r(o), "ppi": _r(p)}
           for d, n, o, p in zip(t.date, t.n, t.rho_ours, t.rho_ppi)]
    return {"won": int((t.rho_ours > t.rho_ppi).sum()), "of": int(len(t)), "median_ours": _r(t.rho_ours.median()),
            "median_ppi": _r(t.rho_ppi.median()), "points": pts}


def item3_boxes(t):
    out = []
    for (crop, event), g in t.groupby(["crop", "event"], sort=True):
        q = g.diff_days.quantile([0.25, 0.5, 0.75]).values
        out.append({"crop": crop, "event": event, "q1": _r(q[0], 1), "median": _r(q[1], 1), "q3": _r(q[2], 1), "n": int(len(g))})
    return out


def item4_series(before, after):
    """Per area: [date, CGLS mean, ours before fix 2, ours after fix 2] over the 300 m cells of each 10-day date."""
    a = before[before["mode"] == "all"].groupby(["area", "date"])["ours"].mean()
    b = after[after["mode"] == "all"].groupby(["area", "date"])[["ours", "cgls"]].mean()
    out = {}
    for (area, date), row in b.iterrows():
        out.setdefault(area, []).append([date, _r(row.cgls), _r(a.get((area, date), np.nan)), _r(row.ours)])
    return out


def _fix1():
    rows, verdict = [], {}
    for line in open(os.path.join(FV2, "fix1_summary_rf_out.txt"), encoding="utf-8"):
        m = re.match(r"\s*(lai|ndvi)\s+(\w+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
        if m:
            rows.append({"target": m[1], "area": m[2], "winter_before": float(m[4]), "winter_after": float(m[5]),
                         "summer_before": float(m[6]), "summer_after": float(m[7])})
        m = re.match(r"(lai|ndvi): Nov-Feb better in (\d)/4 areas.*-> (pass|fail)", line)
        if m:
            verdict[m[1]] = {"winter_better_areas": int(m[2]), "pass": m[3] == "pass"}
    return {"adopted": False, "rows": rows, "verdict": verdict}


def _fix2():
    g, c = pd.read_csv(os.path.join(BENCH, "fix2_groups.csv.gz")), pd.read_csv(os.path.join(BENCH, "fix2_counts.csv"))
    out = {}
    for t in ("lai", "ndvi"):
        gt, ct = g[g.target == t], c[c.target == t]
        share = ct.flagged.sum() / ct.measured.sum()
        f, u = gt[gt.flagged].absdiff.median(), gt[~gt.flagged].absdiff.median()
        out[t] = {"flagged_share": _r(share, 4), "flagged_vs_landsat": _r(f), "unflagged_vs_landsat": _r(u),
                  "ratio": _r(f / u, 2), "adopted": bool(f >= 2 * u and share < 0.02)}
    r = pd.read_csv(os.path.join(BENCH, "fix2_rebuild_summary.csv"))
    out["rebuild"] = {k: _r(r[k].mean(), 2) for k in ("changed_pct", "lifted_pct", "lowered_pct", "mean_change",
                                                       "mean_where_changed", "measured_before", "measured_after")}
    return out


def _item4_stats():
    out = {}
    for tag, f in (("before", "lai300_summary.csv"), ("after", "lai300_summary_after_fix2.csv")):
        s = pd.read_csv(os.path.join(BENCH, f), index_col=0)
        out[tag] = {k: {c: _r(s.loc[k, c]) for c in ("n", "bias", "rmse", "r")} for k in s.index}
    return out


def build():
    summ = pd.read_csv(os.path.join(FV2, "area_test_summary.csv"))
    i1 = pd.read_csv(os.path.join(BENCH, "hrvpp_item1_summary.csv"))
    data = {
        "area_test": {t: area_test_rows(summ, t) for t in ("lai", "ndvi")},
        "uncertainty": json.load(open(os.path.join(FV2, "uncertainty_V2.json"))),
        "fix1": _fix1(),
        "fix2": _fix2(),
        "item1": [{"crop": c, "season": int(s), "rho": _r(r), "n": int(n)} for c, s, r, n in zip(i1.crop, i1.season, i1.spearman, i1.n)],
        "item2": item2_points(pd.read_csv(os.path.join(BENCH, "hrvpp_item2_primary.csv"))),
        "item3": item3_boxes(pd.read_csv(os.path.join(BENCH, "hrvpp_item3_pairs.csv.gz"))),
        "item4": {"stats": _item4_stats(),
                  "series": item4_series(pd.read_csv(os.path.join(BENCH, "lai300_pairs.csv.gz")),
                                         pd.read_csv(os.path.join(BENCH, "lai300_pairs_after_fix2.csv.gz")))},
    }
    with open(OUT + ".part", "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    os.replace(OUT + ".part", OUT)
    idx_path = os.path.join(os.path.dirname(OUT), "index.json")           # version stamp: browsers fetch the new file
    idx = json.load(open(idx_path, encoding="utf-8"))
    idx["benchmark"] = time.strftime("%Y%m%d%H%M%S")
    with open(idx_path + ".part", "w", encoding="utf-8") as f:
        json.dump(idx, f, indent=1)
    os.replace(idx_path + ".part", idx_path)
    return os.path.getsize(OUT)


if __name__ == "__main__":
    print(build(), "bytes")
