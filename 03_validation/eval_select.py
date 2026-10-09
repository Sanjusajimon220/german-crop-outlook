"""v10 combined test: pre-registered selection (project log 2026-10-08), training years only.

For every crop and combination: period A = fitted <= 2002, scored 2003-2008 (cut2002 file, years <= 2008);
period B = fitted <= 2008, scored 2009-2017 (cut2008 file). Per period: RMSE, stress gap (bias in dry+hot
sensitive window minus bias in normal years; classes as stress_matrix.py from training-year thirds).
Rule: chosen only if RMSE beats the current model in BOTH periods and the stress gap does not grow
(either period, tolerance 0.02); among those, lowest mean RMSE; ties < 1 % -> fewer ingredients.
District offsets: an independent extra step; its effect on period B is measured with offsets learned
from the SAME combination's period-A errors (year part removed, shrinkage k = 3).
Writes data/processed/combined_select.csv and prints the choice per crop.
"""
import glob
import os

import numpy as np
import pandas as pd

from stress_matrix import STAGE, windows

P = os.path.join("data", "processed")
K = 3
BASE = {"potato": "potato_canopy"}


def classes(crop, w, codes):
    m = pd.read_csv(os.path.join(P, f"master_{crop}.csv"), dtype={"district": str})
    m = m[m["year"] <= 2017].copy()
    st = STAGE[crop] if STAGE[crop] in m.columns else "doy_heading"
    m[st] = m[st].fillna(m.groupby("district")[st].transform("median")).fillna(m[st].median())
    m["wb"], m["hot"] = windows(w, codes, m, st)
    lo, hi = np.nanpercentile(m["wb"], [33.3, 66.7])
    m["dry"], m["wet"], m["hotc"] = m["wb"] < lo, m["wb"] > hi, m["hot"] >= 3
    return m[["district", "year", "dry", "wet", "hotc"]]


def scores(p):
    e = p["pred"] - p["yield_t_ha"]
    normal = ~p["dry"] & ~p["wet"] & ~p["hotc"]
    return np.sqrt((e ** 2).mean()), e[p["dry"] & p["hotc"]].mean() - e[normal].mean()


def main():
    w = dict(np.load(os.path.join(P, "weather_daily_v8.npz"), allow_pickle=True))
    codes = list(w["codes"])
    rows = []
    for crop in ("winter_wheat", "winter_barley", "grain_maize", "silage_maize", "potato"):
        cl = classes(crop, w, codes)
        base = BASE.get(crop, "")
        for fb in sorted(glob.glob(os.path.join(P, f"forward_predictions_{crop}_v8*_cut2008.csv"))):
            name = os.path.basename(fb)[len(f"forward_predictions_{crop}_v8"):-len("_cut2008.csv")].strip("_")
            fa = fb.replace("_cut2008.csv", "_cut2002.csv")
            if not os.path.exists(fa):
                continue
            ingred = [x for x in name.split("_") if x] if name else []
            label = name.replace(base, "").strip("_") if base else name
            label = label.replace("drought_heat", "droughtheat")
            ing = [x for x in label.split("_") if x]
            if any(x not in ("soil", "lst", "stressboost", "droughtheat") for x in ing):
                continue                                   # older single-variant files (step 3)
            a = pd.read_csv(fa, dtype={"district": str})
            a = a[a["year"] <= 2008].dropna(subset=["yield_t_ha", "pred"]).merge(cl, on=["district", "year"], how="left")
            b = pd.read_csv(fb, dtype={"district": str}).dropna(subset=["yield_t_ha", "pred"]).merge(cl, on=["district", "year"], how="left")
            ra, ga = scores(a)
            rb, gb = scores(b)
            ea = a["pred"] - a["yield_t_ha"]
            ea = ea - ea.groupby(a["year"]).transform("mean")
            off = ea.groupby(a["district"]).sum() / (ea.groupby(a["district"]).count() + K)
            bo = b.assign(pred=b["pred"] - b["district"].map(off).fillna(0))
            rbo, gbo = scores(bo)
            rows.append(dict(crop=crop, combination="+".join(ing) or "current", n_ingredients=len(ing),
                             rmse_A=ra, rmse_B=rb, gap_A=ga, gap_B=gb, rmse_B_offsets=rbo, gap_B_offsets=gbo))
    s = pd.DataFrame(rows)
    out = []
    for crop, g in s.groupby("crop", sort=False):
        cur = g[g["combination"] == "current"].iloc[0]
        g = g.assign(beats_A=g["rmse_A"] < cur["rmse_A"], beats_B=g["rmse_B"] < cur["rmse_B"],
                     gap_ok=(g["gap_A"] <= cur["gap_A"] + 0.02) & (g["gap_B"] <= cur["gap_B"] + 0.02),
                     mean_rmse=(g["rmse_A"] + g["rmse_B"]) / 2,
                     change_A_pct=100 * (g["rmse_A"] / cur["rmse_A"] - 1), change_B_pct=100 * (g["rmse_B"] / cur["rmse_B"] - 1),
                     change_B_with_offsets_pct=100 * (g["rmse_B_offsets"] / cur["rmse_B"] - 1))
        ok = g[g["beats_A"] & g["beats_B"] & g["gap_ok"] & (g["combination"] != "current")].sort_values("mean_rmse")
        choice = "current"
        if len(ok):
            best = ok.iloc[0]
            near = ok[ok["mean_rmse"] <= best["mean_rmse"] * 1.01].sort_values(["n_ingredients", "mean_rmse"])
            choice = near.iloc[0]["combination"]
        g["chosen"] = g["combination"] == choice
        out.append(g)
    s = pd.concat(out).round(3)
    s.to_csv(os.path.join(P, "combined_select.csv"), index=False)
    cols = ["crop", "combination", "rmse_A", "change_A_pct", "rmse_B", "change_B_pct", "gap_A", "gap_B",
            "change_B_with_offsets_pct", "beats_A", "beats_B", "gap_ok", "chosen"]
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(s[cols].to_string(index=False))


if __name__ == "__main__":
    main()
