"""Pre-registered YieldSAT field-level test (docs/project_log.md, 2026-10-06). One look.
Run `python phase4_yieldsat_score.py` after phase4_yieldsat_run.py.
"""
import os

import numpy as np
import pandas as pd

YS = os.path.join("data", "processed", "yieldsat")


def main():
    f = pd.read_csv(os.path.join(YS, "field_results_v5.csv"), dtype={"district": str})
    meta = pd.read_csv(os.path.join("data", "raw", "yieldsat", "wheat_metadata.csv"))
    f = f.merge(meta[["field_shared_name", "yield_ground_truth", "yieldmap_quality"]], left_on="field",
                right_on="field_shared_name")
    f["truth"] = f["yield_ground_truth"] * 0.85 / 0.86                # 15 % -> 14 % moisture
    y = pd.read_csv(os.path.join("data", "processed", "yields_long.csv"), dtype={"district": str})
    y = y[y["crop"] == "winter_wheat"][["district", "year", "yield_t_ha"]].rename(columns={"yield_t_ha": "official"})
    f = f.merge(y, on=["district", "year"], how="left")
    # B2: satellite index, leave-one-year-out
    f["b2_satindex"] = np.nan
    for yr in f["year"].unique():
        tr = f[(f["year"] != yr) & f["glad_after"].notna()]
        te = (f["year"] == yr) & f["glad_after"].notna()
        a, b = np.polyfit(tr["glad_after"], tr["truth"], 1)
        f.loc[te, "b2_satindex"] = a * f.loc[te, "glad_after"] + b
    f["b2_satindex"] = f["b2_satindex"].fillna(f["blend"])
    preds = {"v5 field model": "v5_yield_t_ha", "B1 district model": "blend",
             "B2 satellite index (LOYO)": "b2_satindex", "B3 official district yield (oracle)": "official"}
    f["dy"] = f["district"] + "_" + f["year"].astype(str)
    rng = np.random.default_rng(0)

    def table(s, label):
        print(f"\n{label}: {len(s)} fields, {s['dy'].nunique()} district-years, truth mean {s['truth'].mean():.2f} t/ha")
        for name, col in preds.items():
            e = s[col] - s["truth"]
            wr = [g[col].corr(g["truth"]) for _, g in s.groupby("dy") if len(g) >= 4 and g[col].std() > 0]
            print(f"  {name:36s} RMSE {np.sqrt((e ** 2).mean()):.2f}  mean error {e.mean():+.2f}  "
                  f"r {s[col].corr(s['truth']):.2f}  within district-year r {np.nanmean(wr) if wr else np.nan:.2f} "
                  f"({len(wr)} groups)")
        dys = s["dy"].unique()
        for base in ("blend", "b2_satindex"):
            diffs = []
            for _ in range(2000):
                pick = rng.choice(dys, len(dys))
                b = pd.concat([s[s["dy"] == k] for k in pick])
                diffs.append(np.sqrt(((b["v5_yield_t_ha"] - b["truth"]) ** 2).mean())
                             - np.sqrt(((b[base] - b["truth"]) ** 2).mean()))
            lo, hi = np.percentile(diffs, [5, 95])
            print(f"  v5 minus {base}: RMSE difference 90 % interval {lo:+.2f} .. {hi:+.2f}")

    table(f[f["year"] >= 2018], "MAIN: harvests 2018-2022")
    table(f[f["year"] <= 2017], "Secondary: 2016-2017 (district anchor in-sample)")
    for q, s in f[f["year"] >= 2018].groupby("yieldmap_quality"):
        e = s["v5_yield_t_ha"] - s["truth"]
        print(f"  map quality {q}: n {len(s)}, v5 RMSE {np.sqrt((e ** 2).mean()):.2f}, "
              f"B1 RMSE {np.sqrt(((s['blend'] - s['truth']) ** 2).mean()):.2f}")
    f.to_csv(os.path.join(YS, "yieldsat_scores.csv"), index=False)


if __name__ == "__main__":
    main()
