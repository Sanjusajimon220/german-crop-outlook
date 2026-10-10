"""v11.1 FINAL assembly (items 1-4, pre-registered docs/project_log.md 2026-10-10 13:45).

Step a  forecast/forecast_<crop>_v11i_leadNN.csv = v10 in-season forecasts 2018-2025 + v10 in-season 2026
        (check2026/inseason2026_*), then `ranges_v11.py _v11i stateyr` gives v10 ranges with the adopted
        state-year part for 2018-2026 (ranges_v11i_stateyr_final).          -> run: build_v11_1_final.py prepare
Step b  S2 correction (frozen v11_1/model.json) at the five adopted cells: 2019-2023 leave-one-year-out values
        (v11_1/v11_1_<crop>_leadNN.csv), 2024 from DLR-labelled cells, 2025/2026 from classified cells
        (s2_features_cells.csv; only crops that passed the classification check). District medians and ranges
        shift by the correction; state / national medians and ranges shift by the area-weighted mean district
        correction (districts without S2 features: 0).                     -> run: build_v11_1_final.py assemble
Output  data/processed/ranges_v11_1_final/ranges_<crop>_leadNN.csv (+ column s2_corr), 2026 frozen with md5.
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

from build_v11_1 import apply
from phase5_benchmarks import census_area

P = os.path.join("data", "processed")
F = os.path.join(P, "forecast")
CROPS = ("winter_wheat", "winter_barley", "grain_maize", "silage_maize", "potato")
LEADS = (12, 8, 6, 4, 2, 0)
# 2026-10-10 14:24: overlap check FAILED -> no S2 correction for 2024+ (pre-registered consequence);
# S2 corrections only in the 2019-2023 backtest (leave-one-year-out values). Set True only after a DLR refit.
LIVE_S2 = False


def prepare():
    for crop in CROPS:
        for lead in LEADS:
            a = pd.read_csv(os.path.join(F, f"forecast_{crop}_v10_lead{lead:02d}.csv"), dtype={"district": str})
            b = pd.read_csv(os.path.join(P, "check2026", f"inseason2026_{crop}_lead{lead:02d}.csv"), dtype={"district": str})
            b = b.assign(year=2026, role="check_2026", obs=np.nan)[["district", "year", "role", "obs", "scenario", "blend"]]
            pd.concat([a[a["year"] <= 2025][["district", "year", "role", "obs", "scenario", "blend"]], b],
                      ignore_index=True).to_csv(os.path.join(F, f"forecast_{crop}_v11i_lead{lead:02d}.csv"), index=False)
    print("v11i forecast files written; next: ranges_v11.py _v11i stateyr")


def assemble():
    model = json.load(open(os.path.join(P, "v11_1", "model.json")))
    cells = (pd.read_csv(os.path.join(P, "s2_features_cells.csv"), dtype={"district": str, "state": str})
             if LIVE_S2 else pd.DataFrame(columns=["crop", "lead", "district", "year"]))
    y = pd.read_csv(os.path.join(P, "yields_long.csv"), dtype={"district": str})
    out = os.path.join(P, "ranges_v11_1_final")
    os.makedirs(out, exist_ok=True)
    for crop in CROPS:
        yc = y[y["crop"] == crop].copy()
        yc["area"] = census_area(yc)
        alast = yc.dropna(subset=["area"]).sort_values("year").groupby("district")["area"].last()
        for lead in LEADS:
            r = pd.read_csv(os.path.join(P, "ranges_v11i_stateyr_final", f"ranges_{crop}_lead{lead:02d}.csv"), dtype={"region": str})
            r["s2_corr"] = 0.0
            key = f"{crop}_lead{lead:02d}"
            if key in model["cells"]:
                dist = r["level"] == "district"
                loyo = pd.read_csv(os.path.join(P, "v11_1", f"v11_1_{key}.csv"), dtype={"district": str})
                loyo = loyo[loyo["year"].between(2019, 2023)]      # leave-one-year-out values of the fit years
                c = loyo.set_index(["district", "year"])["corr"]
                f = cells[(cells["crop"] == crop) & (cells["lead"] == lead)] if LIVE_S2 else cells.iloc[:0]
                if len(f):
                    c = pd.concat([c, pd.Series(apply(model["cells"][key], f), index=pd.MultiIndex.from_frame(f[["district", "year"]]))])
                c = c[~c.index.duplicated(keep="first")]
                r.loc[dist, "s2_corr"] = [c.get((d, t), 0.0) for d, t in zip(r.loc[dist, "region"], r.loc[dist, "year"])]
                dd = r[dist].assign(a=r.loc[dist, "region"].map(alast)).dropna(subset=["a"])
                for lvl, k in (("state", dd["region"].str[:2]), ("national", pd.Series("DE", index=dd.index))):
                    agg = dd.assign(k=k).groupby(["k", "year"]).apply(lambda g: np.average(g["s2_corr"], weights=g["a"]))
                    m = r["level"] == lvl
                    r.loc[m, "s2_corr"] = [agg.get((g, t), 0.0) for g, t in zip(r.loc[m, "region"], r.loc[m, "year"])]
            for col in ("median", "low80", "high80"):
                r[col] = r[col] + r["s2_corr"]
            r.to_csv(os.path.join(out, f"ranges_{crop}_lead{lead:02d}.csv"), index=False)
        print(crop, "assembled", flush=True)
    z = pd.concat([pd.read_csv(f, dtype={"region": str}).assign(file=os.path.basename(f))
                   for f in sorted(glob.glob(os.path.join(out, "ranges_*.csv")))])
    z26 = z[z["year"] == 2026]
    z26.to_csv(os.path.join(P, "check2026", "v11_1_2026_frozen.csv"), index=False)
    md5 = hashlib.md5(open(os.path.join(P, "check2026", "v11_1_2026_frozen.csv"), "rb").read()).hexdigest()
    open(os.path.join(P, "check2026", "frozen_2026_v11_1_md5.txt"), "w").write(md5 + "  v11_1_2026_frozen.csv\n")
    print("2026 frozen:", len(z26), "rows, md5", md5)


if __name__ == "__main__":
    {"prepare": prepare, "assemble": assemble}[sys.argv[1]]()
