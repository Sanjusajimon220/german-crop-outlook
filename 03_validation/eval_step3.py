"""v10 step 3: stress check of the maize variants (training years only).

For each variant's forward predictions (fitted <= 2008, years 2009-2017, v8 weather) the same weather
classes as stress_matrix.py (flowering window -10..+20 d: dry / normal / wet by the training-year
thirds of the climatic water balance; hot = 3+ days with Tmax >= 30 C; whole-summer water balance).
Reported per variant: RMSE, mean bias, bias in dry+hot and in dry summers, and the 'stress gap' =
bias(dry+hot) - bias(normal / not hot) (0 = the model reacts to stress as strongly as reality).
Writes data/processed/step3_variants_stress.csv.
"""
import glob
import os

import numpy as np
import pandas as pd

from stress_matrix import STAGE, season_wb, windows

P = os.path.join("data", "processed")


def main():
    w = dict(np.load(os.path.join(P, "weather_daily_v8.npz"), allow_pickle=True))
    codes = list(w["codes"])
    rows = []
    for crop in ("grain_maize", "silage_maize"):
        m = pd.read_csv(os.path.join(P, f"master_{crop}.csv"), dtype={"district": str})
        m = m[m["year"] <= 2017].copy()
        st = STAGE[crop]
        m[st] = m[st].fillna(m.groupby("district")[st].transform("median")).fillna(m[st].median())
        m["wb"], m["hot"] = windows(w, codes, m, st)
        m["swb"] = season_wb(w, codes, m)
        lo, hi = np.nanpercentile(m["wb"], [33.3, 66.7])
        slo = np.nanpercentile(m["swb"], 33.3)
        files = sorted(glob.glob(os.path.join(P, f"forward_predictions_{crop}_v8*_cut2008.csv")))
        for f in files:
            name = os.path.basename(f)[len(f"forward_predictions_{crop}_v8"):-len("_cut2008.csv")].strip("_") or "current"
            p = pd.read_csv(f, dtype={"district": str}).dropna(subset=["yield_t_ha", "pred"])
            p = p.merge(m[["district", "year", "wb", "hot", "swb"]], on=["district", "year"], how="left")
            e = p["pred"] - p["yield_t_ha"]
            dry, wet, hot = p["wb"] < lo, p["wb"] > hi, p["hot"] >= 3
            normal = ~dry & ~wet & ~hot
            rows.append(dict(crop=crop, variant=name, n=len(p), rmse=np.sqrt((e ** 2).mean()), bias=e.mean(),
                             bias_dry_hot=e[dry & hot].mean(), n_dry_hot=int((dry & hot).sum()),
                             rmse_dry_hot=np.sqrt((e[dry & hot] ** 2).mean()),
                             bias_normal=e[normal].mean(), bias_dry_summer=e[p["swb"] < slo].mean(),
                             stress_gap=e[dry & hot].mean() - e[normal].mean()))
    s = pd.DataFrame(rows).round(3)
    s.to_csv(os.path.join(P, "step3_variants_stress.csv"), index=False)
    with pd.option_context("display.width", 220):
        print(s.to_string(index=False))


if __name__ == "__main__":
    main()
