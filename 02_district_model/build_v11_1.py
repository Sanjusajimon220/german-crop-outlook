"""v11.1 = v10 + crop-specific Sentinel-2 correction at the five crop x lead cells adopted by the pre-registered
test (s2_eval.py, 2026-10-10): winter_wheat 12 / 6, winter_barley 8, silage_maize 6 / 4 weeks before harvest.

Same model as the test: ridge (alpha 10) on standardised S2 features (peak, last3, integral + state anomalies),
target = v10 district residual, correction = 0.5 * prediction, fitted on 2019-2023 without the held-out states.
Outputs (data/processed/v11_1/):
  model.json                     frozen coefficients per cell (apply to any new year with S2 features)
  v11_1_<crop>_lead<LL>.csv      v10 district rows + leave-one-year-out v11.1 median / range for 2019-2023
                                 (range = v10 range shifted by the correction)
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

P = os.path.join("data", "processed")
OUT = os.path.join(P, "v11_1")
F = ["peak", "last3", "integral", "peak_anom", "last3_anom", "integral_anom"]
HELD = {"01", "08", "12"}
CELLS = [("winter_wheat", 12), ("winter_wheat", 6), ("winter_barley", 8), ("silage_maize", 6), ("silage_maize", 4)]
SHRINK = 0.5


def fit(d):
    sc = StandardScaler().fit(d[F])
    m = Ridge(alpha=10).fit(sc.transform(d[F]), d["res"])
    return dict(mean=sc.mean_.tolist(), scale=sc.scale_.tolist(), coef=m.coef_.tolist(), intercept=float(m.intercept_))


def apply(model, x):
    z = (x[F].values - np.array(model["mean"])) / np.array(model["scale"])
    return SHRINK * (z @ np.array(model["coef"]) + model["intercept"])


def main():
    os.makedirs(OUT, exist_ok=True)
    s2 = pd.read_csv(os.path.join(P, "s2_features.csv"), dtype={"district": str, "state": str})
    models = {}
    for crop, lead in CELLS:
        v = pd.read_csv(os.path.join(P, "ranges_v10_final", f"ranges_{crop}_lead{lead:02d}.csv"), dtype={"region": str})
        v = v[v["level"] == "district"].rename(columns={"region": "district"})
        g = s2[(s2["crop"] == crop) & (s2["lead"] == lead)]
        d = v.merge(g[["district", "year", "state"] + F], on=["district", "year"], how="left")
        d["res"] = d["obs"] - d["median"]
        ok = d[F].notna().all(axis=1) & d["res"].notna()
        train = d[ok & ~d["state"].isin(HELD)]
        models[f"{crop}_lead{lead:02d}"] = dict(crop=crop, lead=lead, n=len(train), years=sorted(train["year"].unique().tolist()),
                                                **fit(train))
        d["corr"] = 0.0
        for y in sorted(train["year"].unique()):
            te = ok & (d["year"] == y)
            d.loc[te, "corr"] = apply(fit(train[train["year"] != y]), d[te])
        for c in ("median", "low80", "high80"):
            d[c + "_v11_1"] = d[c] + d["corr"]
        d.drop(columns=["res"]).to_csv(os.path.join(OUT, f"v11_1_{crop}_lead{lead:02d}.csv"), index=False)
        k = ok & ~d["state"].isin(HELD)
        r = lambda c: np.sqrt(((d.loc[k, c] - d.loc[k, "obs"]) ** 2).mean())
        print(f"{crop:14s} lead {lead:2d}: n {k.sum():4d}  v10 {r('median'):.3f}  v11.1 (LOYO) {r('median_v11_1'):.3f}  "
              f"corr sd {d.loc[ok, 'corr'].std():.3f}")
    mf = os.path.join(OUT, "model.json")
    json.dump(dict(features=F, shrink=SHRINK, alpha=10, cells=models), open(mf, "w"), indent=1)
    md5 = hashlib.md5(open(mf, "rb").read()).hexdigest()
    open(os.path.join(OUT, "model_md5.txt"), "w").write(md5 + "  model.json\n")
    print("frozen model.json md5", md5)


if __name__ == "__main__":
    main()
