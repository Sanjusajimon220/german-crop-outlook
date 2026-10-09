"""Forward test (training years only): do district offsets learned from past errors carry over?
Offsets from the forward errors 2003-2008 (model fitted <= 2002), shrunk towards 0:
  offset_d = sum(e_d) / (n_d + k)  (k = 3 fixed a priori: offset halved with 3 years of data)
applied to the 2009-2017 forward predictions (model fitted <= 2008): pred - offset.
Also state offsets (same rule) for comparison. Prints RMSE 2009-2017 before / after."""
import os
import numpy as np
import pandas as pd
P = os.path.join("data", "processed")
K = 3
for crop in ("winter_wheat", "winter_barley", "grain_maize", "silage_maize", "potato"):
    a = pd.read_csv(os.path.join(P, f"forward_predictions_{crop}_cut2002.csv"), dtype={"district": str})
    a = a[a["year"] <= 2008].dropna(subset=["yield_t_ha", "pred"])
    b = pd.read_csv(os.path.join(P, f"forward_predictions_{crop}_cut2008.csv"), dtype={"district": str}).dropna(subset=["yield_t_ha", "pred"])
    a["e"] = a["pred"] - a["yield_t_ha"]
    a["e"] -= a.groupby("year")["e"].transform("mean")          # remove year-wide part (not district specific)
    g = a.groupby("district")["e"]
    off = g.sum() / (g.count() + K)
    st = a.groupby(a["district"].str[:2])["e"]
    soff = st.sum() / (st.count() + K)
    e0 = b["pred"] - b["yield_t_ha"]
    e1 = e0 - b["district"].map(off).fillna(0)
    e2 = e0 - b["district"].str[:2].map(soff).fillna(0)
    r = lambda e: np.sqrt((e ** 2).mean())
    print(f"{crop:13s} RMSE 2009-2017: model {r(e0):.3f}  + district offsets {r(e1):.3f} ({100*(r(e1)/r(e0)-1):+.1f} %)"
          f"  + state offsets {r(e2):.3f} ({100*(r(e2)/r(e0)-1):+.1f} %)  districts with offset {off.notna().sum()}")
