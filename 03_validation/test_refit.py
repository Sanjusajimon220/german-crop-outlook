"""Pre-registered refit-schedule test (project log 2026-10-09), training years only.
Frozen: cut-2008 model predicts 2009-2017. Refit every 3 years: cut 2008 -> 2009-2011, cut 2011 -> 2012-2014,
cut 2014 -> 2015-2017. Both with the annual district offsets (D: errors up to t-2, year part removed, k = 3;
history = period-A errors 2003-2008 + the same stream's earlier errors). Adopted if 2009-2017 RMSE drops > 1 %."""
import os
import numpy as np
import pandas as pd
P = os.path.join("data", "processed")
K = 3
BASE = {"potato": "_potato_canopy"}


def load(crop, cut, years):
    f = pd.read_csv(os.path.join(P, f"forward_predictions_{crop}_v8{BASE.get(crop, '')}_cut{cut}.csv"), dtype={"district": str})
    return f[f["year"].isin(years)].dropna(subset=["yield_t_ha", "pred"])


def with_offsets(stream, hist):
    allf = pd.concat([hist, stream], ignore_index=True)
    allf["e"] = allf["pred"] - allf["yield_t_ha"]
    allf["e_loc"] = allf["e"] - allf.groupby("year")["e"].transform("mean")
    out = []
    for t, g in stream.groupby("year"):
        h = allf[allf["year"] <= t - 2]
        off = h.groupby("district")["e_loc"].sum() / (h.groupby("district")["e_loc"].count() + K)
        out.append(g["pred"].values - g["district"].map(off).fillna(0).values - g["yield_t_ha"].values)
    return np.concatenate(out)


rows = []
for crop in ("winter_wheat", "winter_barley", "grain_maize", "silage_maize", "potato"):
    hist = load(crop, 2002, range(2003, 2009))
    frozen = load(crop, 2008, range(2009, 2018))
    refit = pd.concat([load(crop, 2008, range(2009, 2012)), load(crop, 2011, range(2012, 2015)),
                       load(crop, 2014, range(2015, 2018))])
    r = lambda e: float(np.sqrt(np.mean(e ** 2)))
    ef, er = frozen["pred"] - frozen["yield_t_ha"], refit["pred"] - refit["yield_t_ha"]
    fo, ro = with_offsets(frozen, hist), with_offsets(refit, hist)
    yb = lambda s, e: " ".join(f"{v:+.2f}" for v in e.groupby(s["year"].values).mean().values)
    rows.append(dict(crop=crop, frozen=r(ef), refit=r(er), frozen_D=r(fo), refit_D=r(ro),
                     change_with_D_pct=100 * (r(ro) / r(fo) - 1), bias_frozen=ef.mean(), bias_refit=er.mean()))
    print(f"{crop:13s} yearly bias frozen: {yb(frozen, ef)}\n{'':13s} yearly bias refit : {yb(refit, er)}")
s = pd.DataFrame(rows).round(3)
s["adopt"] = s["change_with_D_pct"] < -1
s.to_csv(os.path.join(P, "refit_test.csv"), index=False)
print(s.to_string(index=False))
