"""v10b ranges for the frozen v10 forecasts (pre-registered 2026-10-09): as phase2_ranges_v10.py, but
  forecasts  forecast_<crop>_v10_leadNN.csv (district updates, barley refits)
  year part  from the rolling forward errors 1991-2017 (v8 cuts 1990/1996/2002/2008) after the annual
             district offsets (errors <= t-2, k = 3), i.e. 27 years instead of 9
  local part v9 forward errors 2009-2017 minus offsets learned from that rolling history (<= t-2)
Original description (method B, chosen on training years: eval_ranges_v10.py, project log 2026-10-08).

Forecast distribution for every test year and forecast date from K joint draws:
  weather   one of the 39 weather scenarios, the SAME for all districts in a draw (weather is shared)
  year      a year-wide model error shared by all districts: bias + sd_year * sqrt(1 + 1/n) * t(n - 1)
            from the v9 forward errors 2009-2017 (fitted <= 2008; n = 9 years)
  local     each district's own error, drawn from the local part of the same forward errors; for
            district-years in held-out states ('test_both') widened by the cross-validation spread
            ratio (as in v8)
Outputs per district-year and lead: median, 10/90 % range, P(yield < district's previous 5-year
official mean); per state and Germany the same from the area-weighted draws (areas: latest census).
Writes data/processed/ranges_v10/ranges_<crop>_lead<NN>.csv (district, state, national rows).
Scoring is separate (score_ranges_v10.py) and is the one test look.
"""
import os
import sys

import numpy as np
import pandas as pd

from phase2_forecast import LEADS_WEEKS, OUT
from phase2_intervals import training_residuals
from phase2_growth import FORWARD_WEIGHT, P, VTAG
from phase5_benchmarks import census_area

K = 2000
DEST = os.path.join(P, "ranges_v10b")


def rolling(crop):
    b = "_potato_canopy" if crop == "potato" else ""
    parts = []
    for cut, lo, hi in ((1990, 1991, 1996), (1996, 1997, 2002), (2002, 2003, 2008), (2008, 2009, 2017)):
        f = pd.read_csv(os.path.join(P, f"forward_predictions_{crop}_v8{b}_cut{cut}.csv"), dtype={"district": str})
        parts.append(f[f["year"].between(lo, hi)])
    f = pd.concat(parts).dropna(subset=["yield_t_ha", "pred"])
    return pd.DataFrame({"district": f["district"].values, "year": f["year"].values, "e": (f["pred"] - f["yield_t_ha"]).values})


def offsets_upto(hist, t):
    h = hist[hist["year"] <= t - 2]
    loc = h["e"] - h.groupby("year")["e"].transform("mean")
    return loc.groupby(h["district"]).sum() / (loc.groupby(h["district"]).count() + 3)


def forward_errors(crop):
    """Year means (27 years, after offsets) and local errors (v9 2009-2017 after offsets); errors here are
    observed - predicted as in phase2_ranges_v10 (pred - obs negated)."""
    roll = rolling(crop)
    corr = []
    for t, g in roll.groupby("year"):
        off = offsets_upto(roll, t)
        corr.append(g["e"].values - g["district"].map(off).fillna(0).values)
    roll["ec"] = np.concatenate(corr)
    year_means = -roll.groupby("year")["ec"].mean().values
    name = f"forward_predictions_{crop}{'_potato_canopy' if crop == 'potato' else ''}_modis_cut2008.csv"
    f = pd.read_csv(os.path.join(P, name), dtype={"district": str}).dropna(subset=["yield_t_ha", "pred"])
    f = f.sort_values("year", kind="stable").reset_index(drop=True)
    off = np.concatenate([f.loc[f["year"] == t, "district"].map(offsets_upto(roll, t)).fillna(0).values
                          for t in sorted(f["year"].unique())])
    e = (f["pred"] - f["yield_t_ha"]).values - off
    d = pd.DataFrame({"e": -e, "year": f["year"].values})
    local = (d["e"] - d.groupby("year")["e"].transform("mean")).values
    return year_means, local


def main(crop):
    assert VTAG == "_v9", "run with VISTA_WEATHER=v8 VISTA_MODIS=1"
    os.makedirs(DEST, exist_ok=True)
    rng = np.random.default_rng(0)
    year_means, local = forward_errors(crop)
    n = len(year_means)
    w = FORWARD_WEIGHT.get(crop, 0.8)
    r_years, r_both = training_residuals(crop, w, "years"), training_residuals(crop, w, "both")
    ratio = np.std(r_both) / np.std(r_years)
    print(f"{crop}: year sd {year_means.std(ddof=1):.2f} (n={n}, bias {year_means.mean():+.2f}), local sd "
          f"{local.std():.2f}, both-new local x{ratio:.2f}", flush=True)
    y = pd.read_csv(os.path.join(P, "yields_long.csv"), dtype={"district": str})
    y = y[y["crop"] == crop].copy()
    y["area"] = census_area(y)
    hist = y.set_index(["district", "year"])["yield_t_ha"]
    area = y.set_index(["district", "year"])["area"]
    for lead in LEADS_WEEKS:
        f = pd.read_csv(os.path.join(OUT, f"forecast_{crop}_v10_lead{lead:02d}.csv"), dtype={"district": str})
        rows = []
        for yr, g in f.groupby("year"):
            sc = g.pivot_table(index="district", columns="scenario", values="blend")
            dists = sc.index.values
            role = g.drop_duplicates("district").set_index("district")["role"].reindex(dists).values
            obs = g.drop_duplicates("district").set_index("district")["obs"].reindex(dists).values
            s_idx = rng.integers(0, sc.shape[1], K)
            yr_err = year_means.mean() + year_means.std(ddof=1) * np.sqrt(1 + 1 / n) * rng.standard_t(n - 1, (K, 1))
            loc = rng.choice(local, size=(K, len(dists)))
            loc = np.where(role[None, :] == "test_both", loc * ratio, loc)
            dr = sc.values.T[s_idx] + yr_err + loc                       # K x districts
            thr = np.array([np.nanmean([hist.get((d, yr - k), np.nan) for k in range(1, 6)]) for d in dists])
            q = np.percentile(dr, [10, 50, 90], axis=0)
            for i, d in enumerate(dists):
                rows.append(dict(level="district", region=d, year=yr, role=role[i], obs=obs[i], median=q[1, i],
                                 low80=q[0, i], high80=q[2, i], normal=thr[i],
                                 p_below=(dr[:, i] < thr[i]).mean() if np.isfinite(thr[i]) else np.nan,
                                 pit=(dr[:, i] < obs[i]).mean() if np.isfinite(obs[i]) else np.nan))
            a = np.array([area.get((d, yr), np.nan) for d in dists])
            a = np.nan_to_num(a)
            states = np.array([d[:2] for d in dists])
            for level, keys in (("national", np.array(["DE"] * len(dists))), ("state", states)):
                for key in np.unique(keys):
                    s = (keys == key) & (a > 0)
                    if s.sum() == 0:
                        continue
                    wt = a[s] / a[s].sum()
                    ad = dr[:, s] @ wt
                    has = s & np.isfinite(obs)
                    ao = float(obs[has] @ (a[has] / a[has].sum())) if has.sum() and a[has].sum() > 0 else np.nan
                    th = thr[s & np.isfinite(thr)]
                    tw = a[s & np.isfinite(thr)]
                    nt = float(th @ (tw / tw.sum())) if tw.sum() > 0 else np.nan
                    q10, q50, q90 = np.percentile(ad, [10, 50, 90])
                    rows.append(dict(level=level, region=key, year=yr, role="", obs=ao, median=q50, low80=q10,
                                     high80=q90, normal=nt, p_below=(ad < nt).mean(),
                                     pit=(ad < ao).mean() if np.isfinite(ao) else np.nan,
                                     districts=int(s.sum()), obs_area_share=float(a[has].sum() / a[s].sum())))
        pd.DataFrame(rows).to_csv(os.path.join(DEST, f"ranges_{crop}_lead{lead:02d}.csv"), index=False)
        print(f"  lead {lead}: {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
