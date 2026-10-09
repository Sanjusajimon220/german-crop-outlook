"""Phase 4, step 3b: satellite yield signals per field beyond plain LAI (no yields used).

Per field and season, from our PROSAIL retrieval (data/processed/s2/retrieval_115_<year>.csv),
with heading and maturity of the field's district from the frozen crop calendar:
  lai_heading       green LAI at heading (linear between satellite dates)
  glad_after        green leaf area duration after heading: integral of LAI from heading to
                    heading + 45 days (LAI x days) - light the crop can still catch for the grain
  stay_green_days   days after heading until LAI falls below 1.5 (how long the canopy stays green)
  ccc_heading       canopy chlorophyll (LAI x leaf chlorophyll) around heading (+-10 days, median):
                    linked to canopy nitrogen
  cab_heading       leaf chlorophyll around heading (nitrogen per leaf area)
  lai_peak          highest LAI of the season (90th percentile of the dates, robust to outliers)
Features need a satellite date within 12 days on both sides of the window; otherwise empty.
These are stored for learning once field yields are available (YieldSAT, yield maps); here they
are only compared with the model's field yields (v2) to see what each signal adds.

Also writes v2 field yields (frozen rule, docs/project_log.md): v1 field yields rescaled so that
their area-weighted mean per district equals the district model - the satellite only spreads
the district yield over the fields.
Run `python phase4_field_features.py 2021 2022 2023`.
"""
import os
import sys

import numpy as np
import pandas as pd

from phase2_growth import P, load_growth
from phase4_field_assim import CROP, FD, observations


def district_dates(year):
    m, d = load_growth(CROP)
    out = {}
    for i in m.index[m["year"] == year]:
        sow = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=round(m.at[i, "doy_sowing"]) - 1)
        head = int((d["dev"][i] >= d["th_h"][i]).float().argmax())
        mat = int((d["dev"][i] >= d["th_m"][i]).float().argmax())
        out[m.at[i, "district"]] = (sow + pd.Timedelta(days=head), sow + pd.Timedelta(days=mat))
    return out


def features(g, head):
    t = (g["date"] - head).dt.days.values.astype(float)
    lai, ccc, cab = g["lai"].values, g["ccc"].values, g["cab"].values
    f = {"lai_peak": np.percentile(lai, 90), "n_dates": len(t)}
    if (t <= 0).any() and (t >= 0).any() and min(abs(t[t <= 0]).min(), t[t >= 0].min()) <= 12:
        f["lai_heading"] = np.interp(0, t, lai)
    if (t <= 0).any() and (t >= 45).any() and abs(t[t <= 0]).max() >= 0 and \
            np.diff(np.concatenate([[0], t[(t > 0) & (t < 45)], [45]])).max() <= 24:
        days = np.arange(0, 46)
        f["glad_after"] = np.trapezoid(np.interp(days, t, lai), days)
        below = days[np.interp(days, t, lai) < 1.5]
        f["stay_green_days"] = below.min() if len(below) else 45
    near = np.abs(t) <= 10
    if near.any():
        f["ccc_heading"], f["cab_heading"] = np.median(ccc[near]), np.median(cab[near])
    return f


def main(years):
    pred = pd.read_csv(os.path.join(P, f"growth_predictions_{CROP}.csv"), dtype={"district": str})
    for y in years:
        dates = district_dates(y)
        obs = observations(y).sort_values("date")
        rows = []
        for (fid, dist), g in obs.groupby(["field_id", "district"]):
            if dist in dates:
                rows.append({"field_id": fid, "district": dist, **features(g, dates[dist][0])})
        feat = pd.DataFrame(rows)
        # v2: same field pattern as v1, district mean = district model
        f = pd.read_csv(os.path.join(FD, f"field_results_{CROP}_{y}.csv"), dtype={"district": str})
        blend = pred[pred["year"] == y].set_index("district")["blend"]
        mean_v1 = f.groupby("district").apply(lambda g: np.average(g["yield_t_ha"], weights=g["area_ha"]),
                                              include_groups=False)
        k = f["district"].map(blend) / f["district"].map(mean_v1)
        for c in ("yield_t_ha", "yield_p10", "yield_p90"):
            f[c.replace("yield", "v2_yield")] = f[c] * k
        f = f.merge(feat, on=["field_id", "district"], how="left")
        f.to_csv(os.path.join(FD, f"field_results_v2_{CROP}_{y}.csv"), index=False)
        cols = ["lai_peak", "lai_heading", "glad_after", "stay_green_days", "ccc_heading", "cab_heading"]
        print(f"\n{y}: {len(f)} fields; feature available (%):",
              {c: round(100 * f[c].notna().mean()) for c in cols})
        print("  median:", {c: round(f[c].median(), 2) for c in cols})
        # within-district correlation with the model's field yield (no yields involved)
        z = f[cols + ["v2_yield_t_ha"]].groupby(f["district"]).transform(lambda s: s - s.mean())
        print("  correlation with model field yield (within districts):",
              {c: round(z[c].corr(z["v2_yield_t_ha"]), 2) for c in cols})


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]])
