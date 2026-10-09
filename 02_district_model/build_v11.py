"""Assemble v11 in-season forecasts = v10 + fusion with the latest published JRC MARS forecast.

Rules (project log 2026-10-09, fusion chosen on training years 2009-2017, fusion_train.csv):
- for each crop, forecast year and forecast date (typical harvest day - 7 x lead weeks), take the latest MARS
  bulletin of that year whose information date is on or before the forecast date;
- if fusion is adopted for the crop and the bulletin's season phase (Mar-Jun / Jul / Aug-Oct), the national
  forecast becomes w * ours + (1 - w) * MARS with w = w_ours from the training years (50 % shrinkage);
- every district (and every weather scenario) is multiplied by fused / ours (district pattern kept).
- no adopted fusion or no bulletin yet -> v11 = v10.
Writes data/processed/forecast/forecast_<crop>_v11_leadNN.csv and v11_fusion_applied_<crop>.csv.
"""
import os

import numpy as np
import pandas as pd

import phase5_benchmarks as pb

F = os.path.join(pb.P, "forecast")
LEADS = (12, 8, 6, 4, 2, 0)


def phase(month):
    return "early (Mar-Jun)" if month <= 6 else "mid (Jul)" if month == 7 else "late (Aug-Oct)"


def main(crops=pb.CROPS):
    fus = pd.read_csv(os.path.join(pb.P, "fusion_train.csv"))
    fus = fus[fus["adopt"]].set_index(["crop", "month"])["w_ours"]
    t = pb.truth()
    off = pb.official(t)
    mars = off[off["source"].str.startswith("mars") & (off["region"] == "DE")]
    y = pd.read_csv(os.path.join(pb.P, "yields_long.csv"), dtype={"district": str})
    for crop in crops:
        harvest = int(pd.read_csv(os.path.join(pb.P, f"master_{crop}.csv"), usecols=["doy_harvest"])["doy_harvest"].median())
        yc = y[y["crop"] == crop].copy()
        yc["area"] = pb.census_area(yc)
        area = yc.set_index(["district", "year"])["area"]
        applied = []
        for lead in LEADS:
            f = pd.read_csv(os.path.join(F, f"forecast_{crop}_v10_lead{lead:02d}.csv"), dtype={"district": str})
            med = f.groupby(["district", "year"])["blend"].median().reset_index()
            factor = {}
            for yr, g in med.groupby("year"):
                date = pd.Timestamp(f"{yr}-01-01") + pd.Timedelta(days=harvest - 7 * lead - 1)
                b = mars[(mars["crop"] == crop) & (mars["year"] == yr) & (mars["date"] <= date)].sort_values("date")
                if b.empty:
                    continue
                last = b.iloc[-1]
                key = (crop, phase(last["date"].month))
                if key not in fus.index:
                    continue
                a = np.array([area.get((d, yr), np.nan) for d in g["district"]])
                ok = np.isfinite(a) & (a > 0)
                if not ok.any():
                    continue
                ours = float(np.average(g["blend"].values[ok], weights=a[ok]))
                w = float(fus.loc[key])
                fused = w * ours + (1 - w) * float(last["value"])
                factor[yr] = fused / ours
                applied.append(dict(lead=lead, year=yr, forecast_date=date.date(), mars_date=last["date"].date(),
                                    phase=key[1], w_ours=w, ours_national=ours, mars=float(last["value"]),
                                    fused=fused, factor=fused / ours))
            f["blend"] = f["blend"] * f["year"].map(factor).fillna(1.0)
            f.to_csv(os.path.join(F, f"forecast_{crop}_v11_lead{lead:02d}.csv"), index=False)
        a = pd.DataFrame(applied)
        a.to_csv(os.path.join(F, f"v11_fusion_applied_{crop}.csv"), index=False)
        print(f"{crop}: fusion applied in {len(a)} lead-years" + (f", factor {a['factor'].min():.3f}-{a['factor'].max():.3f}" if len(a) else ""), flush=True)


if __name__ == "__main__":
    import sys
    main(tuple(sys.argv[1:]) or pb.CROPS)
