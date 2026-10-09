"""Step 11: standard district outputs per crop (what a platform shows), every district-year 1979-2025.

  yield_pred_t_ha      final model (blend) - for test years an out-of-sample prediction
  yield_low80/high80   80 % range: prediction + 10/90 % of the out-of-fold training errors
                       (leave-year-blocks-out, phase2_intervals.training_residuals)
  normal_t_ha          the district's mean OFFICIAL yield of the previous 5 years (known before the
                       season); deviation_t_ha / deviation_pct = prediction - normal
  maturity_date_pred   ripening (wheat/barley: yellow ripeness; maize: dough ripeness) from the frozen
                       crop calendar run on that year's weather
  harvest_date_pred    ripening + the district's typical gap to harvest (median over training years)
  water_demand_mm      crop water demand over the season (simulated potential evapotranspiration)
  water_use_mm         simulated actual evapotranspiration; water_ratio = use / demand (1 = no stress)
  official_t_ha        official yield where published (for comparison)
Writes data/processed/outputs/district_outputs_<crop>.csv. Run `python phase2_outputs.py winter_wheat`.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

import phase2_phenology as phen
from phase2_growth import GROWTH, P, VTAG
from phase2_intervals import training_residuals

OUT = os.path.join(P, "outputs")


def main(crop, weight=0.8):
    os.makedirs(OUT, exist_ok=True)
    pred = pd.read_csv(os.path.join(P, f"growth_predictions_{crop}{VTAG}.csv"), dtype={"district": str})
    if VTAG:        # v8 range method: errors when predicting the future (forward 2009-2017)
        from phase2_intervals import forward_pool
        res = forward_pool(crop)
    else:
        res = training_residuals(crop, weight, "years")
    lo, hi = np.percentile(res, [10, 90])
    # calendar: ripening date per district-year
    stages, gate, vp, _ = phen.CROPS[crop]
    m, d = phen.load(crop)
    cal = phen.Calendar(len(stages), stages.index(gate), vp,
                        json.load(open(os.path.join(P, f"phenology_model_{crop}.json"))))
    with torch.no_grad():
        days = np.concatenate([cal({k: v[a:a + 2000] for k, v in d.items() if v.dim() and len(v) == len(m)}).numpy()
                               for a in range(0, len(m), 2000)])
    mat_stage = GROWTH[crop]["maturity"]
    if mat_stage not in stages:          # potato: calendar ends at flowering; harvest = flowering + typical gap
        mat_stage = GROWTH[crop]["heading"]
    m["doy_maturity_pred"] = m["doy_sowing"] + days[:, stages.index(mat_stage)]
    tr = m["role"] == "train"
    gap = (m["doy_harvest"] - m[f"doy_{mat_stage}"])[tr].groupby(m.loc[tr, "district"]).median()
    m["doy_harvest_pred"] = m["doy_maturity_pred"] + m["district"].map(gap).fillna(gap.median())
    to_date = lambda y, doy: (pd.to_datetime(y.astype(str) + "-01-01") + pd.to_timedelta(doy.round() - 1, unit="D")).dt.date
    m["maturity_date_pred"] = to_date(m["year"], m["doy_maturity_pred"])
    m["harvest_date_pred"] = to_date(m["year"], m["doy_harvest_pred"])
    o = pred.merge(m[["district", "year", "maturity_date_pred", "harvest_date_pred"]], on=["district", "year"], how="left")
    o = o.sort_values(["district", "year"])
    o["normal_t_ha"] = o.groupby("district")["yield_t_ha"].transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
    out = pd.DataFrame({
        "district": o["district"], "year": o["year"], "role": o["role"],
        "yield_pred_t_ha": o["blend"].round(2), "yield_low80": (o["blend"] + lo).round(2),
        "yield_high80": (o["blend"] + hi).round(2), "normal_t_ha": o["normal_t_ha"].round(2),
        "deviation_t_ha": (o["blend"] - o["normal_t_ha"]).round(2),
        "deviation_pct": (100 * (o["blend"] / o["normal_t_ha"] - 1)).round(1),
        "maturity_date_pred": o["maturity_date_pred"], "harvest_date_pred": o["harvest_date_pred"],
        "water_demand_mm": o["phys_etp_mm"].round(0), "water_use_mm": o["phys_eta_mm"].round(0),
        "water_ratio": o["phys_water_ratio"].round(3), "official_t_ha": o["yield_t_ha"]})
    out.to_csv(os.path.join(OUT, f"district_outputs_{crop}{VTAG}.csv"), index=False)
    t = out[out["role"] != "train"]
    print(f"{crop}: {len(out)} district-years; 80 % range = prediction {lo:+.2f} / {hi:+.2f} t/ha")
    print(t.groupby("year")[["yield_pred_t_ha", "normal_t_ha", "deviation_pct", "water_demand_mm",
                             "water_use_mm", "water_ratio"]].median().round(2).to_string())
    hd = pd.to_datetime(t["harvest_date_pred"])
    print("predicted harvest date (median day of year) per test year:", hd.dt.dayofyear.groupby(t["year"]).median().to_dict())


if __name__ == "__main__":
    main(sys.argv[1])
