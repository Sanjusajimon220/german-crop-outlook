"""Pre-registered 2026 check, step 1: district predictions for harvest 2026 (no 2026 yield is used).

Run with VISTA_WEATHER=v8_2026 (v8) or additionally VISTA_MODIS=1 (v9). Rules (docs/project_log.md):
  rows       every district with the crop in the master table; year 2026, no yield
  weather    weather_daily_v8_2026.npz (HYRAS to 6 Oct, SARAH, station wind, climatology afterwards)
  stages     all predicted by the frozen crop calendar (2026 observations are not yet published);
             sowing = district median (as for any district-year without a sowing observation)
  nitrogen   each district's 2025 value
  model      the frozen final model of the version (physics settings from file; correction and
             boosting refitted on the training rows exactly as in phase2_forecast.fit_final)
  state      area-weighted mean of district predictions, weights = district crop area of the most
             recent census year with an area (2020; changed before any look - areas exist only in
             census years)
Writes data/processed/check2026/pred2026_<crop><VTAG>.csv and state_pred2026_<crop><VTAG>.csv.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

import phase2_phenology as phen
from phase0_master import monthly_features
from phase2_forecast import adjust, fit_final
from phase2_growth import FORWARD_WEIGHT, GROWTH, MODIS, N_COLUMN, P, VTAG, load_growth, simulate

OUT = os.path.join(P, "check2026")


def extra_rows(crop):
    m = pd.read_csv(os.path.join(P, f"master_{crop}.csv"), dtype={"district": str})
    last = m.sort_values("year").groupby("district").tail(1).copy()
    keep = [c for c in ("district", "name", "paw_1m_mm", "paw_2m_mm") if c in last.columns]
    e = last[keep].copy()
    e["year"], e["role"], e["fold"] = 2026, "check_2026", -1
    w = phen.weather()
    months = sorted({int(c[-2:]) for c in m.columns if c.startswith("tas_m")})
    mf = monthly_features(w, w["codes"], months, "tas_winter" in m.columns)
    mf = mf.reindex(pd.MultiIndex.from_frame(e[["district", "year"]]))
    for c in mf.columns:
        e[c] = mf[c].values
    return e


def main(crop):
    os.makedirs(OUT, exist_ok=True)
    phen.EXTRA[crop] = extra_rows(crop)
    weight = FORWARD_WEIGHT.get(crop, 0.8)
    model, corr, bst, feats, bench, _ = fit_final(crop)
    m, d = load_growth(crop)
    rows = (m["year"] == 2026).values
    nit = pd.read_csv(os.path.join(P, "nitrogen_districts.csv"), dtype={"district": str})
    n25 = nit[nit["year"] == 2025].set_index("district")[N_COLUMN[crop]]
    n26 = m.loc[rows, "district"].map(n25).fillna(n25.median()).values
    sub = {k: v[rows] for k, v in d.items() if v.dim() and len(v) == len(m)}
    sub["n_supply"] = torch.tensor(n26, dtype=torch.float32)
    mt = m[rows].reset_index(drop=True)
    mt["n_crop_kg_ha"] = n26
    sim = adjust(simulate(model, sub, GROWTH[crop]["dry_matter"]), model, mt["year"].values, crop)
    stages, gate, vp, _ = phen.CROPS[crop]
    cal = phen.Calendar(len(stages), stages.index(gate), vp,
                        json.load(open(os.path.join(P, f"phenology_model_{crop}.json"))))
    with torch.no_grad():
        days = cal(sub).numpy()
    for k, st in enumerate(stages):
        if f"doy_{st}" in mt.columns:
            mt[f"doy_{st}"] = mt["doy_sowing"] + days[:, k]
    tr = m[(m["role"] == "train")]
    if "doy_harvest" in mt.columns:
        mt["doy_harvest"] = mt["district"].map(tr.groupby("district")["doy_harvest"].median())
    mm = pd.concat([mt, sim], axis=1)
    phys = mm["phys_yield"].clip(lower=0.5).values
    hybrid = phys * np.exp(corr.predict(mm[feats]))
    b = bst.predict(mm[bench].values.astype(float))
    p = pd.DataFrame({"district": mt["district"], "physics": phys, "hybrid": hybrid, "boosting": b,
                      "blend": weight * hybrid + (1 - weight) * b, "water_ratio": sim["phys_water_ratio"]})
    p.to_csv(os.path.join(OUT, f"pred2026_{crop}{VTAG}.csv"), index=False)
    aggregate(crop, VTAG)


def aggregate(crop, vtag):
    """State means of the district predictions, weighted by the district's crop area in the most
    recent census year with an area (2020; official areas exist only in census years)."""
    p = pd.read_csv(os.path.join(OUT, f"pred2026_{crop}{vtag}.csv"), dtype={"district": str})
    y = pd.read_csv(os.path.join(P, "yields_long.csv"), dtype={"district": str})
    y = y[(y["crop"] == crop) & (y["area_ha"] > 0)].sort_values("year")
    area = y.groupby("district")["area_ha"].last()
    p["area"] = p["district"].map(area)
    p["state"] = p["district"].str[:2]
    q = p.dropna(subset=["area"])
    s = q.groupby("state").apply(lambda g: pd.Series({c: np.average(g[c], weights=g["area"])
                                                      for c in ("physics", "hybrid", "boosting", "blend")}
                                                     | {"area_ha": g["area"].sum(), "districts": len(g)}),
                                 include_groups=False)
    s.round(3).to_csv(os.path.join(OUT, f"state_pred2026_{crop}{vtag}.csv"))
    print(f"{crop}{vtag}: {len(p)} districts ({len(q)} with area), national area-weighted blend "
          f"{np.average(q['blend'], weights=q['area']):.2f} t/ha, boosting "
          f"{np.average(q['boosting'], weights=q['area']):.2f}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "aggregate":
        aggregate(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
    else:
        main(sys.argv[1])
