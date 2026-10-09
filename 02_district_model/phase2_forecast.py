"""Phase 2, step 3: in-season yield forecasts and their accuracy against lead time.

At a forecast date F (e.g. 8 weeks before the typical harvest) only the weather up to F is known.
The rest of the season is filled with the weather of each training year 1979-2017 in turn
(39 scenarios, same calendar days), and every model is run on each scenario:
  - crop calendar and growth model (physics, fitted settings of the final model)
  - learned correction and blend (as in the final model)
  - Phase 1 gradient boosting (same scenario weather), the benchmark
Stage dates not yet observed at F are replaced by the calendar's prediction for that scenario.
The forecast is the median over scenarios; the 10-90 % range is the uncertainty.

Scored on the test district-years (new years 2018-2025 and both-new) for several lead times:
RMSE of the median, bias, and how often the observed yield falls inside the 10-90 % range
(should be close to 80 %).

Run `python phase2_forecast.py winter_wheat`. Resumable: one file per forecast date in
data/processed/forecast/; then a summary forecast_skill_<crop>.csv.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

import phase2_phenology as phen
from phase0_master import monthly_features
from phase1_baselines import BOOST, boost, features
from phase2_growth import (MODIS, VTAG, DRIFTING, drifting, GROWTH, N_COLUMN, P, TECH_RULE, Growth, booster, load_growth,
                           simulate)

OUT = os.path.join(P, "forecast")
SCENARIOS = range(1979, 2018)                   # training years only: no test weather leaks in
LEADS_WEEKS = (12, 8, 6, 4, 2, 0)
TEST_ROLES = ("test_years", "test_both")
TARGET_YEARS = range(2018, 2026)
LAST = 2017
FORWARD = int(os.environ.get("VISTA_FORWARD", "0"))
if FORWARD:     # v11: forecasts of training years FORWARD+1..2017 with models fitted <= FORWARD;
    SCENARIOS = range(1979, FORWARD + 1)        # scenario weather only from years <= the cut
    TEST_ROLES, TARGET_YEARS, LAST = ("fwd",), range(FORWARD + 1, 2018), FORWARD


def spliced(w, cutoff_doy, scenario, years):
    """Weather where, in each target year, the days after cutoff_doy come from `scenario`."""
    dates = pd.DatetimeIndex(w["dates"])
    out = {k: (v.copy() if k in ("tas", "tasmax", "tasmin", "hurs", "pr", "rsds", "wind2", "et0") else v)
           for k, v in w.items()}
    src_index = {(d.year, d.month, d.day): i for i, d in enumerate(dates) if d.year == scenario}
    for y in years:
        tgt = np.flatnonzero((dates.year == y) & (dates.dayofyear > cutoff_doy))
        src = [src_index.get((y2, dates[i].month, min(dates[i].day, 28 if dates[i].month == 2 else 31)))
               for i in tgt for y2 in [scenario]]
        ok = [s is not None for s in src]
        tgt, src = tgt[ok], np.array([s for s in src if s is not None])
        for k in ("tas", "tasmax", "tasmin", "hurs", "pr", "rsds", "wind2", "et0"):
            out[k][tgt] = w[k][src]
    return out


def fit_final(crop):
    """Final model as in phase2_growth.main(): physics settings from file, correction + boosting
    refitted on the training rows with the true weather."""
    m, d = load_growth(crop)
    path = os.path.join(P, f"growth_model_{crop}_v7{VTAG}.json")
    if FORWARD:     # the forward model of the v9 validation (fitted <= FORWARD, MODIS)
        import glob
        cands = sorted(glob.glob(os.path.join(P, f"growth_model_{crop}_forward{FORWARD}_v7*modis*.json")))
        cands = [c for c in cands if ("potato_canopy" in c) == (crop == "potato")]
        path = min(cands, key=len)
        print(f"  forward model {os.path.basename(path)}", flush=True)
    model = Growth(json.load(open(path)))
    tr = ((m["role"] == "train") & m["yield_t_ha"].notna()).values
    bench = [c for c in features(m, m[tr]) if c != "n_crop_kg_ha"]
    sim = adjust(simulate(model, d, GROWTH[crop]["dry_matter"]), model, m["year"].values, crop)
    mm = pd.concat([m.reset_index(drop=True), sim], axis=1)
    feats = [c for c in bench + list(sim.columns) if c not in drifting(crop)]
    phys = mm["phys_yield"].clip(lower=0.5).values
    corr = booster({"max_depth": 6, "learning_rate": 0.05}).fit(
        mm.loc[tr, feats], np.log(mm.loc[tr, "yield_t_ha"].values / phys[tr]))
    bst = boost(mm.loc[tr, bench].values.astype(float), mm.loc[tr, "yield_t_ha"].values, BOOST[1])
    return model, corr, bst, feats, bench, int(m.loc[tr, "year"].max())


def adjust(sim, model, years, crop, last=None):
    """Technology rule of the final model (frozen: level held at the last training year)."""
    with torch.no_grad():
        st = {k: v.item() for k, v in model.settings().items()}
    last = LAST if last is None else last
    tech = lambda yr: np.exp(st["tech"] * (1 - np.exp(-(yr - 1979) / st["tau"])))
    if TECH_RULE.get(crop, "frozen") == "frozen":
        sim["phys_yield"] = sim["phys_yield"] * tech(np.minimum(years, last)) / tech(years)
    return sim


MODIS_SERIES = None


def forecast_date(crop, model, corr, bst, feats, bench, cutoff, weight):
    global MODIS_SERIES
    if MODIS and MODIS_SERIES is None:
        from phase0_modis_features import series
        MODIS_SERIES = series()
    stages, gate, vp, _ = phen.CROPS[crop]
    w = np.load(os.path.join(P, phen.WEATHER_FILE))
    w = {k: w[k] for k in w.files}
    codes = w["codes"]
    rows = []
    for s in SCENARIOS:
        t0 = time.time()
        phen.WEATHER = spliced(w, cutoff, s, TARGET_YEARS)
        m, d = load_growth(crop)
        test = m["role"].isin(TEST_ROLES).values
        sub = {k: v[test] for k, v in d.items() if v.dim() and len(v) == len(m)}
        mt = m[test].reset_index(drop=True)
        sim = adjust(simulate(model, sub, GROWTH[crop]["dry_matter"]), model, mt["year"].values, crop)
        # monthly weather features and not-yet-observed stage dates from the scenario
        months = sorted({int(c[-2:]) for c in mt.columns if c.startswith("tas_m")})
        mf = monthly_features(phen.WEATHER, codes, months, "tas_winter" in mt.columns)
        mf = mf.reindex(pd.MultiIndex.from_frame(mt[["district", "year"]]))
        for c in mf.columns:
            if c in mt.columns:
                mt[c] = mf[c].values
        cal = phen.Calendar(len(stages), stages.index(gate), vp,
                            json.load(open(os.path.join(P, f"phenology_model_{crop}.json"))))
        with torch.no_grad():
            days = cal(sub).numpy()
        for k, st in enumerate(stages):
            col = f"doy_{st}"
            if col in mt.columns:
                late = mt[col].isna() | (mt[col] > cutoff)
                mt.loc[late, col] = (mt["doy_sowing"] + days[:, k])[late.values]
        if "doy_harvest" in mt.columns:
            late = mt["doy_harvest"].isna() | (mt["doy_harvest"] > cutoff)
            mt.loc[late, "doy_harvest"] = mt.loc[late, "district"].map(
                mt.groupby("district")["doy_harvest"].median())
        if MODIS:      # satellite greenness only up to the forecast date
            from phase0_modis_features import features as modis_features
            mf2 = modis_features(crop, cutoff, MODIS_SERIES).set_index(["district", "year"])
            mf2 = mf2.reindex(pd.MultiIndex.from_frame(mt[["district", "year"]]))
            for c in mf2.columns:
                if c in mt.columns:
                    mt[c] = mf2[c].values
        mm = pd.concat([mt, sim], axis=1)
        phys = mm["phys_yield"].clip(lower=0.5).values
        hybrid = phys * np.exp(corr.predict(mm[feats]))
        b = bst.predict(mm[bench].values.astype(float))
        rows.append(pd.DataFrame({"district": mt["district"], "year": mt["year"], "role": mt["role"],
                                  "obs": mt["yield_t_ha"], "scenario": s, "physics": phys,
                                  "hybrid": hybrid, "boosting": b,
                                  "blend": weight * hybrid + (1 - weight) * b}))
        print(f"    scenario {s}: {time.time() - t0:.0f} s", flush=True)
    phen.WEATHER = None
    return pd.concat(rows, ignore_index=True)


def main():
    crop = sys.argv[1] if len(sys.argv) > 1 else "winter_wheat"
    weight = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8      # blend weight of the final model
    os.makedirs(OUT, exist_ok=True)
    harvest = int(pd.read_csv(os.path.join(P, f"master_{crop}.csv"))["doy_harvest"].median())
    print(f"{crop}: typical harvest day {harvest}; fitting the final model", flush=True)
    model, corr, bst, feats, bench, _ = fit_final(crop)
    for lead in LEADS_WEEKS:
        path = os.path.join(OUT, f"forecast_{crop}{VTAG}_lead{lead:02d}.csv")
        if os.path.exists(path):
            continue
        cutoff = harvest - 7 * lead
        print(f"  lead {lead} weeks (forecast date = day {cutoff})", flush=True)
        forecast_date(crop, model, corr, bst, feats, bench, cutoff, weight).to_csv(path, index=False)
    summary = []
    for lead in LEADS_WEEKS:
        f = pd.read_csv(os.path.join(OUT, f"forecast_{crop}{VTAG}_lead{lead:02d}.csv"), dtype={"district": str})
        g = f.groupby(["district", "year", "role", "obs"])
        for name in ("physics", "hybrid", "boosting", "blend"):
            q = g[name].quantile([0.1, 0.5, 0.9]).unstack().reset_index()
            for role in TEST_ROLES:
                r = q[(q["role"] == role) & q["obs"].notna()]
                e = r[0.5] - r["obs"]
                summary.append(dict(crop=crop, lead_weeks=lead, data=role, model=name, n=len(r),
                                    rmse=round(float(np.sqrt((e ** 2).mean())), 3), bias=round(float(e.mean()), 3),
                                    coverage_80=round(float(((r["obs"] >= r[0.1]) & (r["obs"] <= r[0.9])).mean()), 3)))
    s = pd.DataFrame(summary)
    s.to_csv(os.path.join(P, f"forecast_skill_{crop}{VTAG}.csv"), index=False)
    print(s[s["data"] == "test_years"].pivot(index="lead_weeks", columns="model", values="rmse").to_string())
    print(s[s["data"] == "test_years"].pivot(index="lead_weeks", columns="model", values="coverage_80").to_string())


if __name__ == "__main__":
    main()
