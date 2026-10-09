"""Phase 2, step 4: calibrated forecast ranges (weather uncertainty + model error).

The scenario forecasts (phase2_forecast.py) only express weather uncertainty: 39 possible endings
of the season. The model is also wrong when the weather is perfectly known (management, disease,
local effects, data noise). That model error is measured on the training years 1979-2017 by
cross-validation with the final model's blend and the true weather (residual = observed -
predicted, t/ha): leaving out blocks of years (error in new years, used for the new-years test)
and leaving out a state and a block of years together (error in a new region in new years, used
for the both-new test).

Combined forecast distribution for a district-year at a forecast date = every weather scenario
prediction + every model residual (all combinations). Its 10 and 90 % quantiles form the 80 %
range; the median stays the point forecast. Nothing from the test years is used to build it.

Checked on the test district-years: share of observed yields inside the 80 % range (target 80 %)
and the average width of the range.

Run `python phase2_intervals.py winter_wheat [blend weight]` after phase2_forecast.py.
Writes data/processed/forecast_intervals_<crop>.csv and prints the coverage table.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from phase1_baselines import BOOST, boost, features
from phase2_forecast import LEADS_WEEKS, OUT, TEST_ROLES, adjust
from phase2_growth import VTAG, DRIFTING, drifting, GROWTH, P, Growth, booster, load_growth, simulate


def training_residuals(crop, weight, by="years"):
    m, d = load_growth(crop)
    model = Growth(json.load(open(os.path.join(P, f"growth_model_{crop}_v7{VTAG}.json"))))
    tr = ((m["role"] == "train") & m["yield_t_ha"].notna()).values
    bench = [c for c in features(m, m[tr]) if c != "n_crop_kg_ha"]
    sim = adjust(simulate(model, d, GROWTH[crop]["dry_matter"]), model, m["year"].values, crop)
    mm = pd.concat([m.reset_index(drop=True), sim], axis=1)
    feats = [c for c in bench + list(sim.columns) if c not in drifting(crop)]
    phys = mm["phys_yield"].clip(lower=0.5).values
    y = mm["yield_t_ha"].values
    pred = np.full(len(mm), np.nan)
    # groups left out in turn: year blocks (error in new years) or states (error in new regions)
    state, fold = mm["district"].str[:2], mm["fold"]
    if by == "years":
        splits = [((fold != f).values, (fold == f).values) for f in sorted(fold[tr].unique())]
    elif by == "states":
        splits = [((state != s).values, (state == s).values) for s in sorted(state[tr].unique())]
    else:   # both new: train without that state AND without that block of years
        splits = [(((state != s) & (fold != f)).values, ((state == s) & (fold == f)).values)
                  for s in sorted(state[tr].unique()) for f in sorted(fold[tr].unique())]
    for keep, out in splits:
        a, b = tr & keep, tr & out
        if b.sum() == 0:
            continue
        corr = booster({"max_depth": 6, "learning_rate": 0.05}).fit(mm.loc[a, feats], np.log(y[a] / phys[a]))
        hybrid = phys[b] * np.exp(corr.predict(mm.loc[b, feats]))
        bst = boost(mm.loc[a, bench].values.astype(float), y[a], BOOST[1]).predict(
            mm.loc[b, bench].values.astype(float))
        pred[b] = weight * hybrid + (1 - weight) * bst
    r = (y - pred)[tr]
    r = r[np.isfinite(r)]
    print(f"training residuals (leaving out {by}): n={len(r)}, mean {r.mean():+.2f}, "
          f"sd {r.std():.2f}, 10/90 % {np.percentile(r, 10):+.2f} / {np.percentile(r, 90):+.2f} t/ha")
    return r


def forward_pool(crop):
    """v8 range method (chosen on training data, eval_range_method.py): model errors when predicting
    the future - fitted <= 2008, errors of 2009-2017 (observed - predicted, t/ha)."""
    name = f"forward_predictions_{crop}_cut2008.csv"
    if VTAG and crop == "potato":        # v8 potato physics (tubers from canopy closure)
        name = "forward_predictions_potato_potato_canopy_cut2008.csv"
    if VTAG == "_v9":                    # v9: forward errors of the model with MODIS features
        name = f"forward_predictions_{crop}{'_potato_canopy' if crop == 'potato' else ''}_modis_cut2008.csv"
    f = pd.read_csv(os.path.join(P, name)).dropna(subset=["yield_t_ha"])
    return (f["yield_t_ha"] - f["pred"]).values


def main():
    crop = sys.argv[1] if len(sys.argv) > 1 else "winter_wheat"
    weight = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
    method = "forward" if "forward" in sys.argv else "cv"
    rng = np.random.default_rng(0)
    pools = {by: rng.choice(r, size=min(400, len(r)), replace=False)
             for by, r in ((by, training_residuals(crop, weight, by)) for by in ("years", "both"))}
    if method == "forward":              # v8: forward errors; both-new widened by the CV spread ratio
        fwd = forward_pool(crop)
        ratio = np.std(pools["both"]) / np.std(pools["years"])
        centre = np.median(fwd)
        pools = {"years": rng.choice(fwd, size=min(400, len(fwd)), replace=False)}
        pools["both"] = centre + (pools["years"] - centre) * ratio
        print(f"range method: forward errors 2009-2017 (n={len(fwd)}, 10/90 % "
              f"{np.percentile(fwd, 10):+.2f}/{np.percentile(fwd, 90):+.2f}); both-new widened x{ratio:.2f}")
    pool_for = {"test_years": "years", "test_both": "both", "test_regions": "states"}
    rows = []
    for lead in LEADS_WEEKS:
        f = pd.read_csv(os.path.join(OUT, f"forecast_{crop}{VTAG}_lead{lead:02d}.csv"), dtype={"district": str})
        f = f[f["obs"].notna()]
        for (dist, year, role, obs), g in f.groupby(["district", "year", "role", "obs"]):
            sc = g["blend"].values
            res = pools[pool_for[role]]
            combined = (sc[:, None] + res[None, :]).ravel()
            q10, q50, q90 = np.percentile(combined, [10, 50, 90])
            w10, w90 = np.percentile(sc, [10, 90])
            rows.append(dict(district=dist, year=year, role=role, lead_weeks=lead, obs=obs,
                             forecast=np.median(sc), low80=q10, high80=q90, weather_low80=w10,
                             weather_high80=w90))
    out = pd.DataFrame(rows)
    tag = (VTAG or "_v8") if method == "forward" else ""      # v8 or v9 files
    out.to_csv(os.path.join(P, f"forecast_intervals_{crop}{tag}.csv"), index=False)
    out["inside"] = (out["obs"] >= out["low80"]) & (out["obs"] <= out["high80"])
    out["inside_weather_only"] = (out["obs"] >= out["weather_low80"]) & (out["obs"] <= out["weather_high80"])
    out["width"] = out["high80"] - out["low80"]
    t = out.groupby(["role", "lead_weeks"]).agg(n=("obs", "size"), coverage=("inside", "mean"),
                                                 coverage_weather_only=("inside_weather_only", "mean"),
                                                 width_t_ha=("width", "mean")).round(3)
    print(t.sort_index(level=1, ascending=False).to_string())
    t.reset_index().to_csv(os.path.join(P, f"forecast_coverage_{crop}{tag}.csv"), index=False)


if __name__ == "__main__":
    main()
