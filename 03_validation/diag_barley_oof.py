"""Diagnosis (training years only): out-of-fold errors of the final barley blend per year, with
year blocks left out (same folds as the model) - is the miss a drought effect or a trend effect?"""
import json, os
import numpy as np, pandas as pd
from phase1_baselines import BOOST, boost, features
from phase2_forecast import adjust
from phase2_growth import GROWTH, P, Growth, booster, drifting, load_growth, simulate

crop = "winter_barley"
m, d = load_growth(crop)
model = Growth(json.load(open(os.path.join(P, f"growth_model_{crop}_v7.json"))))
tr = ((m["role"] == "train") & m["yield_t_ha"].notna()).values
bench = [c for c in features(m, m[tr]) if c != "n_crop_kg_ha"]
sim = adjust(simulate(model, d, GROWTH[crop]["dry_matter"]), model, m["year"].values, crop)
mm = pd.concat([m.reset_index(drop=True), sim], axis=1)
feats = [c for c in bench + list(sim.columns) if c not in drifting(crop)]
phys = mm["phys_yield"].clip(lower=0.5).values
y = mm["yield_t_ha"].values
pred = np.full(len(mm), np.nan); hyb = np.full(len(mm), np.nan)
for f in sorted(mm.loc[tr, "fold"].unique()):
    a, b = tr & (mm["fold"] != f).values, tr & (mm["fold"] == f).values
    corr = booster({"max_depth": 6, "learning_rate": 0.05}).fit(mm.loc[a, feats], np.log(y[a] / phys[a]))
    hyb[b] = phys[b] * np.exp(corr.predict(mm.loc[b, feats]))
    bst = boost(mm.loc[a, bench].values.astype(float), y[a], BOOST[1]).predict(mm.loc[b, bench].values.astype(float))
    pred[b] = 0.8 * hyb[b] + 0.2 * bst
t = mm[tr].assign(e_blend=(pred - y)[tr], e_hybrid=(hyb - y)[tr], e_phys=(phys - y)[tr])
g = t.groupby("year")[["e_phys", "e_hybrid", "e_blend", "phys_water_ratio", "phys_grainfill_stress"]].mean().round(2)
g.to_csv(os.path.join(P, "diag_barley_oof_by_year.csv"))
print(g.to_string())
dry = g["phys_water_ratio"] < g["phys_water_ratio"].quantile(0.2)
print("\nmean OOF blend error: dry years (lowest 20 % water ratio)", g.loc[dry, "e_blend"].mean().round(2),
      "| other years", g.loc[~dry, "e_blend"].mean().round(2))
print("trend of OOF blend error per decade:", np.polyfit(g.index, g["e_blend"], 1)[0].round(3) * 10)
