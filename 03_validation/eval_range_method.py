"""Range method check (training data only): which error pool gives honest 80 % ranges for new years?
Check data: forward predictions fitted <= 2008 for 2009-2017 (no weather uncertainty: at harvest).
  A current: out-of-fold training residuals (phase2_intervals.training_residuals, year blocks)
  B new:     forward errors fitted <= 2002 for 2003-2008
Coverage = share of 2009-2017 district-years whose official yield lies within prediction + 10/90 % of the pool."""
import numpy as np, pandas as pd, os
from phase2_intervals import training_residuals
from phase2_growth import FORWARD_WEIGHT
P = "data/processed"
rows = []
for crop in ("winter_wheat", "winter_barley", "grain_maize", "silage_maize", "potato"):
    w = FORWARD_WEIGHT.get(crop, 0.8)
    a = training_residuals(crop, w, "years")
    f02 = pd.read_csv(f"{P}/forward_predictions_{crop}_cut2002.csv")
    f02 = f02[(f02["year"] <= 2008) & f02["yield_t_ha"].notna()]
    b = (f02["yield_t_ha"] - f02["pred"]).values
    chk = pd.read_csv(f"{P}/forward_predictions_{crop}_cut2008.csv").dropna(subset=["yield_t_ha"])
    e = (chk["yield_t_ha"] - chk["pred"]).values
    for name, pool in (("A current (cross-validation)", a), ("B new (forward 2003-2008)", b)):
        lo, hi = np.percentile(pool, [10, 90])
        inside = ((e >= lo) & (e <= hi)).mean()
        yr = pd.Series((e >= lo) & (e <= hi)).groupby(chk["year"].values).mean()
        rows.append(dict(crop=crop, method=name, low=round(lo, 2), high=round(hi, 2), width=round(hi - lo, 2),
                         coverage_2009_2017=round(inside, 3), worst_year=round(yr.min(), 2)))
t = pd.DataFrame(rows)
t.to_csv(f"{P}/range_method_check.csv", index=False)
print(t.to_string(index=False))
