"""v10 deep diagnosis: where does the remaining error come from? (training years only)

Forward errors (v8 model fitted <= 2002 for 2003-2008, fitted <= 2008 for 2009-2017), e = pred - official.
Variance split (sums of squares, area not weighted):
  year        mean error of all districts in a year (shared nationally)
  state-year  state mean minus year mean
  district    district's mean error over the 15 years after removing year and state-year parts
  rest        what remains (random, incl. statistics noise)
Also the noise floor hint: correlation of a district's residual with its neighbours' (spatially smooth
error = information gap; uncorrelated = noise).
Writes data/processed/error_decomposition.csv.
"""
import os
import numpy as np
import pandas as pd

P = os.path.join("data", "processed")
rows = []
for crop in ("winter_wheat", "winter_barley", "grain_maize", "silage_maize", "potato"):
    f02 = pd.read_csv(os.path.join(P, f"forward_predictions_{crop}_cut2002.csv"), dtype={"district": str})
    f08 = pd.read_csv(os.path.join(P, f"forward_predictions_{crop}_cut2008.csv"), dtype={"district": str})
    f = pd.concat([f02[f02["year"] <= 2008], f08]).dropna(subset=["yield_t_ha", "pred"]).copy()
    f["e"] = f["pred"] - f["yield_t_ha"]
    f["state"] = f["district"].str[:2]
    tot = (f["e"] ** 2).sum()
    yr = f.groupby("year")["e"].transform("mean")
    sy = f.groupby(["year", "state"])["e"].transform("mean") - yr
    r1 = f["e"] - yr - sy
    dist = r1.groupby(f["district"]).transform("mean")
    rest = r1 - dist
    mean_bias = f["e"].mean()
    share = lambda x: float((x ** 2).sum() / tot)
    rows.append(dict(crop=crop, n=len(f), years=f["year"].nunique(), rmse=np.sqrt((f["e"] ** 2).mean()),
                     mean_bias=mean_bias, share_year=share(yr), share_state_year=share(sy),
                     share_district=share(dist), share_rest=share(rest),
                     year_sd=yr.groupby(f["year"]).first().std(), rest_sd=rest.std()))
s = pd.DataFrame(rows).round(3)
s.to_csv(os.path.join(P, "error_decomposition.csv"), index=False)
with pd.option_context("display.width", 200):
    print(s.to_string(index=False))
