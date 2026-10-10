"""v11 plan B, route 1: Sentinel-2 crop NDVI -> MODIS cropland NDVI bridge (satellite vs satellite, no yields).

Question: can the district MODIS cropland greenness the model uses be reproduced from the crop-specific
Sentinel-2 samples, so the model can keep running when MODIS (Terra/Aqua) ends?
Per district and MODIS 16-day date (Apr-Sep): S2 dekad NDVI of wheat, barley, maize, potato interpolated to the
MODIS composite mid-date; missing crop -> mean of available crops. Model: ridge (alpha 1) MODIS ~ 4 crop NDVIs +
day of year, leave-one-year-out. Output data/processed/s2_bridge.csv (per left-out year r, RMSE, bias) and
s2_bridge_pairs.csv.gz.
"""
import glob
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from s2_features import dekads

P = os.path.join("data", "processed")
S2 = os.path.join(P, "s2_district")
CROPS = ("wheat", "barley", "maize", "potato")


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(S2, "sample_*_n25_cheap.csv"))):
        tag = os.path.basename(f)[7:-4]
        tsf = os.path.join(S2, f"sample_{tag}", "timeseries.csv")
        mf = os.path.join(P, "modis", f"modis_ndvi_{tag.split('_')[1]}.npz")
        if not (os.path.exists(tsf) and os.path.exists(mf)):
            continue
        year = int(tag.split("_")[1])
        md = np.load(mf)
        mdoy = pd.to_datetime(md["dates"]).dayofyear + 8
        col = {c: i for i, c in enumerate(md["codes"])}
        smp = pd.read_csv(f, dtype={"AGS": str})
        smp["crop"] = smp["crop"].replace({"potatoes": "potato"})
        ts = pd.read_csv(tsf).rename(columns={"band_unnamed": "ndvi"}).dropna(subset=["ndvi"])
        for ags, g in smp.groupby("AGS"):
            if ags not in col:
                continue
            curves = {}
            for i, x in g.iterrows():
                s = dekads(ts[ts["feature_index"] == i], year)
                if x["cells"] >= 5 and len(s) >= 4:
                    curves[x["crop"]] = s
            if not curves:
                continue
            for t, doy in enumerate(mdoy):
                if not 100 <= doy <= 270 or not np.isfinite(md["ndvi"][t, col[ags]]):
                    continue
                v = {c: np.interp(doy, s.index, s.values, left=np.nan, right=np.nan) for c, s in curves.items()}
                v = {c: x for c, x in v.items() if np.isfinite(x)}
                if not v:
                    continue
                m = np.mean(list(v.values()))
                rows.append(dict(district=ags, year=year, doy=doy, modis=md["ndvi"][t, col[ags]], ncrops=len(v),
                                 **{c: v.get(c, m) for c in CROPS}))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(P, "s2_bridge_pairs.csv.gz"), index=False)
    X = lambda z: np.column_stack([z[list(CROPS)].values, (z["doy"].values - 180) / 50])
    out = []
    for y in sorted(d["year"].unique()):
        tr, te = d[d["year"] != y], d[d["year"] == y]
        p = pd.Series(Ridge(alpha=1).fit(X(tr), tr["modis"]).predict(X(te)), index=te.index)
        e = p - te["modis"]
        anom = lambda z: z - z.groupby(te["doy"]).transform("mean")
        out.append(dict(year=y, n=len(te), r=np.corrcoef(p, te["modis"])[0, 1], rmse=np.sqrt((e ** 2).mean()), bias=e.mean(),
                        r_within_date=np.corrcoef(anom(p), anom(te["modis"]))[0, 1]))
    o = pd.DataFrame(out).round(3)
    o.to_csv(os.path.join(P, "s2_bridge.csv"), index=False)
    print(o.to_string(index=False))


if __name__ == "__main__":
    main()
