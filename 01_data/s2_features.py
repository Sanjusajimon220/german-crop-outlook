"""v11.1 Sentinel-2 crop-specific district features (pre-registered in docs/project_log.md, 2026-10-09 16:25).

Input  data/processed/s2_district/sample_<state>_<year>_n25_cheap.csv (+ /timeseries.csv: date, feature_index, NDVI)
Output data/processed/s2_features.csv: crop, district, year, lead, peak, last3, integral and their anomalies vs
       the state mean of the same crop and year (suffix _anom). Only dekads ending before the forecast date.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

S2 = os.path.join("data", "processed", "s2_district")
CROPMAP = {"wheat": ["winter_wheat"], "barley": ["winter_barley"], "maize": ["grain_maize", "silage_maize"],
           "potatoes": ["potato"], "potato": ["potato"]}
LEADS = (12, 8, 6, 4, 2, 0)


def dekads(ts, year):
    """Dekad median of valid dates -> Series indexed by dekad end doy (Apr-Sep)."""
    d = pd.to_datetime(ts["date"]).dt.tz_localize(None)
    dk = (d.dt.month - 1) * 3 + np.minimum((d.dt.day - 1) // 10, 2)
    v = ts["ndvi"].groupby(dk.values).median()
    ends = {k: (pd.Timestamp(year, k // 3 + 1, [10, 20, 1][k % 3]) + (pd.offsets.MonthEnd(0) if k % 3 == 2 else pd.Timedelta(0))).dayofyear
            for k in v.index}
    return pd.Series(v.values, index=[ends[k] for k in v.index]).sort_index()


def feats(s, doy):
    s = s[s.index < doy].dropna()
    if len(s) < 4:
        return None
    full = s.reindex(range(s.index.min(), s.index.max() + 1)).interpolate()
    return dict(peak=s.max(), last3=s.iloc[-3:].mean(), integral=np.clip(full - 0.2, 0, None).sum() / 10)


def main():
    meta = json.load(open(os.path.join("platform", "data", "meta.json")))["crops"]
    rows = []
    for f in sorted(glob.glob(os.path.join(S2, "sample_*_n25_cheap.csv"))):
        tag = os.path.basename(f)[7:-4]
        tsf = os.path.join(S2, f"sample_{tag}", "timeseries.csv")
        if not os.path.exists(tsf):
            continue
        state, year = tag.split("_")[0], int(tag.split("_")[1])
        smp = pd.read_csv(f, dtype={"AGS": str})
        ts = pd.read_csv(tsf).rename(columns={"band_unnamed": "ndvi"}).dropna(subset=["ndvi"])
        for i, x in smp.iterrows():
            if x["cells"] < 5:
                continue
            s = dekads(ts[ts["feature_index"] == i], year)
            for crop in CROPMAP[x["crop"]]:
                for lead in LEADS:
                    fe = feats(s, meta[crop]["harvest_doy"] - 7 * lead)
                    if fe:
                        rows.append(dict(crop=crop, district=x["AGS"], state=state, year=year, lead=lead, cells=x["cells"], **fe))
    d = pd.DataFrame(rows)
    for c in ("peak", "last3", "integral"):
        d[c + "_anom"] = d[c] - d.groupby(["crop", "state", "year", "lead"])[c].transform("mean")
    d.to_csv(os.path.join("data", "processed", "s2_features.csv"), index=False)
    print(d.groupby(["crop", "year"]).size().unstack(), "\n", d.groupby(["crop", "lead"])[["peak", "integral"]].mean().round(2))


if __name__ == "__main__":
    main()
