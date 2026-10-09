"""Crop features from the MODIS district greenness series (phase0_modis_districts.py).

For every district and harvest year, from the 16-day NDVI composites (composite centre = start + 8 days)
inside the crop's window:
  modis_mean     mean greenness in the window
  modis_peak     highest greenness;  modis_peak_doy  day of year of the peak
  modis_drop     fall from the peak to the last composite of the window (browning / ripening speed)
  modis_area     sum of greenness above 0.3 over the window (green area duration)
  modis_anom     window mean minus the district's mean over the TRAINING years 2000-2017 (an unusual
                 season; computed from training years only, so no test information enters)
  modis_n        number of composites used
In-season forecasts: features(crop, cutoff_doy) uses only composites that END before the cutoff.
Windows (day of year): winter wheat / barley 60-212 (Mar-Jul), grain / silage maize 152-273
(Jun-Sep), potato 135-243 (mid May - Aug).
Writes data/processed/modis/modis_features_<crop>.csv. Run `python phase0_modis_features.py`.
"""
import glob
import os

import numpy as np
import pandas as pd

D = os.path.join("data", "processed", "modis")
WINDOWS = {"winter_wheat": (60, 212), "winter_barley": (60, 212), "grain_maize": (152, 273),
           "silage_maize": (152, 273), "potato": (135, 243)}


def series():
    rows = []
    for f in sorted(glob.glob(os.path.join(D, "modis_ndvi_*.npz"))):
        d = np.load(f)
        dates = pd.to_datetime(d["dates"])
        for k, t in enumerate(dates):
            rows.append(pd.DataFrame({"district": d["codes"], "start": t, "ndvi": d["ndvi"][k], "n": d["n"][k]}))
    s = pd.concat(rows, ignore_index=True)
    s["centre"] = s["start"] + pd.Timedelta(days=8)
    s["end"] = s["start"] + pd.Timedelta(days=15)
    s["year"], s["doy"] = s["centre"].dt.year, s["centre"].dt.dayofyear
    return s[s["n"] >= 50].dropna(subset=["ndvi"])


def features(crop, cutoff_doy=None, s=None):
    s = series() if s is None else s
    a, b = WINDOWS[crop]
    w = s[(s["doy"] >= a) & (s["doy"] <= b)]
    if cutoff_doy is not None:
        w = w[w["end"].dt.dayofyear < cutoff_doy]
    cols = ["district", "year", "modis_mean", "modis_peak", "modis_peak_doy", "modis_drop", "modis_area",
            "modis_n", "modis_anom"]
    if w.empty:                 # forecast date before the crop's window: no satellite information yet
        return pd.DataFrame(columns=cols)
    g = w.sort_values("doy").groupby(["district", "year"])
    f = pd.DataFrame({"modis_mean": g["ndvi"].mean(), "modis_peak": g["ndvi"].max(),
                      "modis_peak_doy": g.apply(lambda x: x.loc[x["ndvi"].idxmax(), "doy"], include_groups=False),
                      "modis_drop": g["ndvi"].max() - g["ndvi"].last(),
                      "modis_area": g["ndvi"].apply(lambda v: np.clip(v - 0.3, 0, None).sum()),
                      "modis_n": g["ndvi"].size()}).reset_index()
    base = f[(f["year"] >= 2000) & (f["year"] <= 2017)].groupby("district")["modis_mean"].mean()
    f["modis_anom"] = f["modis_mean"] - f["district"].map(base)
    return f


def main():
    s = series()
    for crop in WINDOWS:
        f = features(crop, s=s)
        f.to_csv(os.path.join(D, f"modis_features_{crop}.csv"), index=False)
        print(f"{crop}: {len(f)} district-years, years {f['year'].min()}-{f['year'].max()}; "
              f"national mean anomaly by year: " + " ".join(
                  f"{y}:{v:+.3f}" for y, v in f.groupby("year")["modis_anom"].mean().items()))


if __name__ == "__main__":
    main()
