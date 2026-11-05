"""MODIS Aqua -> Terra calibration of the district cropland NDVI (review step 4, pre-registered 2026-11-05).

Aqua (MYD13Q1) continues after Terra's composites end (2025-06-26). Both are 16-day composites, phased 8 days apart, so
each Terra composite is paired with the nearest Aqua composite of the same district within 8 days.
Calibration Terra ~ a + b x Aqua, one line or one per month (Feb-Oct): the form with the lower leave-one-year-out RMSE on
2003-2017 is used, then fitted on all of 2003-2017. Check on 2018-2024: PASS if RMSE <= 0.02 and r >= 0.95.
Writes data/processed/modis/aqua_calibration.json and modis_aqua_cal_<year>.npz (calibrated Aqua, Terra's layout).
Run from D:\\vista after phase0_modis_districts.py with MODIS_SENSOR=MYD13Q1: python phase0_modis_aqua_calib.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd

OUT = os.path.join("data", "processed", "modis")
FIT_YEARS, CHECK_YEARS = range(2003, 2018), range(2018, 2025)
MAX_RMSE, MIN_R = 0.02, 0.95


def pair(terra, aqua, max_days=8):
    """terra / aqua: long tables (date, district, terra | aqua). Each Terra row with the nearest Aqua composite of the same
    district within max_days."""
    out = []
    for d, g in terra.groupby("district"):
        a = aqua[aqua["district"] == d].sort_values("date")
        if a.empty:
            continue
        m = pd.merge_asof(g.sort_values("date"), a[["date", "aqua"]], on="date", direction="nearest",
                          tolerance=pd.Timedelta(days=max_days))
        out.append(m.dropna(subset=["aqua", "terra"]))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["date", "district", "terra", "aqua"])


def fit(df, by_month):
    """Least-squares line terra = a + b aqua; {month: (a, b)} or {0: (a, b)} for one line."""
    groups = df.groupby("month") if by_month else [(0, df)]
    return {int(k): tuple(np.polyfit(g["aqua"], g["terra"], 1)[::-1]) for k, g in groups}


def apply(coefs, df):
    key = df["month"] if set(coefs) != {0} else pd.Series(0, index=df.index)
    a = key.map(lambda k: coefs[int(k)][0]).values
    b = key.map(lambda k: coefs[int(k)][1]).values
    return a + b * df["aqua"].values


def scores(pred, truth):
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    return {"n": int(len(pred)), "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))), "bias": float(np.mean(pred - truth)),
            "r": float(np.corrcoef(pred, truth)[0, 1])}


def _long(prefix, col):
    rows = []
    for f in sorted(glob.glob(os.path.join(OUT, f"{prefix}_[0-9]*.npz"))):
        z = np.load(f)
        dates = pd.to_datetime(z["dates"])
        v = z["ndvi"]
        k = np.isfinite(v)
        t_, d_ = np.nonzero(k)
        rows.append(pd.DataFrame({"date": dates[t_], "district": z["codes"][d_], col: v[k]}))
    return pd.concat(rows, ignore_index=True)


def main():
    terra, aqua = _long("modis_ndvi", "terra"), _long("modis_aqua_ndvi", "aqua")
    p = pair(terra, aqua)
    p["year"], p["month"] = p["date"].dt.year, p["date"].dt.month
    fit_d = p[p["year"].isin(FIT_YEARS)]
    loyo = {}
    for by_month in (False, True):
        err = []
        for y in FIT_YEARS:
            tr, te = fit_d[fit_d["year"] != y], fit_d[fit_d["year"] == y]
            if len(te):
                err.append(apply(fit(tr, by_month), te) - te["terra"].values)
        loyo[by_month] = float(np.sqrt(np.mean(np.concatenate(err) ** 2)))
    by_month = loyo[True] < loyo[False]
    coefs = fit(fit_d, by_month)
    chk = p[p["year"].isin(CHECK_YEARS)]
    raw, cal = scores(chk["aqua"], chk["terra"]), scores(apply(coefs, chk), chk["terra"])
    passed = cal["rmse"] <= MAX_RMSE and cal["r"] >= MIN_R
    res = {"form": "per month" if by_month else "one line", "loyo_rmse_one_line": loyo[False], "loyo_rmse_per_month": loyo[True],
           "coefs": {str(k): v for k, v in coefs.items()}, "check_2018_2024_raw": raw, "check_2018_2024_calibrated": cal,
           "pass": bool(passed), "rule": f"RMSE <= {MAX_RMSE} and r >= {MIN_R}"}
    json.dump(res, open(os.path.join(OUT, "aqua_calibration.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    if passed:                                          # calibrated Aqua per year, in the layout of modis_ndvi_<year>.npz
        for f in sorted(glob.glob(os.path.join(OUT, "modis_aqua_ndvi_[0-9]*.npz"))):
            z = dict(np.load(f))
            m = pd.DatetimeIndex(z["dates"]).month.values
            v = z["ndvi"]
            key = m[:, None].repeat(v.shape[1], 1) if by_month else np.zeros_like(v, int)
            a = np.vectorize(lambda k: coefs.get(int(k), (np.nan, np.nan))[0])(key)
            b = np.vectorize(lambda k: coefs.get(int(k), (np.nan, np.nan))[1])(key)
            z["ndvi"] = np.where(np.isfinite(v), a + b * v, np.nan).astype(np.float32)
            np.savez_compressed(f.replace("modis_aqua_ndvi_", "modis_aqua_cal_"), **z)


if __name__ == "__main__":
    main()
