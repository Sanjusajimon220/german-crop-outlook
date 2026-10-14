"""Field v2 step 3 (pre-registered 2026-10-14): Sentinel-1 as a PROXY for missing Sentinel-2 dates, per 10 m pixel,
pilot cell 38_706, season 2024. Sentinel-2 stays primary; S1 only fills gaps > 15 days and flags suspect S2 values.

S1: 55 ascending dates (VV, VH sigma0, same grid). dB after a 3 x 3 mean in linear units (speckle); change since the
previous S1 date; VH-VV. Context: crop group (NRW record of the pixel's main field), day of season, rain day 0/1 and
days since wheat heading (district 05358 Dueren weather / crop calendar: the cell lies in Dueren / Heinsberg).
Model: gradient boosting per target (LAI, NDVI), trained WITHIN 2024 on clear S2 pixel-dates with an S1 date within
+-2 days. Check: 5 folds by FIELD; 30 % of the clear S2 pixel-dates of test fields hidden; S1 estimate vs per-pixel
linear interpolation between kept clear dates; by gap class and phase. Adopt per target if S1 beats interpolation on
gaps 16-30 and > 30 days. Correction: clear S2 value flagged 'suspect' if below the S1 estimate by > 2.5 x S1 RMSE and
below both neighbouring clear values of the pixel by > 0.3 LAI (0.05 NDVI).
Outputs check_s1proxy.csv, s1proxy_summary.csv and (if adopted) series_2024.npz: continuous series per pixel on all S1
and S2 dates with a source flag (1 measured S2, 2 interpolated <= 15 d, 3 S1 estimate, 4 corrected).
"""
import glob
import os

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import uniform_filter
from sklearn.ensemble import HistGradientBoostingRegressor

from phase4_field_features import district_dates

P = os.path.join("data", "processed")
OUT = os.path.join(P, "field_v2", "pilot_2024")
RAW = os.path.join("data", "raw", "field_v2", "pilot_2024", "s1")
DIST, YEAR = "05358", 2024
FEATS = ["VV_db", "VH_db", "ratio_db", "dVV", "dVH", "dratio", "crop", "doy", "rain0", "rain1", "since_heading"]
CROPS = ["wheat", "barley", "other cereal", "maize", "rapeseed", "sugar beet", "potato", "legume", "grass / fodder",
         "fallow", "other"]
rng = np.random.default_rng(11)


def s1_stack(rows, cols):
    files = sorted(glob.glob(os.path.join(RAW, "*.tif")))
    dates, vv, vh = [], [], []
    for f in files:
        with rasterio.open(f) as r:
            a = r.read().astype(np.float32)
        a[a <= 0] = np.nan
        sm = []
        for b in a:
            valid = np.isfinite(b)
            num = uniform_filter(np.where(valid, b, 0), 3)
            den = uniform_filter(valid.astype(np.float32), 3)
            sm.append(np.where(den > 0, num / np.maximum(den, 1e-6), np.nan))
        dates.append(pd.Timestamp(os.path.basename(f)[7:17]))
        vv.append(10 * np.log10(sm[0][rows, cols]))
        vh.append(10 * np.log10(sm[1][rows, cols]))
    vv, vh = np.stack(vv), np.stack(vh)
    rat = vh - vv
    d = lambda x: np.vstack([np.full((1, x.shape[1]), np.nan), np.diff(x, axis=0)])
    return pd.DatetimeIndex(dates), dict(VV_db=vv, VH_db=vh, ratio_db=rat, dVV=d(vv), dVH=d(vh), dratio=d(rat))


def context(dates):
    w = np.load(os.path.join(P, "weather_daily_v8.npz"))
    wd = pd.DatetimeIndex(w["dates"])
    col = list(w["codes"]).index(DIST)
    pr = pd.Series(w["pr"][:, col], index=wd)
    head = district_dates(YEAR)[DIST][0]
    start = pd.Timestamp(f"{YEAR - 1}-09-01")
    return pd.DataFrame({"doy": (dates - start).days, "rain0": pr.reindex(dates).values,
                         "rain1": pr.reindex(dates - pd.Timedelta(days=1)).values,
                         "since_heading": (dates - head).days}, index=dates)


def features(s1, ctx, k, idx, crop):
    """Feature matrix for S1 date index k and pixel indices idx."""
    x = pd.DataFrame({n: s1[n][k, idx] for n in ("VV_db", "VH_db", "ratio_db", "dVV", "dVH", "dratio")})
    x["crop"] = crop[idx]
    for c in ("doy", "rain0", "rain1", "since_heading"):
        x[c] = ctx.iloc[k][c]
    return x[FEATS]


def prev_next(valid):
    """For a [time x pixel] mask: index of the previous and next valid time step (strictly before / after; -1 if none)."""
    T, N = valid.shape
    t = np.arange(T)[:, None]
    last = np.where(valid, t, -1)
    last = np.maximum.accumulate(last, axis=0)
    prev = np.vstack([np.full((1, N), -1), last[:-1]])
    nxtv = np.where(valid, t, T)
    nxtv = np.minimum.accumulate(nxtv[::-1], axis=0)[::-1]
    nxt = np.vstack([nxtv[1:], np.full((1, N), T)])
    nxt[nxt == T] = -1
    return prev, nxt


def interp_at(days, y, valid, prev, nxt):
    """Linear interpolation from the previous / next valid value; NaN where either is missing."""
    cols = np.arange(y.shape[1])[None, :]
    ok = (prev >= 0) & (nxt >= 0)
    p, n = np.where(ok, prev, 0), np.where(ok, nxt, 0)
    d0, d1 = days[p], days[n]
    w = np.where(d1 > d0, (days[:, None] - d0) / np.maximum(d1 - d0, 1), 0)
    v = y[p, cols] + w * (y[n, cols] - y[p, cols])
    return np.where(ok, v, np.nan), np.where(ok, d1 - d0, 999)


def main():
    z = np.load(os.path.join(OUT, "s2_pixels.npz"))
    s2d, rows, cols = pd.DatetimeIndex(z["dates"]), z["rows"], z["cols"]
    tg = {"lai": z["lai"], "ndvi": z["ndvi"]}
    px = pd.read_csv(os.path.join(OUT, "pixels_nrw.csv.gz"))
    fl = pd.read_csv(os.path.join(OUT, "fields_nrw.csv"))
    main_f = px.sort_values("share").drop_duplicates(["row", "col"], keep="last").set_index(["row", "col"])["field_id"]
    fid = main_f.reindex(pd.MultiIndex.from_arrays([rows, cols])).values
    grp = pd.Series(fid).map(fl.set_index("field_id")["crop_group"]).fillna("other").values
    crop = np.array([CROPS.index(g) if g in CROPS else CROPS.index("other") for g in grp])
    s1d, s1 = s1_stack(rows, cols)
    ctx = context(s1d)
    print(f"pixels {len(rows)}, S2 dates {len(s2d)}, S1 dates {len(s1d)}", flush=True)
    # S1 date nearest to each S2 date (within +-2 days)
    near = {}
    for i, d in enumerate(s2d):
        k = int(np.argmin(np.abs((s1d - d).days)))
        if abs((s1d[k] - d).days) <= 2:
            near[i] = k
    folds = {f: i % 5 for i, f in enumerate(rng.permutation(np.unique(fid)))}
    fold = np.array([folds[f] for f in fid])
    # hide 30 % of clear S2 pixel-dates (never the first / last clear date of a pixel)
    clear = np.isfinite(tg["lai"])
    hide = clear & (rng.random(clear.shape) < 0.3)
    first = np.argmax(clear, axis=0)
    last = clear.shape[0] - 1 - np.argmax(clear[::-1], axis=0)
    hide[first, np.arange(clear.shape[1])] = False
    hide[last, np.arange(clear.shape[1])] = False
    kept = clear & ~hide
    results, summaries, models = [], [], {}
    for t in ("lai", "ndvi"):
        y = tg[t]
        # training pairs: kept clear pixel-dates with a near S1 date (sampled to 400k)
        Xs, ys, fs = [], [], []
        for i, k in near.items():
            idx = np.flatnonzero(kept[i])
            Xs.append(features(s1, ctx, k, idx, crop))
            ys.append(y[i, idx])
            fs.append(fold[idx])
        X, Y, F = pd.concat(Xs, ignore_index=True), np.concatenate(ys), np.concatenate(fs)
        ok = np.isfinite(X[["VV_db", "VH_db"]]).all(axis=1).values & np.isfinite(Y)
        X, Y, F = X[ok].reset_index(drop=True), Y[ok], F[ok]
        if len(X) > 400000:
            s = rng.choice(len(X), 400000, replace=False)
            X, Y, F = X.iloc[s].reset_index(drop=True), Y[s], F[s]
        # hidden pixel-dates: S1 estimate (model without that fold) vs interpolation
        days = (s2d - s2d[0]).days.values
        ev = []
        prv, nxt = prev_next(kept)
        ip_all, gap_all = interp_at(days, y, kept, prv, nxt)
        for f in range(5):
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, categorical_features=[FEATS.index("crop")],
                                              random_state=0).fit(X[F != f], Y[F != f])
            for i, k in near.items():
                idx = np.flatnonzero(hide[i] & (fold == f) & np.isfinite(ip_all[i]))
                if not len(idx):
                    continue
                est = m.predict(features(s1, ctx, k, idx, crop))
                ev.append(pd.DataFrame({"target": t, "pixel": idx, "date_i": i, "obs": y[i, idx], "s1": est,
                                        "interp": ip_all[i, idx], "gap": gap_all[i, idx],
                                        "since_heading": ctx.iloc[k]["since_heading"]}))
        e = pd.concat(ev, ignore_index=True).dropna()
        e["gap_class"] = pd.cut(e["gap"], [0, 15, 30, 400], labels=["<=15 d", "16-30 d", ">30 d"])
        e["phase"] = np.where(e["since_heading"] < 0, "before heading", "after heading")
        rm = lambda v: float(np.sqrt(np.mean(np.square(v))))
        for key, g in [("all", e)] + [(f"gap {c}", g) for c, g in e.groupby("gap_class", observed=True)] + list(e.groupby("phase")):
            summaries.append(dict(target=t, subset=key, n=len(g), rmse_s1=rm(g["s1"] - g["obs"]), rmse_interp=rm(g["interp"] - g["obs"]),
                                  r_s1=g["s1"].corr(g["obs"]), r_interp=g["interp"].corr(g["obs"])))
        results.append(e)
        models[t] = (HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, categorical_features=[FEATS.index("crop")],
                                                   random_state=0).fit(X, Y), rm(e["s1"] - e["obs"]))
    s = pd.DataFrame(summaries).round(3)
    s.to_csv(os.path.join(OUT, "s1proxy_summary.csv"), index=False)
    pd.concat(results).to_csv(os.path.join(OUT, "check_s1proxy.csv.gz"), index=False)
    print(s.to_string(index=False))
    adopt = {}
    for t in ("lai", "ndvi"):
        q = s[(s["target"] == t) & s["subset"].isin(["gap 16-30 d", "gap >30 d"])]
        adopt[t] = bool(len(q) == 2 and (q["rmse_s1"] < q["rmse_interp"]).all())
        print(f"RESULT {t}: {'ADOPT' if adopt[t] else 'NOT adopted'} (S1 vs interpolation on gaps > 15 days)")
    # continuous series on the union of S1 and S2 dates, with source flags
    alld = s2d.union(s1d)
    for t in ("lai", "ndvi"):
        if not adopt[t]:
            continue
        model, rmse = models[t]
        est = np.full((len(s1d), len(rows)), np.nan, np.float32)
        for k in range(len(s1d)):
            est[k] = model.predict(features(s1, ctx, k, np.arange(len(rows)), crop))
        est[~np.isfinite(s1["VV_db"])] = np.nan
        ser = np.full((len(alld), len(rows)), np.nan, np.float32)
        src = np.zeros((len(alld), len(rows)), np.int8)
        di = {d: n for n, d in enumerate(alld)}
        meas = tg[t]
        dd = (alld - alld[0]).days.values
        sd2 = np.array([di[d] for d in s2d])
        # suspect S2 values (job 3): below the S1 estimate by > 2.5 RMSE and below both neighbouring clear values
        thr = 0.3 if t == "lai" else 0.05
        valid = np.isfinite(meas)
        pv, nx = prev_next(valid)
        cols_ = np.arange(len(rows))[None, :]
        pval = np.where(pv >= 0, meas[np.maximum(pv, 0), cols_], np.nan)
        nval = np.where(nx >= 0, meas[np.maximum(nx, 0), cols_], np.nan)
        e_at = np.full_like(meas, np.nan)
        for i, k in near.items():
            e_at[i] = est[k]
        sus = valid & (e_at - meas > 2.5 * rmse) & (meas < pval - thr) & (meas < nval - thr)
        good = valid & ~sus
        # put measured (non-suspect) S2 on the union grid, then fill
        meas_u = np.full((len(alld), len(rows)), np.nan, np.float32)
        meas_u[sd2] = np.where(good, meas, np.nan)
        vu = np.isfinite(meas_u)
        pu, nu = prev_next(vu)
        ipu, gapu = interp_at(dd, meas_u, vu, pu, nu)
        est_u = np.full_like(meas_u, np.nan)
        for k, d in enumerate(s1d):
            est_u[di[d]] = est[k]
        sus_u = np.zeros_like(vu)
        sus_u[sd2] = sus
        ser = np.where(vu, meas_u, np.nan)
        src = np.where(vu, 1, 0).astype(np.int8)
        use_ip = ~vu & (gapu <= 15) & np.isfinite(ipu)
        ser[use_ip], src[use_ip] = ipu[use_ip], 2
        use_s1 = ~vu & ~use_ip & np.isfinite(est_u)
        ser[use_s1], src[use_s1] = est_u[use_s1], np.where(sus_u[use_s1], 4, 3)
        np.savez_compressed(os.path.join(OUT, f"series_{t}_2024.npz"), dates=np.array(alld, dtype="datetime64[D]"),
                            rows=rows, cols=cols, value=ser, source=src)
        mon = pd.Series(sus.sum(1), index=s2d.strftime("%Y-%m")).groupby(level=0).sum()
        print(f"{t}: series {ser.shape}; source share measured {np.mean(src == 1):.2f}, interpolated {np.mean(src == 2):.2f}, "
              f"S1 {np.mean(src == 3):.2f}, corrected {np.mean(src == 4):.4f}; suspect S2 values by month {mon.to_dict()}")


if __name__ == "__main__":
    main()
