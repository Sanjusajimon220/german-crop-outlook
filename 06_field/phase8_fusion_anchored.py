"""Layer A data fusion: anchored, radar-guided gap filling ('1+4'), pilot cell 38_706, season 2024
(pre-registered docs/project_log.md 2026-10-14). Sentinel-2 stays the reference; radar only describes change.

LIVE:        LAI(t) = LAI(a) + f(radar change since a, days, LAI(a), crop, season, rain, days since heading)
RE-ANALYSIS: LAI(t) = interp(t between a and b) + g(radar changes vs a and b, position, gap, LAI(a), LAI(b), ...)
Radar: field mean of linear VV / VH per S1 date, rolling mean over 3 S1 dates; VV, VH, VH-VV (dB), RVI.
Check: hide 30 % of clear pixel-dates (seed 7 as step 3), 5 folds by field; RMSE vs interpolation by gap class, crop.
Outputs data/processed/field_v2/pilot_2024/fusion_check.csv.gz, fusion_summary.csv, fusion_by_crop.csv
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from phase7_step3_s1proxy import CROPS, context, prev_next, s1_stack

P = os.path.join("data", "processed")
OUT = os.path.join(P, "field_v2", "pilot_2024")
MAIN6 = ["wheat", "barley", "sugar beet", "potato", "maize", "rapeseed"]
RADAR = ["VV", "VH", "RAT", "RVI"]


def field_radar(fid, rows, cols):
    """Field-mean radar per S1 date (linear), rolling mean over 3 dates, then dB features. -> dates, {feat: [T x fields]}"""
    s1d, s1 = s1_stack(rows, cols)                              # pixel dB after 3x3 mean
    lin = {b: 10 ** (s1[f"{b}_db"] / 10) for b in ("VV", "VH")}
    fids, inv = np.unique(fid, return_inverse=True)
    out = {}
    for b in ("VV", "VH"):
        x = lin[b]
        ok = np.isfinite(x)
        s = np.stack([np.bincount(inv, weights=np.where(ok[k], x[k], 0), minlength=len(fids)) for k in range(len(s1d))])
        n = np.stack([np.bincount(inv, weights=ok[k].astype(float), minlength=len(fids)) for k in range(len(s1d))])
        m = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        out[b] = pd.DataFrame(m).rolling(3, min_periods=1).mean().values      # rolling over S1 dates
    vv, vh = out["VV"], out["VH"]
    feats = {"VV": 10 * np.log10(vv), "VH": 10 * np.log10(vh), "RAT": 10 * np.log10(vh / vv), "RVI": 4 * vh / (vv + vh)}
    return s1d, feats, inv


def nearest_s1(s1d, d, tol=3):
    k = int(np.argmin(np.abs((s1d - d).days)))
    return k if abs((s1d[k] - d).days) <= tol else -1


def main():
    z = np.load(os.path.join(OUT, "s2_pixels.npz"))
    s2d, rows, cols = pd.DatetimeIndex(z["dates"]), z["rows"], z["cols"]
    px = pd.read_csv(os.path.join(OUT, "pixels_nrw.csv.gz"))
    fl = pd.read_csv(os.path.join(OUT, "fields_nrw.csv"))
    main_f = px.sort_values("share").drop_duplicates(["row", "col"], keep="last").set_index(["row", "col"])["field_id"]
    fid = main_f.reindex(pd.MultiIndex.from_arrays([rows, cols])).values
    grp = pd.Series(fid).map(fl.set_index("field_id")["crop_group"]).fillna("other").values
    crop = np.array([CROPS.index(g) if g in CROPS else CROPS.index("other") for g in grp])
    s1d, rad, inv = field_radar(fid, rows, cols)
    ctx = context(s2d)                                           # season day, rain, days since heading at S2 dates
    w = np.load(os.path.join(P, "weather_daily_v8.npz"))
    pr = pd.Series(w["pr"][:, list(w["codes"]).index("05358")], index=pd.DatetimeIndex(w["dates"]))
    rain_cum = pr.cumsum()
    k_s1 = np.array([nearest_s1(s1d, d) for d in s2d])
    days = (s2d - s2d[0]).days.values
    rng = np.random.default_rng(7)                               # same hiding as step 3
    folds = {f: i % 5 for i, f in enumerate(rng.permutation(np.unique(fid)))}
    fold = np.array([folds[f] for f in fid])
    results = []
    for target in ("lai", "ndvi"):
        y = z[target]
        clear = np.isfinite(y)
        hide = clear & (np.random.default_rng(70).random(clear.shape) < 0.3)
        first = np.argmax(clear, axis=0)
        last = clear.shape[0] - 1 - np.argmax(clear[::-1], axis=0)
        hide[first, np.arange(clear.shape[1])] = False
        hide[last, np.arange(clear.shape[1])] = False
        kept = clear & ~hide

        def radar_at(i, idx):
            k = k_s1[i]
            if k < 0:
                return np.full((len(idx), len(RADAR)), np.nan)
            return np.stack([rad[f][k, inv[idx]] for f in RADAR], 1)

        def base(i, idx):
            c = ctx.iloc[i]
            return dict(crop=crop[idx], dsin=np.sin(2 * np.pi * c["doy"] / 365), dcos=np.cos(2 * np.pi * c["doy"] / 365),
                        since_heading=c["since_heading"])

        def live_rows(i, a, idx):
            r = radar_at(i, idx) - radar_at(a, idx)
            d = dict(base(i, idx), days=days[i] - days[a], y_a=y[a, idx],
                     rain=rain_cum[s2d[i]] - rain_cum[s2d[a]], **{f"d{f}": r[:, n] for n, f in enumerate(RADAR)})
            return pd.DataFrame(d)

        def re_rows(i, a, b, idx):
            ra, rb = radar_at(i, idx) - radar_at(a, idx), radar_at(i, idx) - radar_at(b, idx)
            pos = (days[i] - days[a]) / (days[b] - days[a])
            d = dict(base(i, idx), pos=pos, gap=days[b] - days[a], y_a=y[a, idx], y_b=y[b, idx],
                     rain_a=rain_cum[s2d[i]] - rain_cum[s2d[a]], rain_b=rain_cum[s2d[b]] - rain_cum[s2d[i]],
                     **{f"da{f}": ra[:, n] for n, f in enumerate(RADAR)}, **{f"db{f}": rb[:, n] for n, f in enumerate(RADAR)})
            return pd.DataFrame(d)

        # training sets from KEPT dates only: live pairs (a among previous 1-3 kept), re-analysis triples
        T = len(s2d)
        Ltr, Lt, Rtr, Rt, Lf, Rf = [], [], [], [], [], []
        for i in range(T):
            for a in range(max(0, i - 3), i):
                idx = np.flatnonzero(kept[i] & kept[a])
                if len(idx) > 40000:
                    idx = rng.choice(idx, 40000, replace=False)
                if len(idx):
                    Ltr.append(live_rows(i, a, idx)); Lt.append(y[i, idx] - y[a, idx]); Lf.append(fold[idx])
            for a in range(max(0, i - 2), i):
                for b in range(i + 1, min(T, i + 3)):
                    idx = np.flatnonzero(kept[i] & kept[a] & kept[b])
                    if len(idx) > 30000:
                        idx = rng.choice(idx, 30000, replace=False)
                    if len(idx):
                        lin = y[a, idx] + (days[i] - days[a]) / (days[b] - days[a]) * (y[b, idx] - y[a, idx])
                        Rtr.append(re_rows(i, a, b, idx)); Rt.append(y[i, idx] - lin); Rf.append(fold[idx])
        XL, yL, fL = pd.concat(Ltr, ignore_index=True), np.concatenate(Lt), np.concatenate(Lf)
        XR, yR, fR = pd.concat(Rtr, ignore_index=True), np.concatenate(Rt), np.concatenate(Rf)
        print(f"{target}: training pairs live {len(XL)}, re-analysis triples {len(XR)}", flush=True)
        # hidden pixel-dates: previous / next kept clear date
        prv, nxt = prev_next(kept)
        ev = []
        for f in range(5):
            mL = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=0,
                                               categorical_features=[XL.columns.get_loc("crop")]).fit(XL[fL != f], yL[fL != f])
            mR = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=0,
                                               categorical_features=[XR.columns.get_loc("crop")]).fit(XR[fR != f], yR[fR != f])
            for i in range(T):
                sel = hide[i] & (fold == f) & (prv[i] >= 0) & (nxt[i] >= 0)
                for (a, b), g in pd.DataFrame({"a": prv[i, sel], "b": nxt[i, sel], "p": np.flatnonzero(sel)}).groupby(["a", "b"]):
                    idx = g["p"].values
                    lin = y[a, idx] + (days[i] - days[a]) / (days[b] - days[a]) * (y[b, idx] - y[a, idx])
                    live = y[a, idx] + mL.predict(live_rows(i, a, idx))
                    re = lin + mR.predict(re_rows(i, a, b, idx))
                    ev.append(pd.DataFrame({"target": target, "pixel": idx, "date_i": i, "obs": y[i, idx], "interp": lin,
                                            "live": live, "reanalysis": re, "gap": days[b] - days[a], "crop": grp[idx]}))
        e = pd.concat(ev, ignore_index=True)
        e["gap_class"] = pd.cut(e["gap"], [0, 15, 30, 400], labels=["<=15 d", "16-30 d", ">30 d"])
        results.append(e)
    e = pd.concat(results, ignore_index=True)
    e.to_csv(os.path.join(OUT, "fusion_check.csv.gz"), index=False)
    rm = lambda v: float(np.sqrt(np.mean(np.square(v))))
    summ, byc = [], []
    for (t, gc), g in e.groupby(["target", "gap_class"], observed=True):
        summ.append(dict(target=t, gap=gc, n=len(g), interp=rm(g.interp - g.obs), live=rm(g.live - g.obs), reanalysis=rm(g.reanalysis - g.obs)))
        for c in MAIN6:
            h = g[g.crop == c]
            if len(h):
                byc.append(dict(target=t, gap=gc, crop=c, n=len(h), interp=rm(h.interp - h.obs), live=rm(h.live - h.obs),
                                reanalysis=rm(h.reanalysis - h.obs)))
    s, b = pd.DataFrame(summ).round(3), pd.DataFrame(byc).round(3)
    s.to_csv(os.path.join(OUT, "fusion_summary.csv"), index=False)
    b.to_csv(os.path.join(OUT, "fusion_by_crop.csv"), index=False)
    print(s.to_string(index=False))
    print(b[b["gap"] != "<=15 d"].to_string(index=False))
    for t in ("lai", "ndvi"):
        for gc in ("16-30 d", ">30 d"):
            for v in ("live", "reanalysis"):
                row = s[(s.target == t) & (s.gap == gc)]
                cr = b[(b.target == t) & (b.gap == gc)]
                ok = len(row) and row[v].iloc[0] < row["interp"].iloc[0] and int((cr[v] < cr["interp"]).sum()) >= 4
                print(f"RESULT {t} {gc} {v}: {'ADOPT' if ok else 'not adopted'} "
                      f"(crops better {int((cr[v] < cr['interp']).sum())}/{len(cr)})")


if __name__ == "__main__":
    main()
