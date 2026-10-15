"""Step 1 (pre-registered 2026-10-15): v2 fusion + PHYSICS features for wheat fields (model-expected change from the
field model v4 prior ensemble, prior_lai_wheat.csv.gz). Same random draws as phase8b -> paired comparison.
Layer A data fusion v2: as phase8_fusion_anchored.py, but radar from BOTH orbits (ascending + descending, separate
features) after a multi-temporal speckle filter (radar_tools.quegan), field level. Pre-registered 2026-10-15.
Original description of v1 follows.
Layer A data fusion: anchored, radar-guided gap filling ('1+4'), pilot cell 38_706, season 2024
(pre-registered docs/project_log.md 2026-10-14). Sentinel-2 stays the reference; radar only describes change.

LIVE:        LAI(t) = LAI(a) + f(radar change since a, days, LAI(a), crop, season, rain, days since heading)
RE-ANALYSIS: LAI(t) = interp(t between a and b) + g(radar changes vs a and b, position, gap, LAI(a), LAI(b), ...)
Radar: field mean of linear VV / VH per S1 date, rolling mean over 3 S1 dates; VV, VH, VH-VV (dB), RVI.
Check: hide 30 % of clear pixel-dates (seed 7 as step 3), 5 folds by field; RMSE vs interpolation by gap class, crop.
Outputs data/processed/field_v2/pilot_2024/fusion3_check.csv.gz, fusion3_summary.csv, fusion3_by_crop.csv
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from phase7_step3_s1proxy import CROPS, context, prev_next
from radar_tools import field_radar as orbit_radar
RAW = os.path.join("data", "raw", "field_v2", "pilot_2024")

P = os.path.join("data", "processed")
OUT = os.path.join(P, "field_v2", "pilot_2024")
MAIN6 = ["wheat", "barley", "sugar beet", "potato", "maize", "rapeseed"]
RADAR = ["VV", "VH", "RAT", "RVI"]
ORBITS = {"asc": "s1", "desc": "s1_desc"}


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
    fids, inv = np.unique(fid, return_inverse=True)
    orb = {o: orbit_radar(os.path.join(RAW, folder), rows, cols, inv) for o, folder in ORBITS.items()}
    for o, (d_, _) in orb.items():
        print(f"orbit {o}: {len(d_)} radar dates", flush=True)
    ctx = context(s2d)
    pri = pd.read_csv(os.path.join(OUT, "prior_lai_wheat.csv.gz"), parse_dates=["date"])
    pri = pri[pri["date"].isin(s2d)].pivot(index="date", columns="field_id", values="prior_lai").reindex(s2d)
    PRI = np.full((len(s2d), len(rows)), np.nan, np.float32)        # prior LAI per S2 date and pixel (wheat fields)
    has = np.isin(fid, pri.columns)
    PRI[:, has] = pri[fid[has]].values
    print(f"pixels with a physics prior (wheat): {has.sum()}", flush=True)                                           # season day, rain, days since heading at S2 dates
    w = np.load(os.path.join(P, "weather_daily_v8.npz"))
    pr = pd.Series(w["pr"][:, list(w["codes"]).index("05358")], index=pd.DatetimeIndex(w["dates"]))
    rain_cum = pr.cumsum()
    k_s1 = {o: np.array([nearest_s1(orb[o][0], d) for d in s2d]) for o in ORBITS}
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
            cols_ = []
            for o in ORBITS:
                k = k_s1[o][i]
                for f in RADAR:
                    cols_.append(np.full(len(idx), np.nan) if k < 0 else orb[o][1][f][k, inv[idx]])
            return np.stack(cols_, 1)

        def base(i, idx):
            c = ctx.iloc[i]
            return dict(crop=crop[idx], dsin=np.sin(2 * np.pi * c["doy"] / 365), dcos=np.cos(2 * np.pi * c["doy"] / 365),
                        since_heading=c["since_heading"])

        def live_rows(i, a, idx):
            r = radar_at(i, idx) - radar_at(a, idx)
            d = dict(base(i, idx), days=days[i] - days[a], y_a=y[a, idx], m_chg=PRI[i, idx] - PRI[a, idx], prior_t=PRI[i, idx],
                     rain=rain_cum[s2d[i]] - rain_cum[s2d[a]], **{f"d{o}_{f}": r[:, n] for n, (o, f) in enumerate((o, f) for o in ORBITS for f in RADAR)})
            return pd.DataFrame(d)

        def re_rows(i, a, b, idx):
            ra, rb = radar_at(i, idx) - radar_at(a, idx), radar_at(i, idx) - radar_at(b, idx)
            pos = (days[i] - days[a]) / (days[b] - days[a])
            m_lin = PRI[a, idx] + pos * (PRI[b, idx] - PRI[a, idx])
            d = dict(base(i, idx), pos=pos, gap=days[b] - days[a], y_a=y[a, idx], y_b=y[b, idx],
                     m_dev=PRI[i, idx] - m_lin, prior_t=PRI[i, idx],
                     rain_a=rain_cum[s2d[i]] - rain_cum[s2d[a]], rain_b=rain_cum[s2d[b]] - rain_cum[s2d[i]],
                     **{f"da{o}_{f}": ra[:, n] for n, (o, f) in enumerate((o, f) for o in ORBITS for f in RADAR)},
                     **{f"db{o}_{f}": rb[:, n] for n, (o, f) in enumerate((o, f) for o in ORBITS for f in RADAR)})
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
    e.to_csv(os.path.join(OUT, "fusion3_check.csv.gz"), index=False)
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
    s.to_csv(os.path.join(OUT, "fusion3_summary.csv"), index=False)
    b.to_csv(os.path.join(OUT, "fusion3_by_crop.csv"), index=False)
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
