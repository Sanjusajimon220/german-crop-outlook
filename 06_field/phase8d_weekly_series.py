"""Step 1b (pre-registered 2026-10-15): weekly LAI / NDVI series for every pilot field, season 2024.

Monday grid Oct 2023 - Aug 2024. Per PIXEL and week W, using all clear S2 values (nothing hidden in production):
  measured     W is a clear S2 date of the pixel (+-3 days)
  interpolated gap between the previous and next clear dates <= 15 days
  re-analysis  both neighbouring clear dates exist (gap > 15 d): line + model deviation
  live         only an earlier clear date: last clear value + model change (gap > 30 d per the adopted rule;
               for 16-30 d live gaps the value stays interpolated from the last two clear dates is NOT possible,
               so live 16-30 d is labelled 'live-short' with the live model and its larger uncertainty)
Models: the adopted fusion versions, refitted on ALL clear pixel-dates of 2024 (radar from radar_tools, both orbits;
physics features for wheat if adopted). Field value = area-weighted mean of its pixels (main-field pixels, step-1
shares); source = the source of most of the field's area; uncertainty = test RMSE of that source and gap class
(fusion test tables). Output field_v2/pilot_2024/weekly_field_series_2024.csv
Run `python phase8d_weekly_series.py [physics]`.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from phase7_step3_s1proxy import CROPS, context, prev_next
from radar_tools import field_radar as orbit_radar

P = os.path.join("data", "processed")
OUT = os.path.join(P, "field_v2", "pilot_2024")
RAW = os.path.join("data", "raw", "field_v2", "pilot_2024")
RADAR = ["VV", "VH", "RAT", "RVI"]
ORBITS = {"asc": "s1", "desc": "s1_desc"}
PHYS_ARG = "physics" in sys.argv          # physics features for LAI only (adopted 2026-10-15; NDVI without)
PHYS = False


def nearest(dates, d, tol=3):
    k = int(np.argmin(np.abs((dates - d).days)))
    return k if abs((dates[k] - d).days) <= tol else -1


def main():
    z = np.load(os.path.join(OUT, "s2_pixels.npz"))
    s2d, rows, cols = pd.DatetimeIndex(z["dates"]), z["rows"], z["cols"]
    px = pd.read_csv(os.path.join(OUT, "pixels_nrw.csv.gz"))
    fl = pd.read_csv(os.path.join(OUT, "fields_nrw.csv"))
    mainf = px.sort_values("share").drop_duplicates(["row", "col"], keep="last").set_index(["row", "col"])
    key = pd.MultiIndex.from_arrays([rows, cols])
    fid, share = mainf["field_id"].reindex(key).values, mainf["share"].reindex(key).values
    grp = pd.Series(fid).map(fl.set_index("field_id")["crop_group"]).fillna("other").values
    crop = np.array([CROPS.index(g) if g in CROPS else CROPS.index("other") for g in grp])
    fids, inv = np.unique(fid, return_inverse=True)
    orb = {o: orbit_radar(os.path.join(RAW, f), rows, cols, inv) for o, f in ORBITS.items()}
    weeks = pd.date_range("2023-10-02", "2024-08-26", freq="W-MON")
    alld = s2d.union(weeks)                                      # S2 dates + weekly dates
    ctx = context(alld)
    w = np.load(os.path.join(P, "weather_daily_v8.npz"))
    rain_cum = pd.Series(w["pr"][:, list(w["codes"]).index("05358")], index=pd.DatetimeIndex(w["dates"])).cumsum()
    kr = {o: np.array([nearest(orb[o][0], d) for d in alld]) for o in ORBITS}
    if PHYS_ARG:
        pri = pd.read_csv(os.path.join(OUT, "prior_lai_wheat.csv.gz"), parse_dates=["date"])
        pri = pri[pri["date"].isin(alld)].pivot(index="date", columns="field_id", values="prior_lai").reindex(alld)
        PRI = np.full((len(alld), len(rows)), np.nan, np.float32)
        has = np.isin(fid, pri.columns)
        PRI[:, has] = pri[fid[has]].values
    days = (alld - alld[0]).days.values
    si = np.array([alld.get_loc(d) for d in s2d])

    def radar_at(i, idx):
        c = []
        for o in ORBITS:
            k = kr[o][i]
            for f in RADAR:
                c.append(np.full(len(idx), np.nan) if k < 0 else orb[o][1][f][k, inv[idx]])
        return np.stack(c, 1)

    def base(i, idx):
        c = ctx.iloc[i]
        return dict(crop=crop[idx], dsin=np.sin(2 * np.pi * c["doy"] / 365), dcos=np.cos(2 * np.pi * c["doy"] / 365),
                    since_heading=c["since_heading"])

    names = [f"{o}_{f}" for o in ORBITS for f in RADAR]

    def live_rows(i, a, idx, y):
        r = radar_at(i, idx) - radar_at(a, idx)
        d = dict(base(i, idx), days=days[i] - days[a], y_a=y[a, idx])
        if PHYS:
            d.update(m_chg=PRI[i, idx] - PRI[a, idx], prior_t=PRI[i, idx])
        d.update(rain=rain_cum[alld[i]] - rain_cum[alld[a]], **{f"d{n}": r[:, k] for k, n in enumerate(names)})
        return pd.DataFrame(d)

    def re_rows(i, a, b, idx, y):
        ra, rb = radar_at(i, idx) - radar_at(a, idx), radar_at(i, idx) - radar_at(b, idx)
        pos = (days[i] - days[a]) / (days[b] - days[a])
        d = dict(base(i, idx), pos=pos, gap=days[b] - days[a], y_a=y[a, idx], y_b=y[b, idx])
        if PHYS:
            d.update(m_dev=PRI[i, idx] - (PRI[a, idx] + pos * (PRI[b, idx] - PRI[a, idx])), prior_t=PRI[i, idx])
        d.update(rain_a=rain_cum[alld[i]] - rain_cum[alld[a]], rain_b=rain_cum[alld[b]] - rain_cum[alld[i]],
                 **{f"da{n}": ra[:, k] for k, n in enumerate(names)}, **{f"db{n}": rb[:, k] for k, n in enumerate(names)})
        return pd.DataFrame(d)

    rng = np.random.default_rng(7)
    rmse = pd.read_csv(os.path.join(OUT, "fusion3_summary.csv" if PHYS_ARG else "fusion2_summary.csv"))
    out = {}
    global PHYS
    for t in ("lai", "ndvi"):
        PHYS = PHYS_ARG and t == "lai"
        y = np.full((len(alld), len(rows)), np.nan, np.float32)
        y[si] = z[t]
        clear = np.isfinite(y)
        # production models on ALL clear values (pairs / triples along the S2 dates)
        L, Ly, R, Ry = [], [], [], []
        for n, i in enumerate(si):
            for a in si[max(0, n - 3):n]:
                idx = np.flatnonzero(clear[i] & clear[a])
                idx = rng.choice(idx, min(len(idx), 40000), replace=False) if len(idx) else idx
                if len(idx):
                    L.append(live_rows(i, a, idx, y)); Ly.append(y[i, idx] - y[a, idx])
            for a in si[max(0, n - 2):n]:
                for b in si[n + 1:n + 3]:
                    idx = np.flatnonzero(clear[i] & clear[a] & clear[b])
                    idx = rng.choice(idx, min(len(idx), 30000), replace=False) if len(idx) else idx
                    if len(idx):
                        lin = y[a, idx] + (days[i] - days[a]) / (days[b] - days[a]) * (y[b, idx] - y[a, idx])
                        R.append(re_rows(i, a, b, idx, y)); Ry.append(y[i, idx] - lin)
        XL, XR = pd.concat(L, ignore_index=True), pd.concat(R, ignore_index=True)
        mL = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=0,
                                           categorical_features=[XL.columns.get_loc("crop")]).fit(XL, np.concatenate(Ly))
        mR = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=0,
                                           categorical_features=[XR.columns.get_loc("crop")]).fit(XR, np.concatenate(Ry))
        prv, nxt = prev_next(clear)
        val = np.full((len(weeks), len(rows)), np.nan, np.float32)
        src = np.zeros((len(weeks), len(rows)), np.int8)            # 1 measured 2 interp 3 re-analysis 4 live
        gapv = np.full((len(weeks), len(rows)), np.nan, np.float32)
        since = np.full((len(weeks), len(rows)), np.nan, np.float32)
        for wi, wd in enumerate(weeks):
            i = alld.get_loc(wd)
            near = [j for j in si if abs(days[j] - days[i]) <= 3]
            meas = np.full(len(rows), np.nan, np.float32)
            for j in near:                                           # a clear S2 value within +-3 days
                meas = np.where(np.isnan(meas) & clear[j], y[j], meas)
            m = np.isfinite(meas)
            val[wi, m], src[wi, m] = meas[m], 1
            a, b = prv[i], nxt[i]
            since[wi] = np.where(a >= 0, days[i] - days[np.maximum(a, 0)], np.nan)
            gap = np.where((a >= 0) & (b >= 0), days[np.maximum(b, 0)] - days[np.maximum(a, 0)], 999)
            gapv[wi] = gap
            for (aa, bb), g in pd.DataFrame({"a": a, "b": b, "p": np.arange(len(rows))})[~m].groupby(["a", "b"]):
                idx = g["p"].values
                if aa < 0:
                    continue
                if bb >= 0:
                    lin = y[aa, idx] + (days[i] - days[aa]) / (days[bb] - days[aa]) * (y[bb, idx] - y[aa, idx])
                    if days[bb] - days[aa] <= 15:
                        val[wi, idx], src[wi, idx] = lin, 2
                    else:
                        val[wi, idx], src[wi, idx] = lin + mR.predict(re_rows(i, aa, bb, idx, y)), 3
                else:
                    val[wi, idx], src[wi, idx] = y[aa, idx] + mL.predict(live_rows(i, aa, idx, y)), 4
        out[t] = (val, src, gapv, since)
        print(f"{t}: weekly pixel series {val.shape}; share measured {np.mean(src == 1):.2f}, interpolated "
              f"{np.mean(src == 2):.2f}, re-analysis {np.mean(src == 3):.2f}, live {np.mean(src == 4):.2f}", flush=True)
    np.savez_compressed(os.path.join(OUT, "weekly_pixel_series_2024.npz"), weeks=np.array(weeks, dtype="datetime64[D]"),
                        rows=rows, cols=cols, lai=out["lai"][0], lai_src=out["lai"][1], lai_gap=out["lai"][2],
                        lai_since=out["lai"][3], ndvi=out["ndvi"][0])
    aggregate(weeks, out, inv, fids, share, fl, rmse)


def aggregate(weeks, out, inv, fids, share, fl, rmse):
    """Field means (area-weighted by the main-field share) + source label + uncertainty, with array sums."""
    gc = lambda g: np.where(g <= 15, "<=15 d", np.where(g <= 30, "16-30 d", ">30 d"))
    lai, lsrc, lgap, lsince = out["lai"]
    ndvi = out["ndvi"][0]
    lai = np.clip(lai, 0, None)                       # green LAI cannot be negative (fix 2026-10-15)
    ndvi = np.clip(ndvi, -1, 1)
    wgt = np.nan_to_num(share).astype(float)
    F = len(fids)
    names = {1: "measured", 2: "interpolated", 3: "re-analysis", 4: "live"}
    recs = []
    for wi, wd in enumerate(weeks):
        ok = np.isfinite(lai[wi]) & (wgt > 0)
        ww = np.where(ok, wgt, 0)
        sw = np.bincount(inv, weights=ww, minlength=F)
        f_lai = np.bincount(inv, weights=np.where(ok, lai[wi], 0) * ww, minlength=F) / np.where(sw > 0, sw, np.nan)
        okn = ok & np.isfinite(ndvi[wi])
        wn = np.where(okn, wgt, 0)
        swn = np.bincount(inv, weights=wn, minlength=F)
        f_ndvi = np.bincount(inv, weights=np.where(okn, ndvi[wi], 0) * wn, minlength=F) / np.where(swn > 0, swn, np.nan)
        area_by_src = np.stack([np.bincount(inv, weights=ww * (lsrc[wi] == k), minlength=F) for k in (1, 2, 3, 4)], 1)
        f_src = np.where(sw > 0, area_by_src.argmax(1) + 1, 0)
        f_gap = np.bincount(inv, weights=np.where(ok, np.nan_to_num(lgap[wi]), 0) * ww, minlength=F) / np.where(sw > 0, sw, np.nan)
        f_since = np.bincount(inv, weights=np.where(ok, np.nan_to_num(lsince[wi]), 0) * ww, minlength=F) / np.where(sw > 0, sw, np.nan)
        keep = sw > 0
        recs.append(pd.DataFrame({"field_id": fids[keep], "week": wd.date(), "lai": f_lai[keep].round(3),
                                  "ndvi": f_ndvi[keep].round(3), "source": [names[k] for k in f_src[keep]],
                                  "gap_days": f_gap[keep].round(0), "days_since_clear": f_since[keep].round(0)}))
    res = pd.concat(recs, ignore_index=True)
    res["crop"] = res["field_id"].map(fl.set_index("field_id")["crop_group"])
    lt = rmse[rmse["target"] == "lai"].set_index("gap")
    unc = []
    for s, gp in zip(res["source"], res["gap_days"]):
        if s == "measured":
            unc.append(0.15)                                   # retrieval uncertainty, field mean
        else:
            col = {"interpolated": "interp", "re-analysis": "reanalysis", "live": "live"}[s]
            unc.append(float(lt.loc[str(gc(np.array([gp]))[0]), col]))
    res["lai_uncertainty"] = np.round(unc, 3)
    res.to_csv(os.path.join(OUT, "weekly_field_series_2024.csv"), index=False)
    print(f"weekly field series: {res['field_id'].nunique()} fields x {res['week'].nunique()} weeks; source shares "
          f"{res['source'].value_counts(normalize=True).round(3).to_dict()}")


if __name__ == "__main__":
    if "aggregate" in sys.argv:                       # reuse the saved pixel series
        z = np.load(os.path.join(OUT, "weekly_pixel_series_2024.npz"))
        px = pd.read_csv(os.path.join(OUT, "pixels_nrw.csv.gz"))
        fl = pd.read_csv(os.path.join(OUT, "fields_nrw.csv"))
        mainf = px.sort_values("share").drop_duplicates(["row", "col"], keep="last").set_index(["row", "col"])
        key = pd.MultiIndex.from_arrays([z["rows"], z["cols"]])
        fid, share = mainf["field_id"].reindex(key).values, mainf["share"].reindex(key).values
        fids, inv = np.unique(fid, return_inverse=True)
        out = {"lai": (z["lai"], z["lai_src"], z["lai_gap"], z["lai_since"]), "ndvi": (z["ndvi"],)}
        rmse = pd.read_csv(os.path.join(OUT, "fusion3_summary.csv" if PHYS_ARG else "fusion2_summary.csv"))
        aggregate(pd.DatetimeIndex(z["weeks"]), out, inv, fids, share, fl, rmse)
    else:
        main()
