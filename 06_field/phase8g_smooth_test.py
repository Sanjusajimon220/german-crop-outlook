"""Smoothing test for v2 long-gap values (pre-registered docs/project_log.md 2026-10-15).

Paired hiding of fusion3/fusion5; v2 pixels (per-field clearness) and per-pixel neighbour training (fix 1b);
re-analysis model only (LAI with physics features as in production, NDVI without). For every hidden pixel-date t with
anchors a, b: predictions at t + o, o in -21..+21 days (7-day steps); outside (a, b) the anchor value is used.
S0 = p(t); S1 = (p-7 + 2 p0 + p+7) / 4; S2 = (p-14 + 2 p-7 + 3 p0 + 2 p+7 + p+14) / 9.
Outputs smooth_check.csv.gz, smooth_summary.csv. Run `python phase8g_smooth_test.py [summary]`.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from phase7_step3_s1proxy import CROPS, context, neighbours, pixel_groups, prev_next
from radar_tools import field_radar as orbit_radar

P = os.path.join("data", "processed")
OUT = os.path.join(P, "field_v2", "pilot_2024")
RAW = os.path.join("data", "raw", "field_v2", "pilot_2024")
MAIN6 = ["wheat", "barley", "sugar beet", "potato", "maize", "rapeseed"]
RADAR = ["VV", "VH", "RAT", "RVI"]
ORBITS = {"asc": "s1", "desc": "s1_desc"}
NP = "nophys" in sys.argv                                       # satellite-only (no physics features)
SFX = "_nophys" if NP else ""
OFFS = [-21, -14, -7, 0, 7, 14, 21]


def nearest_s1(s1d, d, tol=3):
    k = int(np.argmin(np.abs((s1d - d).days)))
    return k if abs((s1d[k] - d).days) <= tol else -1


def main():
    z = np.load(os.path.join(OUT, "s2_pixels_v2.npz"))
    zo = np.load(os.path.join(OUT, "s2_pixels.npz"))
    s2d, rows, cols = pd.DatetimeIndex(z["dates"]), z["rows"], z["cols"]
    assert np.array_equal(zo["rows"], rows) and np.array_equal(zo["cols"], cols)
    D = pd.DatetimeIndex(sorted(set(s2d) | {d + pd.Timedelta(days=o) for d in s2d for o in OFFS}))   # extended date axis
    si = np.array([D.get_loc(d) for d in s2d])
    px = pd.read_csv(os.path.join(OUT, "pixels_nrw.csv.gz"))
    fl = pd.read_csv(os.path.join(OUT, "fields_nrw.csv"))
    main_f = px.sort_values("share").drop_duplicates(["row", "col"], keep="last").set_index(["row", "col"])["field_id"]
    fid = main_f.reindex(pd.MultiIndex.from_arrays([rows, cols])).values
    grp = pd.Series(fid).map(fl.set_index("field_id")["crop_group"]).fillna("other").values
    crop = np.array([CROPS.index(g) if g in CROPS else CROPS.index("other") for g in grp])
    fids, inv = np.unique(fid, return_inverse=True)
    orb = {o: orbit_radar(os.path.join(RAW, f), rows, cols, inv) for o, f in ORBITS.items()}
    ctx = context(D)
    pri = pd.read_csv(os.path.join(OUT, "prior_lai_wheat.csv.gz"), parse_dates=["date"])
    pri = pri[pri["date"].isin(D)].pivot(index="date", columns="field_id", values="prior_lai").reindex(D)
    PF = pri.reindex(columns=fids).values.astype(np.float32)        # prior per date and FIELD (memory)
    w = np.load(os.path.join(P, "weather_daily_v8.npz"))
    rain_cum = pd.Series(w["pr"][:, list(w["codes"]).index("05358")], index=pd.DatetimeIndex(w["dates"])).cumsum()
    k_s1 = {o: np.array([nearest_s1(orb[o][0], d) for d in D]) for o in ORBITS}
    days = (D - D[0]).days.values
    rng = np.random.default_rng(7)
    folds = {f: i % 5 for i, f in enumerate(rng.permutation(np.unique(fid)))}
    fold = np.array([folds[f] for f in fid])
    names = [f"{o}_{f}" for o in ORBITS for f in RADAR]
    res = []
    for target in ("lai", "ndvi"):
        phys = target == "lai" and not NP
        Y = z[target]                                             # on the 20 S2 dates; si maps to D
        clear = np.isfinite(Y)
        co = np.isfinite(zo[target])                              # the fusion3 hiding on the old 12 dates
        ho = co & (np.random.default_rng(70).random(co.shape) < 0.3)
        ho[np.argmax(co, axis=0), np.arange(co.shape[1])] = False
        ho[co.shape[0] - 1 - np.argmax(co[::-1], axis=0), np.arange(co.shape[1])] = False
        hide = np.zeros_like(clear)
        hide[[s2d.get_loc(d) for d in pd.DatetimeIndex(zo["dates"])]] = ho
        assert np.all(clear[hide])
        kept = clear & ~hide

        def radar_at(i, idx):
            c = []
            for o in ORBITS:
                k = k_s1[o][i]
                for f in RADAR:
                    c.append(np.full(len(idx), np.nan) if k < 0 else orb[o][1][f][k, inv[idx]])
            return np.stack(c, 1)

        def re_rows(i, sa, sb, idx):
            """i: index on D; sa, sb: anchor indices on the S2 axis."""
            a, b = si[sa], si[sb]
            ra, rb = radar_at(i, idx) - radar_at(a, idx), radar_at(i, idx) - radar_at(b, idx)
            pos = (days[i] - days[a]) / (days[b] - days[a])
            c = ctx.iloc[i]
            d = dict(crop=crop[idx], dsin=np.sin(2 * np.pi * c["doy"] / 365), dcos=np.cos(2 * np.pi * c["doy"] / 365),
                     since_heading=c["since_heading"], pos=pos, gap=days[b] - days[a], y_a=Y[sa, idx], y_b=Y[sb, idx])
            if phys:
                q = inv[idx]
                d.update(m_dev=PF[i, q] - (PF[a, q] + pos * (PF[b, q] - PF[a, q])), prior_t=PF[i, q])
            d.update(rain_a=rain_cum[D[i]] - rain_cum[D[a]], rain_b=rain_cum[D[b]] - rain_cum[D[i]],
                     **{f"da{n}": ra[:, k] for k, n in enumerate(names)}, **{f"db{n}": rb[:, k] for k, n in enumerate(names)})
            return pd.DataFrame(d)

        pv, nx = neighbours(kept, 3)
        R, Ry, Rf = [], [], []
        for n in range(len(s2d)):
            for s in range(2):
                for u in range(2):
                    for (a, b), idx in pixel_groups(n, kept[n], (pv[s], nx[u]), 30000, rng):
                        A, B, I = days[si[a]], days[si[b]], days[si[n]]
                        lin = Y[a, idx] + (I - A) / (B - A) * (Y[b, idx] - Y[a, idx])
                        R.append(re_rows(si[n], a, b, idx)); Ry.append(Y[n, idx] - lin); Rf.append(fold[idx])
        XR, yR, fR = pd.concat(R, ignore_index=True), np.concatenate(Ry), np.concatenate(Rf)
        print(f"{target}: re-analysis triples {len(XR)}", flush=True)
        prv, nxt = prev_next(kept)
        for f in range(5):
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=0,
                                              categorical_features=[XR.columns.get_loc("crop")]).fit(XR[fR != f], yR[fR != f])
            for n in range(len(s2d)):
                sel = hide[n] & (fold == f) & (prv[n] >= 0) & (nxt[n] >= 0)
                for (a, b), g in pd.DataFrame({"a": prv[n, sel], "b": nxt[n, sel], "p": np.flatnonzero(sel)}).groupby(["a", "b"]):
                    idx = g["p"].values
                    A, B = days[si[a]], days[si[b]]
                    rec = {"target": target, "pixel": idx, "date": s2d[n], "obs": Y[n, idx], "gap": B - A, "crop": grp[idx]}
                    for o in OFFS:
                        j = D.get_loc(s2d[n] + pd.Timedelta(days=o))
                        if days[j] <= A:
                            v = Y[a, idx]
                        elif days[j] >= B:
                            v = Y[b, idx]
                        else:
                            lin = Y[a, idx] + (days[j] - A) / (B - A) * (Y[b, idx] - Y[a, idx])
                            v = lin + m.predict(re_rows(j, a, b, idx))
                        rec[f"p{o:+d}"] = v
                    res.append(pd.DataFrame(rec))
            print(f"{target}: fold {f} done", flush=True)
    e = pd.concat(res, ignore_index=True)
    e.to_csv(os.path.join(OUT, f"smooth_check{SFX}.csv.gz"), index=False)


def summarise():
    e = pd.read_csv(os.path.join(OUT, f"smooth_check{SFX}.csv.gz"), parse_dates=["date"])
    v1 = pd.read_csv(os.path.join(OUT, "fusion2_check.csv.gz" if NP else "fusion3_check.csv.gz"))
    d1 = pd.DatetimeIndex(np.load(os.path.join(OUT, "s2_pixels.npz"))["dates"])
    v1["date"] = d1[v1["date_i"].values]
    e = e.merge(v1[["target", "pixel", "date", "gap"]].rename(columns={"gap": "gap_v1"}), on=["target", "pixel", "date"])
    p = {o: e[f"p{o:+d}"] for o in OFFS}
    s1 = lambda c: (p[c - 7] + 2 * p[c] + p[c + 7]) / 4
    s2 = lambda c: (p[c - 14] + 2 * p[c - 7] + 3 * p[c] + 2 * p[c + 7] + p[c + 14]) / 9
    e["S0"], e["S1"] = p[0], s1(0)
    e["S2"] = (p[-14] + 2 * p[-7] + 3 * p[0] + 2 * p[7] + p[14]) / 9
    rough = {"S0": (p[7] - 2 * p[0] + p[-7]).abs(), "S1": (s1(7) - 2 * s1(0) + s1(-7)).abs()}
    # S2 roughness would need -28 .. +28; reported only for S0 / S1
    rm = lambda v: float(np.sqrt(np.mean(np.square(v))))
    rows = []
    for t, g in e[e.gap_v1 > 30].groupby("target"):
        for c in ["all"] + MAIN6:
            h = g if c == "all" else g[g.crop == c]
            rows.append(dict(target=t, crop=c, n=len(h), **{k: rm(h[k] - h.obs) for k in ("S0", "S1", "S2")},
                             **{f"rough_{k}": float(rough[k][h.index].mean()) for k in ("S0", "S1")}))
    s = pd.DataFrame(rows).round(4)
    s.to_csv(os.path.join(OUT, f"smooth_summary{SFX}.csv"), index=False)
    print(s.to_string(index=False))
    choice = {}
    for t, thr in (("lai", 0.005), ("ndvi", 0.001)):
        a = s[(s.target == t) & (s.crop == "all")].iloc[0]
        cr = s[(s.target == t) & (s.crop != "all")]
        best = "S1" if a.S1 <= a.S2 else "S2"
        ok = a[best] <= a.S0 - thr and int((cr[best] < cr.S0).sum()) >= 4
        choice[t] = {"use": best if ok else "none", "rmse": float(a[best] if ok else a.S0)}
        print(f"RESULT {t}: best {best} {a[best]:.4f} vs S0 {a.S0:.4f}, crops better {int((cr[best] < cr.S0).sum())}/6 -> "
              f"{'ADOPT ' + best if ok else 'no smoothing'}")
    import json
    json.dump(choice, open(os.path.join(OUT, f"smooth_choice{SFX}.json"), "w"), indent=1)


if __name__ == "__main__":
    if "summary" not in sys.argv:
        main()
    summarise()
