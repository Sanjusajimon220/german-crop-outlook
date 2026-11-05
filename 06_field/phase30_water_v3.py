"""Field water v3: v2 (FAO-56 dual Kc, field ET0, Kcb from weekly satellite LAI) as an ensemble of 2,000 versions, weighted
every week by Sentinel-1 radar soil wetness, with a forecast from cut dates (pre-registered docs/project_log.md 2026-11-05).

Versions vary TAW, maximum root depth, Kcb_mid, start depletion and TEW. Radar: change-detection wetness index of the
field's VV per orbit (5th / 95th percentile of the season), compared with each version's surface wetness 1 - De / TEW on
radar dates with field LAI < 2. Forecast tests F1 (tower ET) and F2 (soil sensors) at ICOS DE-RuS.
Run from D:\\vista: python phase30_water_v3.py -> data/processed/field_v2/water_v3/.
"""
import os

import numpy as np
import pandas as pd

from phase29_water_v2 import KC_MAX, ZE_REW

N_VERSIONS, MIN_ESS, RADAR_ERR, RADAR_MAX_LAI = 2000, 50, 0.25, 2.0
RANGES = {"taw_f": (0.5, 1.5), "root_f": (0.6, 1.4), "kcb_f": (0.8, 1.2), "dr0": (0.0, 0.5), "tew_f": (0.7, 1.3)}
OUT = os.path.join("data", "processed", "field_v2", "water_v3")


def draw_versions(n=N_VERSIONS, seed=0):
    """Latin hypercube over the uncertain settings (even spread per setting)."""
    rng = np.random.default_rng(seed)
    out = {}
    for k, (lo, hi) in RANGES.items():
        u = (np.arange(n) + rng.random(n)) / n
        rng.shuffle(u)
        out[k] = lo + u * (hi - lo)
    return out


def ensemble_balance(et0, rain, kcb, lai, taw, p, tew, versions, rew=ZE_REW):
    """The v2 dual-Kc balance for all versions at once. taw / kcb: days (scaled by taw_f / kcb_f) or versions x days
    (already per version).
    Returns e, t, et, ks, dr, de, taw as versions x days."""
    n, m = len(et0), len(versions["kcb_f"])
    taw2 = np.asarray(taw, float)
    taw2 = taw2[None, :] * versions["taw_f"][:, None] if taw2.ndim == 1 else taw2
    kcb2 = np.asarray(kcb, float)
    kcb2 = np.minimum(kcb2[None, :] * versions["kcb_f"][:, None] if kcb2.ndim == 1 else kcb2, KC_MAX)
    tew2 = tew * versions["tew_f"]
    out = {k: np.zeros((m, n)) for k in ("e", "t", "et", "ks", "dr", "de")}
    de = np.zeros(m)
    dr = versions["dr0"] * taw2[:, 0]
    for i in range(n):
        de, dr = np.maximum(de - rain[i], 0.0), np.maximum(dr - rain[i], 0.0)
        few = max(1 - (1 - np.exp(-0.5 * max(lai[i], 0.0))), 0.01)
        kr = np.where(de <= rew, 1.0, np.maximum((tew2 - de) / np.maximum(tew2 - rew, 1e-9), 0.0))
        ke = np.minimum(kr * (KC_MAX - kcb2[:, i]), few * KC_MAX)
        e = ke * et0[i]
        raw = p * taw2[:, i]
        ks = np.where(dr <= raw, 1.0, np.maximum((taw2[:, i] - dr) / np.maximum(taw2[:, i] - raw, 1e-9), 0.0))
        t = np.minimum(ks * kcb2[:, i] * et0[i], np.maximum(taw2[:, i] - dr, 0.0))
        de = np.minimum(de + e / few, tew2)
        dr = np.minimum(dr + e + t, taw2[:, i])
        out["e"][:, i], out["t"][:, i], out["et"][:, i], out["ks"][:, i], out["dr"][:, i], out["de"][:, i] = e, t, e + t, ks, dr, de
    out["taw"], out["tew"] = taw2, tew2
    return out


def radar_wetness(vv):
    """Change-detection wetness index 0-1 from a season of VV (dB): 5th / 95th percentile = dry / wet."""
    vv = np.asarray(vv, float)
    lo, hi = np.nanpercentile(vv, 5), np.nanpercentile(vv, 95)
    return np.clip((vv - lo) / max(hi - lo, 1e-6), 0, 1)


def weigh(model, obs, err=RADAR_ERR, min_ess=MIN_ESS):
    """Gaussian weights of versions (rows) against observations (columns), tempered until >= min_ess effective versions."""
    m = model.shape[0]
    if model.shape[1] == 0:
        return np.full(m, 1.0 / m)
    ll = -0.5 * (((model - obs[None, :]) / err) ** 2).sum(1)
    temp = 1.0
    for _ in range(40):
        a = temp * ll
        w = np.exp(a - a.max())
        w /= w.sum()
        if 1.0 / (w ** 2).sum() >= min_ess:
            return w
        temp *= 0.7
    return w


def wquantile(x, w, q):
    """Weighted quantile(s) over versions (rows of x) for every column. q scalar -> columns; list -> len(q) x columns."""
    o = np.argsort(x, axis=0)
    xs = np.take_along_axis(x, o, 0)
    cw = np.cumsum(w[o], axis=0)
    cw = (cw - 0.5 * w[o]) / cw[-1]
    qs = np.atleast_1d(q)
    out = np.array([[np.interp(qq, cw[:, j], xs[:, j]) for j in range(x.shape[1])] for qq in qs])
    return out[0] if np.ndim(q) == 0 else out


def weekly_sums(model, tower, valid, week, min_days=5):
    """Weekly sums over the tower's valid days. model: versions x days. Weeks with < min_days valid days are dropped."""
    weeks = np.unique(week)
    m = np.array([[row[(week == k) & valid].sum() for k in weeks] for row in model])
    t = np.array([tower[(week == k) & valid].sum() for k in weeks])
    keep = np.array([((week == k) & valid).sum() >= min_days for k in weeks])
    return m, t, keep


# ---------------------------------------------------------------- DE-RuS runner (prep = no tower / sensor data; look = one look)
CUTS = ("05-01", "06-01", "07-01")
DECISION = os.path.join(OUT, "potato_2022_decision.json")


def field_wetness(year, fid):
    """Radar wetness index per orbit for one field (field-mean VV as in the weekly series), dates 1 March - 30 September."""
    from fv2 import pixels, radar
    px = pixels("selhausen", year)
    j = int(np.where(px["fids"] == fid)[0][0])
    m = px["inv"] == j
    one = dict(rows=px["rows"][m], cols=px["cols"][m], inv=np.zeros(int(m.sum()), int))
    out = []
    for orbit, (dates, f) in radar("selhausen", year, one).items():
        s = pd.Series(f["VV"][:, 0], index=dates)
        s = s[(s.index >= f"{year}-03-01") & (s.index <= f"{year}-09-30")].dropna()
        out.append(pd.DataFrame({"date": s.index.normalize(), "orbit": orbit, "vv": s.values, "w": radar_wetness(s.values)}))
    return pd.concat(out, ignore_index=True)


def season_ensemble(year, versions, irrigation=None):
    """All versions over 1 March - 30 September: v2 inputs, Kcb_mid and root depth per version."""
    from phase29_water_v2 import (FIELDS, KCB_MID, P_TAB, ROOT_MAX, _soil, _taw, kcb_from_lai, site_season,
                                  site_weather)
    S = site_season(year)
    crop, dates = S["crop"], pd.DatetimeIndex(S["dates"])
    import json
    s = json.load(open(os.path.join("D:/german-crop-outlook-field/docs/field", f"selhausen_{year}", "series.json")))
    f = s["fields"][str(FIELDS[year][0])]
    lai_w = pd.Series([np.nan if v is None else v for v in f["lai"]], index=pd.DatetimeIndex(s["weeks"]))
    lai = lai_w.reindex(lai_w.index.union(dates)).interpolate(limit_area="inside").reindex(dates).fillna(0).values
    w = site_weather(dates)
    rain = w["pr"].values.copy()
    if irrigation is not None:
        rain += pd.Series(irrigation).reindex(dates.strftime("%Y-%m-%d")).fillna(0).values
    paw, tew = _soil()
    frac = np.clip(np.asarray((dates - S["start"]).days) / max((S["peak"] - S["start"]).days, 1), 0, 1)
    depth = 0.1 + (ROOT_MAX[crop] * versions["root_f"][:, None] - 0.1) * frac[None, :]
    taw = np.stack([_taw(paw, d) for d in depth]) * versions["taw_f"][:, None]
    kcb = np.stack([kcb_from_lai(lai, KCB_MID[crop] * k) for k in versions["kcb_f"]])
    ens = ensemble_balance(w["et0"].values, rain, kcb, lai, taw, P_TAB[crop], tew, versions)
    return dict(S=S, dates=dates, lai=lai, ens=ens, tew=tew)


def cut_weights(E, wet, cut):
    """Weights from radar dates before the cut with field LAI < 2 (both orbits, cumulative)."""
    idx = E["dates"]
    ok = (wet["date"] < cut) & wet["date"].isin(idx)
    sel = wet[ok]
    pos = idx.get_indexer(sel["date"])
    sel, pos = sel[E["lai"][pos] < RADAR_MAX_LAI], pos[E["lai"][pos] < RADAR_MAX_LAI]
    surf = 1 - E["ens"]["de"][:, pos] / E["ens"]["tew"][:, None]
    return weigh(surf, sel["w"].values), len(pos)


def prep():
    """Radar wetness, weights and effective versions per season and cut. Uses no tower or soil-sensor data."""
    from phase29_water_v2 import FIELDS
    os.makedirs(OUT, exist_ok=True)
    V = draw_versions()
    rows = []
    for year in sorted(FIELDS):
        wet = field_wetness(year, FIELDS[year][0])
        wet.to_csv(os.path.join(OUT, f"radar_wetness_{year}.csv"), index=False)
        E = season_ensemble(year, V)
        for c in CUTS:
            w, n = cut_weights(E, wet, pd.Timestamp(f"{year}-{c}"))
            top = {k: float(np.sum(w * V[k])) for k in RANGES}
            rows.append(dict(year=year, crop=E["S"]["crop"], cut=c, radar_dates=n, ess=1 / float((w ** 2).sum()),
                             **{f"wmean_{k}": v for k, v in top.items()}))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "prep_weights.csv"), index=False)



def _swc():
    """Measured soil water, daily depth mean of the 5 ICOS sensors (QC <= 1), as in v2 T2."""
    from phase29_water_v2 import SWC
    cols = [f"SWC_F_MDS_{k}" for k in range(1, 6)]
    d = pd.read_csv(SWC, usecols=["TIMESTAMP_END"] + cols + [c + "_QC" for c in cols], na_values=[-9999])
    t = pd.to_datetime(d["TIMESTAMP_END"].astype(str), format="%Y%m%d%H%M")
    day = (t - pd.Timedelta(minutes=1)).dt.normalize()
    v = pd.DataFrame({c: d[c].where(d[c + "_QC"] <= 1) for c in cols})
    return v.groupby(day).mean().mean(axis=1, skipna=True)


def look():
    """THE one look (F1 tower ET, F2 soil water, 80 % band). Refuses to run before the potato 2022 decision is on file."""
    import json
    from phase28_water_job3 import ec_daily_gapfilled
    from phase29_water_v2 import FIELDS, site_season
    if not os.path.exists(DECISION):
        raise SystemExit(f"no potato 2022 decision on file ({DECISION}) - log it before the look")
    dec = json.load(open(DECISION))
    V = draw_versions()
    ec, swc = ec_daily_gapfilled(), _swc()
    f1, f2, band = [], [], []
    for year in sorted(FIELDS):
        if year == 2022 and dec.get("handling") == "leave_out":
            continue
        irr = dec.get("irrigation_mm") if year == 2022 else None
        E = season_ensemble(year, V, irrigation=irr)
        idx = E["dates"]
        single = site_season(year)["single"] if irr is None else None
        if single is None:                                  # plain FAO-56 with the same irrigation added
            from phase28_water_job3 import fao56_et
            from phase29_water_v2 import KC_TAB, P_TAB, _stage_curve, site_weather
            S = E["S"]
            rain = site_weather(idx)["pr"].values + pd.Series(irr).reindex(idx.strftime("%Y-%m-%d")).fillna(0).values
            single = fao56_et(S["et0"], rain, _stage_curve(idx, S["start"], S["peak"], S["end"], *KC_TAB[S["crop"]]),
                              S["taw"], P_TAB[S["crop"]])
        wet = pd.read_csv(os.path.join(OUT, f"radar_wetness_{year}.csv"), parse_dates=["date"])
        rel = 1 - E["ens"]["dr"] / np.maximum(E["ens"]["taw"], 1e-6)
        for c in CUTS:
            cut = pd.Timestamp(f"{year}-{c}")
            w, _ = cut_weights(E, wet, cut)
            win = (idx >= cut) & (idx < cut + pd.Timedelta(days=28))
            wk = ((idx[win] - cut).days // 7).values
            valid = idx[win].isin(ec.index)
            tower = ec.reindex(idx[win]).fillna(0).values
            m, t, keep = weekly_sums(E["ens"]["et"][:, win], tower, valid, wk)
            s1, _, _ = weekly_sums(single[None, win], tower, valid, wk)
            q = wquantile(m, w, [0.1, 0.5, 0.9])
            v2 = np.median(m, axis=0)
            for k in np.where(keep)[0]:
                f1.append(dict(year=year, crop=E["S"]["crop"], cut=c, week=k + 1, tower=t[k], v3=q[1, k], v2=v2[k],
                               fao56=s1[0, k], lo=q[0, k], hi=q[2, k]))
            ms = swc.reindex(idx[win])
            ok = ms.notna().values
            if ok.sum() >= 10:
                r3 = wquantile(rel[:, win][:, ok], w, 0.5)
                r2 = np.median(rel[:, win][:, ok], axis=0)
                f2.append(pd.DataFrame(dict(year=year, cut=c, v3=r3 - r3.mean(), v2=r2 - r2.mean(),
                                            meas=ms.values[ok] - ms.values[ok].mean())))
        print(year, E["S"]["crop"], "done", flush=True)
    F1, F2 = pd.DataFrame(f1), pd.concat(f2)
    rm = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    rows, wins = [], 0
    for c in CUTS:
        d = F1[F1.cut == c]
        r = {m: rm(d[m], d.tower) for m in ("v3", "v2", "fao56")}
        win = r["v3"] < r["v2"] and r["v3"] < r["fao56"]
        wins += win
        rows.append(dict(cut=c, weeks=len(d), **{f"rmse_{k}": v for k, v in r.items()}, v3_best=win))
    r2v3 = float(np.corrcoef(F2.v3, F2.meas)[0, 1]); r2v2 = float(np.corrcoef(F2.v2, F2.meas)[0, 1])
    cover = float(((F1.tower >= F1.lo) & (F1.tower <= F1.hi)).mean())
    ok = wins >= 2 and r2v3 >= 0.7 and cover >= 0.7
    F1.to_csv(os.path.join(OUT, "f1_weeks.csv"), index=False); F2.to_csv(os.path.join(OUT, "f2_days.csv"), index=False)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "f1_scores.csv"), index=False)
    pd.DataFrame([dict(cuts_v3_best=wins, f2_r_v3=r2v3, f2_r_v2=r2v2, band80_cover=cover, potato=dec.get("handling"),
                       passed=ok)]).to_csv(os.path.join(OUT, "look_summary.csv"), index=False)
    print(R.round(3).to_string(index=False))
    print(f"F2 anomaly r: v3 {r2v3:.3f}, v2 {r2v2:.3f}; 80 % band holds the tower week in {cover:.0%}")
    print(f"RESULT v3: {'PASS (indicative)' if ok else 'FAIL'} (v3 best at >= 2 of 3 cuts, F2 r >= 0.7, band >= 70 %)")
    d, e = F1[F1.year != 2022], F2[F2.year != 2022]                    # secondary view, declared before the look
    sec = [dict(cut=c, **{f"rmse_{k}": rm(d[d.cut == c][k], d[d.cut == c].tower) for k in ("v3", "v2", "fao56")})
           for c in CUTS]
    print("SECONDARY (without potato 2022, not deciding):")
    print(pd.DataFrame(sec).round(3).to_string(index=False))
    print(f"  F2 r v3 {np.corrcoef(e.v3, e.meas)[0, 1]:.3f}; band {((d.tower >= d.lo) & (d.tower <= d.hi)).mean():.0%}")
    return ok


if __name__ == "__main__":
    import sys
    {"prep": prep, "look": look}[sys.argv[1]]()
