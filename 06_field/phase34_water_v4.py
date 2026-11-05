"""Field water v4 (thermal, SETMI-style): the v3 ensemble of FAO-56 dual-Kc versions, weighted by TSEB thermal ET from
Landsat 8/9 surface temperature instead of radar wetness. Pre-registered in docs/project_log.md 2026-11-06, judged at the
ICOS flux-tower test sites DE-Kli and DE-Geb (never on the platform).

prep  thermal ET per test season (no tower or soil-sensor data used) -> water_v4/thermal_<area>_<year>.csv, prep log
look  THE one look: F1 weekly tower ET after the cuts, F2 soil-water anomaly r, 80 % band; secondary views
Run from D:\\vista: python phase34_water_v4.py prep|look
"""
import os
import sys
import types

import numpy as np
import pandas as pd

# pyTSEB imports pypro4sail (not on PyPI) only for its 4SAIL variant, which v4 never calls: stand-in that refuses use.
if "pypro4sail" not in sys.modules:
    _p, _f = types.ModuleType("pypro4sail"), types.ModuleType("pypro4sail.four_sail")

    def _no4sail(*a, **k):
        raise NotImplementedError("4SAIL is not used in water v4")
    _f.foursail = _no4sail
    _p.four_sail = _f
    sys.modules["pypro4sail"], sys.modules["pypro4sail.four_sail"] = _p, _f

from pyTSEB import TSEB, net_radiation as rad, resistances as res   # noqa: E402

OUT = os.path.join("data", "processed", "field_v2", "water_v4")
N_VERSIONS, MIN_ESS = 2000, 50
RANGES = {"taw_f": (0.5, 1.5), "root_f": (0.6, 1.4), "kcb_f": (0.7, 1.3), "dr0": (0.0, 0.5), "tew_f": (0.7, 1.3)}
H_MAX = {"wheat": 0.9, "barley": 0.9, "maize": 2.5, "potato": 0.6}
KC_TAB = {"wheat": (0.40, 1.15, 0.30), "barley": (0.30, 1.15, 0.25), "maize": (0.30, 1.20, 0.35),
          "potato": (0.50, 1.15, 0.75)}                                 # FAO-56 Table 12
JUDGED = [("dekli", 2021), ("dekli", 2023), ("dekli", 2025), ("degeb", 2021), ("degeb", 2022), ("degeb", 2024)]
SEPARATE = [("dekli", 2022)]                                            # spring barley
INDICATIVE = [("dekli", 2018), ("degeb", 2020)]                          # maize, potato
CUTS = ("05-01", "06-01", "07-01")
LEAF_WIDTH = 0.05      # ruling 2026-11-06 (pre-registration silent): typical crop leaf width; pyTSEB default 0.1 is for trees
LAMBDA = 2.45e6
SIGMA = 5.670374e-8


# ------------------------------------------------------------------------------------------------- pure functions
def air_temp_at(tmin, tmax, hour, sunrise):
    """Air temperature at local solar hour: sine rise from Tmin at sunrise to Tmax at 14:00."""
    f = np.clip((np.asarray(hour, float) - sunrise) / (14.0 - sunrise), 0, 1)
    return tmin + (tmax - tmin) * np.sin(0.5 * np.pi * f)


def rs_instant(rs_mj, hour, daylength):
    """Instantaneous sunlight (W/m2) from the daily total (MJ/m2) on a sine day shape."""
    h = np.asarray(hour, float) - (12.0 - daylength / 2)
    return np.where((h > 0) & (h < daylength),
                    rs_mj * 1e6 * np.pi / (2 * daylength * 3600) * np.sin(np.pi * h / daylength), 0.0)


def daily_et(le_inst, rs_inst, rs_mj):
    """Daily ET (mm) by the solar ratio: evaporative share of instantaneous sunlight x daily sunlight."""
    return le_inst / rs_inst * rs_mj * 1e6 / LAMBDA


def screen(lst, qa_pixel, st_qa, t_air, max_unc=3.0, cold=5.0):
    """Clear (qa_pixel bit 6, bits 1-4 off), ST uncertainty <= 3 K, not colder than air - 5 K (missed cloud)."""
    q = np.asarray(qa_pixel).astype(np.int64)
    clear = ((q >> 6) & 1 == 1) & (q & 0b11110 == 0)
    return clear & (np.asarray(st_qa) <= max_unc) & (np.asarray(lst) >= t_air - cold) & np.isfinite(lst)


def interior_points(mask, rows, cols, min_m=100.0, px_m=10.0):
    """True where the 10 m pixel (rows, cols) lies >= min_m inside the field mask."""
    from scipy import ndimage
    d = ndimage.distance_transform_edt(mask) * px_m
    inside = (rows >= 0) & (cols >= 0) & (rows < mask.shape[0]) & (cols < mask.shape[1])
    out = np.zeros(len(rows), bool)
    out[inside] = d[rows[inside], cols[inside]] >= min_m
    return out


def obs_error(et):
    return np.maximum(0.3 * np.asarray(et, float), 0.5)


def solar(lat, doy, hour):
    """(sunrise solar hour, daylength h, solar zenith deg at hour) - FAO-56 eqs. 24-25, 34."""
    phi = np.radians(lat)
    dec = 0.409 * np.sin(2 * np.pi / 365 * doy - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
    n = 24 / np.pi * ws
    cz = np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(np.radians(15 * (hour - 12)))
    return 12 - n / 2, n, float(np.degrees(np.arccos(np.clip(cz, -1, 1))))


def tseb(tr_k, ta_k, lai, h_c, u, ea_mb, p_mb, rs, sza, z_u=10.0, z_t=2.0):
    """TSEB Priestley-Taylor for one field and time (pyTSEB 2.5.4). Returns flag and LE / H / Rn parts (W/m2)."""
    a = lambda v: np.array([float(v)])
    lai = max(float(lai), 0.05)
    fc = 1 - np.exp(-0.5 * lai)
    difvis, difnir, fvis, fnir = rad.calc_difuse_ratio(a(rs), a(sza), press=a(p_mb))
    skyl = fvis * difvis + fnir * difnir
    s_dir, s_dif = a(rs) * (1 - skyl), a(rs) * skyl
    sn_c, sn_s = rad.calc_Sn_Campbell(a(lai), a(sza), s_dir, s_dif, fvis, fnir, a(0.07), a(0.08), a(0.32), a(0.33),
                                      a(0.15), a(0.25), x_LAD=a(1))
    l_dn = rad.calc_emiss_atm(a(ea_mb), a(ta_k)) * SIGMA * a(ta_k) ** 4
    z0m, d0 = res.calc_roughness(a(lai), a(h_c), w_C=a(1), landcover=a(res.CROP), f_c=a(fc))
    o = TSEB.TSEB_PT(a(tr_k), a(0.0), a(ta_k), a(max(u, 0.5)), a(ea_mb), a(p_mb), sn_c, sn_s, l_dn, a(lai), a(h_c),
                     a(0.98), a(0.95), z0m, d0, a(z_u), a(z_t), leaf_width=a(LEAF_WIDTH), f_c=a(fc))
    flag, le_c, h_c_, le_s, h_s, g = o[0], o[6], o[7], o[8], o[9], o[10]
    return dict(flag=int(flag[0]), le=float(le_c[0] + le_s[0]), h=float(h_c_[0] + h_s[0]), g=float(g[0]),
                rn=float(le_c[0] + le_s[0] + h_c_[0] + h_s[0] + g[0]))


def draw_versions(n=N_VERSIONS, seed=0):
    """Latin hypercube over the v4 ranges (v3 algorithm; Kcb_mid range widened)."""
    rng = np.random.default_rng(seed)
    out = {}
    for k, (lo, hi) in RANGES.items():
        u = (np.arange(n) + rng.random(n)) / n
        rng.shuffle(u)
        out[k] = lo + u * (hi - lo)
    return out


# ------------------------------------------------------------------------------------------------- data chain
def _info(area, year):
    import json
    from areas import out_dir
    return json.load(open(os.path.join(out_dir(area, year), "testsite_field.json")))


def _interior_px(area, year):
    """Interior Landsat pixels of the season's field (from prep's thermal file); None before prep."""
    p = os.path.join(OUT, f"thermal_{area}_{year}.csv")
    if not os.path.exists(p):
        return None
    try:
        d = pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return 0
    return int(d["interior_px"].iloc[0]) if len(d) else 0


def active(seasons):
    """Seasons kept by the pre-registered size rules: field outline not dropped, >= 4 interior Landsat pixels."""
    out = []
    for a, y in seasons:
        if _info(a, y).get("dropped"):
            continue
        n = _interior_px(a, y)
        if n is not None and n < 4:
            continue
        out.append((a, y))
    return out


def _field(area, year):
    import json
    from areas import out_dir
    o = out_dir(area, year)
    z = np.load(os.path.join(o, "testsite_field.npz"))
    info = json.load(open(os.path.join(o, "testsite_field.json")))
    lai = pd.read_csv(os.path.join(o, "testsite_lai_daily.csv"), index_col=0, parse_dates=True)["lai"]
    return z, info, lai


def thermal(area, year):
    """Thermal ET per clear Landsat date for the tower field (uses no tower data)."""
    import rasterio
    from affine import Affine
    from areas import raw_dir
    from phase10_landsat import grid
    from phase31_landsat_thermal import kelvin
    from phase33_testsite_field import TOWERS, site_weather
    z, info, lai = _field(area, year)
    t = TOWERS[area]
    mask, r0, c0 = z["mask"], int(z["r0"]), int(z["c0"])
    tr10 = Affine(*z["transform"])
    tr30, w30, h30 = grid(area)
    rr, cc = np.indices((h30, w30))
    xs, ys = rasterio.transform.xy(tr30, rr.ravel(), cc.ravel())
    r10, c10 = rasterio.transform.rowcol(tr10, np.array(xs), np.array(ys))
    r10, c10 = np.asarray(r10) - r0, np.asarray(c10) - c0
    keep = interior_points(mask, r10, c10, 100.0)
    idx30 = np.flatnonzero(keep)
    d = os.path.join(raw_dir(area, year), "landsat_st")
    sc = pd.read_csv(os.path.join(d, "scenes.csv"))
    days = pd.date_range(f"{year}-03-01", f"{year}-09-30")
    W = site_weather(area, days)
    rows = []
    for _, s in sc.iterrows():
        f = os.path.join(d, s["file"])
        if not os.path.exists(f) or len(idx30) < 4:
            continue
        day = pd.Timestamp(s["date"])
        if day not in W.index:
            continue
        with rasterio.open(f) as r:
            a = r.read().reshape(3, -1)[:, idx30]
        lst, unc, qa = kelvin(a[0]), a[1].astype(float) * 0.01, a[2]
        hh, mm = map(int, s["time"].split(":"))
        hour = hh + mm / 60 + t["lon"] / 15                          # local solar time
        sunrise, n, sza = solar(t["lat"], day.dayofyear, hour)
        w = W.loc[day]
        ta = float(air_temp_at(w["tasmin"], w["tasmax"], hour, sunrise)) + 273.15
        ok = screen(lst, qa, unc, ta)
        share = ok.mean()
        row = dict(date=day, time=s["time"], scene=s["file"], interior_px=len(idx30), clear_px=int(ok.sum()),
                   clear_share=float(share))
        if share >= 0.5 and ok.sum() >= 4:
            tmean = (w["tasmin"] + w["tasmax"]) / 2
            ea = w["hurs"] / 100 * 6.108 * np.exp(17.27 * tmean / (tmean + 237.3))      # hPa = mb
            p_mb = 1013.25 * ((293 - 0.0065 * t["elev"]) / 293) ** 5.26
            rs = float(rs_instant(w["rs"], hour, n))
            l_ = float(lai.get(day, 0.0))
            h_c = max(0.1, H_MAX[info["crop"]] * min(1.0, l_ / 3))
            tr_ = float(np.median(lst[ok]))
            o = tseb(tr_, ta, l_, h_c, float(w["u10"]), float(ea), p_mb, rs, sza)
            et = daily_et(o["le"], rs, w["rs"]) if rs > 50 else np.nan
            row.update(lst=tr_, t_air=ta, lai=l_, rs_inst=rs, flag=o["flag"], le=o["le"], h=o["h"], g=o["g"],
                       et_thermal=et, et0=float(w["et0"]))
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):                                                     # two overlapping paths on a day: keep the clearer
        df = df.sort_values(["date", "clear_px"]).drop_duplicates("date", keep="last").reset_index(drop=True)
    return df, info


def usable(df):
    if "et_thermal" not in df:
        return pd.DataFrame(columns=["date", "et_thermal", "flag"]).astype({"date": "datetime64[ns]", "et_thermal": float})
    return df[df["et_thermal"].notna() & (df["et_thermal"] > 0) & (df["flag"] < 255)]


def season_ensemble(area, year, versions):
    """All versions 1 March - 30 September at a test site: v2 equations, Kcb_mid and root depth per version."""
    from phase29_water_v2 import KCB_MID, P_TAB, ROOT_MAX, _taw, kcb_from_lai
    from phase30_water_v3 import ensemble_balance
    from phase33_testsite_field import site_weather, soil
    z, info, lai_s = _field(area, year)
    crop = info["crop"]
    dates = pd.date_range(f"{year}-03-01", f"{year}-09-30")
    lai = lai_s.reindex(dates).fillna(0).values
    start, peak, end = (pd.Timestamp(info[k]) for k in ("start", "peak", "end"))
    w = site_weather(area, dates)
    paw, tew = soil(area)
    frac = np.clip(np.asarray((dates - start).days) / max((peak - start).days, 1), 0, 1)
    depth = 0.1 + (ROOT_MAX[crop] * versions["root_f"][:, None] - 0.1) * frac[None, :]
    taw = np.stack([_taw(paw, dd) for dd in depth]) * versions["taw_f"][:, None]
    kcb = np.stack([kcb_from_lai(lai, KCB_MID[crop] * k) for k in versions["kcb_f"]])
    ens = ensemble_balance(w["et0"].values, w["pr"].values, kcb, lai, taw, P_TAB[crop], tew, versions)
    from phase28_water_job3 import fao56_et
    from phase29_water_v2 import _stage_curve
    taw1 = _taw(paw, 0.1 + (ROOT_MAX[crop] - 0.1) * frac)
    single = fao56_et(w["et0"].values, w["pr"].values, _stage_curve(dates, start, peak, end, *KC_TAB[crop]), taw1,
                      P_TAB[crop])
    return dict(info=info, crop=crop, dates=dates, lai=lai, ens=ens, single=single, et0=w["et0"].values)


def cut_weights(E, th, cut):
    """Weights from thermal dates before the cut (cumulative)."""
    th = th[(th["date"] < cut) & th["date"].isin(E["dates"])]
    if len(th) == 0:
        from phase30_water_v3 import weigh
        return weigh(np.zeros((len(E["ens"]["et"]), 0)), np.zeros(0)), 0
    from phase30_water_v3 import weigh
    pos = E["dates"].get_indexer(th["date"])
    obs = th["et_thermal"].values
    return weigh(E["ens"]["et"][:, pos], obs, err=obs_error(obs)[None, :], min_ess=MIN_ESS), len(pos)


def prep():
    """Thermal ET for every test season and the weights it gives (no tower or soil-sensor data)."""
    os.makedirs(OUT, exist_ok=True)
    V = draw_versions()
    summ = []
    for area, year in active(JUDGED + SEPARATE + INDICATIVE):
        df, info = thermal(area, year)
        df.to_csv(os.path.join(OUT, f"thermal_{area}_{year}.csv"), index=False)
        u = usable(df)
        E = season_ensemble(area, year, V)
        for c in CUTS:
            w, n = cut_weights(E, u, pd.Timestamp(f"{year}-{c}"))
            summ.append(dict(area=area, year=year, crop=info["crop"], cut=c, thermal_dates=n, ess=1 / float((w ** 2).sum()),
                             **{f"wmean_{k}": float(np.sum(w * V[k])) for k in RANGES}))
        print(f"{area} {year} {info['crop']}: interior Landsat px {int(df['interior_px'].iloc[0]) if len(df) else 0}, "
              f"scenes {len(df)}, usable thermal dates {len(u)}, thermal ET median "
              f"{u['et_thermal'].median() if len(u) else float('nan'):.2f} mm/d", flush=True)
    pd.DataFrame(summ).to_csv(os.path.join(OUT, "prep_weights.csv"), index=False)


def _tower(area):
    import glob
    p = glob.glob(os.path.join("data", "raw", "icos", "extract", f"fx_{'Kli' if area == 'dekli' else 'Geb'}_inner",
                               "*FLUXMET_HH*.csv"))[0]
    return p


def ec_daily(path, col="LE_F_MDS", min_good=0.8):
    """Daily tower ET (mm) as Job 3 / v3: days with >= 80 % good half-hours (QC <= 1). col LE_CORR: closed balance."""
    d = pd.read_csv(path, usecols=["TIMESTAMP_END", col, "LE_F_MDS_QC"], na_values=[-9999])
    t = pd.to_datetime(d["TIMESTAMP_END"].astype(str), format="%Y%m%d%H%M")
    day = (t - pd.Timedelta(minutes=1)).dt.normalize()
    good = (d["LE_F_MDS_QC"] <= 1) & d[col].notna()
    g = pd.DataFrame({"day": day, "le": d[col], "good": good}).groupby("day")
    share, n = g["good"].mean(), g["le"].count()
    et = g["le"].sum() * 1800 / LAMBDA
    return et[(share >= min_good) & (n == 48)]


def swc_daily(path):
    """Depth mean of all SWC_F_MDS sensors in the file (QC <= 1), daily."""
    cols = [c for c in pd.read_csv(path, nrows=0).columns if c.startswith("SWC_F_MDS_") and not c.endswith("_QC")]
    d = pd.read_csv(path, usecols=["TIMESTAMP_END"] + cols + [c + "_QC" for c in cols], na_values=[-9999])
    t = pd.to_datetime(d["TIMESTAMP_END"].astype(str), format="%Y%m%d%H%M")
    day = (t - pd.Timedelta(minutes=1)).dt.normalize()
    v = pd.DataFrame({c: d[c].where(d[c + "_QC"] <= 1) for c in cols})
    return v.groupby(day).mean().mean(axis=1, skipna=True)


def _season_rows(area, year, V, ec, swc, tag):
    from phase30_water_v3 import weekly_sums, wquantile
    th = usable(pd.read_csv(os.path.join(OUT, f"thermal_{area}_{year}.csv"), parse_dates=["date"]))
    E = season_ensemble(area, year, V)
    idx = E["dates"]
    rel = 1 - E["ens"]["dr"] / np.maximum(E["ens"]["taw"], 1e-6)
    f1, f2 = [], []
    for c in CUTS:
        cut = pd.Timestamp(f"{year}-{c}")
        w, nth = cut_weights(E, th, cut)
        win = (idx >= cut) & (idx < cut + pd.Timedelta(days=28))
        wk = ((idx[win] - cut).days // 7).values
        out = {}
        for name, e in ec.items():
            valid = idx[win].isin(e.index)
            tower = e.reindex(idx[win]).fillna(0).values
            m, t, keep = weekly_sums(E["ens"]["et"][:, win], tower, valid, wk)
            s1, _, _ = weekly_sums(E["single"][None, win], tower, valid, wk)
            out[name] = (m, t, keep, s1)
        m, t, keep, s1 = out["LE_F_MDS"]
        q = wquantile(m, w, [0.1, 0.5, 0.9])
        v2 = np.median(m, axis=0)
        mc, tc, keepc, s1c = out["LE_CORR"]
        qc = wquantile(mc, w, 0.5)
        for k in np.where(keep)[0]:
            f1.append(dict(group=tag, area=area, year=year, crop=E["crop"], cut=c, week=k + 1, thermal_dates=nth,
                           tower=t[k], v4=q[1, k], v2=v2[k], fao56=s1[0, k], lo=q[0, k], hi=q[2, k],
                           tower_corr=tc[k] if keepc[k] else np.nan, v4_corr=qc[k], v2_corr=np.median(mc, 0)[k],
                           fao56_corr=s1c[0, k]))
        ms = swc.reindex(idx[win])
        ok = ms.notna().values
        if ok.sum() >= 10:
            r4 = wquantile(rel[:, win][:, ok], w, 0.5)
            r2 = np.median(rel[:, win][:, ok], axis=0)
            f2.append(pd.DataFrame(dict(group=tag, area=area, year=year, cut=c, v4=r4 - r4.mean(), v2=r2 - r2.mean(),
                                        meas=ms.values[ok] - ms.values[ok].mean())))
    e = ec["LE_F_MDS"]
    th_t = th[th["date"].isin(e.index)]
    td = pd.DataFrame(dict(group=tag, area=area, year=year, date=th_t["date"].values, thermal=th_t["et_thermal"].values,
                           tower=e.reindex(th_t["date"]).values, tower_corr=ec["LE_CORR"].reindex(th_t["date"]).values))
    return f1, f2, td


def look():
    """THE one look (pre-registered 2026-11-06)."""
    os.makedirs(OUT, exist_ok=True)
    V = draw_versions()
    f1, f2, td = [], [], []
    for group, seasons in (("judged", JUDGED), ("spring_barley", SEPARATE), ("indicative", INDICATIVE)):
        for area, year in active(seasons):
            p = _tower(area)
            ec = {"LE_F_MDS": ec_daily(p), "LE_CORR": ec_daily(p, "LE_CORR")}
            a, b, c = _season_rows(area, year, V, ec, swc_daily(p), group)
            f1 += a; f2 += b; td.append(c)
            print(area, year, group, "done", flush=True)
    F1, F2, TD = pd.DataFrame(f1), pd.concat(f2), pd.concat(td)
    F1.to_csv(os.path.join(OUT, "f1_weeks.csv"), index=False)
    F2.to_csv(os.path.join(OUT, "f2_days.csv"), index=False)
    TD.to_csv(os.path.join(OUT, "thermal_vs_tower.csv"), index=False)
    rm = lambda x, y: float(np.sqrt(np.nanmean((x - y) ** 2)))
    lines = []

    def report(name, d, e, deciding):
        rows, wins = [], 0
        for c in CUTS:
            dd = d[d.cut == c]
            r = {m: rm(dd[m], dd.tower) for m in ("v4", "v2", "fao56")}
            win = r["v4"] < r["v2"] and r["v4"] < r["fao56"]
            wins += win
            rows.append(dict(cut=c, weeks=len(dd), tower_mean=float(dd.tower.mean()),
                             **{f"rmse_{k}": v for k, v in r.items()}, v4_best=win))
        R = pd.DataFrame(rows)
        r4 = float(np.corrcoef(e.v4, e.meas)[0, 1]) if len(e) > 2 else np.nan
        r2 = float(np.corrcoef(e.v2, e.meas)[0, 1]) if len(e) > 2 else np.nan
        cover = float(((d.tower >= d.lo) & (d.tower <= d.hi)).mean()) if len(d) else np.nan
        ok = wins >= 2 and r4 >= 0.7 and cover >= 0.7
        lines.append(f"== {name}{' (DECIDING)' if deciding else ' (not deciding)'}")
        lines.append(R.round(3).to_string(index=False))
        lines.append(f"F2 anomaly r: v4 {r4:.3f}, v2 {r2:.3f}; 80 % band holds the tower week in {cover:.0%}; "
                     f"below band {((d.tower < d.lo).mean() if len(d) else np.nan):.0%}")
        if deciding:
            lines.append(f"RESULT v4: {'PASS' if ok else 'FAIL'} (v4 best at >= 2 of 3 cuts [{wins}], F2 r >= 0.7, band >= 70 %)")
        return R, dict(group=name, cuts_v4_best=wins, f2_r_v4=r4, f2_r_v2=r2, band80_cover=cover, passed=ok)

    J = F1[F1.group == "judged"]
    R, S = report("judged winter cereals", J, F2[F2.group == "judged"], True)
    R.to_csv(os.path.join(OUT, "f1_scores.csv"), index=False)
    summ = [S]
    for g in ("spring_barley", "indicative"):
        if (F1.group == g).any():
            summ.append(report(g, F1[F1.group == g], F2[F2.group == g], False)[1])
    Jc = J.dropna(subset=["tower_corr"])
    lines.append("== secondary: tower LE_CORR (closed energy balance), judged, not deciding")
    lines.append(pd.DataFrame([dict(cut=c, **{f"rmse_{m}": rm(Jc[Jc.cut == c][f"{m}_corr"], Jc[Jc.cut == c].tower_corr)
                                              for m in ("v4", "v2", "fao56")}) for c in CUTS]).round(3).to_string(index=False))
    for g in ("judged", "spring_barley", "indicative"):
        t = TD[(TD.group == g)].dropna(subset=["tower"])
        if len(t):
            lines.append(f"== thermal ET vs tower on thermal dates ({g}): n {len(t)}, RMSE {rm(t.thermal, t.tower):.2f} mm/d, "
                         f"bias {float((t.thermal - t.tower).mean()):+.2f}, r {np.corrcoef(t.thermal, t.tower)[0, 1]:.2f}; "
                         f"vs LE_CORR RMSE {rm(t.thermal, t.tower_corr):.2f}, bias {float((t.thermal - t.tower_corr).mean()):+.2f}")
    pd.DataFrame(summ).to_csv(os.path.join(OUT, "look_summary.csv"), index=False)
    txt = "\n".join(lines)
    open(os.path.join(OUT, "look_log.txt"), "w").write(txt + "\n")
    print(txt)
    return summ[0]["passed"]


if __name__ == "__main__":
    {"prep": prep, "look": look}[sys.argv[1]]()
