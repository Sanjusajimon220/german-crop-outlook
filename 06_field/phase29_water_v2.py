"""Field water v2: FAO-56 dual crop coefficient with field-specific inputs (pre-registered docs/project_log.md 2026-11-05).

ET = (Ks x Kcb + Ke) x ET0 per field and day (Allen et al. 1998, FAO-56 ch. 7):
  ET0  FAO Penman-Monteith from the field's HYRAS 1 km cell (tmin, tmax, humidity), SARAH-3 sunlight (0.05 deg),
       DWD station wind (10 m -> 2 m)
  Kcb  from the field's weekly satellite green LAI (variant 'sat') or the FAO Table 17 curve on the stages ('table')
  Ke   soil evaporation from a 10 cm surface layer after rain, every day of the year
  Ks   root-zone water stress from a bucket with the field's plant-available water
Tests T1 (flux tower, new site-years) and T2 (measured soil water) at ICOS DE-RuS. Run from D:\\vista:
python phase29_water_v2.py t1 | t2  -> data/processed/field_v2/water_v2/.
"""
import os

import numpy as np
import pandas as pd

KCB_MIN, KC_MAX, ZE_REW = 0.15, 1.20, 9.0
OUT = os.path.join("data", "processed", "field_v2", "water_v2")


def u2_from_u10(u10):
    """Wind at 2 m from a 10 m measurement (FAO-56 eq. 47)."""
    return np.asarray(u10, float) * 4.87 / np.log(67.8 * 10 - 5.42)


def et0_pm(tmin, tmax, rh, rs, u2, elev, lat, doy):
    """FAO-56 Penman-Monteith reference ET (mm/day), daily step. rh: mean relative humidity (%), rs: MJ/m2/day."""
    tmin, tmax, rh, rs, u2, doy = (np.asarray(x, float) for x in (tmin, tmax, rh, rs, u2, doy))
    t = (tmin + tmax) / 2
    p = 101.3 * ((293 - 0.0065 * elev) / 293) ** 5.26                      # eq. 7
    gamma = 0.000665 * p                                                   # eq. 8
    es_ = lambda x: 0.6108 * np.exp(17.27 * x / (x + 237.3))               # eq. 11
    es = (es_(tmax) + es_(tmin)) / 2                                       # eq. 12
    ea = rh / 100 * es                                                     # eq. 19
    delta = 4098 * es_(t) / (t + 237.3) ** 2                               # eq. 13
    phi = np.radians(lat)
    dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * doy)                         # eq. 23
    dec = 0.409 * np.sin(2 * np.pi / 365 * doy - 1.39)                     # eq. 24
    ws = np.arccos(-np.tan(phi) * np.tan(dec))                             # eq. 25
    ra = 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))
    rso = (0.75 + 2e-5 * elev) * ra                                        # eq. 37
    rns = 0.77 * rs                                                        # eq. 38
    sb = 4.903e-9
    rnl = sb * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2 * (0.34 - 0.14 * np.sqrt(ea)) \
        * (1.35 * np.clip(rs / rso, 0, 1) - 0.35)                          # eq. 39
    rn = rns - rnl
    return np.maximum((0.408 * delta * rn + gamma * 900 / (t + 273) * u2 * (es - ea)) /
                      (delta + gamma * (1 + 0.34 * u2)), 0.0)              # eq. 6, G = 0 daily


def kcb_from_lai(lai, kcb_mid, k=0.7):
    """Basal crop coefficient from green LAI: Kcb_min at bare soil, approaching the FAO Table 17 Kcb_mid at full cover."""
    lai = np.nan_to_num(np.asarray(lai, float))
    return KCB_MIN + (kcb_mid - KCB_MIN) * (1 - np.exp(-k * lai))


def dual_kc_balance(et0, rain, kcb, lai, taw, p, tew, rew=ZE_REW, dr0=0.0):
    """FAO-56 dual crop coefficient daily water balance (eq. 69-86). Buckets start full (root zone: depleted by dr0 mm). Returns daily e (soil evaporation),
    t (transpiration), et, ks, dr (root-zone depletion, mm) and de (surface-layer depletion)."""
    n = len(et0)
    out = {k: np.zeros(n) for k in ("e", "t", "et", "ks", "dr", "de")}
    de, dr = 0.0, float(dr0)
    for i in range(n):
        de, dr = max(de - rain[i], 0.0), max(dr - rain[i], 0.0)            # rain refills, excess drains
        fc = 1 - np.exp(-0.5 * max(lai[i], 0.0))
        few = max(1 - fc, 0.01)
        kr = 1.0 if de <= rew else max((tew - de) / max(tew - rew, 1e-9), 0.0)   # eq. 74
        ke = min(kr * (KC_MAX - kcb[i]), few * KC_MAX)                     # eq. 71
        e = ke * et0[i]
        raw = p * taw[i]
        ks = 1.0 if dr <= raw else max((taw[i] - dr) / max(taw[i] - raw, 1e-9), 0.0)   # eq. 84
        t = ks * kcb[i] * et0[i]
        t = min(t, max(taw[i] - dr, 0.0))
        de = min(de + e / few, tew)                                        # eq. 77 (evaporation from the exposed fraction)
        dr = min(dr + e + t, taw[i])                                       # eq. 85
        out["e"][i], out["t"][i], out["et"][i], out["ks"][i], out["dr"][i], out["de"][i] = e, t, e + t, ks, dr, de
    return out


# ---- DE-RuS (ICOS Selhausen) inputs and tests T1 / T2 ----
SITE = dict(lat=50.866104, lon=6.446836, elev=104.0, cell=276024, district="05358", soil_ratio=1.04, wind_station="03623")
FIELDS = {2021: (5367327, "sugar beet"), 2022: (6268159, "potato"), 2023: (7182626, "wheat"), 2024: (7561227, "sugar beet"),
          2025: (8398633, "wheat")}
KCB_MID = {"wheat": 1.10, "barley": 1.10, "maize": 1.15, "potato": 1.10, "sugar beet": 1.15}
KCB_TAB = {"wheat": (0.15, 1.10, 0.25), "potato": (0.15, 1.10, 0.65), "sugar beet": (0.15, 1.15, 0.50)}   # FAO Table 17
KC_TAB = {"wheat": (0.40, 1.15, 0.30), "potato": (0.50, 1.15, 0.75), "sugar beet": (0.35, 1.20, 0.70)}    # FAO Table 12
P_TAB = {"wheat": 0.55, "barley": 0.55, "maize": 0.55, "potato": 0.35, "sugar beet": 0.55}                # FAO Table 22
ROOT_MAX = {"wheat": 1.5, "barley": 1.5, "maize": 1.0, "potato": 0.5, "sugar beet": 0.9}
SWC = os.path.join("data", "raw", "icos", "extract", "fx_inner", "ICOSETC_DE-RuS_FLUXNET_FLUXMET_HH_2019-2026_v1.3_r1.csv")


def _wind():
    import glob
    import io
    import zipfile
    parts = []
    for f in glob.glob(os.path.join("data", "raw", "dwd_stations*", f"tageswerte_KL_{SITE['wind_station']}_*.zip")):
        z = zipfile.ZipFile(f)
        d = pd.read_csv(io.BytesIO(z.read([n for n in z.namelist() if n.startswith("produkt")][0])), sep=";",
                        skipinitialspace=True, na_values=[-999])
        parts.append(pd.Series(d["FM"].values, index=pd.to_datetime(d["MESS_DATUM"].astype(str))))
    s = pd.concat(parts)
    return s[~s.index.duplicated(keep="last")].sort_index()


def site_weather(dates):
    """Daily inputs for the DE-RuS cell: rain, tmin, tmax, rh, rs (MJ), u2 and the field ET0."""
    out = []
    for y in sorted({d.year for d in dates}):
        h = np.load(os.path.join("data", "processed", "field", f"hyras_cells_{y}.npz"))
        j = list(h["cells"]).index(SITE["cell"])
        out.append(pd.DataFrame({k: h[k][:, j] for k in ("tasmin", "tasmax", "hurs", "pr")}, index=pd.DatetimeIndex(h["dates"])))
    w = pd.concat(out).reindex(dates)
    s = np.load(os.path.join("data", "processed", "field", "sarah3_rur.npz"))
    i, k = np.argmin(np.abs(s["lat"] - SITE["lat"])), np.argmin(np.abs(s["lon"] - SITE["lon"]))
    w["rs"] = pd.Series(s["sis"][:, i, k] * 0.0864, index=pd.DatetimeIndex(s["dates"])).reindex(dates).values   # W/m2 -> MJ
    wind = _wind().reindex(dates)
    w["u2"] = u2_from_u10(wind.fillna(wind.median()).values)
    w["et0"] = et0_pm(w["tasmin"], w["tasmax"], w["hurs"], w["rs"], w["u2"], SITE["elev"], SITE["lat"], dates.dayofyear)
    return w


def _soil():
    s = pd.read_csv(os.path.join("data", "processed", "soil_districts.csv"), dtype={"district": str}).set_index("district").loc[SITE["district"]]
    paw = np.array([(s[f"fc_{k}"] - s[f"wp_{k}"]) * 100 for k in range(1, 21)]) * SITE["soil_ratio"]   # mm per 10 cm
    tew = 1000 * (s["fc_1"] - 0.5 * s["wp_1"]) * 0.10                      # FAO-56 eq. 73, Ze 0.10 m
    return paw, tew


def _taw(paw, root_m):
    top = np.arange(len(paw)) * 10.0
    share = np.clip((np.asarray(root_m)[:, None] * 100 - top[None, :]) / 10.0, 0, 1)
    return share @ paw


def _stage_curve(dates, start, peak, end, ini, mid, endv):
    """FAO stage shape on the satellite season: initial 25 % of start-peak, development to the peak, mid to half of
    peak-end, late to the end, end value after."""
    t = pd.DatetimeIndex(dates)
    dev0 = start + (peak - start) * 0.25
    lat0 = peak + (end - peak) * 0.5
    f = lambda a, b: np.clip((t - a).days / max((b - a).days, 1), 0, 1)
    v = np.where(t < dev0, ini, np.where(t < peak, ini + (mid - ini) * f(dev0, peak),
                 np.where(t < lat0, mid, mid + (endv - mid) * f(lat0, end))))
    return np.where(t > end, endv, v)


def site_season(year):
    """One DE-RuS season: daily inputs, satellite LAI, stages, runs of v2-sat, v2-table and plain FAO-56 single Kc."""
    import json
    from phase28_water_job3 import fao56_et
    fid, crop = FIELDS[year]
    s = json.load(open(os.path.join("D:/german-crop-outlook-field/docs/field", f"selhausen_{year}", "series.json")))
    f = s["fields"][str(fid)]
    weeks = pd.DatetimeIndex(s["weeks"])
    lai_w = pd.Series([np.nan if v is None else v for v in f["lai"]], index=weeks)
    season = np.asarray(f["season"], bool)
    start, end = weeks[season].min(), weeks[season].max()
    peak = lai_w[season].idxmax()
    dates = pd.date_range(f"{year}-03-01", f"{year}-09-30")
    lai = lai_w.reindex(lai_w.index.union(dates)).interpolate(limit_area="inside").reindex(dates).fillna(0).values
    w = site_weather(dates)
    paw, tew = _soil()
    frac = np.clip((dates - start).days / max((peak - start).days, 1), 0, 1)
    taw = _taw(paw, 0.1 + (ROOT_MAX[crop] - 0.1) * frac)
    p = P_TAB[crop]
    rain = w["pr"].values
    sat = dual_kc_balance(w["et0"].values, rain, kcb_from_lai(lai, KCB_MID[crop]), lai, taw, p, tew)
    kcb_t = _stage_curve(dates, start, peak, end, *KCB_TAB[crop])
    fc_t = np.clip((kcb_t - KCB_MIN) / (KC_MAX - KCB_MIN), 0, 1) ** (1 + 0.5 * 0.5)   # FAO-56 eq. 76, h 0.5 m
    lai_t = -np.log(np.clip(1 - fc_t, 1e-3, 1)) / 0.5                        # same few as the sat variant
    tab = dual_kc_balance(w["et0"].values, rain, kcb_t, lai_t, taw, p, tew)
    single = fao56_et(w["et0"].values, rain, _stage_curve(dates, start, peak, end, *KC_TAB[crop]), taw, p)
    return dict(year=year, crop=crop, field=fid, dates=dates, start=start, peak=peak, end=end, et0=w["et0"].values,
                sat=sat, table=tab, single=single, taw=taw)


def t1():
    """Flux-tower test on site-years not used in Job 3 (beet 2021, potato 2022, beet 2024); wheat 2023 / 2025 supporting."""
    from phase28_water_job3 import ec_daily_gapfilled, scores
    os.makedirs(OUT, exist_ok=True)
    ec_all = ec_daily_gapfilled()
    rows, pool = [], {"sat": [], "table": [], "single": [], "ec": []}
    for year in sorted(FIELDS):
        S = site_season(year)
        idx = pd.DatetimeIndex(S["dates"])
        ec = ec_all[(ec_all.index >= f"{year}-04-01") & (ec_all.index <= S["end"])]
        ec = ec[ec.index.isin(idx)]
        pos = idx.get_indexer(ec.index)
        judged = year in (2021, 2022, 2024)
        for m, v in (("sat", S["sat"]["et"]), ("table", S["table"]["et"]), ("single", S["single"])):
            rows.append(dict(year=year, crop=S["crop"], judged=judged, method=m, **scores(v[pos], ec.values)))
            if judged:
                pool[m].append(v[pos])
        if judged:
            pool["ec"].append(ec.values)
        print(year, S["crop"], "season", S["start"].date(), "-", S["end"].date(), "peak", S["peak"].date(),
              "ET0 mean Apr-Aug %.2f" % S["et0"][(idx.month >= 4) & (idx.month <= 8)].mean(), flush=True)
    e = np.concatenate(pool["ec"])
    pooled = {m: scores(np.concatenate(pool[m]), e) for m in ("sat", "table", "single")}
    rows += [dict(year="POOLED (judged)", crop="", judged=True, method=m, **v) for m, v in pooled.items()]
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(OUT, "t1_scores.csv"), index=False)
    a, b = pooled["sat"], pooled["single"]
    ok = a["rmse"] <= b["rmse"] and a["r"] >= 0.7 and 0.85 <= a["ratio"] <= 1.15
    with pd.option_context("display.width", 200):
        print(r.round(3).to_string(index=False))
    print(f"RESULT T1: {'PASS' if ok else 'FAIL'} (pooled v2-sat RMSE <= plain FAO-56, r >= 0.7, ratio 0.85-1.15)")
    return ok


def t2():
    """Soil-water test: model root-zone relative storage vs measured SWC (depth mean), daily anomalies per season."""
    os.makedirs(OUT, exist_ok=True)
    cols = [f"SWC_F_MDS_{k}" for k in range(1, 6)]
    d = pd.read_csv(SWC, usecols=["TIMESTAMP_END"] + cols + [c + "_QC" for c in cols], na_values=[-9999])
    t = pd.to_datetime(d["TIMESTAMP_END"].astype(str), format="%Y%m%d%H%M")
    day = (t - pd.Timedelta(minutes=1)).dt.normalize()
    v = pd.DataFrame({c: d[c].where(d[c + "_QC"] <= 1) for c in cols})
    swc = v.groupby(day).mean().mean(axis=1, skipna=True)
    rs, lim = [], []
    for year in sorted(FIELDS):
        S = site_season(year)
        idx = pd.DatetimeIndex(S["dates"])
        win = (idx >= pd.Timestamp(f"{year}-04-01")) & (idx <= S["end"])
        rel = pd.Series(1 - S["sat"]["dr"] / np.maximum(S["taw"], 1e-6), index=idx)[win]
        meas = swc.reindex(rel.index)
        ok = meas.notna()
        a, b = rel[ok] - rel[ok].mean(), meas[ok] - meas[ok].mean()
        rs.append(pd.DataFrame({"year": year, "model": a.values, "meas": b.values}))
        ks = pd.Series(S["sat"]["ks"], index=idx)[win][ok]
        dry = meas[ok] <= meas[ok].quantile(1 / 3)
        lim.append(pd.DataFrame({"year": year, "limited": (ks < 1).values, "dry_third": dry.values}))
        print(year, S["crop"], "days", int(ok.sum()), "r %.3f" % np.corrcoef(a, b)[0, 1], "water-limited days",
              int((ks < 1).sum()), flush=True)
    R, L = pd.concat(rs), pd.concat(lim)
    r = float(np.corrcoef(R.model, R.meas)[0, 1])
    lim_days = L[L.limited]
    share = float(lim_days.dry_third.mean()) if len(lim_days) else float("nan")
    ok = r >= 0.7 and len(lim_days) > 0 and share >= 0.7
    pd.DataFrame([dict(pooled_r=r, limited_days=len(lim_days), share_in_driest_third=share, passed=ok)]).to_csv(
        os.path.join(OUT, "t2_scores.csv"), index=False)
    print(f"pooled r {r:.3f}; water-limited days {len(lim_days)}, share in the driest measured third {share:.2f}")
    print(f"RESULT T2: {'PASS' if ok else 'FAIL'} (r >= 0.7 AND >= 70 % of water-limited days in the driest third)")
    return ok


if __name__ == "__main__":
    import sys
    {"t1": t1, "t2": t2}[sys.argv[1]]()
