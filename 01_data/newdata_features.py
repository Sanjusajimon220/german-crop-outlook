"""v10 new inputs as district-year features (used by phase2_growth variants 'soil' and 'lst').

soil (DWD AMBAV crop soil moisture, % nFK, 0-60 cm; wheat file for wheat / barley, maize file for maize /
      potato; 1991+): monthly means April-September (sm_m04..sm_m09), summer minimum June-August
      (sm_min_summer), mean in the sensitive window (stage -10..+20 days, win_sm)
lst  (MODIS daytime land surface temperature over cropland, Aqua MYD11A2 preferred, Terra MOD11A2 where
      Aqua is missing; 2000+): canopy minus air temperature dT = LST - mean HYRAS/v8 Tmax of the 8-day
      composite; monthly means May-August (lst_dt_m05..m08) and in the sensitive window (win_lst_dt)
As ANOMALIES (amended 2026-10-08 after the first screening: raw levels with NaN before 1991 let the
models read 'missing' as 'old era' and levels as district potential): each feature minus the district's
mean in a fixed reference period that precedes both forward test periods (soil 1991-2002; LST: all
years up to 2008, because LST starts 2000/2002 - caveat for the cut-2002 period); years without data -> 0
(= normal), so missingness carries no information.
"""
import os

import numpy as np
import pandas as pd

P = os.path.join("data", "processed")
SM_CROP = {"winter_wheat": "wheat", "winter_barley": "wheat", "grain_maize": "maize",
           "silage_maize": "maize", "potato": "maize"}


def _window_mean(dates, values, col, yr, stage_doy, before=10, after=20):
    if not np.isfinite(stage_doy):
        return np.nan
    t0 = np.datetime64(f"{yr}-01-01") + np.timedelta64(int(stage_doy) - 1 - before, "D")
    s = (dates >= t0) & (dates < t0 + np.timedelta64(before + after, "D"))
    v = values[s, col]
    return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan


def anomalies(df, m, first, last):
    """Feature minus the district mean over reference years first..last; missing -> 0 (normal)."""
    ref = (m["year"] >= first) & (m["year"] <= last)
    out = df.copy()
    for c in df.columns:
        mean = df.loc[ref, c].groupby(m.loc[ref, "district"]).mean()
        base = m["district"].map(mean).fillna(df.loc[ref, c].mean())
        out[c] = (df[c] - base).fillna(0.0)
    return out


def soil_features(crop, m, stage):
    out = {c: np.full(len(m), np.nan) for c in [f"sm_m{k:02d}" for k in range(4, 10)] + ["sm_min_summer", "win_sm"]}
    cache = {}
    for i, (d, yr, sd) in enumerate(zip(m["district"].values, m["year"].values, m[stage].values)):
        if yr not in cache:
            f = os.path.join(P, "soil_moisture", f"sm_{SM_CROP[crop]}_{yr}.npz")
            if os.path.exists(f):
                z = np.load(f)
                cache[yr] = (z["dates"], z["sm"], {c: j for j, c in enumerate(z["codes"])},
                             z["dates"].astype("datetime64[M]").astype(int) % 12 + 1)
            else:
                cache[yr] = None
        c = cache[yr]
        if c is None or d not in c[2]:
            continue
        dates, sm, col, mon = c
        j = col[d]
        for k in range(4, 10):
            out[f"sm_m{k:02d}"][i] = np.nanmean(sm[mon == k, j])
        out["sm_min_summer"][i] = np.nanmin(sm[(mon >= 6) & (mon <= 8), j])
        out["win_sm"][i] = _window_mean(dates, sm, j, yr, sd)
    return anomalies(pd.DataFrame(out, index=m.index), m, 1991, 2002)


def lst_features(crop, m, stage, weather):
    out = {c: np.full(len(m), np.nan) for c in [f"lst_dt_m{k:02d}" for k in range(5, 9)] + ["win_lst_dt"]}
    wdates = np.asarray(weather["dates"])
    tmax = np.asarray(weather["tasmax"])
    wcol = {c: j for j, c in enumerate(weather["codes"])}
    cache = {}
    for i, (d, yr, sd) in enumerate(zip(m["district"].values, m["year"].values, m[stage].values)):
        if yr not in cache:
            z = None
            for sensor in ("MYD", "MOD"):
                f = os.path.join(P, "modis_lst", f"lst_{sensor}_{yr}.npz")
                if os.path.exists(f):
                    z = np.load(f)
                    break
            if z is None:
                cache[yr] = None
            else:
                dates = z["dates"]
                i0 = ((dates - wdates[0]).astype(int))[:, None] + np.arange(8)[None, :]
                i0 = np.clip(i0, 0, len(wdates) - 1)
                air = tmax[i0].mean(1)                                   # composites x districts (weather order)
                codes = list(z["codes"])
                order = [wcol.get(c, -1) for c in codes]
                air = np.stack([air[:, k] if k >= 0 else np.full(len(dates), np.nan) for k in order], 1)
                dt = z["lst"] - air
                cache[yr] = (dates, dt, {c: j for j, c in enumerate(codes)},
                             dates.astype("datetime64[M]").astype(int) % 12 + 1)
        c = cache[yr]
        if c is None or d not in c[2]:
            continue
        dates, dt, col, mon = c
        j = col[d]
        for k in range(5, 9):
            v = dt[mon == k, j]
            out[f"lst_dt_m{k:02d}"][i] = np.nanmean(v) if np.isfinite(v).any() else np.nan
        out["win_lst_dt"][i] = _window_mean(dates, dt, j, yr, sd, before=12, after=24)
    return anomalies(pd.DataFrame(out, index=m.index), m, 2000, 2008)
