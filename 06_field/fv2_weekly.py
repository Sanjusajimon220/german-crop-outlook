"""Weekly pixel series per area-season with the tested design (plan 2026-10-17, Task 2).

Per pixel and Monday: measured (clear S2 value within +-3 days, nearest) > straight line (gap <= 30 d) > radar V2
re-analysis (gap > 30 d; line instead if the area test did not adopt the design) > live V2 after the last clear image.
LAI clear values flagged by fix 2 (single bad clear images) are treated as cloudy.
Run `python fv2_weekly.py all` from D:\vista (resumable).
"""
import numpy as np

SRC_NONE, SRC_MEASURED, SRC_LINE, SRC_RADAR, SRC_AFTER = 0, 1, 2, 3, 4
MEASURED_DAYS = 3
LINE_MAX_GAP = 30


def assign_week(days, values, sd, week_day, models=None):
    """One pixel, one week. days / values / sd: its clear S2 values (ascending days). Returns (value, unc, src, gap).
    models: optional callables {"re": f(a, b) -> value, "live": f(a) -> value} for the radar cases (indices into days);
    without them the radar cases fall back to the straight line / last value (used by the pure tests)."""
    if len(days) == 0:
        return np.nan, np.nan, SRC_NONE, 0
    d = np.abs(days - week_day)
    k = int(np.argmin(d))
    if d[k] <= MEASURED_DAYS:
        return float(values[k]), float(sd[k]), SRC_MEASURED, 0
    before, after = np.flatnonzero(days < week_day), np.flatnonzero(days > week_day)
    if len(before) == 0:
        return np.nan, np.nan, SRC_NONE, 0
    a = before[-1]
    if len(after) == 0:
        v = models["live"](a) if models else values[a]
        return float(v), np.nan, SRC_AFTER, int(week_day - days[a])
    b = after[0]
    gap = int(days[b] - days[a])
    line = values[a] + (week_day - days[a]) / gap * (values[b] - values[a])
    if gap <= LINE_MAX_GAP:
        return float(line), np.nan, SRC_LINE, gap
    v = models["re"](a, b) if models else line
    return float(v), np.nan, SRC_RADAR, gap


# ---- full area-season run (needs Task 1 models and uncertainty table) ----
import json
import os
import sys

import joblib
import pandas as pd

import fv2
import phase11_fusion_test as F
from areas import AREAS, out_dir
from fv2_produce import NDVI_MEASURED_SD, design_adopted
from phase7_step3_s1proxy import prev_next

MODELS = os.path.join("data", "processed", "field_v2", "models")
UNC = os.path.join("data", "processed", "field_v2", "uncertainty_V2.json")
OUTFILE = "weekly_pixels_V2.npz"
FIX2_DIR = os.path.join("data", "processed", "field_v2", "benchmark")
FIX2_TARGETS = {"lai"}                                         # fix 2 adopted for LAI only (project log 2026-10-21 12:25)


def drop_flagged(Y, flags):
    """Clear values flagged by fix 2 (single bad clear images) treated as cloudy; returns a copy."""
    if flags.shape != Y.shape:
        raise ValueError(f"fix-2 flags {flags.shape} do not match the season grid {Y.shape}")
    out = Y.copy()
    out[flags] = np.nan
    return out


def mondays(year):
    return pd.date_range(f"{year - 1}-10-01", f"{year}-08-31", freq="W-MON")


def season_on_weeks(area, year, target, weeks):
    """Season with nothing hidden, its date axis extended by the weekly Mondays (no values on those rows)."""
    S = F.Season(area, year, target, landsat=False, date_hide=True, hide_share=0.0)
    if target in FIX2_TARGETS:
        S.Y = drop_flagged(S.Y, np.load(os.path.join(FIX2_DIR, f"fix2_flags_{target}_{area}_{year}.npz"))["flags"])
    s2 = S.dates
    D = s2.union(weeks)
    Y = np.full((len(D), S.Y.shape[1]), np.nan, np.float32)
    Y[[D.get_loc(d) for d in s2]] = S.Y
    S.dates, S.Y = D, Y
    S.kept = np.isfinite(Y)
    S.k_s1 = {o: np.array([F.nearest(S.orb[o][0], d) for d in D]) for o in fv2.ORBITS}
    S.ctx, S.rain = fv2.context(D, year, fv2.district_of(area, year, S.px))
    S.days = (D - D[0]).days.values
    S.s2_rows = np.array([D.get_loc(d) for d in s2])
    return S


def weekly_target(S, weeks, target, unc, adopted, measured_sd):
    mL = joblib.load(os.path.join(MODELS, f"{target}_live_V2.joblib"))
    mR = joblib.load(os.path.join(MODELS, f"{target}_re_V2.joblib"))
    P = S.Y.shape[1]
    T = len(weeks)
    val = np.full((T, P), np.nan, np.float32); u = np.full((T, P), np.nan, np.float32)
    src = np.zeros((T, P), np.int8); gap = np.zeros((T, P), np.int16)
    prv, nxt = prev_next(S.kept)
    s2days = S.days[S.s2_rows]
    for w, wd in enumerate(weeks):
        i = S.dates.get_loc(wd)
        near = S.s2_rows[np.argsort(np.abs(s2days - S.days[i]))]
        near = [j for j in near if abs(S.days[j] - S.days[i]) <= MEASURED_DAYS]
        m = np.zeros(P, bool)
        for j in near:                                         # nearest clear value within +-3 days
            take = ~m & np.isfinite(S.Y[j])
            val[w, take] = S.Y[j, take]; u[w, take] = measured_sd[j][take]; src[w, take] = SRC_MEASURED
            m |= take
        a_all, b_all = prv[i], nxt[i]
        rest = ~m & (a_all >= 0)
        for (a, b), g in pd.DataFrame({"a": a_all[rest], "b": b_all[rest], "p": np.flatnonzero(rest)}).groupby(["a", "b"]):
            idx = g["p"].values
            if b >= 0:
                gp = int(S.days[b] - S.days[a])
                line = S.lin(i, a, b, idx)
                gap[w, idx] = gp
                if gp <= LINE_MAX_GAP or not adopted:
                    val[w, idx], src[w, idx] = line, SRC_LINE
                    u[w, idx] = unc["line_le15"] if gp <= 15 else unc["line_16_30"] if gp <= 30 else unc["line_gt30"]
                else:
                    val[w, idx] = line + F.pred(mR, S.re_rows(i, a, b, idx))
                    src[w, idx], u[w, idx] = SRC_RADAR, unc["radar_gt30"]
            else:
                since = int(S.days[i] - S.days[a])
                val[w, idx] = S.Y[a, idx] + F.pred(mL, S.live_rows(i, a, idx))
                src[w, idx], gap[w, idx] = SRC_AFTER, since
                u[w, idx] = unc["live_le30"] if since <= 30 else unc["live_gt30"]
    val = np.clip(val, 0, None) if target == "lai" else np.clip(val, -1, 1)
    return val, u, src, gap


def weekly_pixels(area, year):
    out = os.path.join(out_dir(area, year), OUTFILE)
    if os.path.exists(out):
        return out
    weeks = mondays(year)
    unc = json.load(open(UNC))
    res = {}
    for target in ("lai", "ndvi"):
        S = season_on_weeks(area, year, target, weeks)
        if target == "lai":
            sdz = np.load(os.path.join(out_dir(area, year), "s2_pixels_fv2.npz"))["lai_sd"]
            msd = np.full(S.Y.shape, np.nan, np.float32); msd[S.s2_rows] = sdz
        else:
            msd = np.full(S.Y.shape, NDVI_MEASURED_SD, np.float32)
        v, u, s, g = weekly_target(S, weeks, target, unc[target], design_adopted(target), msd)
        res.update({target: v, f"{target}_unc": u, f"{target}_src": s, f"{target}_gap": g})
        rows, cols = S.px["rows"], S.px["cols"]
        del S
    tmp = out + ".part.npz"
    np.savez_compressed(tmp, weeks=np.array(weeks, dtype="datetime64[D]"), rows=rows, cols=cols, **res)
    os.replace(tmp, out)
    s = res["lai_src"]
    print(f"{area} {year}: weeks {len(weeks)}; LAI sources measured {np.mean(s == 1):.2f}, line {np.mean(s == 2):.2f}, "
          f"radar {np.mean(s == 3):.2f}, after last {np.mean(s == 4):.2f}, none {np.mean(s == 0):.2f}", flush=True)
    return out


if __name__ == "__main__" and sys.argv[1:] == ["all"]:
    for a in AREAS:
        for y in range(2021, 2026):
            weekly_pixels(a, y)
