"""Step 4, fix 2: single bad clear Sentinel-2 values (haze / cloud edges missed by the mask), rule pre-registered 2026-10-21.

A measured pixel value is flagged when it lies below both neighbouring clear values of that pixel by more than 3 x its
uncertainty and the radar estimate for that date does not show the drop. Test against Landsat within 1 day (field means of
flagged vs unflagged values on the same pixels). Adopted per target if median |S2 - Landsat| of flagged >= 2 x unflagged and
< 2 % of measured values are flagged. Run from D:\vista: python phase15_fix2_dips.py -> benchmark/fix2_*.csv, flags npz, log.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

K, MIN_PX, LANDSAT_DAYS = 3.0, 10, 1
OUT = r"D:\vista\data\processed\field_v2\benchmark"


def candidates(Y, sd):
    """(time, pixel, prev, next) of clear values lying more than K x sd below both neighbouring clear values."""
    from phase7_step3_s1proxy import prev_next
    ok = np.isfinite(Y)
    pv, nx = prev_next(ok)
    t, p = np.nonzero(ok & (pv >= 0) & (nx >= 0))
    a, b = pv[t, p], nx[t, p]
    y, s = Y[t, p], sd[t, p]
    dip = (Y[a, p] - y > K * s) & (Y[b, p] - y > K * s)
    return t[dip], p[dip], a[dip], b[dip]


def dip_flags(y_a, y_i, y_b, sd, radar):
    """True where the radar estimate does not show the drop (stays above min(neighbours) - K x sd)."""
    return radar >= np.minimum(y_a, y_b) - K * sd


def compare_groups(s2, ls, flag, field, min_px=MIN_PX):
    """Field means of S2 and Landsat on the same pixels, separately for flagged / unflagged; groups with >= min_px pixels."""
    ok = np.isfinite(s2) & np.isfinite(ls)
    df = pd.DataFrame({"s2": s2[ok], "ls": ls[ok], "flagged": flag[ok], "field": field[ok]})
    g = df.groupby(["field", "flagged"]).agg(s2=("s2", "mean"), ls=("ls", "mean"), n=("s2", "size")).reset_index()
    g = g[g.n >= min_px].copy()
    g["absdiff"] = (g.s2 - g.ls).abs()
    return g


def run_season(area, year, target):
    import joblib
    import phase11_fusion_test as F
    from areas import out_dir
    from fv2_produce import NDVI_MEASURED_SD
    S = F.Season(area, year, target, landsat=False, date_hide=True, hide_share=0.0)
    Y = S.Y
    if target == "lai":
        sd = np.load(os.path.join(out_dir(area, year), "s2_pixels_fv2.npz"))["lai_sd"].astype(float)
    else:
        sd = np.full(Y.shape, NDVI_MEASURED_SD)
    t, p, a, b = candidates(Y, sd)
    mR = joblib.load(os.path.join(r"D:\vista\data\processed\field_v2\models", f"{target}_re_V2.joblib"))
    flag = np.zeros(len(t), bool)
    for (i, aa, bb), g in pd.DataFrame({"t": t, "a": a, "b": b, "k": np.arange(len(t))}).groupby(["t", "a", "b"]):
        idx_k = g["k"].values; idx = p[idx_k]
        radar = S.lin(i, aa, bb, idx) + F.pred(mR, S.re_rows(i, aa, bb, idx))
        flag[idx_k] = dip_flags(Y[aa, idx], Y[i, idx], Y[bb, idx], sd[i, idx], radar)
    F_ = np.zeros(Y.shape, bool)
    F_[t[flag], p[flag]] = True
    n_meas = int(np.isfinite(Y).sum())
    # Landsat within 1 day of an S2 date
    l = np.load(os.path.join(out_dir(area, year), "landsat_pixels.npz"))
    ld = pd.DatetimeIndex(l["dates"])
    field = S.px["inv"]
    rows = []
    for i, d in enumerate(S.dates):
        k = np.argmin(np.abs((ld - d).days)) if len(ld) else -1
        if k < 0 or abs((ld[k] - d).days) > LANDSAT_DAYS:
            continue
        g = compare_groups(Y[i].astype(float), l[target][k].astype(float), F_[i], field)
        g["date"], g["area"], g["season"], g["target"] = str(d.date()), area, year, target
        rows.append(g)
    return F_, n_meas, int(len(t)), (pd.concat(rows) if rows else pd.DataFrame())


def run():
    from areas import AREAS
    os.makedirs(OUT, exist_ok=True)
    allg, counts = [], []
    for target in ("lai", "ndvi"):
        for area in AREAS:
            for year in range(2021, 2026):
                Fl, n, ncand, g = run_season(area, year, target)
                np.savez_compressed(os.path.join(OUT, f"fix2_flags_{target}_{area}_{year}.npz"), flags=Fl)
                counts.append(dict(target=target, area=area, season=year, measured=n, candidates=ncand, flagged=int(Fl.sum())))
                allg.append(g)
                print(target, area, year, "measured", n, "candidates", ncand, "flagged", int(Fl.sum()), flush=True)
    G = pd.concat(allg); C = pd.DataFrame(counts)
    G.to_csv(os.path.join(OUT, "fix2_groups.csv.gz"), index=False); C.to_csv(os.path.join(OUT, "fix2_counts.csv"), index=False)
    verdict = {}
    for target in ("lai", "ndvi"):
        g, c = G[G.target == target], C[C.target == target]
        mf, mu = g[g.flagged].absdiff.median(), g[~g.flagged].absdiff.median()
        share = c.flagged.sum() / c.measured.sum()
        ok = bool(mf >= 2 * mu and share < 0.02)
        verdict[target] = ok
        print(f"{target}: flagged share {share:.4%}; median |S2-Landsat| flagged {mf:.4f} (groups {int(g.flagged.sum())}) vs "
              f"unflagged {mu:.4f} (groups {int((~g.flagged).sum())}); ratio {mf / mu:.2f} -> {'pass' if ok else 'fail'}")
    print("RESULT fix2:", ", ".join(f"{t} {'ADOPTED' if v else 'NOT adopted'}" for t, v in verdict.items()))


if __name__ == "__main__":
    run()
