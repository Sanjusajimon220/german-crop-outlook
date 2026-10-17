"""Fix 2 (pre-registered docs/project_log.md 2026-10-15): physics-shaped ends of long gaps, wheat LAI re-analysis.

shape = line(a, b) + k * m_dev   (m_dev = physics prior at t minus the prior's own line between the anchors)
final = re + w * (shape - re)    k, w fitted per fold on the hidden wheat rows of the other 4 folds (grid search)
Uses the fusion3 test (same hiding, folds and predictions) and the independent partly cloudy 15 June check.
Output fix2_params.json (k, w fitted on all hidden wheat rows), fix2_check.csv.gz
"""
import json
import os

import numpy as np
import pandas as pd

from phase7_step3_s1proxy import prev_next

P = os.path.join("data", "processed")
OUT = os.path.join(P, "field_v2", "pilot_2024")
K = np.round(np.arange(0.5, 1.51, 0.1), 2)
W = np.round(np.arange(0.0, 1.01, 0.1), 2)


def pixel_fields():
    px = pd.read_csv(os.path.join(OUT, "pixels_nrw.csv.gz"))
    return px.sort_values("share").drop_duplicates(["row", "col"], keep="last").set_index(["row", "col"])


def prior_matrix(dates, fid):
    pri = pd.read_csv(os.path.join(OUT, "prior_lai_wheat.csv.gz"), parse_dates=["date"])
    pri = pri[pri["date"].isin(dates)].pivot(index="date", columns="field_id", values="prior_lai").reindex(dates)
    M = np.full((len(dates), len(fid)), np.nan, np.float32)
    has = np.isin(fid, pri.columns)
    M[:, has] = pri[fid[has]].values
    return M


def rm(v):
    return float(np.sqrt(np.mean(np.square(v))))


def fit(e):
    best = (np.inf, 1.0, 0.0)
    for k in K:
        shape = e["lin"] + k * e["m_dev"]
        for w in W:
            r = rm(e["re"] + w * (shape - e["re"]) - e["obs"])
            if r < best[0]:
                best = (r, k, w)
    return best[1], best[2]


def main():
    z = np.load(os.path.join(OUT, "s2_pixels.npz"))
    s2d, rows, cols = pd.DatetimeIndex(z["dates"]), z["rows"], z["cols"]
    mainf = pixel_fields()
    key = pd.MultiIndex.from_arrays([rows, cols])
    fid, share = mainf["field_id"].reindex(key).values, mainf["share"].reindex(key).values
    days = (s2d - s2d[0]).days.values
    y = z["lai"]
    clear = np.isfinite(y)                                          # same hiding as fusion3 (lai)
    hide = clear & (np.random.default_rng(70).random(clear.shape) < 0.3)
    first = np.argmax(clear, axis=0)
    last = clear.shape[0] - 1 - np.argmax(clear[::-1], axis=0)
    hide[first, np.arange(clear.shape[1])] = False
    hide[last, np.arange(clear.shape[1])] = False
    prv, nxt = prev_next(clear & ~hide)
    rng = np.random.default_rng(7)
    folds = {f: i % 5 for i, f in enumerate(rng.permutation(np.unique(fid)))}
    PRI = prior_matrix(s2d, fid)
    c = pd.read_csv(os.path.join(OUT, "fusion3_check.csv.gz"))
    e = c[(c.target == "lai") & (c.crop == "wheat")].copy()
    p, i = e["pixel"].values, e["date_i"].values
    a, b = prv[i, p], nxt[i, p]
    pos = (days[i] - days[a]) / (days[b] - days[a])
    e["pos"] = pos
    e["m_dev"] = PRI[i, p] - (PRI[a, p] + pos * (PRI[b, p] - PRI[a, p]))
    e["lin"], e["re"], e["fold"] = e["interp"], e["reanalysis"], [folds[f] for f in fid[p]]
    e = e[np.isfinite(e["m_dev"])].copy()
    lin_chk = y[a, p] + pos * (y[b, p] - y[a, p])
    print(f"hidden wheat LAI rows: {len(c[(c.target == 'lai') & (c.crop == 'wheat')])}, with physics {len(e)}; anchors reproduce the line: {np.allclose(lin_chk[np.isfinite(PRI[i, p] - PRI[a, p] - PRI[b, p])], e['lin'], atol=1e-3)}")
    e["fixed"] = np.nan
    for f in range(5):
        k, w = fit(e[e.fold != f])
        m = e.fold == f
        e.loc[m, "fixed"] = e.loc[m, "re"] + w * (e.loc[m, "lin"] + k * e.loc[m, "m_dev"] - e.loc[m, "re"])
        print(f"fold {f}: k {k}, w {w}", flush=True)
    kf, wf = fit(e)
    json.dump({"k": float(kf), "w": float(wf)}, open(os.path.join(OUT, "fix2_params.json"), "w"))
    e[["pixel", "date_i", "obs", "lin", "re", "fixed", "gap", "pos", "m_dev", "fold"]].to_csv(
        os.path.join(OUT, "fix2_check.csv.gz"), index=False)
    out = {}
    for name, g in (("16-30 d", e[(e.gap > 15) & (e.gap <= 30)]), (">30 d", e[e.gap > 30]),
                    (">30 d, last third", e[(e.gap > 30) & (e.pos > 2 / 3)]), (">30 d, first third", e[(e.gap > 30) & (e.pos < 1 / 3)])):
        r = dict(n=len(g), line=rm(g.lin - g.obs), current=rm(g.re - g.obs), fixed=rm(g.fixed - g.obs),
                 bias_current=float((g.re - g.obs).mean()), bias_fixed=float((g.fixed - g.obs).mean()))
        out[name] = r
        print(f"{name:20s} n {r['n']:7d}  RMSE line {r['line']:.3f} current {r['current']:.3f} fixed {r['fixed']:.3f}  "
              f"bias current {r['bias_current']:+.3f} fixed {r['bias_fixed']:+.3f}")
    # independent 15 June (partly cloudy scene; never used): weekly pixel series (current re-analysis) -> fixed
    d = pd.Timestamp("2024-06-15")
    wk = np.load(os.path.join(OUT, "weekly_pixel_series_2024.npz"))
    weeks = pd.DatetimeIndex(wk["weeks"])
    w0 = weeks.get_loc(pd.Timestamp("2024-06-10"))
    re15 = wk["lai"][w0] + (5 / 7) * (wk["lai"][w0 + 1] - wk["lai"][w0])
    alld = s2d.union([d]); j = alld.get_loc(d)
    yy = np.full((len(alld), len(rows)), np.nan, np.float32); yy[[alld.get_loc(t) for t in s2d]] = y
    dd = (alld - alld[0]).days.values
    pv, nx = prev_next(np.isfinite(yy))
    A, B = pv[j], nx[j]
    ok = (A >= 0) & (B >= 0)
    A, B = np.where(ok, A, 0), np.where(ok, B, 0)
    pp = (dd[j] - dd[A]) / (dd[B] - dd[A])
    cc = np.arange(len(rows))
    PR = prior_matrix(alld, fid)
    lin = yy[A, cc] + pp * (yy[B, cc] - yy[A, cc])
    mdev = PR[j] - (PR[A, cc] + pp * (PR[B, cc] - PR[A, cc]))
    fixed = re15 + wf * (lin + kf * mdev - re15)
    chk = pd.read_csv(os.path.join(OUT, "check_late_gap.csv"), parse_dates=["date"])
    chk = chk[(chk.date == d) & (chk.crop == "wheat")]
    rec = []
    for f_, ob in zip(chk.field_id, chk.obs):
        m = (fid == f_) & ok & np.isfinite(fixed) & np.isfinite(re15)
        wgt = share[m]
        rec.append(dict(field_id=f_, obs=ob, current=np.average(re15[m], weights=wgt), fixed=np.average(fixed[m], weights=wgt)))
    r15 = pd.DataFrame(rec)
    print(r15.round(2).to_string(index=False))
    for k_ in ("current", "fixed"):
        print(f"15 June wheat (independent, n {len(r15)}): {k_} bias {(r15[k_] - r15.obs).mean():+.3f}, RMSE {rm(r15[k_] - r15.obs):.3f}")
    g30, g16, gl = out[">30 d"], out["16-30 d"], out[">30 d, last third"]
    ok_rule = (g30["fixed"] < g30["current"] and g16["fixed"] <= g16["current"] + 0.01
               and abs(gl["bias_fixed"]) <= abs(gl["bias_current"])
               and abs((r15.fixed - r15.obs).mean()) < abs((r15.current - r15.obs).mean()))
    print(f"final k {kf}, w {wf}\nRESULT fix 2: {'ADOPT' if ok_rule else 'not adopted'}")


if __name__ == "__main__":
    main()
