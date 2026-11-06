"""Water v6 STRESS SIDE (pre-registered docs/project_log.md 2026-11-06): v5 + root-depth scale r and depletion-fraction
shift dp, fitted on all 22 seen deciding seasons (DE-RuS, DE-Kli, BE-Lon, FR-Gri) against LE_CORR weekly ET.
Seen data only = descriptive. The unseen one look (FR-EM2 / CZ-KrP / CH-Oe2) is a separate step after its own download OK.
Run from D:\\vista: python phase39_water_v6.py -> data/processed/field_v2/water_v6/
"""
import os
import pickle

import numpy as np
import pandas as pd

OUT = os.path.join("data", "processed", "field_v2", "water_v6")
R_GRID = np.round(np.arange(1.0, 2.0001, 0.1), 2)
DP_GRID = np.round(np.arange(0.0, 0.3001, 0.05), 2)
V5 = (0.60, 0.45)                 # f_e, k
DEPTH_CAP, P_CAP, STRESS_KS = 2.0, 0.80, 0.8


# ------------------------------------------------------------------------------------------------- pure functions
def root_depth(root_max, r, frac):
    """Linear root growth 0.1 m -> min(r x FAO max, 2 m profile) at peak LAI."""
    return 0.1 + (min(r * root_max, DEPTH_CAP) - 0.1) * np.asarray(frac, float)


def shifted_p(p, dp):
    return min(p + dp, P_CAP)


def choose(scores, tol=0.01):
    """Lowest pooled RMSE; settings within 1 % of it are ties -> closest to v5 (r 1.0, dp 0.0)."""
    best = min(scores.values())
    tied = [s for s, v in scores.items() if v <= best * (1 + tol)]
    return min(tied, key=lambda s: (abs(s[0] - 1.0) + abs(s[1]) / 0.3, scores[s]))


def stressed_log(m, t, ks, thr=STRESS_KS):
    """Mean log ratio model / tower over weeks whose mean v5 Ks < thr, and their number."""
    sel = np.asarray(ks) < thr
    if not sel.any():
        return np.nan, 0
    return float(np.mean(np.log(np.asarray(m)[sel] / np.asarray(t)[sel]))), int(sel.sum())


def verdict(rmse6, rmse5, log6, s6, s5, n_stress, min_stress=20):
    if n_stress < min_stress:
        return "NOT TESTABLE"
    ok = rmse6 <= 0.95 * rmse5 and abs(log6) <= np.log(1.10) and abs(s6) < abs(s5)
    return "PASS" if ok else "FAIL"


def weekly_with_ks(model, tower, ks, valid, start, min_days=5):
    """Weekly sums (as phase37 weekly_sums) plus the mean Ks of the valid days of each kept week."""
    wk = (model.index - start).days // 7
    df = pd.DataFrame(dict(m=model.where(valid), t=tower.where(valid), k=pd.Series(ks, index=model.index).where(valid),
                           v=valid.astype(int), wk=wk))
    g = df.groupby("wk").agg(m=("m", "sum"), t=("t", "sum"), k=("k", "mean"), v=("v", "sum"))
    g = g[(g.v >= min_days) & (g.t > 0)]
    return g.m.values, g.t.values, g.k.values


# ------------------------------------------------------------------------------------------------- data
def _paw(site):
    if site == "selhausen":
        from phase29_water_v2 import _soil
        return _soil()[0]
    if site in ("belon", "frgri"):
        from phase38_water_heldout import silt_loam_soil
        return silt_loam_soil()[0]
    from phase33_testsite_field import soil
    return soil(site)[0]


def seasons(cache=True):
    """All seen seasons (v5 fit + DE-Geb + held-out), each with its PAW profile and frac (0 at start, 1 at peak)."""
    path = os.path.join(OUT, "seasons.pkl")
    if cache and os.path.exists(path):
        return pickle.load(open(path, "rb"))
    import phase37_water_v5 as F
    import phase38_water_heldout as H
    S = {s: F.season_inputs(*s) for s in F.FIT + F.OTHER}
    for s in H.SEASONS:
        x = H.season(*s)
        if x is not None:
            S[s] = x
    for (site, _), x in S.items():
        x["paw"] = _paw(site)
        x["frac"] = np.clip(np.asarray((x["dates"] - x["start"]).days) / max((x["peak"] - x["start"]).days, 1), 0, 1)
    os.makedirs(OUT, exist_ok=True)
    pickle.dump(S, open(path, "wb"))
    return S


def run_model(S, r, dp, f_e=V5[0], k=V5[1]):
    from phase29_water_v2 import ROOT_MAX, _taw, dual_kc_balance, kcb_from_lai
    taw = _taw(S["paw"], root_depth(ROOT_MAX[S["crop"]], r, S["frac"]))
    return dual_kc_balance(S["et0"], S["rain"], kcb_from_lai(S["lai"], S["kcb_mid"], k), S["lai"], taw,
                           shifted_p(S["p"], dp), S["tew"], ke_scale=f_e)


def _weeks(S, out, ks_ref, truth="corr"):
    valid = pd.Series(S["inside"], index=S["dates"]) & S[truth].notna()
    return weekly_with_ks(pd.Series(out["et"], index=S["dates"]), S[truth].fillna(0), ks_ref, valid, S["start"])


def _swc_r(S, dr):
    """Not deciding: correlation of model root-zone depletion with the tower's mean soil-moisture drawdown in season."""
    from phase35_water_bias import tower_path
    cols = [c for c in pd.read_csv(tower_path(S["site"]), nrows=0).columns if c.startswith("SWC_F_MDS_") and not c.endswith("QC")]
    if not cols:
        return np.nan
    d = pd.read_csv(tower_path(S["site"]), usecols=["TIMESTAMP_END"] + cols, na_values=[-9999])
    d.index = pd.to_datetime(d.TIMESTAMP_END.astype(str), format="%Y%m%d%H%M") - pd.Timedelta(minutes=1)
    swc = d[cols].mean(axis=1).resample("D").mean().reindex(S["dates"])
    ok = S["inside"] & swc.notna().values
    return float(np.corrcoef(-np.asarray(dr)[ok], swc.values[ok])[0, 1]) if ok.sum() > 10 else np.nan


def run():
    from phase37_water_v5 import deciding
    os.makedirs(OUT, exist_ok=True)
    S = seasons()
    for s, x in S.items():                                   # regression: r 1.0 dp 0.0 reproduces v5 exactly
        from phase29_water_v2 import ROOT_MAX, _taw
        assert np.abs(_taw(x["paw"], root_depth(ROOT_MAX[x["crop"]], 1.0, x["frac"])) - x["taw"]).max() < 1e-9, s
    fit = [s for s, x in S.items() if deciding(x["ebr"]) and x["corr"][x["inside"]].notna().any()]
    v5 = {s: run_model(S[s], 1.0, 0.0) for s in S}
    grid = {}
    for r in R_GRID:
        for dp in DP_GRID:
            m, t = [], []
            for s in fit:
                a, b, _ = _weeks(S[s], run_model(S[s], r, dp), v5[s]["ks"])
                m.append(a); t.append(b)
            m, t = np.concatenate(m), np.concatenate(t)
            grid[(float(r), float(dp))] = float(np.sqrt(np.mean((m - t) ** 2)))
    pd.Series(grid).rename_axis(["r", "dp"]).rename("rmse").reset_index().to_csv(os.path.join(OUT, "grid_rmse.csv"), index=False)
    pick = choose(grid)
    rows, pooled = [], {"v5": ([], [], []), "v6": ([], [], [])}
    for s, x in S.items():
        out6 = run_model(x, *pick)
        row = dict(site=s[0], year=s[1], crop=x["crop"], ebr=x["ebr"], deciding=s in fit)
        for name, out in (("v5", v5[s]), ("v6", out6)):
            for truth in ("corr", "raw"):
                m, t, k = _weeks(x, out, v5[s]["ks"], truth)
                row[f"{name}_rmse_{truth}"] = float(np.sqrt(np.mean((m - t) ** 2))) if len(m) else np.nan
                row[f"{name}_factor_{truth}"] = float(np.exp(np.mean(np.log(m / t)))) if len(m) else np.nan
                if truth == "corr" and s in fit and len(m):
                    for i, v in enumerate((m, t, k)):
                        pooled[name][i].append(v)
            row[f"{name}_stress_days"] = int((out["ks"][x["inside"]] < 0.9).sum())
            row[f"{name}_swc_r"] = _swc_r(x, out["dr"])
        rows.append(row)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "seasons.csv"), index=False)
    res = {}
    for name in ("v5", "v6"):
        m, t, k = (np.concatenate(v) for v in pooled[name])
        res[name] = dict(rmse=float(np.sqrt(np.mean((m - t) ** 2))), log=float(np.mean(np.log(m / t))), weeks=len(m),
                         stress=stressed_log(m, t, k), k=k, m=m, t=t)
    cls = pd.cut(res["v5"]["k"], [0, 0.6, 0.8, 0.95, 1.01], labels=["<0.6", "0.6-0.8", "0.8-0.95", ">=0.95"])
    by = pd.DataFrame({"cls": cls, "v5": np.log(res["v5"]["m"] / res["v5"]["t"]), "v6": np.log(res["v6"]["m"] / res["v6"]["t"])})
    by = by.groupby("cls", observed=True).agg(weeks=("v5", "size"), v5=("v5", "mean"), v6=("v6", "mean"))
    by[["v5", "v6"]] = np.exp(by[["v5", "v6"]])
    D = R[R.deciding]
    L = [f"Fit seasons {len(fit)}; pooled weeks {res['v6']['weeks']}",
         f"Chosen r {pick[0]:.1f}, dp {pick[1]:.2f} (grid r {R_GRID[0]}-{R_GRID[-1]}, dp {DP_GRID[0]}-{DP_GRID[-1]})",
         f"Pooled weekly RMSE vs LE_CORR: v5 {res['v5']['rmse']:.2f} -> v6 {res['v6']['rmse']:.2f} "
         f"({100 * (res['v6']['rmse'] / res['v5']['rmse'] - 1):+.1f} %)",
         f"Pooled factor: v5 {np.exp(res['v5']['log']):.2f} -> v6 {np.exp(res['v6']['log']):.2f}",
         f"Stressed weeks (v5 Ks < 0.8, n {res['v5']['stress'][1]}): factor v5 {np.exp(res['v5']['stress'][0]):.2f} -> "
         f"v6 {np.exp(res['v6']['stress'][0]):.2f}",
         f"v6 better than v5 in {int((D.v6_rmse_corr < D.v5_rmse_corr).sum())} of {len(D)} fit seasons",
         f"Stress days (Ks < 0.9) in fit seasons: v5 {int(D.v5_stress_days.sum())} -> v6 {int(D.v6_stress_days.sum())}",
         "Factor by v5 weekly Ks class:", by.round(2).to_string(),
         "Seen data only: descriptive, NOT proof. Unseen one look pending (own download OK)."]
    txt = "\n".join(L)
    open(os.path.join(OUT, "fit_log.txt"), "w").write(txt + "\n")
    print(txt)
    return pick


if __name__ == "__main__":
    run()
