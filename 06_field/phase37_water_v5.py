"""Water v5: v2 with two physically motivated settings fitted at the towers (pre-registered docs/project_log.md
2026-11-06): soil-evaporation scale f_e (residue / crust) and the Kcb-LAI extinction k (partial cover).
Truth = energy-balance-corrected tower ET (LE_CORR); raw LE_F_MDS reported as lower bound; towers with energy balance
ratio > 1.05 never decide. Fit on DE-RuS 2021-2025 + DE-Kli 2018/2021/2023/2025; DE-Geb reported.
This is a FIT on seen data (descriptive). The proof is the held-out test (BE-Lon, FR-Gri, DE-Kli 2017), pre-registered.
Run from D:\\vista: python phase37_water_v5.py -> data/processed/field_v2/water_v5/
"""
import os

import numpy as np
import pandas as pd

OUT = os.path.join("data", "processed", "field_v2", "water_v5")
FE_GRID = np.round(np.arange(0.50, 1.0001, 0.05), 2)
K_GRID = np.round(np.arange(0.30, 0.7001, 0.05), 2)
V2 = (1.0, 0.70)
FIT = [("selhausen", y) for y in (2021, 2022, 2023, 2024, 2025)] + [("dekli", y) for y in (2018, 2021, 2023, 2025)]
OTHER = [("degeb", y) for y in (2021, 2022, 2024)]


# ------------------------------------------------------------------------------------------------- pure functions
def deciding(ebr):
    return ebr <= 1.05


def choose(scores):
    """Lowest score; ties -> the setting closest to v2 (f_e 1.0, k 0.7)."""
    best = min(scores.values())
    tied = [s for s, v in scores.items() if np.isclose(v, best)]
    return min(tied, key=lambda s: abs(s[0] - V2[0]) + abs(s[1] - V2[1]))


def weekly_scores(model, tower, valid, start, min_days=5):
    """(RMSE of weekly sums, number of weeks, mean log ratio model / tower) over weeks from start with >= min_days valid."""
    m, t = weekly_sums(model, tower, valid, start, min_days)
    if len(m) == 0:
        return np.nan, 0, np.nan
    return float(np.sqrt(np.mean((m - t) ** 2))), len(m), float(np.mean(np.log(m / t)))


def weekly_sums(model, tower, valid, start, min_days=5):
    wk = (model.index - start).days // 7
    df = pd.DataFrame(dict(m=model.where(valid), t=tower.where(valid), v=valid.astype(int), wk=wk))
    g = df.groupby("wk").agg(m=("m", "sum"), t=("t", "sum"), v=("v", "sum"))
    g = g[(g.v >= min_days) & (g.t > 0)]
    return g.m.values, g.t.values


# ------------------------------------------------------------------------------------------------- data
def season_inputs(site, year):
    """Daily inputs of the v2 single run, the tower truth and the energy balance ratio for one season."""
    from phase29_water_v2 import KCB_MID, P_TAB
    from phase34_water_v4 import ec_daily
    from phase35_water_bias import ONE, ebr, model_season, tower_path, tower_season
    if site == "selhausen":
        from phase30_water_v3 import season_ensemble
        E = season_ensemble(year, ONE)
        tew = float(E["tew"])
    else:
        from phase34_water_v4 import season_ensemble
        E = season_ensemble(site, year, ONE)
        tew = float(E["ens"]["tew"][0])
    m, I = model_season(site, year)
    t = tower_season(site, m.index, I["lat"], I["elev"])
    days = m.index
    inside = (days >= I["start"]) & (days <= I["end"])
    corr = ec_daily(tower_path(site), "LE_CORR").reindex(days)
    return dict(site=site, year=year, crop=I["crop"], dates=days, start=I["start"], peak=I["peak"], inside=inside,
                et0=m["et0_m"].values, rain=m["rain_m"].values, lai=m["lai"].values, taw=E["ens"]["taw"][0], tew=tew,
                p=P_TAB[I["crop"]], kcb_mid=KCB_MID[I["crop"]], et_v2=m["et_m"].values,
                corr=corr, raw=t["et_t"], ebr=float(ebr(site).get(year, np.nan)))


def run_model(S, f_e, k):
    from phase29_water_v2 import dual_kc_balance, kcb_from_lai
    kcb = kcb_from_lai(S["lai"], S["kcb_mid"], k)
    return dual_kc_balance(S["et0"], S["rain"], kcb, S["lai"], S["taw"], S["p"], S["tew"], ke_scale=f_e)


def season_score(S, et, truth="corr"):
    tower = S[truth]
    valid = pd.Series(S["inside"], index=S["dates"]) & tower.notna()
    return weekly_sums(pd.Series(et, index=S["dates"]), tower.fillna(0), valid, S["start"])


def run():
    from phase35_water_bias import stages
    os.makedirs(OUT, exist_ok=True)
    seasons = {s: season_inputs(*s) for s in FIT + OTHER}
    for s, S in seasons.items():                                         # regression: v2 single run reproduced
        d = np.abs(run_model(S, *V2)["et"] - S["et_v2"]).max()
        assert d < 1e-9, f"{s}: v2 not reproduced ({d})"
    fit = [s for s in FIT if deciding(seasons[s]["ebr"])]
    grid = {}
    for fe in FE_GRID:
        for k in K_GRID:
            m, t = [], []
            for s in fit:
                a, b = season_score(seasons[s], run_model(seasons[s], fe, k)["et"])
                m.append(a); t.append(b)
            m, t = np.concatenate(m), np.concatenate(t)
            grid[(float(fe), float(k))] = float(np.sqrt(np.mean((m - t) ** 2)))
    best = choose(grid)
    G = pd.Series(grid).unstack()
    G.to_csv(os.path.join(OUT, "grid_rmse.csv"))
    L = [f"Fit seasons (closure rule): {fit}",
         f"CHOSEN f_e {best[0]:.2f}, k {best[1]:.2f}: pooled weekly RMSE {grid[best]:.2f} mm/week (v2 {grid[V2]:.2f})",
         "Grid (rows f_e, columns k), pooled weekly RMSE vs LE_CORR:", G.round(2).to_string()]
    rows, stage_rows, days_out = [], [], []
    for s, S in seasons.items():
        r2, r5 = run_model(S, *V2), run_model(S, *best)
        row = dict(site=s[0], year=s[1], crop=S["crop"], ebr=S["ebr"], fit=s in fit)
        for name, out in (("v2", r2), ("v5", r5)):
            for truth in ("corr", "raw"):
                rm, n, lr = weekly_scores(pd.Series(out["et"], index=S["dates"]), S[truth].fillna(0),
                                          pd.Series(S["inside"], index=S["dates"]) & S[truth].notna(), S["start"])
                row[f"{name}_rmse_{truth}"], row[f"{name}_factor_{truth}"], row["weeks_" + truth] = rm, np.exp(lr), n
            row[f"{name}_et_season"] = float(out["et"][S["inside"]].sum())
            row[f"{name}_stress_days"] = int((out["ks"][S["inside"]] < 0.9).sum())
            row[f"{name}_first_stress"] = (str(S["dates"][S["inside"]][out["ks"][S["inside"]] < 0.9][0].date())
                                           if (out["ks"][S["inside"]] < 0.9).any() else "-")
        ok = S["inside"] & S["corr"].notna().values
        row["tower_corr_season_valid"] = float(S["corr"].values[ok].sum())
        row["v2_et_valid"], row["v5_et_valid"] = float(r2["et"][ok].sum()), float(r5["et"][ok].sum())
        rows.append(row)
        st = stages(S["dates"], S["lai"], S["peak"])
        days_out.append(pd.DataFrame(dict(site=s[0], year=s[1], date=S["dates"], stage=st, v2=r2["et"], v5=r5["et"],
                                          ks_v2=r2["ks"], ks_v5=r5["ks"], e_v2=r2["e"], e_v5=r5["e"], corr=S["corr"].values,
                                          raw=S["raw"].values, ok=ok, fit=s in fit)))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "seasons.csv"), index=False)
    D = pd.concat(days_out)
    D.to_csv(os.path.join(OUT, "days.csv"), index=False)
    L.append("\n== Per season (weekly RMSE mm/week and weekly log-mean factor model / tower; LE_CORR truth, raw in brackets)")
    for _, r in R.iterrows():
        L.append(f"{r.site:9s} {r.year} {r.crop:10s} EBR {r.ebr:.2f}{' fit' if r.fit else '    '} | RMSE v2 {r.v2_rmse_corr:5.2f} "
                 f"v5 {r.v5_rmse_corr:5.2f} ({r.v2_rmse_raw:.2f}/{r.v5_rmse_raw:.2f}) | factor v2 {r.v2_factor_corr:.2f} "
                 f"v5 {r.v5_factor_corr:.2f} ({r.v2_factor_raw:.2f}/{r.v5_factor_raw:.2f}) | season ET valid days "
                 f"tower {r.tower_corr_season_valid:.0f} v2 {r.v2_et_valid:.0f} v5 {r.v5_et_valid:.0f} | stress days v2 "
                 f"{r.v2_stress_days} ({r.v2_first_stress}) v5 {r.v5_stress_days} ({r.v5_first_stress})")
    L.append("\n== By stage, fit seasons (sum ET on valid days; ratio to tower LE_CORR)")
    F = D[D.fit & D.ok]
    sg = F.groupby("stage").agg(days=("corr", "size"), tower=("corr", "sum"), v2=("v2", "sum"), v5=("v5", "sum"))
    sg["v2_ratio"], sg["v5_ratio"], sg["change"] = sg.v2 / sg.tower, sg.v5 / sg.tower, sg.v5 / sg.v2 - 1
    L.append(sg.reindex(["lai<1", "lai1-3", "lai>=3", "after_peak"]).round(2).to_string())
    Rf = R[R.fit]
    L.append(f"\nFit seasons pooled factor (mean of season factors) v2 {np.exp(np.log(Rf.v2_factor_corr).mean()):.2f} -> "
             f"v5 {np.exp(np.log(Rf.v5_factor_corr).mean()):.2f}; raw truth v2 {np.exp(np.log(Rf.v2_factor_raw).mean()):.2f}"
             f" -> v5 {np.exp(np.log(Rf.v5_factor_raw).mean()):.2f}; v5 better than v2 (RMSE, LE_CORR) in "
             f"{int((Rf.v5_rmse_corr < Rf.v2_rmse_corr).sum())} of {len(Rf)} fit seasons")
    txt = "\n".join(L)
    open(os.path.join(OUT, "fit_log.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    run()
