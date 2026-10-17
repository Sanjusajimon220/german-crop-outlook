"""Step 2 (pre-registered docs/project_log.md 2026-10-15): satellite assimilation for WHEAT fields, pilot 2024.

Crop model: field model v4 PRIOR ensemble (3,000 members per weather group; physics enters only here).
Observations: final satellite-only weekly series weekly_field_series_2024_sat.csv (Dec / Jan dropped as v4).
Variants: P prior only; B measured weeks only; C all weekly values with their test uncertainty.
Look-ahead test at 04-01, 05-01, 06-01 (see log); production run with the preferred variant.
Outputs field_v2/pilot_2024/assim_wheat_check.csv, assim_wheat_2024.csv, assim_wheat_weekly_2024.csv
Run `python phase9_assim_wheat.py [check|run|both]`.
"""
import os
import sys

import numpy as np
import pandas as pd

import phase4_field_assim_v4 as v4
from phase2_growth import P
from phase4_field_assim import CROP, FD, wquant
from phase4_field_assim_v3 import logprior_for, soil_ratio, weights

OUT = os.path.join(P, "field_v2", "pilot_2024")
YEAR = 2024
CUTS = ["04-01", "05-01", "06-01"]


def wheat_fields():
    fl = pd.read_csv(os.path.join(OUT, "fields_nrw.csv"))
    wheat = fl.loc[fl["crop"].str.contains("Winterweichweizen", na=False), ["field_id", "area_ha"]]
    cell = pd.read_csv(os.path.join(FD, "fields_hyras_cell.csv"))
    cell = cell[(cell["VALIDFROM"] == YEAR) & (cell["CODE"] == 115)].set_index("ID")["cell"]
    dist = pd.read_csv(os.path.join(FD, "fields_district.csv"), dtype={"district": str})
    dist = dist[(dist["VALIDFROM"] == YEAR) & (dist["CODE"] == 115)].set_index("ID")["district"]
    f = wheat.copy()
    f["cell"], f["district"] = f["field_id"].map(cell), f["field_id"].map(dist)
    return f.dropna(subset=["cell", "district"]).astype({"cell": int})


def observations():
    s = pd.read_csv(os.path.join(OUT, "weekly_field_series_2024_sat.csv"), parse_dates=["week"])
    s = s[(s["crop"] == "wheat") & ~s["week"].dt.month.isin([12, 1])].rename(columns={"week": "date"})
    s["lai_sd"] = s["lai_uncertainty"]
    return s


def days_of(g, e):
    day = (g["date"] - e["sowing"]).dt.days.values
    ok = (day >= 0) & (day <= e["mature_day"]) & (day < e["lai"].shape[1])
    return ok, day


def setup():
    f = wheat_fields()
    obs = observations()
    f = f[f["field_id"].isin(obs["field_id"])]
    print(f"wheat fields with weather cell, district and weekly series: {len(f)}", flush=True)
    ens = v4.ensembles(YEAR, f)
    ratio = soil_ratio()
    return f, obs, ens, ratio


def group_of(ens, x):
    E = ens.get(x["district"])
    if E is None or x["cell"] not in E["cell_group"]:
        return None, None
    return E, E["groups"][E["cell_group"][x["cell"]]]


def check(f, obs, ens, ratio):
    rows = []
    for _, x in f.iterrows():
        E, e = group_of(ens, x)
        if e is None:
            continue
        g = obs[obs["field_id"] == x["field_id"]].sort_values("date")
        ok, day = days_of(g, e)
        lp = logprior_for(e, ratio.get((x["field_id"], YEAR)))
        meas = (g["source"] == "measured").values
        for cut in CUTS:
            c = pd.Timestamp(f"{YEAR}-{cut}")
            before = (g["date"] < c).values
            if not (meas & before).any():
                continue
            last_meas = g["date"][meas & before].max()
            target = ok & meas & ~before
            if not target.any():
                continue
            persist = g["lai"].values[meas & before][-1]
            use = {"P": np.zeros(len(g), bool), "B": ok & meas & before,
                   "C": ok & before & (g["date"] <= last_meas).values}
            pred = {}
            for v, u in use.items():
                w = weights(e["lai"], day[u], g["lai"].values[u], g["lai_sd"].values[u], lp)
                pred[v] = w @ e["lai"][:, day[target]]
            rows.append(pd.DataFrame({"field_id": x["field_id"], "cut": cut, "date": g["date"].values[target],
                                      "obs": g["lai"].values[target], "persistence": persist,
                                      **{v: p for v, p in pred.items()}, "n_B": use["B"].sum(), "n_C": use["C"].sum()}))
    r = pd.concat(rows, ignore_index=True)
    r.to_csv(os.path.join(OUT, "assim_wheat_check.csv"), index=False)
    rm = lambda v: float(np.sqrt(np.mean(np.square(v))))
    summ = []
    for cut, g in r.groupby("cut"):
        summ.append(dict(cut=cut, fields=g["field_id"].nunique(), n=len(g), **{f"rmse_{k}": rm(g[k] - g.obs) for k in ("persistence", "P", "B", "C")},
                         **{f"bias_{k}": float((g[k] - g.obs).mean()) for k in ("persistence", "P", "B", "C")}))
    s = pd.DataFrame(summ).round(3)
    print(s.to_string(index=False))
    useful = {v: int(sum((s[f"rmse_{v}"] < s["rmse_P"]) & (s[f"rmse_{v}"] < s["rmse_persistence"]))) for v in ("B", "C")}
    c_better = int((s["rmse_C"] < s["rmse_B"]).sum())
    print(f"RESULT: cuts where B beats prior AND persistence {useful['B']}/3, C {useful['C']}/3 -> assimilation "
          f"{'USEFUL' if max(useful.values()) >= 2 else 'not useful'}; C better than B at {c_better}/3 -> preferred "
          f"{'C' if c_better >= 2 else 'B'}")
    return "C" if c_better >= 2 else "B"


def run(f, obs, ens, ratio, variant):
    pred = pd.read_csv(os.path.join(P, f"growth_predictions_{CROP}.csv"), dtype={"district": str})
    blend = pred[pred["year"] == YEAR].set_index("district")["blend"]
    rows, weekly = [], []
    for _, x in f.iterrows():
        E, e = group_of(ens, x)
        if e is None:
            continue
        g = obs[obs["field_id"] == x["field_id"]].sort_values("date")
        ok, day = days_of(g, e)
        u = ok & ((g["source"] == "measured").values if variant == "B" else True)
        lp = logprior_for(e, ratio.get((x["field_id"], YEAR)))
        w = weights(e["lai"], day[u], g["lai"].values[u], g["lai_sd"].values[u], lp)
        w0 = weights(e["lai"], day[:0], g["lai"].values[:0], g["lai_sd"].values[:0], lp)    # prior (soil only)
        t = e["tot"]
        rel = t["phys_yield"].values / E["prior_mean"]
        r = dict(field_id=x["field_id"], district=x["district"], area_ha=x["area_ha"], n_obs=int(u.sum()), ess=1 / (w ** 2).sum(),
                 rel=w @ rel, rel_p10=wquant(rel, w, 0.1), rel_p90=wquant(rel, w, 0.9), rel_prior=w0 @ rel,
                 sowing=e["sowing"].date())
        for k in ("phys_lai_max", "phys_eta_mm", "phys_etp_mm", "cap_f", "n_supply"):
            r[k] = w @ t[k].values
            r[f"{k}_prior"] = w0 @ t[k].values
        rows.append(r)
        L = e["lai"]
        for k in range(0, min(L.shape[1] - 1, e["mature_day"]) + 1, 7):    # weekly, sowing to maturity
            d = e["sowing"] + pd.Timedelta(days=k)
            weekly.append((x["field_id"], d.date(), w @ L[:, k], wquant(L[:, k], w, 0.1), wquant(L[:, k], w, 0.9), w0 @ L[:, k]))
    res = pd.DataFrame(rows)
    if res["district"].isin(blend.index).any():
        res["k"] = res["district"].map(blend) / res["district"].map(E_prior_scale(res))
    else:
        res["k"] = np.nan
    for c in ("rel", "rel_p10", "rel_p90", "rel_prior"):
        res[c.replace("rel", "yield_t_ha")] = res[c] * res["k"]
    res.to_csv(os.path.join(OUT, "assim_wheat_2024.csv"), index=False)
    pd.DataFrame(weekly, columns=["field_id", "date", "lai_post", "lai_p10", "lai_p90", "lai_prior"]).round(3).to_csv(
        os.path.join(OUT, "assim_wheat_weekly_2024.csv"), index=False)
    print(f"production ({variant}): {len(res)} fields; median ESS {res['ess'].median():.0f}; "
          f"yield t/ha 10/50/90 %: {np.round(res['yield_t_ha'].quantile([.1, .5, .9]).values, 2)} "
          f"(prior {np.round(res['yield_t_ha_prior'].quantile([.1, .5, .9]).values, 2)}); "
          f"ET mm 10/50/90 %: {np.round(res['phys_eta_mm'].quantile([.1, .5, .9]).values, 0)}; "
          f"LAI max median {res['phys_lai_max'].median():.2f} (prior {res['phys_lai_max_prior'].median():.2f})")


def E_prior_scale(res):
    """Area-weighted district mean of the posterior relative yield (as v4): field yields average to the district model."""
    return res.groupby("district").apply(lambda g: np.average(g["rel"], weights=g["area_ha"]), include_groups=False)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    f, obs, ens, ratio = setup()
    best = check(f, obs, ens, ratio) if mode in ("check", "both") else (sys.argv[2] if len(sys.argv) > 2 else "B")
    if mode in ("run", "both"):
        run(f, obs, ens, ratio, best)
