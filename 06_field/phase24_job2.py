"""Job 2 test: green LAI forecast per field (design and rule docs/project_log.md 2026-10-22, amendments 1-3).

Observations: our weekly field means (fix 2), measured weeks only (>= 50 % of the field's pixels measured that week),
inside the field's crop season, December / January never used. Cuts 1 April, 1 May, 1 June: weeks before the cut are
used, every later measured week of the season is a target.
  A persistence (last used week)          B our model, no satellite (soil prior only)
  C WOFOST 1,000 + particle weights        D our model 3,000 + particle weights (soil prior, as tested 15 Oct)
  E1 normal WOFOST (1 run)                 E2 / E3 WOFOST / our model + ensemble Kalman filter (100, phase24b)
Particle weights identical for C and D (phase4 v3: Student-t, error 0.5, tempering to ESS >= 50).
PASS per crop: D lower RMSE than A, B and C at >= 2 of 3 cuts AND |bias D at 1 June| <= 0.3 LAI.
Scenario report (amendment 4): D against all lines per season, area, soil third, spring rain half, field size and
forecast distance. Outputs field_v2/job2/job2_rows.csv.gz, job2_summary.csv, job2_scenarios.csv. Run from D:\\vista: python phase24_job2.py [years]
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CUTS = ("04-01", "05-01", "06-01")
RULE = ("A", "B", "C", "D")
EXTRA = ("E1", "E2", "E3")
MIN_SHARE = 0.5
SOIL_SD = 0.15
OUT = os.path.join("data", "processed", "field_v2", "job2")


def split_weeks(weeks, share, mask, cut, min_share=MIN_SHARE):
    """Measured weeks of the season before the cut (no December / January) and measured season weeks from the cut on."""
    w = weeks.astype("datetime64[D]")
    month = w.astype("datetime64[M]").astype(int) % 12 + 1
    meas = np.nan_to_num(share) >= min_share
    meas &= np.asarray(mask, bool)
    used = meas & (w < cut) & ~np.isin(month, (12, 1))
    return used, meas & (w >= cut)


def summarise(rows, methods):
    out = []
    for (crop, cut), g in rows.groupby(["crop", "cut"]):
        r = dict(crop=crop, cut=cut, n=len(g))
        if "field" in g:
            r["fields"] = g["field"].nunique()
        for m in methods:
            e = (g[m] - g["obs"]).dropna()
            r[f"n_{m}"], r[f"rmse_{m}"], r[f"bias_{m}"] = len(e), float(np.sqrt(np.mean(e ** 2))), float(e.mean())
        out.append(r)
    return pd.DataFrame(out)


def verdict(s):
    v = {}
    for crop, g in s.groupby("crop"):
        g = g.set_index("cut")
        wins = int(sum(g.at[c, "rmse_D"] < min(g.at[c, f"rmse_{m}"] for m in ("A", "B", "C")) for c in g.index))
        bias = round(float(g.at["06-01", "bias_D"]), 3)
        v[crop] = dict(wins=wins, bias_0601=bias, **{"pass": wins >= 2 and abs(bias) <= 0.3})
        if all(f"rmse_{m}" in g for m in ("E1", "E2")):     # amendment 5: claim against all WOFOST lines
            ww = int(sum(g.at[c, "rmse_D"] < min(g.at[c, f"rmse_{m}"] for m in ("C", "E1", "E2")) for c in g.index))
            v[crop].update(wins_wofost=ww, claim_wofost=ww >= 2 and abs(bias) <= 0.3)
    return v


def soil_logprior(cap_f, ratio):
    if ratio is None or not np.isfinite(ratio):
        return np.zeros(len(cap_f))
    return -0.5 * ((np.log(cap_f) - np.log(np.clip(ratio, 0.5, 1.5))) / SOIL_SD) ** 2


def _quantile(x, w, q):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return np.interp(q, c - w[o] / 2, x[o])


def field_rows(t, F, k, W, O, lp, E2=None, E3=None):
    """Forecast rows of one field-season for the three cuts."""
    from phase4_field_assim_v3 import weights
    weeks, y, share, mask = F["weeks"], F["lai"][:, k], F["meas"][:, k], F["mask"][:, k]
    if W is not None:
        wl = W["lai"].astype(float)
        wl = wl[np.isfinite(wl).all(1)]
    ol = O["lai"].astype(float)
    rows = []
    for j, cut in enumerate(CUTS):
        used, target = split_weeks(weeks, share, mask, np.datetime64(f"{t['year']}-{cut}"))
        used &= np.isfinite(y)
        target &= np.isfinite(y)
        if not used.any() or not target.any():
            continue
        u = np.flatnonzero(used)
        z = np.zeros(len(u))
        w0 = weights(ol, u[:0], y[u][:0], z[:0], lp)
        wc = weights(wl, u, y[u], z, np.zeros(len(wl))) if W is not None else None
        wd = weights(ol, u, y[u], z, lp)
        tg = np.flatnonzero(target)
        d = wd @ ol[:, tg]
        r = pd.DataFrame(dict(crop=t["crop"], area=t["area"], year=t["year"], field=t["field"], cut=cut,
                              week=weeks[tg], obs=y[tg], n_used=len(u), A=y[u][-1], B=w0 @ ol[:, tg], C=wc @ wl[:, tg] if W is not None else np.nan,
                              D=d, E1=W["normal_lai"][tg].astype(float) if W is not None else np.nan,
                              E2=E2[j][:, tg].astype(float).mean(0) if E2 is not None else np.nan,
                              E3=E3[j][:, tg].astype(float).mean(0) if E3 is not None else np.nan,
                              D_p10=[_quantile(ol[:, i], wd, 0.1) for i in tg], D_p90=[_quantile(ol[:, i], wd, 0.9) for i in tg],
                              ess_D=1 / (wd ** 2).sum(), ess_C=1 / (wc ** 2).sum() if W is not None else np.nan))
        rows.append(r)
    return rows


def add_levels(r, tasks):
    """Scenario levels per row: season, area, soil third, spring-rain half, field size, forecast distance."""
    from phase4_field_assim_v4 import cell_weather
    t = tasks.drop_duplicates(["crop", "year", "field"]).set_index(["crop", "year", "field"])
    key = pd.MultiIndex.from_frame(r[["crop", "year", "field"]])
    nfk = t["nfk_mm"].reindex(key).values
    fields = t.reset_index()
    thirds = {c: np.nanquantile(g["nfk_mm"], [1 / 3, 2 / 3]) for c, g in fields.groupby("crop")}
    rain = {}
    for year in sorted(r.year.unique()):
        dates, cells, w = cell_weather(year)
        sel = (dates >= np.datetime64(f"{year}-03-01")) & (dates <= np.datetime64(f"{year}-06-30"))
        col = {c: i for i, c in enumerate(cells)}
        f = fields[fields.year == year]
        rain.update({(c, year, fl): w["pr"][sel, col[cl]].sum() if cl in col else np.nan
                     for c, fl, cl in zip(f.crop, f.field, f.cell)})
    rr = pd.Series(rain)
    med = rr.groupby(level=[0, 1]).median()
    sizes = {}
    for (area, year), g in fields.groupby(["area", "year"]):
        fl = pd.read_csv(os.path.join("data", "processed", "field_v2", f"{area}_{year}", "fields_nrw.csv")).set_index("field_id")["area_ha"]
        sizes.update({(year, f): fl.get(f, np.nan) for f in g.field})
    out = r.copy()
    out["season"] = out.year.astype(str)
    out["soil"] = [("low" if v <= thirds[c][0] else "middle" if v <= thirds[c][1] else "high") if np.isfinite(v) else "unknown"
                   for c, v in zip(out.crop, nfk)]
    out["spring_rain"] = ["drier" if rain.get((c, y, f), np.nan) < med.get((c, y), np.nan) else "wetter"
                          for c, y, f in zip(out.crop, out.year, out.field)]
    out["size"] = ["< 5 ha" if sizes.get((y, f), np.nan) < 5 else ">= 5 ha" for y, f in zip(out.year, out.field)]
    lead = (pd.to_datetime(out.week) - pd.to_datetime(out.year.astype(str) + "-" + out.cut)).dt.days / 7
    out["distance"] = np.where(lead < 4, "1-4 weeks", "> 4 weeks")
    return out


SCENARIOS = ("season", "area", "soil", "spring_rain", "size", "distance")


def scenarios(r):
    out = []
    for sc in SCENARIOS:
        for lev, g in r.groupby(sc):
            out.append(summarise(g, RULE + EXTRA).assign(scenario=sc, level=lev))
    return pd.concat(out, ignore_index=True)


def run(years=(2024, 2025), early=False):
    """early: disclosed look at the finished lines (A, B, D, E3) before the WOFOST lines exist; no verdict."""
    from phase16_hrvpp_items import _field_table
    import phase22b_wofost_run as WR
    import phase23_ours_field as OF
    tasks = pd.DataFrame(json.load(open(os.path.join(WR.W.OUT, "tasks.json"))))
    tasks = tasks[tasks.year.isin(years)]
    ratio = OF.soil_ratios()
    rows, missing = [], 0
    for (area, year), T in tasks.groupby(["area", "year"]):
        F = _field_table(area, year)
        pos = {int(f): k for k, f in enumerate(F["fids"])}
        for t in T.to_dict("records"):
            wp, op = WR.out_path(t), OF.out_path(t)
            k = pos.get(t["field"])
            if k is None or not os.path.exists(op) or not (early or os.path.exists(wp)):
                missing += 1
                continue
            W, O = (None if early else np.load(wp)), np.load(op)
            assert (O["weeks"] == F["weeks"]).all() and (W is None or (W["weeks"] == F["weeks"]).all())
            E = [np.load(p)["lai"] if os.path.exists(p) and not (early and d == "enkf") else None
                 for d, p in ((d, os.path.join(OUT, d, str(year), f"{t['crop']}_{t['field']}.npz")) for d in ("enkf", "enkf_ours"))]
            lp = soil_logprior(O["cap_f"], ratio[t["crop"]].get((t["field"], year)))
            rows += field_rows(t, F, k, W, O, lp, *E)
    r = pd.concat(rows, ignore_index=True)
    os.makedirs(OUT, exist_ok=True)
    tag = "_early" if early else ""
    r.to_csv(os.path.join(OUT, f"job2_rows{tag}.csv.gz"), index=False)
    s = summarise(r, RULE + EXTRA)
    s.to_csv(os.path.join(OUT, f"job2_summary{tag}.csv"), index=False)
    pd.set_option("display.width", 250)
    print(f"field-seasons without both ensembles (skipped): {missing}")
    print(s.round(3).to_string(index=False))
    for y_, g in r.groupby("year"):
        print(f"\nseason {y_} (supporting):")
        print(summarise(g, RULE + EXTRA)[["crop", "cut", "n"] + [f"rmse_{m}" for m in RULE + EXTRA]].round(3).to_string(index=False))
    sc = scenarios(add_levels(r, tasks))
    sc.to_csv(os.path.join(OUT, f"job2_scenarios{tag}.csv"), index=False)
    print("\nscenarios (reported, RMSE):\n" + sc[["crop", "scenario", "level", "cut", "n"] + [f"rmse_{m}" for m in RULE + EXTRA]].round(3).to_string(index=False))
    inside = ((r.obs >= r.D_p10) & (r.obs <= r.D_p90)).groupby([r.crop, r.cut]).mean()
    print("\nD 80 % band holds the observed week (reported):\n" + inside.round(3).to_string())
    if early:
        return None
    v = verdict(s)
    for crop, x in v.items():
        print(f"RESULT {crop}: D best at {x['wins']}/3 cuts, bias at 1 June {x['bias_0601']:+.3f} -> {'PASS' if x['pass'] else 'FAIL'}"
              + (f"; CLAIM vs WOFOST (C, E1, E2): D best at {x['wins_wofost']}/3 -> {'HOLDS' if x['claim_wofost'] else 'does not hold'}"
                 if "claim_wofost" in x else ""))
    json.dump(v, open(os.path.join(OUT, "job2_verdict.json"), "w"), indent=1)
    return v


if __name__ == "__main__":
    os.chdir(r"D:\vista")
    sys.path.insert(0, r"D:\vista")
    args = [a for a in sys.argv[1:] if a != "early"]
    run(tuple(int(a) for a in args) or (2024, 2025), early="early" in sys.argv)
