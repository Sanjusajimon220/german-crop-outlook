"""Phase 4, step 3d: field model v4 = v3 (own soil water) + own weather (DWD HYRAS 1 km).

The field's 1 km cell (data/processed/field/fields_hyras_cell.csv, hyras_cells_<year>.npz) gives
daily minimum / maximum temperature and rain. To keep the run time small, the cells of a district
that hold wheat fields are grouped into up to K weather groups (k-means on season rain, spring/
summer rain and mean temperature); each group gets the mean daily weather of its cells and its own
ensemble (same members and settings as v1), with
  - development (crop calendar, frozen) recomputed from the group's temperatures and rain
  - growth, water balance and heat stress driven by the group's temperatures and rain
Light (PAR) and reference evaporation (ET0) stay from the district weather (HYRAS radiation ends
2020; satellite radiation SARAH-3 later). Soil prior and satellite weighting as v3; yields
rescaled as v2 (area-weighted district mean = district model).
Run `python phase4_field_assim_v4.py run 2021` or `check 2022 06-01`.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy.cluster.vq import kmeans2

import phase2_phenology as phen
from phase2_growth import GROWTH, P, TAU, load_growth
from phase2_phenology import CROPS, Calendar
import phase4_field_assim as v1
from phase4_field_assim import CROP, FD, FieldGrowth, N_MEMBERS, draw, field_days, field_settings, observations, wquant
from phase4_field_assim_v3 import logprior_for, soil_ratio, weights

K = 20


def cell_weather(year):
    """Daily tmin, tmax, pr per HYRAS cell for the season (1 Sep previous year - 31 Aug)."""
    parts = [np.load(os.path.join(FD, f"hyras_cells_{y}.npz")) for y in (year - 1, year)]
    dates = np.concatenate([p["dates"] for p in parts])
    w = {v: np.concatenate([p[v] for p in parts]) for v in ("tasmin", "tasmax", "pr")}
    return dates, parts[0]["cells"], w


def groups(year, fields, dates, cells, w):
    """Weather groups per district: k-means on rain Oct-Jul, rain Apr-Jul, mean temp Oct-Jul."""
    col = {c: i for i, c in enumerate(cells)}
    season = (dates >= np.datetime64(f"{year - 1}-10-01")) & (dates <= np.datetime64(f"{year}-07-31"))
    spring = (dates >= np.datetime64(f"{year}-04-01")) & (dates <= np.datetime64(f"{year}-07-31"))
    out = {}
    for dist, g in fields.groupby("district"):
        cs = np.unique(g["cell"].values)
        idx = [col[c] for c in cs]
        feat = np.column_stack([w["pr"][season][:, idx].sum(0), w["pr"][spring][:, idx].sum(0),
                                ((w["tasmin"] + w["tasmax"]) / 2)[season][:, idx].mean(0)])
        k = min(K, len(cs))
        z = (feat - feat.mean(0)) / (feat.std(0) + 1e-6)
        if k > 1:
            _, lab = kmeans2(z, k, seed=1, minit="++")
        else:
            lab = np.zeros(len(cs), int)
        out[dist] = dict(cell_group=dict(zip(cs, lab)),
                         series={j: {v: w[v][:, [idx[i] for i in np.flatnonzero(lab == j)]].mean(1)
                                     for v in ("tasmin", "tasmax", "pr")} for j in np.unique(lab)})
    return out


def ensembles(year, fields, adjust=None):
    """adjust(dist, j, sub, dates_window) may change the group's drivers (v5: sunlight, ET0)."""
    m, d = load_growth(CROP)
    settings = field_settings()
    dates, cells, w = cell_weather(year)
    grp = groups(year, fields, dates, cells, w)
    stages, gate, vp, horizon = CROPS[CROP]
    g_ = GROWTH[CROP]
    cal = Calendar(len(stages), stages.index(gate), vp,
                   json.load(open(os.path.join(P, f"phenology_model_{CROP}.json"))))
    ie, ih, im = (stages.index(g_[k]) for k in ("emergence", "heading", "maturity"))
    ens = {}
    for i in m.index[(m["year"] == year) & m["district"].isin(list(grp))]:
        dist = m.at[i, "district"]
        sow = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=round(m.at[i, "doy_sowing"]) - 1)
        first = int((np.datetime64(sow.date()) - dates[0]).astype(int))
        n_days = d["tmin"].shape[1]
        pert = draw(N_MEMBERS, seed=int(dist) + year)
        pt = {k: torch.tensor(v.values, dtype=torch.float32) for k, v in pert.items()}
        prior_mean, per_group = [], {}
        for j, s in grp[dist]["series"].items():
            sub = {k: v[i:i + 1].clone() for k, v in d.items() if v.dim() and len(v) == len(m)}
            ok = slice(first, first + n_days)
            for key, var in (("tmin", "tasmin"), ("tmax", "tasmax"), ("pr", "pr")):
                series = s[var][ok]
                if len(series) == n_days and np.isfinite(series).all():
                    sub[key] = torch.tensor(series, dtype=torch.float32)[None, :]
            if adjust is not None:
                adjust(dist, j, sub, dates[ok], grp[dist], sub_district={k: d[k][i:i + 1] for k in ('tmin', 'tmax')})
            with torch.no_grad():
                dev, th = cal({"tmin": sub["tmin"], "tmax": sub["tmax"], "daylen": sub["daylen"],
                               "rain": torch.cumsum(sub["pr"], 1), "decade": sub["decade"]}, return_dev=True)
            sub = {k: v.repeat(N_MEMBERS, *([1] * (v.dim() - 1))) for k, v in sub.items()}
            devp = dev.repeat(N_MEMBERS, 1) * pt["dev_f"][:, None]
            e, h, mt = th[0, ie], th[0, ih], th[0, im]
            sub["em"] = torch.sigmoid((devp - e) / TAU)
            sub["ph"] = torch.clamp((devp - e) / (h - e), 0, 1)
            sub["post"] = torch.sigmoid((devp - h) / TAU)
            sub["gf"] = torch.clamp((devp - h) / (mt - h), 0, 1)
            sub["mat"] = torch.sigmoid((devp - mt) / TAU)
            sub["cap"] = sub["cap"] * pt["cap_f"][:, None]
            sub["n_supply"] = pt["n_supply"]
            with torch.no_grad():
                _, out = FieldGrowth(settings, pt)(sub, g_["dry_matter"], record=True)
            tot = pd.concat([pert, pd.DataFrame({k: v.numpy() for k, v in out.items() if k != "lai_daily"})], axis=1)
            mature = int((dev[0] >= mt).float().argmax())
            per_group[j] = dict(lai=out["lai_daily"].numpy(), tot=tot, sowing=sow, mature_day=mature)
            prior_mean.append(tot["phys_yield"].mean())
        ens[dist] = dict(groups=per_group, cell_group=grp[dist]["cell_group"], prior_mean=np.mean(prior_mean))
        print(f"  {dist}: {len(per_group)} weather groups", flush=True)
    return ens


def field_table(year):
    obs = observations(year)
    cell = pd.read_csv(os.path.join(FD, "fields_hyras_cell.csv"))
    cell = cell[(cell["VALIDFROM"] == year) & (cell["CODE"] == v1.CODE)].set_index("ID")["cell"]
    obs["cell"] = obs["field_id"].map(cell)
    return obs.dropna(subset=["cell"]).astype({"cell": int})


def run(year, ahead=None, adjust=None, tag="v4"):
    obs = field_table(year)
    ens = ensembles(year, obs.drop_duplicates("field_id")[["field_id", "district", "cell"]], adjust)
    ratio = soil_ratio()
    pred = pd.read_csv(os.path.join(P, f"growth_predictions_{CROP}.csv"), dtype={"district": str})
    blend = pred[pred["year"] == year].set_index("district")["blend"]
    rows, errs = [], []
    for (fid, dist), g in obs.groupby(["field_id", "district"]):
        if dist not in ens or dist not in blend.index:
            continue
        E = ens[dist]
        e = E["groups"][E["cell_group"][g["cell"].iloc[0]]]
        day, y, sd = field_days(g.sort_values("date"), e)
        lp = logprior_for(e, ratio.get((fid, year)))
        if ahead:                                 # look-ahead check (no yields)
            cut = (pd.Timestamp(f"{year}-{ahead}") - e["sowing"]).days
            use, hide = day < cut, day >= cut
            if use.sum() >= 3 and hide.sum():
                w = weights(e["lai"], day[use], y[use], sd[use], lp)
                errs += list(w @ e["lai"][:, day[hide]] - y[hide])
            continue
        w = weights(e["lai"], day, y, sd, lp)
        t = e["tot"]
        rel = t["phys_yield"].values / E["prior_mean"]
        r = dict(field_id=fid, district=dist, cell=g["cell"].iloc[0], n_obs=len(day), ess=1 / (w ** 2).sum(),
                 area_ha=g["area_ha"].iloc[0], soil_ratio=ratio.get((fid, year)), rel=w @ rel,
                 rel_p10=wquant(rel, w, 0.1), rel_p90=wquant(rel, w, 0.9))
        for k in ("phys_lai_max", "phys_eta_mm", "phys_etp_mm", "cap_f", "n_supply"):
            r[k] = w @ t[k].values
        rows.append(r)
    if ahead:
        errs = np.array(errs)
        print(f"look-ahead from {ahead}: LAI RMSE {np.sqrt((errs ** 2).mean()):.3f} (n {len(errs)})")
        return
    f = pd.DataFrame(rows)
    mean_rel = f.groupby("district").apply(lambda g: np.average(g["rel"], weights=g["area_ha"]), include_groups=False)
    k = f["district"].map(blend) / f["district"].map(mean_rel)
    f[f"{tag}_yield_t_ha"], f[f"{tag}_yield_p10"], f[f"{tag}_yield_p90"] = f["rel"] * k, f["rel_p10"] * k, f["rel_p90"] * k
    f.to_csv(os.path.join(FD, f"field_results_{tag}_{CROP}_{year}.csv"), index=False)
    prev = "v3" if tag == "v4" else "v4"
    p = pd.read_csv(os.path.join(FD, f"field_results_{prev}_{CROP}_{year}.csv"))[["field_id", f"{prev}_yield_t_ha"]]
    m = f.merge(p, on="field_id")
    print(f"{year}: {len(f)} fields; {tag} vs {prev} field yields r {m[f'{tag}_yield_t_ha'].corr(m[f'{prev}_yield_t_ha']):.3f}, "
          f"mean |diff| {(m[f'{tag}_yield_t_ha'] - m[f'{prev}_yield_t_ha']).abs().mean():.2f} t/ha; "
          f"field yield 10/50/90 %: {np.round(f[f'{tag}_yield_t_ha'].quantile([.1, .5, .9]).values, 2)}")


if __name__ == "__main__":
    t0 = time.time()
    mode, year = sys.argv[1], int(sys.argv[2])
    run(year, sys.argv[3] if mode == "check" else None)
    print(f"({time.time() - t0:.0f} s)")
