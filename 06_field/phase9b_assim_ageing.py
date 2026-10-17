"""Fix early leaf ageing, wheat assimilation pilot 2024 (pre-registered docs/project_log.md 2026-10-15).

B0 current; B1 development scaled per district so heading = DWD-observed 2024 heading (only for cuts after it);
B2 dev_f range 0.88-1.12 (stretched draws); B3 = B1 + B2. Variant B = measured weeks only.
Look-ahead test on the B0 target set; production with the adopted variant.
Outputs field_v2/pilot_2024/ageing_check.csv, (production) assim_wheat_2024.csv / assim_wheat_weekly_2024.csv if adopted.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

import phase4_field_assim_v4 as v4
import phase9_assim_wheat as A
from phase2_growth import GROWTH, P, TAU, load_growth
from phase2_phenology import CROPS, Calendar
from phase4_field_assim import CROP, FieldGrowth, N_MEMBERS, draw, field_settings
from phase4_field_assim_v3 import logprior_for, weights

OUT = A.OUT
YEAR = 2024


def obs_heading():
    ph = pd.read_csv(os.path.join(P, "phenology_predictions_winter_wheat.csv"), dtype={"district": str})
    ph = ph[ph["year"] == YEAR].set_index("district")
    return {d: (pd.Timestamp(f"{YEAR}-01-01") + pd.Timedelta(days=round(r["obs_heading"]) - 1), r["obs_heading"])
            for d, r in ph.iterrows() if np.isfinite(r["obs_heading"])}


def ensembles(year, fields, heading=False, wide=False):
    """As v4.ensembles; heading: scale development so the heading threshold is met on the observed day;
    wide: dev_f range stretched from +-6 % to +-12 %."""
    m, d = load_growth(CROP)
    settings = field_settings()
    dates, cells, w = v4.cell_weather(year)
    grp = v4.groups(year, fields, dates, cells, w)
    stages, gate, vp, horizon = CROPS[CROP]
    g_ = GROWTH[CROP]
    cal = Calendar(len(stages), stages.index(gate), vp, json.load(open(os.path.join(P, f"phenology_model_{CROP}.json"))))
    ie, ih, im = (stages.index(g_[k]) for k in ("emergence", "heading", "maturity"))
    oh = obs_heading()
    ens = {}
    for i in m.index[(m["year"] == year) & m["district"].isin(list(grp))]:
        dist = m.at[i, "district"]
        sow = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=round(m.at[i, "doy_sowing"]) - 1)
        first = int((np.datetime64(sow.date()) - dates[0]).astype(int))
        n_days = d["tmin"].shape[1]
        pert = draw(N_MEMBERS, seed=int(dist) + year)
        if wide:
            pert["dev_f"] = 1 + (pert["dev_f"] - 1) * 2.0
        pt = {k: torch.tensor(v.values, dtype=torch.float32) for k, v in pert.items()}
        prior_mean, per_group = [], {}
        for j, s in grp[dist]["series"].items():
            sub = {k: v[i:i + 1].clone() for k, v in d.items() if v.dim() and len(v) == len(m)}
            ok = slice(first, first + n_days)
            for key, var in (("tmin", "tasmin"), ("tmax", "tasmax"), ("pr", "pr")):
                series = s[var][ok]
                if len(series) == n_days and np.isfinite(series).all():
                    sub[key] = torch.tensor(series, dtype=torch.float32)[None, :]
            with torch.no_grad():
                dev, th = cal({"tmin": sub["tmin"], "tmax": sub["tmax"], "daylen": sub["daylen"],
                               "rain": torch.cumsum(sub["pr"], 1), "decade": sub["decade"]}, return_dev=True)
            e, h, mt = th[0, ie], th[0, ih], th[0, im]
            k_h = 1.0
            if heading and dist in oh:
                hd = (oh[dist][0] - sow).days                      # observed heading, days after sowing
                k_h = float(h / dev[0, hd]) if dev[0, hd] > 0 else 1.0
                dev = dev * k_h
            sub = {k: v.repeat(N_MEMBERS, *([1] * (v.dim() - 1))) for k, v in sub.items()}
            devp = dev.repeat(N_MEMBERS, 1) * pt["dev_f"][:, None]
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
            per_group[j] = dict(lai=out["lai_daily"].numpy(), tot=tot, sowing=sow, mature_day=mature, k_h=k_h)
            prior_mean.append(tot["phys_yield"].mean())
        ens[dist] = dict(groups=per_group, cell_group=grp[dist]["cell_group"], prior_mean=np.mean(prior_mean))
        g0 = next(iter(per_group.values()))
        print(f"  {dist}: heading={heading} wide={wide}: k_h {g0['k_h']:.3f}, maturity "
              f"{(sow + pd.Timedelta(days=g0['mature_day'])).date()}", flush=True)
    return ens


def main():
    f = A.wheat_fields()
    obs = A.observations()
    f = f[f["field_id"].isin(obs["field_id"])]
    ratio = A.soil_ratio()
    E = {"B0": ensembles(YEAR, f), "B1": ensembles(YEAR, f, heading=True), "B2": ensembles(YEAR, f, wide=True),
         "B3": ensembles(YEAR, f, heading=True, wide=True)}
    oh = obs_heading()
    rows = []
    for _, x in f.iterrows():
        _, e0 = A.group_of(E["B0"], x)
        if e0 is None:
            continue
        g = obs[obs["field_id"] == x["field_id"]].sort_values("date")
        ok0, day = A.days_of(g, e0)
        meas = (g["source"] == "measured").values
        for cut in A.CUTS:
            c = pd.Timestamp(f"{YEAR}-{cut}")
            before = (g["date"] < c).values
            target = ok0 & meas & ~before                            # B0 target set for every variant
            if not (meas & before).any() or not target.any():
                continue
            rec = {"field_id": x["field_id"], "cut": cut, "date": g["date"].values[target], "obs": g["lai"].values[target]}
            for v, ens in E.items():
                if v in ("B1", "B3") and c <= oh.get(x["district"], (pd.Timestamp.max,))[0]:
                    ens = E["B0"] if v == "B1" else E["B2"]              # heading not yet observed at this cut
                _, e = A.group_of(ens, x)
                ok, dv = A.days_of(g, e)
                u = ok & meas & before
                lp = logprior_for(e, ratio.get((x["field_id"], YEAR)))
                w = weights(e["lai"], dv[u], g["lai"].values[u], g["lai_sd"].values[u], lp)
                rec[v] = w @ e["lai"][:, np.minimum(dv[target], e["lai"].shape[1] - 1)]
            rows.append(pd.DataFrame(rec))
    r = pd.concat(rows, ignore_index=True)
    r.to_csv(os.path.join(OUT, "ageing_check.csv"), index=False)
    rm = lambda v: float(np.sqrt(np.mean(np.square(v))))
    s = pd.DataFrame([dict(cut=c, n=len(g), **{f"rmse_{v}": rm(g[v] - g.obs) for v in E},
                           **{f"bias_{v}": float((g[v] - g.obs).mean()) for v in E}) for c, g in r.groupby("cut")]).round(3)
    print(s.to_string(index=False))
    s = s.set_index("cut")
    best = min(("B1", "B2", "B3"), key=lambda v: s.at["06-01", f"rmse_{v}"])
    ok = (s.at["06-01", f"rmse_{best}"] <= s.at["06-01", "rmse_B0"] - 0.03
          and abs(s.at["06-01", f"bias_{best}"]) < abs(s.at["06-01", "bias_B0"])
          and all(s.at[c, f"rmse_{best}"] <= s.at[c, "rmse_B0"] + 0.02 for c in ("04-01", "05-01")))
    print(f"RESULT: best {best} at 06-01 -> {'ADOPT ' + best if ok else 'keep B0'}")
    json.dump({"variant": best if ok else "B0"}, open(os.path.join(OUT, "ageing_choice.json"), "w"))
    return (best if ok else "B0"), f, obs, E, ratio


if __name__ == "__main__":
    main()
