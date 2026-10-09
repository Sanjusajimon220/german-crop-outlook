"""Phase 4, step 3c: field model v3 = v2 + each field's own soil water (BK50 soil map NRW).

v1/v2 (phase4_field_assim.py, frozen) treat soil water as unknown: every member draws a soil
water multiplier cap_f between 0.5 and 1.5 of the district soil. v3 knows the field's soil from
BK50 (data/processed/field/fields_soil_bk50.csv): its plant-available water (nfk_mm) relative to
the median of the wheat fields in its district,
  ratio = field nfk / district median nfk   (wheat fields 2021-2026; 10-90 %: about 0.6-1.4).
Members whose cap_f is close to that ratio are made more likely before the satellite is used
(prior weight ~ normal on log scale, sd 0.15 = map uncertainty at 1:50,000):
  weight ~ soil prior x satellite likelihood (tempered as in v1, ESS >= 50).
Fields without BK50 value (outside the map) keep the v1 prior. Yields rescaled as in v2 (district
mean = district model). Run `python phase4_field_assim_v3.py run 2021` or `check 2021 [MM-DD]`.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

import phase4_field_assim as v1
from phase4_field_assim import CROP, FD, MIN_ESS, NU, OBS_ERR, P, district_ensembles, field_days, observations, wquant

SOIL_SD = 0.15


def soil_ratio():
    s = pd.read_csv(os.path.join(FD, "fields_soil_bk50.csv"))
    s = s[(s["CODE"] == v1.CODE) & (s["share_mapped"] >= 0.5) & (s["nfk_mm"] > 0)]
    d = pd.read_csv(os.path.join(FD, "fields_district.csv"), dtype={"district": str})[["ID", "VALIDFROM", "district"]]
    s = s.merge(d, on=["ID", "VALIDFROM"])
    s["ratio"] = s["nfk_mm"] / s.groupby("district")["nfk_mm"].transform("median")
    return s.set_index(["ID", "VALIDFROM"])["ratio"]


def weights(lai, days, obs, sd, logprior):
    ll = np.zeros(lai.shape[0])
    if len(days):
        scale = np.sqrt(sd ** 2 + OBS_ERR ** 2)
        z = (obs[None, :] - lai[:, days]) / scale[None, :]
        ll = (-(NU + 1) / 2 * np.log1p(z ** 2 / NU)).sum(1)
    temp = 1.0
    for _ in range(30):
        a = logprior + temp * ll
        w = np.exp(a - a.max())
        w /= w.sum()
        if 1.0 / (w ** 2).sum() >= MIN_ESS:
            return w
        temp *= 0.7
    return w


def logprior_for(e, ratio):
    if ratio is None or not np.isfinite(ratio):
        return np.zeros(len(e["tot"]))
    return -0.5 * ((np.log(e["tot"]["cap_f"].values) - np.log(np.clip(ratio, 0.5, 1.5))) / SOIL_SD) ** 2


def check(year, ahead=None):
    obs = observations(year)
    ens = district_ensembles(year, sorted(obs["district"].unique()))
    ratio = soil_ratio()
    rows = []
    for (fid, dist), g in obs.groupby(["field_id", "district"]):
        e = ens.get(dist)
        if e is None:
            continue
        day, y, sd = field_days(g.sort_values("date"), e)
        if len(day) < 4:
            continue
        if ahead:
            cut = (pd.Timestamp(f"{year}-{ahead}") - e["sowing"]).days
            use, hide = day < cut, day >= cut
            if use.sum() < 3 or hide.sum() == 0:
                continue
        else:
            use, hide = np.arange(len(day)) % 2 == 0, np.arange(len(day)) % 2 == 1
        lp = logprior_for(e, ratio.get((fid, year)))
        for name, prior in (("v1", np.zeros_like(lp)), ("v3", lp)):
            w = weights(e["lai"], day[use], y[use], sd[use], prior)
            post = w @ e["lai"][:, day[hide]]
            rows += [dict(version=name, err=p - o) for p, o in zip(post, y[hide])]
    c = pd.DataFrame(rows)
    print(c.groupby("version")["err"].apply(lambda e: f"LAI RMSE {np.sqrt((e ** 2).mean()):.3f}, n {len(e)}").to_string())


def run(year):
    obs = observations(year)
    ens = district_ensembles(year, sorted(obs["district"].unique()))
    ratio = soil_ratio()
    pred = pd.read_csv(os.path.join(P, f"growth_predictions_{CROP}.csv"), dtype={"district": str})
    blend = pred[pred["year"] == year].set_index("district")["blend"]
    rows = []
    for (fid, dist), g in obs.groupby(["field_id", "district"]):
        e = ens.get(dist)
        if e is None or dist not in blend.index:
            continue
        day, y, sd = field_days(g.sort_values("date"), e)
        r_soil = ratio.get((fid, year))
        w = weights(e["lai"], day, y, sd, logprior_for(e, r_soil))
        t = e["tot"]
        rel = t["phys_yield"].values / t["phys_yield"].mean()
        r = dict(field_id=fid, district=dist, n_obs=len(day), ess=1 / (w ** 2).sum(), soil_ratio=r_soil,
                 area_ha=g["area_ha"].iloc[0], rel=w @ rel, rel_p10=wquant(rel, w, 0.1), rel_p90=wquant(rel, w, 0.9))
        for k in ("phys_lai_max", "phys_eta_mm", "phys_etp_mm", "cap_f", "n_supply"):
            r[k] = w @ t[k].values
        rows.append(r)
    f = pd.DataFrame(rows)
    mean_rel = f.groupby("district").apply(lambda g: np.average(g["rel"], weights=g["area_ha"]), include_groups=False)
    k = f["district"].map(blend) / f["district"].map(mean_rel)
    f["v3_yield_t_ha"], f["v3_yield_p10"], f["v3_yield_p90"] = f["rel"] * k, f["rel_p10"] * k, f["rel_p90"] * k
    f.to_csv(os.path.join(FD, f"field_results_v3_{CROP}_{year}.csv"), index=False)
    v2 = pd.read_csv(os.path.join(FD, f"field_results_v2_{CROP}_{year}.csv"))[["field_id", "v2_yield_t_ha"]]
    m = f.merge(v2, on="field_id")
    print(f"{year}: {len(f)} fields, soil known for {100 * f['soil_ratio'].notna().mean():.0f} %; "
          f"cap_f follows soil: r {m['cap_f'].corr(m['soil_ratio']):.2f}; v3 vs v2 field yields r "
          f"{m['v3_yield_t_ha'].corr(m['v2_yield_t_ha']):.3f}, mean |diff| "
          f"{(m['v3_yield_t_ha'] - m['v2_yield_t_ha']).abs().mean():.2f} t/ha; yield vs soil ratio r "
          f"{m['v3_yield_t_ha'].corr(m['soil_ratio']):.2f} (v2: {m['v2_yield_t_ha'].corr(m['soil_ratio']):.2f})")


if __name__ == "__main__":
    t0 = time.time()
    {"check": check, "run": run}[sys.argv[1]](int(sys.argv[2]), *sys.argv[3:])
    print(f"({time.time() - t0:.0f} s)")
