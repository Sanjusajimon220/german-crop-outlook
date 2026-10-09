"""Phase 4, step 3: field growth model + satellite assimilation (winter wheat, Rur region).

Idea: every field starts from the district model (same weather, soil and crop calendar as its
district - finer field inputs later), but the things that really differ between fields are
unknown. They are drawn at random to make an ensemble of possible crops:
  rue_f      light use efficiency x 0.8-1.2       (variety, crop health)
  lai0_f     leaf area at emergence x 0.4-2.5     (sowing density, establishment)
  rgr_f      early leaf growth speed x 0.7-1.3
  sla_f      leaf area per leaf mass x 0.8-1.2
  sen_f      leaf ageing speed x 0.5-1.6
  cap_f      plant-available soil water x 0.5-1.5 (soil depth / texture within the district)
  n_supply   mineral N 60-240 kg/ha
  dev_f      development speed x 0.94-1.06        (sowing date, variety: about +-1 week at heading)
Each district gets one shared prior ensemble (N members). For each field, every member is scored
by how well its daily green LAI matches the field's satellite LAI (our PROSAIL retrieval) - a
particle filter / importance weighting:
  weight ~ product over dates of Student-t likelihood(obs - member LAI; scale = sqrt(sd_net^2 + 0.7^2))
(0.7 = satellite vs ground LAI error, TR32 check; Student-t so a leftover cloud does not dominate).
Daily model errors are correlated, so the weights are tempered until at least 50 members keep
weight (effective sample size). The field result is the weighted ensemble: LAI curve, yield,
biomass, water use (actual and demand), N factor, with 10/90 % ranges.

Used dates: after sowing, Feb-Nov (Dec/Jan: low sun, frost and snow) and up to the predicted
maturity date of the district (after that, stubble and weeds).

Yield: the physics ensemble gives relative differences; the district level is the district model
(blend of hybrid and boosting, growth_predictions_<crop>.csv):
  field yield = district blend x (field weighted physics yield / district prior mean physics yield)

Development check WITHOUT yields (yields of 2021+ are test data): every other satellite date of a
field is hidden, the rest assimilated, and the hidden LAI predicted; compared with the
no-satellite (open-loop) prediction.

Run `python phase4_field_assim.py check 2021` or `python phase4_field_assim.py run 2021`.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

from phase2_growth import GROWTH, P, Growth, load_growth

CROP, CODE = "winter_wheat", 115
FD = os.path.join(P, "field")
S2 = os.path.join(P, "s2")
N_MEMBERS, MIN_ESS, OBS_ERR, NU = 3000, 50, 0.5, 4.0
# ranges and error chosen on the 2022 hidden-date check (wider upwards, error 0.5), confirmed on 2021
PERTURB = {"rue_f": (0.8, 1.35), "lai0_f": (0.4, 4.0), "rgr_f": (0.7, 1.3), "sla_f": (0.8, 1.5),
           "sen_f": (0.5, 1.6), "cap_f": (0.5, 1.5), "n_supply": (60.0, 240.0), "dev_f": (0.94, 1.06)}
LOG_SCALE = {"lai0_f", "sen_f"}


SHAPE = {"rgr": (0.001, 0.02), "lai0": (0.005, 0.3), "sla": (0.008, 0.035), "r_sen": (0.0, 0.3),
         "c_frost": (0.0, 0.2), "t_frost": (-10.0, 0.0), "g0": (0.0, 0.8)}
SHAPE_FILE = os.path.join(P, f"growth_lai_shape_{CROP}.json")


def bounded(raw, lo, hi):
    return lo + (hi - lo) * torch.sigmoid(raw)


class ShapeGrowth(Growth):
    """District model v7 with only the leaf-area (LAI shape) settings free, for fitting to satellite LAI."""
    def __init__(self, settings):
        super().__init__(settings)
        for p in self.raw.values():
            p.requires_grad_(False)
        start = {"c_frost": 0.02, "t_frost": -3.0, "g0": 0.2}
        self.shape = torch.nn.ParameterDict()
        for k, (lo, hi) in SHAPE.items():
            v = settings.get(k, start.get(k))
            p = min(max((v - lo) / (hi - lo), 1e-3), 1 - 1e-3)
            self.shape[k] = torch.nn.Parameter(torch.tensor(float(np.log(p / (1 - p)))))

    def settings(self):
        s = super().settings()
        s.update({k: bounded(self.shape[k], *lim) for k, lim in SHAPE.items()})
        return s


def field_settings():
    s = json.load(open(os.path.join(P, f"growth_model_{CROP}_v7.json")))
    if os.path.exists(SHAPE_FILE):
        s.update(json.load(open(SHAPE_FILE)))
    return s


class FieldGrowth(Growth):
    """The fitted district model (+ satellite-fitted LAI shape) with per-member multipliers."""
    def __init__(self, settings, pert):
        super().__init__(settings)
        self.extra = {k: torch.tensor(float(settings[k])) for k in ("c_frost", "t_frost", "g0") if k in settings}
        self.pert = pert

    def settings(self):
        s = super().settings()
        s.update(self.extra)
        for k, f in (("rue", "rue_f"), ("lai0", "lai0_f"), ("rgr", "rgr_f"), ("sla", "sla_f"),
                     ("r_sen", "sen_f"), ("r_dry", "sen_f")):
            s[k] = s[k] * self.pert[f]
        return s


def draw(n, seed):
    rng = np.random.default_rng(seed)
    out = {}
    for k, (lo, hi) in PERTURB.items():
        u = (np.arange(n) + rng.random(n)) / n           # Latin hypercube: even spread per setting
        rng.shuffle(u)
        out[k] = np.exp(np.log(lo) + u * np.log(hi / lo)) if k in LOG_SCALE else lo + u * (hi - lo)
    return pd.DataFrame(out)


def district_ensembles(year, districts):
    """Prior ensemble per district: daily LAI (members x days) + season totals per member."""
    m, d = load_growth(CROP)
    settings = field_settings()
    sel = m.index[(m["year"] == year) & m["district"].isin(districts)]
    tau = 0.3
    ens = {}
    for i in sel:
        dist = m.at[i, "district"]
        pert = draw(N_MEMBERS, seed=int(dist) + year)
        n = N_MEMBERS
        sub = {k: v[i:i + 1].repeat(n, *([1] * (v.dim() - 1))) for k, v in d.items()
               if v.dim() and len(v) == len(m)}
        dev = sub["dev"] * torch.tensor(pert["dev_f"].values, dtype=torch.float32)[:, None]
        e, h, mt = sub["th_e"][:, None], sub["th_h"][:, None], sub["th_m"][:, None]
        sub["em"] = torch.sigmoid((dev - e) / tau)
        sub["ph"] = torch.clamp((dev - e) / (h - e), 0, 1)
        sub["post"] = torch.sigmoid((dev - h) / tau)
        sub["gf"] = torch.clamp((dev - h) / (mt - h), 0, 1)
        sub["mat"] = torch.sigmoid((dev - mt) / tau)
        sub["cap"] = sub["cap"] * torch.tensor(pert["cap_f"].values, dtype=torch.float32)[:, None]
        sub["n_supply"] = torch.tensor(pert["n_supply"].values, dtype=torch.float32)
        model = FieldGrowth(settings, {k: torch.tensor(v.values, dtype=torch.float32) for k, v in pert.items()})
        with torch.no_grad():
            _, out = model(sub, GROWTH[CROP]["dry_matter"], record=True)
        sow = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=round(m.at[i, "doy_sowing"]) - 1)
        mature = int((d["dev"][i] >= d["th_m"][i]).float().argmax())
        tot = pd.concat([pert, pd.DataFrame({k: v.numpy() for k, v in out.items() if k != "lai_daily"})], axis=1)
        ens[dist] = dict(lai=out["lai_daily"].numpy(), tot=tot, sowing=sow, mature_day=mature)
        print(f"  {dist}: sown {sow.date()}, maturity day {mature} ({(sow + pd.Timedelta(days=mature)).date()}), "
              f"prior LAI max median {np.median(tot['phys_lai_max']):.2f}, "
              f"prior yield {tot['phys_yield'].median():.2f} t/ha", flush=True)
    return ens


def observations(year):
    r = pd.read_csv(os.path.join(S2, f"retrieval_{CODE}_{year}.csv"))
    r["date"] = pd.to_datetime(r["date"]).dt.tz_localize(None).dt.normalize()
    r = r[~r["date"].dt.month.isin([12, 1])]
    fd = pd.read_csv(os.path.join(FD, "fields_district.csv"), dtype={"district": str})
    fd = fd[(fd["CODE"] == CODE) & (fd["VALIDFROM"] == year)][["ID", "district"]]
    return r.merge(fd.dropna(subset=["district"]).rename(columns={"ID": "field_id"}), on="field_id")


def weights(lai, days, obs, sd):
    """Importance weights of the members for one field (tempered to keep >= MIN_ESS members)."""
    if len(days) == 0:
        return np.full(lai.shape[0], 1.0 / lai.shape[0])
    scale = np.sqrt(sd ** 2 + OBS_ERR ** 2)
    z = (obs[None, :] - lai[:, days]) / scale[None, :]
    ll = (-(NU + 1) / 2 * np.log1p(z ** 2 / NU)).sum(1)
    temp = 1.0
    for _ in range(30):
        w = np.exp(temp * (ll - ll.max()))
        w /= w.sum()
        if 1.0 / (w ** 2).sum() >= MIN_ESS:
            return w
        temp *= 0.7
    return w


def wquant(x, w, q):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return np.interp(q, c - w[o] / 2, x[o])


def field_days(g, e):
    day = (g["date"] - e["sowing"]).dt.days.values
    ok = (day >= 0) & (day <= e["mature_day"]) & (day < e["lai"].shape[1])
    return day[ok], g["lai"].values[ok], g["lai_sd"].values[ok]


def check(year, ahead=None):
    """Hide every other satellite date, assimilate the rest, predict the hidden ones. With
    ahead="MM-DD": assimilate only dates before that day and predict all later dates (baseline:
    the last satellite value carried forward)."""
    obs = observations(year)
    ens = district_ensembles(year, sorted(obs["district"].unique()))
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
        w = weights(e["lai"], day[use], y[use], sd[use])
        prior = e["lai"][:, day[hide]].mean(0)
        interp = np.interp(day[hide], day[use], y[use])      # baseline: line between neighbouring dates
                                                             # (beyond the last date: last value)
        post = w @ e["lai"][:, day[hide]]
        lo = np.array([wquant(e["lai"][:, k], w, 0.05) for k in day[hide]])
        hi = np.array([wquant(e["lai"][:, k], w, 0.95) for k in day[hide]])
        tot = np.sqrt((hi - lo) ** 2 / 3.29 ** 2 + sd[hide] ** 2 + OBS_ERR ** 2) * 1.645  # + obs error
        for k in range(hide.sum()):
            rows.append(dict(field_id=fid, district=dist, day=day[hide][k], obs=y[hide][k], prior=prior[k],
                             post=post[k], inside90=abs(y[hide][k] - post[k]) <= tot[k],
                             interp=interp[k], date=e["sowing"] + pd.Timedelta(days=int(day[hide][k]))))
    c = pd.DataFrame(rows)
    c["month"] = pd.to_datetime(c["date"]).dt.strftime("%Y-%m")
    c.to_csv(os.path.join(FD, f"check_{'ahead' if ahead else 'hidden_dates'}_{year}.csv"), index=False)
    rm = lambda a: np.sqrt(((c[a] - c["obs"]) ** 2).mean())
    print(f"\nhidden satellite dates: {len(c)} ({c['field_id'].nunique()} fields)")
    print(f"  LAI RMSE without satellite (district model) {rm('prior'):.2f}, with satellite {rm('post'):.2f}; "
          f"90 % range holds the hidden value {100 * c['inside90'].mean():.0f} %; "
          f"baseline (line between neighbouring satellite dates) {rm('interp'):.2f}")
    t = c.groupby("month").apply(lambda g: pd.Series({
        "n": len(g), "obs": g["obs"].median(), "rmse_prior": np.sqrt(((g["prior"] - g["obs"]) ** 2).mean()),
        "rmse_post": np.sqrt(((g["post"] - g["obs"]) ** 2).mean()),
        "rmse_interp": np.sqrt(((g["interp"] - g["obs"]) ** 2).mean()), "bias_prior": (g["prior"] - g["obs"]).mean(),
        "bias_post": (g["post"] - g["obs"]).mean(), "inside90": g["inside90"].mean()}), include_groups=False)
    print(t.round(2).to_string())


def fit_shape(year):
    """Fit the LAI-shape settings so each district's model LAI follows the median satellite LAI
    of its fields (dates with >= 30 fields). Uses satellite LAI only - no yields."""
    m, d = load_growth(CROP)
    obs = observations(year)
    sel = [i for i in m.index[(m["year"] == year)] if m.at[i, "district"] in set(obs["district"])]
    sub = {k: v[sel] for k, v in d.items() if v.dim() and len(v) == len(m)}
    rows, days, target, wt = [], [], [], []
    for j, i in enumerate(sel):
        sow = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=round(m.at[i, "doy_sowing"]) - 1)
        mature = int((d["dev"][i] >= d["th_m"][i]).float().argmax())
        g = obs[obs["district"] == m.at[i, "district"]].groupby("date")["lai"].agg(["median", "size"])
        g = g[g["size"] >= 30]
        dd = (g.index - sow).days
        ok = (dd >= 0) & (dd <= mature + 14)
        rows += [j] * ok.sum(); days += list(dd[ok]); target += list(g["median"][ok]); wt += list(np.sqrt(g["size"][ok]))
    rows, days = torch.tensor(rows), torch.tensor(days)
    target, wt = torch.tensor(target, dtype=torch.float32), torch.tensor(wt, dtype=torch.float32)
    model = ShapeGrowth(json.load(open(os.path.join(P, f"growth_model_{CROP}_v7.json"))))
    opt = torch.optim.Adam(model.shape.parameters(), lr=0.05)
    for it in range(150):
        opt.zero_grad()
        _, out = model(sub, GROWTH[CROP]["dry_matter"], record=True)
        pred = out["lai_daily"][rows, days]
        loss = (wt * torch.nn.functional.huber_loss(pred, target, reduction="none", delta=0.5)).sum() / wt.sum()
        loss.backward()
        opt.step()
        if it % 25 == 0 or it == 149:
            rmse = torch.sqrt(((pred - target) ** 2 * wt).sum() / wt.sum()).item()
            print(f"  iteration {it:3d}: district median LAI RMSE {rmse:.2f}", flush=True)
    fitted = {k: round(float(bounded(model.shape[k], *lim)), 5) for k, lim in SHAPE.items()}
    json.dump(fitted, open(SHAPE_FILE, "w"), indent=1)
    print("fitted LAI shape (season", year, "):", fitted)


def run(year):
    obs = observations(year)
    ens = district_ensembles(year, sorted(obs["district"].unique()))
    pred = pd.read_csv(os.path.join(P, f"growth_predictions_{CROP}.csv"), dtype={"district": str})
    blend = pred[pred["year"] == year].set_index("district")["blend"]
    rows = []
    for (fid, dist), g in obs.groupby(["field_id", "district"]):
        e = ens.get(dist)
        if e is None or dist not in blend.index:
            continue
        day, y, sd = field_days(g.sort_values("date"), e)
        w = weights(e["lai"], day, y, sd)
        t = e["tot"]
        rel = t["phys_yield"].values / t["phys_yield"].mean()
        r = dict(field_id=fid, district=dist, n_obs=len(day), ess=1 / (w ** 2).sum(),
                 area_ha=g["area_ha"].iloc[0], inner_ha=g["inner_ha"].iloc[0])
        r["yield_t_ha"] = blend[dist] * (w @ rel)
        r["yield_p10"], r["yield_p90"] = (blend[dist] * wquant(rel, w, q) for q in (0.1, 0.9))
        for k in ("phys_lai_max", "phys_biomass_t_ha", "phys_eta_mm", "phys_etp_mm", "phys_n_factor",
                  "n_supply", "cap_f", "dev_f", "rue_f"):
            r[k] = w @ t[k].values
        rows.append(r)
    f = pd.DataFrame(rows)
    f.to_csv(os.path.join(FD, f"field_results_{CROP}_{year}.csv"), index=False)
    print(f"\n{len(f)} fields; median observations used {f['n_obs'].median():.0f}, median ESS {f['ess'].median():.0f}")
    print(f[["yield_t_ha", "yield_p10", "yield_p90", "phys_lai_max", "phys_eta_mm", "phys_etp_mm",
             "n_supply", "cap_f"]].describe(percentiles=[0.1, 0.5, 0.9]).round(2).to_string())


if __name__ == "__main__":
    t0 = time.time()
    {"check": check, "run": run, "shape": fit_shape}[sys.argv[1]](int(sys.argv[2]), *sys.argv[3:])
    print(f"({time.time() - t0:.0f} s)")
