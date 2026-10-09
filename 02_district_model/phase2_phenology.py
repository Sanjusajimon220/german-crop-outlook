"""Phase 2, step 1: the crop calendar model (development stages), written in PyTorch.

From sowing onwards the crop accumulates "development units" every day:

    before emergence:  rate = f_T,pre x f_W (moisture: rain since sowing)
    before heading:    rate = f_T,pre x f_V(vernalisation) x f_P(day length)
    after heading:     rate = f_T,post

  f_T  temperature response (Wang & Engel beta function): 0 below Tmin and above Tmax, 1 at Topt,
       averaged over 6 temperatures across the day between the daily minimum and maximum
  f_V  vernalisation (winter cereals): cold days (most effective 3-10 C) are counted; development
       before flowering is slowed until enough cold days have passed (VSAT)
  f_P  day length (cereals): exp(-omega x (DLopt - day length)) when days are shorter than DLopt
  f_W  emergence waits for moisture: sigmoid((rain since sowing - R0) / 5 mm), R0 fitted
  Variety trend (version 3): breeding changes how much development a variety needs. The stage
  thresholds are scaled by exp(trend x (year - 2000) / 10), with one trend for the stages up to
  heading/flowering and one for the stages after; trends are fitted like every other setting.
  The phases are separated by smooth switches on the development reached so far, and the
  temperature response has separate settings before and after heading/flowering (version 2).

A stage is reached when the accumulated units pass that stage's threshold. To make this
differentiable, the predicted day is the soft count of days still below the threshold
(a sigmoid instead of a hard step). All settings (cardinal temperatures, VSAT, DLopt, omega,
thresholds) are fitted by gradient descent on the training district-years only.

Compared with: district climatology (each district's median date in the training years; the
simplest forecast) and WOFOST after our calibration (flowering and maturity).

Run `python phase2_phenology.py`. Writes data/processed/phenology_model_<crop>.json (settings)
and data/processed/phenology_predictions_<crop>.csv.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

from wofost_runner import district_latitudes

P = os.path.join("data", "processed")
TAU = 0.3        # softness of the stage crossing, in development units
# crop: (stages predicted in order, stage that ends vernalisation/day-length effects, uses them?,
#        horizon in days after sowing)
CROPS = {
    "winter_wheat": (["emergence", "stem_extension", "heading", "milk_ripe", "yellow_ripe"],
                     "heading", True, 330),
    "winter_barley": (["emergence", "stem_extension", "heading", "yellow_ripe"], "heading", True, 330),
    "silage_maize": (["emergence", "tassel", "flowering", "milk_ripe", "dough_ripe"], "flowering",
                     False, 200),
    "grain_maize": (["emergence", "tassel", "flowering", "milk_ripe", "dough_ripe"], "flowering",
                    False, 200),
    "potato": (["emergence", "canopy_closed", "flowering"], "flowering", False, 160),
}
torch.manual_seed(0)
WEATHER = None       # forecast mode: a daily weather dict to use instead of weather_daily.npz
# weather version: v7 (default) or v8 (SARAH-3 sunlight from 2021; set environment VISTA_WEATHER=v8)
REFIT = int(os.environ.get("VISTA_REFIT", "0"))      # v10: refit year (0 = none)
FORWARD = int(os.environ.get("VISTA_FORWARD", "0"))  # v11: in-season forecasts of training years after this cut
WEATHER_FILE = {"v8": "weather_daily_v8.npz", "v8_2026": "weather_daily_v8_2026.npz"}.get(
    os.environ.get("VISTA_WEATHER"), "weather_daily.npz")
EXTRA = {}           # crop -> DataFrame of extra rows without yields (2026 check), appended in load()


def weather():
    return WEATHER if WEATHER is not None else np.load(os.path.join(P, WEATHER_FILE))


def load(crop):
    """Rows with sowing date and daily weather from sowing; observed days from sowing per stage."""
    stages, _, _, horizon = CROPS[crop]
    m = pd.read_csv(os.path.join(P, f"master_{crop}.csv"), dtype={"district": str})
    m = m.merge(pd.read_csv(os.path.join(P, "split.csv"), dtype={"district": str}), on=["district", "year"])
    if FORWARD:    # v11: training rows after the cut become forecast targets ('fwd')
        m.loc[(m["role"] == "train") & (m["year"] > FORWARD), "role"] = "fwd"
    if REFIT:      # v10 refit schedule: test-year rows up to the refit year become training rows
        late = (m["role"] == "test_years") & (m["year"] <= REFIT)
        m.loc[late, "role"] = "train"
        m.loc[late, "fold"] = m.loc[late, "year"] % 5          # year blocks for the cross-validation
    if crop in EXTRA:
        m = pd.concat([m, EXTRA[crop]], ignore_index=True)
    m["doy_sowing"] = (m["doy_sowing"].fillna(m.groupby("district")["doy_sowing"].transform("median"))
                       .fillna(m["doy_sowing"].median()))
    w = weather()
    dates, col = w["dates"], {c: i for i, c in enumerate(w["codes"])}
    start = (pd.to_datetime(m["year"].astype(str) + "-01-01")
             + pd.to_timedelta(m["doy_sowing"].round() - 1, unit="D")).values.astype("datetime64[D]")
    first = ((start - dates[0]).astype(int))
    ok = (first >= 0) & (first + horizon <= len(dates))
    m, first = m[ok].reset_index(drop=True), first[ok]
    idx = first[:, None] + np.arange(horizon)[None, :]
    cols = m["district"].map(col).values[:, None]
    tmin, tmax = w["tasmin"][idx, cols], w["tasmax"][idx, cols]
    rain = np.cumsum(w["pr"][idx, cols], 1)
    lat = m["district"].map(district_latitudes()).values
    doy = ((dates[idx] - dates[idx].astype("datetime64[Y]")).astype(int) + 1)
    decl = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    daylen = 24 / np.pi * np.arccos(np.clip(-np.tan(np.radians(lat))[:, None] * np.tan(decl), -1, 1))
    obs = np.stack([m[f"doy_{s}"] - m["doy_sowing"] for s in stages], 1).astype(np.float32)
    weight = np.stack([np.where(m.get(f"src_{s}") == "district", 1.0, 0.5) for s in stages], 1)
    t = lambda a: torch.tensor(np.asarray(a, np.float32))
    return m, {"tmin": t(tmin), "tmax": t(tmax), "daylen": t(daylen), "rain": t(rain),
               "decade": t((m["year"].values - 2000) / 10.0), "obs": t(obs), "weight": t(weight)}


def bounded(raw, lo, hi):
    return lo + (hi - lo) * torch.sigmoid(raw)


class Calendar(torch.nn.Module):
    LIMITS = {"tmin": (-5.0, 10.0), "topt": (15.0, 32.0), "tmax": (33.0, 45.0),
              "tmin_post": (-2.0, 15.0), "topt_post": (15.0, 35.0), "tmax_post": (33.0, 45.0),
              "vsat": (5.0, 80.0), "vbase": (0.0, 0.5), "dlopt": (12.0, 20.0),
              "omega": (0.0, 0.5), "rain0": (0.0, 30.0),
              "trend_pre": (-0.15, 0.15), "trend_post": (-0.15, 0.15)}     # per decade

    def __init__(self, n_stages, gate_stage, vern_photo, start=None):
        super().__init__()
        self.gate, self.vp = gate_stage, vern_photo
        self.raw = torch.nn.ParameterDict({k: torch.nn.Parameter(torch.tensor(0.0)) for k in self.LIMITS})
        self.raw["vbase"].data.fill_(-1.0)
        self.raw["omega"].data.fill_(-1.0)
        self.raw["rain0"].data.fill_(-2.0)
        self.r_steps = torch.nn.Parameter(torch.full((n_stages,), 3.0))
        if start:                                 # continue from earlier fitted settings
            for k, v in start.items():
                if k in self.LIMITS:
                    lo, hi = self.LIMITS[k]
                    p = min(max((v - lo) / (hi - lo), 1e-3), 1 - 1e-3)
                    self.raw[k].data.fill_(float(np.log(p / (1 - p))))
            if "thresholds" in start and len(start["thresholds"]) == n_stages:
                steps = np.diff(np.concatenate([[0.0], start["thresholds"]])) / 10.0
                self.r_steps.data = torch.tensor(np.log(np.expm1(np.maximum(steps, 1e-3))),
                                                 dtype=torch.float32)

    def settings(self):
        s = {k: bounded(self.raw[k], *lim) for k, lim in self.LIMITS.items()}
        s["thresholds"] = torch.cumsum(torch.nn.functional.softplus(self.r_steps) * 10.0, 0)
        return s

    @staticmethod
    def temperature_response(tmin, tmax, lo, opt, hi):
        """Wang-Engel beta function averaged over 6 temperatures between Tmin and Tmax."""
        k = torch.cos(torch.pi * (torch.arange(6) + 0.5) / 6)
        temp = ((tmin + tmax) / 2)[..., None] + ((tmax - tmin) / 2)[..., None] * k
        alpha = np.log(2) / torch.log((hi - lo) / (opt - lo))
        x = torch.clamp(temp - lo, min=1e-3)
        span = opt - lo
        f = (2 * x ** alpha * span ** alpha - x ** (2 * alpha)) / span ** (2 * alpha)
        inside = (temp > lo) & (temp < hi)
        return torch.where(inside, torch.clamp(f, min=0.0), torch.zeros_like(f)).mean(-1)

    def forward(self, d, return_dev=False):
        s = self.settings()
        f_pre = self.temperature_response(d["tmin"], d["tmax"], s["tmin"], s["topt"], s["tmax"])
        f_post = self.temperature_response(d["tmin"], d["tmax"], s["tmin_post"], s["topt_post"],
                                           s["tmax_post"])
        if self.vp:
            tmean = (d["tmin"] + d["tmax"]) / 2
            eff = torch.clamp(torch.minimum((tmean + 4) / 7, (17 - tmean) / 7), 0, 1)
            vd = torch.cumsum(eff, 1)
            vbase = s["vbase"] * s["vsat"]
            f_v = torch.clamp((vd - vbase) / (s["vsat"] - vbase), 0, 1)
            f_p = torch.exp(-s["omega"] * torch.relu(s["dlopt"] - d["daylen"]))
            vp = f_v * f_p
        else:
            vp = torch.ones_like(f_pre)
        moist = torch.sigmoid((d["rain"] - s["rain0"]) / 5.0)
        # variety trend: stages up to the gate stage scale with trend_pre, later ones with trend_post
        n_st = s["thresholds"].shape[0]
        early = (torch.arange(n_st) <= self.gate).float()
        rate = early * s["trend_pre"] + (1 - early) * s["trend_post"]
        th = s["thresholds"][None, :] * torch.exp(d["decade"][:, None] * rate[None, :])   # [rows, stages]
        emerge, gate = th[:, 0], th[:, self.gate]
        dev, total = [], torch.zeros(f_pre.shape[0])
        for day in range(f_pre.shape[1]):        # phase switches depend on development so far
            e = torch.sigmoid((emerge - total) / TAU)
            g = torch.sigmoid((gate - total) / TAU)
            pre = e * moist[:, day] + (1 - e) * vp[:, day]
            total = total + g * f_pre[:, day] * pre + (1 - g) * f_post[:, day]
            dev.append(total)
        dev = torch.stack(dev, 1)                                             # [rows, days]
        if return_dev:                       # daily development curve + stage thresholds (growth model)
            return dev, th
        below = torch.sigmoid((th[:, :, None] - dev[:, None, :]) / TAU)
        return below.sum(-1)                                                  # days to each stage


def fit(model, d, rows, iterations=300, lr=0.05):
    sub = {k: v[rows] for k, v in d.items()}
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iterations)
    mask = torch.isfinite(sub["obs"])
    target = torch.nan_to_num(sub["obs"])
    for it in range(iterations):
        opt.zero_grad()
        pred = model(sub)
        err = torch.nn.functional.huber_loss(pred, target, reduction="none", delta=7.0)
        loss = (err * sub["weight"] * mask).sum() / (sub["weight"] * mask).sum()
        loss.backward()
        opt.step()
        sched.step()
        if it % 50 == 0 or it == iterations - 1:
            rmse = torch.sqrt((((pred - target) ** 2) * mask).sum() / mask.sum()).item()
            print(f"  iteration {it:3d}: RMSE {rmse:5.2f} days", flush=True)
    return model


def wofost_errors(crop, m):
    """WOFOST flowering (vs heading + 5 days / flowering) and maturity (vs yellow ripeness) errors."""
    path = os.path.join(P, f"wofost_{crop}.csv")
    if not os.path.exists(path):
        return None
    w = pd.read_csv(path, dtype={"district": str}).drop_duplicates(["district", "year"])
    x = m.merge(w, on=["district", "year"], how="left")
    jan1 = pd.to_datetime(x["year"].astype(str) + "-01-01")
    out = {}
    pairs = {"winter_wheat": (("heading", 5, "sim_anthesis"), ("yellow_ripe", 0, "sim_maturity")),
             "winter_barley": (("heading", 5, "sim_anthesis"), ("yellow_ripe", 0, "sim_maturity")),
             "silage_maize": (("flowering", 0, "sim_anthesis"),),
             "grain_maize": (("flowering", 0, "sim_anthesis"),), "potato": ()}[crop]
    for stage, offset, col in pairs:
        sim = (pd.to_datetime(x[col]) - jan1).dt.days + 1
        out[stage] = (sim - (x[f"doy_{stage}"] + offset)).values
    return out


def main():
    crops = sys.argv[1:] or list(CROPS)
    for crop in crops:
        stages, gate, vp, _ = CROPS[crop]
        m, d = load(crop)
        train = np.flatnonzero((m["role"] == "train").values)
        if len(train) > 5000:      # 13 settings need far fewer seasons; keeps a fit under 30 minutes
            train = np.sort(np.random.default_rng(0).choice(train, 5000, replace=False))
        print(f"\n{crop}: {len(m)} district-years, {len(train)} for training", flush=True)
        path = os.path.join(P, f"phenology_model_{crop}.json")
        start = json.load(open(path)) if os.path.exists(path) else None
        model = Calendar(len(stages), stages.index(gate), vp, start)
        # from scratch: more iterations and larger steps; continuing: fine-tuning
        model = fit(model, d, train, *((300, 0.05) if start else (500, 0.08)))
        with torch.no_grad():
            pred = model(d).numpy()
            s = {k: (v.tolist() if v.dim() else round(v.item(), 3)) for k, v in model.settings().items()}
        json.dump(s, open(os.path.join(P, f"phenology_model_{crop}.json"), "w"), indent=1)
        obs = d["obs"].numpy()
        # climatology: each district's median days-from-sowing in training years
        clim = np.full_like(obs, np.nan)
        tr = m["role"] == "train"
        for k, st in enumerate(stages):
            days = pd.Series(obs[:, k])
            by_d = days[tr.values].groupby(m.loc[tr, "district"].values).median()
            clim[:, k] = m["district"].map(by_d).fillna(days[tr.values].median()).values
        wof = wofost_errors(crop, m)
        out = m[["district", "year", "role"]].copy()
        for k, st in enumerate(stages):
            out[f"pred_{st}"] = pred[:, k] + m["doy_sowing"].values
            out[f"obs_{st}"] = m[f"doy_{st}"]
        out.to_csv(os.path.join(P, f"phenology_predictions_{crop}.csv"), index=False)

        print(f"  fitted: before heading Tmin/Topt/Tmax {s['tmin']}/{s['topt']}/{s['tmax']} C; after "
              f"{s['tmin_post']}/{s['topt_post']}/{s['tmax_post']} C; emergence needs {s['rain0']} mm rain; "
              f"variety trend per decade {100 * s['trend_pre']:+.1f}% (to heading/flowering), "
              f"{100 * s['trend_post']:+.1f}% (after)" +
              (f"; VSAT {s['vsat']} days, DLopt {s['dlopt']} h, omega {s['omega']}" if vp else ""))
        print(f"  RMSE in days: {'stage':15s}" + "".join(f"{r:>22s}" for r in
              ("training", "test: new years", "test: new regions", "test: both")))
        for k, st in enumerate(stages):
            line = f"                {st:15s}"
            for role in ("train", "test_years", "test_regions", "test_both"):
                sel = (m["role"] == role).values & np.isfinite(obs[:, k])
                e_model = np.sqrt(np.mean((pred[sel, k] - obs[sel, k]) ** 2))
                e_clim = np.sqrt(np.mean((clim[sel, k] - obs[sel, k]) ** 2))
                extra = ""
                if wof and st in wof:
                    we = wof[st][sel]
                    we = we[np.isfinite(we)]
                    extra = f"/W{np.sqrt(np.mean(we ** 2)):4.1f}" if len(we) > 30 else ""
                line += f"   {e_model:5.1f} (clim {e_clim:4.1f}{extra})"
            print(line)
    print("\n(model RMSE; in brackets the district climatology and, where available, WOFOST = W)")


if __name__ == "__main__":
    main()
