"""Phase 2, step 2: crop growth, water use and yield (hybrid physics + learned correction).

Every district-year is simulated day by day from sowing, in PyTorch, so that all uncertain
settings can be fitted together to the observed yields of the training years:

  development   taken from the fitted crop calendar (phase2_phenology.py): emergence, heading,
                grain filling progress and maturity for that district and year
  light         absorbed PAR = 0.5 x global radiation x (1 - exp(-k x green LAI))
  biomass       growth = RUE x absorbed PAR x temperature factor x water factor
  partitioning  before heading: a share to leaves (falling to zero at heading), the rest to stems;
                after heading: a share of new growth plus stem reserves go to the grain
  leaves        young canopies (LAI < 1) expand with warmth (relative growth per degree-day, as in
                WOFOST); later green LAI grows with new leaf mass (specific leaf area) and dies off after heading,
                faster when the crop is short of water
  soil water    10 layers of 20 cm (DWD AMBAV field capacity / wilting point, 0-2 m); rain fills
                the layers from the top; roots grow down until heading; transpiration demand =
                ET0 x Kc x absorbed fraction; it is cut when the root zone runs dry (water factor)
  heat          hot days (Tmax above a threshold) around flowering reduce grain set
  nitrogen      (version 6) mineral N applied to this crop per ha (phase0_nitrogen.py: crop
                fertilisation maps 1979-2019 per state, then national change) scales yield: exp(-c_n x softplus((N_ref - N) / 10))
  wet season    (version 2) rainy days (> 2 mm) from late stem extension to mid grain filling cut
                yield: disease pressure and waterlogging, which drought-only models miss (2023, 2024)
  technology    yield x exp(c x (1 - exp(-(year - 1979) / tau))): breeding and management gains that
                level off (the straight-line trend overshoots after 2000); for years after the
                training data the level is held at the last training year (chosen by forward
                validation: `python phase2_growth.py <crop> forward`)

Then a gradient-boosting model learns what the physics misses (the log ratio of observed to
simulated yield) from the simulated quantities and the monthly weather, trained on training
years only and tuned by leaving out one year block at a time. Scored on the same test rows as
every Phase 1 benchmark (wofost_predictions_<crop>.csv).

Run `python phase2_growth.py winter_wheat [refit]`. Writes growth_model_<crop>.json,
growth_predictions_<crop>.csv and growth_comparison_<crop>.csv in data/processed.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

from phase1_baselines import features
import phase2_phenology as phen
from phase2_phenology import CROPS, P, TAU, Calendar, bounded, load

LAYERS, LAYER_CM = 10, 20
# version 8 (VISTA_WEATHER=v8): SARAH sunlight from 2021, potato tubers from canopy closure, own files
V8 = os.environ.get("VISTA_WEATHER", "").startswith("v8")       # v8 weather (also the 2026 extension)
# MODIS greenness features (VISTA_MODIS=1): district NDVI features 2000+ in the correction and boosting
MODIS = os.environ.get("VISTA_MODIS") == "1"
VTAG = "_v9" if (V8 and MODIS) else "_v8" if V8 else ""      # v9 = v8 + MODIS
if int(os.environ.get("VISTA_REFIT", "0")):                     # v10 barley refit schedule
    VTAG += "_r" + os.environ["VISTA_REFIT"]
if int(os.environ.get("VISTA_FORWARD", "0")):                   # v11 training-year in-season forecasts
    VTAG += "_fwd" + os.environ["VISTA_FORWARD"]
GROWTH = {"winter_wheat": {"emergence": "emergence", "heading": "heading", "maturity": "yellow_ripe",
                           "dry_matter": 0.86},
          "winter_barley": {"emergence": "emergence", "heading": "heading", "maturity": "yellow_ripe",
                            "dry_matter": 0.86},
          # maize: flowering plays the role of heading; grain filling until dough ripeness.
          # Grain maize yield at 14 % moisture; silage maize yield = whole-plant fresh mass at ~33 %
          # dry matter (harvest "biomass": above-ground biomass instead of grain)
          "grain_maize": {"emergence": "emergence", "heading": "flowering", "maturity": "dough_ripe",
                          "dry_matter": 0.86, "harvest": "grain"},
          "silage_maize": {"emergence": "emergence", "heading": "flowering", "maturity": "dough_ripe",
                           "dry_matter": 0.33, "harvest": "biomass"},
          # potato: tubers fill from flowering (tuber bulking) until the end of the crop; the calendar
          # stops at flowering, so the end is the district's typical harvest day (training years,
          # DWD observations) on the development clock. Yield = fresh tubers at ~22 % dry matter
          "potato": {"emergence": "emergence", "heading": "flowering", "maturity": "harvest_typical",
                     "dry_matter": 0.22, "harvest": "grain"}}
# crop-specific setting ranges / starting values (others as Growth.LIMITS / INIT). Maize: C4 crop,
# higher light use, warm base and optimum temperature, heat damage only at higher temperatures
CROP_SETTINGS = {
    "grain_maize": {"limits": {"rue": (2.0, 5.5), "tb": (4.0, 12.0), "topt": (20.0, 34.0), "t_heat": (30.0, 40.0),
                               "k": (0.4, 0.8)},
                    "init": {"rue": 3.5, "tb": 8.0, "topt": 28.0, "t_heat": 35.0, "k": 0.6}},
}
CROP_SETTINGS["silage_maize"] = dict(CROP_SETTINGS["grain_maize"])
if V8:
    GROWTH["potato"]["heading"] = "canopy_closed"     # v8 (forward 2008: blend 6.63 vs 6.72)
CROP_SETTINGS["potato"] = {"limits": {"rue": (1.5, 4.5), "tb": (0.0, 8.0), "topt": (12.0, 26.0), "t_heat": (24.0, 34.0)},
                           "init": {"rue": 3.0, "tb": 4.0, "topt": 18.0, "t_heat": 28.0}}
# grain maize (forward validation 2009-2017): saturating x linear technology (hybrid 1.42 -> 1.30),
# blend weight fixed at 0.2 (forward: 0.2 1.12, 0.4 1.13, 0.9 1.26; boosting alone 1.13)
CROP_SETTINGS["grain_maize"]["tech_form"] = "saturating+linear"
FORWARD_WEIGHT = {"grain_maize": 0.2, "silage_maize": 0.6, "potato": 0.4}   # potato forward: 0.4 6.68, boosting 6.83   # silage forward: 0.6 5.89, 0.4 5.90, boosting 5.99
DROUGHT = (2018, 2019, 2022)
TRAIN_ROWS, BATCH, ITERATIONS = 3000, 1500, 300
SUFFIX = ""
if V8:
    SUFFIX = VTAG
# technology level after the training years, chosen per crop by forward validation (fit to 2008,
# forecast 2009-2017): wheat yields stagnated -> hold the level; barley kept rising -> continue
N_COLUMN = {"winter_wheat": "n_wheat_kg_ha", "winter_barley": "n_cereals_kg_ha",
            "grain_maize": "n_maize_kg_ha", "silage_maize": "n_maize_kg_ha", "potato": "n_potato_kg_ha"}
# inputs that move outside their training range in new years (calendar year, technology level,
# nitrogen): trees cannot extrapolate them, so they stay out of the learned correction and act
# only through the physics (version 7)
DRIFTING = {"year", "phys_tech", "n_crop_kg_ha", "phys_n_factor"}
# barley: forward validation 2009-2017 with the technology level in the correction 0.87 t/ha vs
# 0.95 without (boosting 0.93); beyond its training range the trees hold it at the highest level
TECH_IN_CORRECTION = {"winter_barley"}
TECH_RULE = {"winter_wheat": "extrapolated", "winter_barley": "extrapolated", "grain_maize": "extrapolated",
             "silage_maize": "extrapolated", "potato": "extrapolated"}


def drifting(crop):
    """Inputs kept out of the learned correction for this crop (same rule in every script)."""
    return DRIFTING - ({"phys_tech"} if crop in TECH_IN_CORRECTION else set())   # wheat v6: with crop N, extrapolated 0.73 vs frozen 0.74
torch.manual_seed(0)


def configure(crop):
    """Set the setting ranges and harvest type of Growth for this crop (wheat / barley: defaults)."""
    c = CROP_SETTINGS.get(crop, {})
    if "tech_form" in c:
        Growth.TECH_FORM = c["tech_form"]
    extra = {"tech_lin": (0.0, 0.1)} if Growth.TECH_FORM == "saturating+linear" else {}   # per decade
    Growth.LIMITS = {**Growth.BASE_LIMITS, **c.get("limits", {}), **extra}
    if "kernel_water" in Growth.VARIANTS:
        extra["c_ws"] = (0.0, 2.0)
    if "drought_heat" in Growth.VARIANTS:
        extra["d_dry"] = (0.0, 4.0)
    Growth.LIMITS = {**Growth.LIMITS, **extra}
    Growth.INIT = {**Growth.BASE_INIT, **c.get("init", {}), **({"tech_lin": 0.02} if "tech_lin" in extra else {}),
                   **({"c_ws": 0.2} if "c_ws" in extra else {}), **({"d_dry": 1.0} if "d_dry" in extra else {})}
    Growth.HARVEST = GROWTH[crop].get("harvest", "grain")


def load_growth(crop):
    """Daily weather, soil layers and development phases from sowing for every district-year."""
    configure(crop)
    g = GROWTH[crop]
    stages, gate, vp, horizon = CROPS[crop]
    m, d = load(crop)
    if V8:      # monthly weather features of 2021+ from the v8 weather (SARAH sunlight)
        from phase0_master import monthly_features
        wv = phen.weather()
        months = sorted({int(c[-2:]) for c in m.columns if c.startswith("tas_m")})
        mf = monthly_features(wv, wv["codes"], months, "tas_winter" in m.columns)
        late = (m["year"] >= 2021).values
        mf = mf.reindex(pd.MultiIndex.from_frame(m.loc[late, ["district", "year"]]))
        for c in mf.columns:
            if c in m.columns:
                m.loc[late, c] = mf[c].values
    w = phen.weather()
    if "stressboost" in Growth.VARIANTS:   # v10 step 3: flowering-window stress as model inputs
        from stress_matrix import STAGE, windows
        stage = STAGE[crop] if STAGE[crop] in m.columns else "doy_heading"
        sd = m[stage].fillna(m.groupby("district")[stage].transform("median")).fillna(m[stage].median())
        wd = {k: np.asarray(w[k]) for k in ("dates", "pr", "et0", "tasmax")}
        m["win_water_balance"], m["win_hot_days"] = windows(wd, list(w["codes"]), m.assign(**{stage: sd}), stage)
    if Growth.VARIANTS & {"soil", "lst"}:  # v10: DWD soil moisture / MODIS canopy temperature inputs
        import newdata_features as nf
        from stress_matrix import STAGE
        stage = STAGE[crop] if STAGE[crop] in m.columns else "doy_heading"
        mm = m.assign(**{stage: m[stage].fillna(m.groupby("district")[stage].transform("median")).fillna(m[stage].median())})
        if "soil" in Growth.VARIANTS:
            m = pd.concat([m, nf.soil_features(crop, mm, stage)], axis=1)
        if "lst" in Growth.VARIANTS:
            m = pd.concat([m, nf.lst_features(crop, mm, stage, w)], axis=1)
    dates, col = w["dates"], {c: i for i, c in enumerate(w["codes"])}
    start = (pd.to_datetime(m["year"].astype(str) + "-01-01")
             + pd.to_timedelta(m["doy_sowing"].round() - 1, unit="D")).values.astype("datetime64[D]")
    idx = (start - dates[0]).astype(int)[:, None] + np.arange(horizon)[None, :]
    cols = m["district"].map(col).values[:, None]
    t = lambda a: torch.tensor(np.asarray(a, np.float32))
    d["par"] = t(0.5 * w["rsds"][idx, cols] * 0.0864)                     # MJ/m2/day
    d["et0"] = t(w["et0"][idx, cols])
    d["pr"] = t(w["pr"][idx, cols])
    soil = pd.read_csv(os.path.join(P, "soil_districts.csv"), dtype={"district": str}).set_index("district")
    paw = np.stack([(soil[f"fc_{k}"] - soil[f"wp_{k}"]).clip(lower=0) * 100 for k in range(1, 21)], 1)
    paw = pd.DataFrame(paw[:, 0::2] + paw[:, 1::2], index=soil.index)     # mm per 20 cm layer
    cap = paw.reindex(m["district"]).fillna(paw.median()).values
    d["cap"] = t(np.maximum(cap, 1.0))
    # development from the fitted calendar (frozen): phases as smooth 0..1 switches
    cal_set = json.load(open(os.path.join(P, f"phenology_model_{crop}.json")))
    cal = Calendar(len(stages), stages.index(gate), vp, cal_set)
    devs, ths = [], []
    with torch.no_grad():
        for a in range(0, len(m), 2000):
            dv, th = cal({k: v[a:a + 2000] for k, v in d.items() if v.dim() and len(v) == len(m)},
                         return_dev=True)
            devs.append(dv)
            ths.append(th)
    dev, th = torch.cat(devs), torch.cat(ths)
    e, h = (th[:, stages.index(g[k])][:, None] for k in ("emergence", "heading"))
    if g["maturity"] == "harvest_typical":      # potato: development reached on the typical harvest day
        trn = m["role"] == "train"
        typ = (m["doy_harvest"] - m["doy_sowing"])[trn].groupby(m.loc[trn, "district"]).median()
        day = m["district"].map(typ).fillna(typ.median()).clip(1, dev.shape[1] - 1).round().astype(int).values
        mt = torch.maximum(dev[torch.arange(len(m)), torch.tensor(day)], h[:, 0] + 1.0)[:, None]
    else:
        mt = th[:, stages.index(g["maturity"])][:, None]
    d["em"] = torch.sigmoid((dev - e) / TAU)
    d["ph"] = torch.clamp((dev - e) / (h - e), 0, 1)                      # progress to heading
    d["post"] = torch.sigmoid((dev - h) / TAU)                             # after heading
    d["gf"] = torch.clamp((dev - h) / (mt - h), 0, 1)                      # grain filling progress
    d["mat"] = torch.sigmoid((dev - mt) / TAU)                             # mature
    d["dev"], d["th_e"], d["th_h"], d["th_m"] = dev, e[:, 0], h[:, 0], mt[:, 0]   # kept for field ensembles
    nit = pd.read_csv(os.path.join(P, "nitrogen_districts.csv"), dtype={"district": str})
    col = N_COLUMN[crop]                                    # mineral N applied to this crop
    m = m.merge(nit[["district", "year", col]].rename(columns={col: "n_crop_kg_ha"}),
                on=["district", "year"], how="left")
    m["n_crop_kg_ha"] = m["n_crop_kg_ha"].fillna(m["n_crop_kg_ha"].median())
    d["n_supply"] = t(m["n_crop_kg_ha"].values)
    d["year"] = t(m["year"].values)
    if MODIS:
        mf = pd.read_csv(os.path.join(P, "modis", f"modis_features_{crop}.csv"), dtype={"district": str})
        m = m.merge(mf, on=["district", "year"], how="left")
    d["obs"] = t(m["yield_t_ha"].values)
    return m, d


class Growth(torch.nn.Module):
    LIMITS = {"rue": (1.0, 4.5), "k": (0.35, 0.8), "sla": (0.008, 0.035), "lai0": (0.005, 0.3),
              "fl0": (0.2, 0.85), "tb": (-2.0, 6.0), "topt": (10.0, 24.0), "r_sen": (0.0, 0.3),
              "r_dry": (0.0, 0.2), "fg": (0.3, 1.0), "remob": (0.0, 0.5), "zmax": (60.0, 200.0),
              "kc": (0.8, 1.4), "es": (0.1, 1.0), "q": (0.15, 0.8), "t_heat": (27.0, 36.0),
              "c_heat": (0.0, 0.15), "tech": (0.0, 1.5), "tau": (3.0, 60.0), "rgr": (0.002, 0.02), "c_wet": (0.0, 0.05),
              "c_n": (0.0, 0.3), "n_ref": (50.0, 300.0)}
    INIT = {"rue": 2.8, "k": 0.55, "sla": 0.022, "lai0": 0.05, "fl0": 0.5, "tb": 1.0, "topt": 15.0,
            "r_sen": 0.08, "r_dry": 0.05, "fg": 0.7, "remob": 0.2, "zmax": 130.0, "kc": 1.1,
            "es": 0.5, "q": 0.45, "t_heat": 31.0, "c_heat": 0.03, "tech": 0.5, "tau": 15.0,
            "rgr": 0.009, "c_wet": 0.005, "c_n": 0.05, "n_ref": 150.0}
    BASE_LIMITS, BASE_INIT, HARVEST = dict(LIMITS), dict(INIT), "grain"
    TECH_FORM = "saturating"     # "saturating+linear": breeding progress that keeps rising (barley test)
    # v10 step 3 maize stress variants (off by default = v9): 'kernel_water' water stress around
    # flowering cuts kernel set (c_ws); 'drought_heat' heat window from 10 d before flowering and
    # canopy warming under drought (d_dry, degrees C at full stress); 'stressfeat' stress-window
    # indices as outputs (features of the learned correction)
    VARIANTS = set()

    def __init__(self, start=None):
        super().__init__()
        start = start or self.INIT
        self.raw = torch.nn.ParameterDict()
        for k, (lo, hi) in self.LIMITS.items():
            p = min(max((start.get(k, self.INIT[k]) - lo) / (hi - lo), 1e-3), 1 - 1e-3)
            self.raw[k] = torch.nn.Parameter(torch.tensor(float(np.log(p / (1 - p)))))

    def settings(self):
        return {k: bounded(self.raw[k], *lim) for k, lim in self.LIMITS.items()}

    def forward(self, d, dry_matter, record=False):
        s = self.settings()
        daily = []
        n, days = d["tmin"].shape
        cap = d["cap"]
        soil = cap.clone()                                    # start at field capacity
        ztop = torch.arange(LAYERS) * float(LAYER_CM)
        z = torch.zeros(n)
        lai, w_leaf, w_stem, w_grain, reserve, heat, wet = z, z, z, z, z, z, z
        eta, etp, stress, n_post, lai_max, biomass = z, z, z, z, z, z
        prev_em, prev_post, prev_gf = z, z, z
        kw_stress, kw_n, kw_hot = z, z, z
        for day in range(days):
            tmin, tmax = d["tmin"][:, day], d["tmax"][:, day]
            em, ph, post = d["em"][:, day], d["ph"][:, day], d["post"][:, day]
            gf, mat = d["gf"][:, day], d["mat"][:, day]
            growing = em * (1 - mat)
            lai = lai + s["lai0"] * torch.relu(em - prev_em)
            fapar = 1 - torch.exp(-s["k"] * lai)
            ft = torch.clamp(((tmin + tmax) / 2 - s["tb"]) / (s["topt"] - s["tb"]), 0, 1)
            # water: demand, root zone, uptake, soil evaporation, rain
            tpot = d["et0"][:, day] * s["kc"] * fapar * growing
            epot = d["et0"][:, day] * (1 - fapar) * s["es"]
            root = 10 + (s["zmax"] - 10) * ph
            froot = torch.clamp((root[:, None] - ztop) / LAYER_CM, 0, 1)
            avail, room = (froot * soil).sum(1), (froot * cap).sum(1)
            fw = torch.clamp(avail / (room + 1e-3) / s["q"], 0, 1)
            ta = torch.minimum(tpot * fw, avail)
            soil = soil - ta[:, None] * froot * soil / (avail[:, None] + 1e-6)
            es = torch.minimum(epot * torch.clamp(soil[:, 0] / cap[:, 0], 0, 1), soil[:, 0])
            soil = torch.cat([(soil[:, 0] - es)[:, None], soil[:, 1:]], 1)
            deficit = cap - soil
            before = torch.cumsum(deficit, 1) - deficit
            soil = soil + torch.minimum(torch.relu(d["pr"][:, day][:, None] - before), deficit)
            # growth and partitioning
            grow = s["rue"] * d["par"][:, day] * fapar * ft * fw * growing
            fl = s["fl0"] * (1 - ph)
            reserve = reserve + s["remob"] * w_stem * torch.relu(post - prev_post)
            release = reserve * torch.relu(gf - prev_gf) / torch.clamp(1 - prev_gf, min=0.05)
            reserve = reserve - release
            w_leaf = w_leaf + fl * grow * (1 - post)
            w_stem = w_stem + (1 - fl) * grow * (1 - post) + (1 - s["fg"]) * grow * post
            w_grain = w_grain + s["fg"] * grow * post + release
            biomass = biomass + grow
            juvenile = torch.sigmoid((1.0 - lai) / 0.2) * (1 - post) * em    # young canopy: leaf area
            lai = lai + s["sla"] * fl * grow * (1 - post) + juvenile * s["rgr"] * lai * torch.relu(
                (tmin + tmax) / 2 - s["tb"])                                  # grows with warmth (WOFOST)
            # field version only (defaults off = district model v7): leaf loss on frosty days and a
            # later start of leaf ageing in grain filling (g0), both fitted to satellite LAI
            if "c_frost" in s:
                lai = lai * (1 - torch.clamp(s["c_frost"] * torch.relu(s["t_frost"] - tmin), 0, 0.5) * (1 - post))
            g0 = s.get("g0", 0.0)
            sen = torch.relu(gf - g0) / (1 - g0)
            lai = lai * (1 - post * torch.clamp(s["r_sen"] * sen + s["r_dry"] * (1 - fw), 0, 0.5))
            lai_max = torch.maximum(lai_max, lai)
            if record:
                daily.append(lai)
            flowering = post * torch.clamp(1 - gf / 0.2, 0, 1)
            # kernel-set window: ~10 days before flowering to ~20 days after (v10 variants)
            kwin = torch.sigmoid((ph - 0.9) / 0.03) * (1 - post) + post * torch.clamp(1 - gf / 0.25, 0, 1)
            if "drought_heat" in self.VARIANTS:
                heat = heat + kwin * torch.relu(tmax + s["d_dry"] * (1 - fw) - s["t_heat"])
            else:
                heat = heat + flowering * torch.relu(tmax - s["t_heat"])
            kw_stress = kw_stress + kwin * (1 - fw)
            kw_n = kw_n + kwin
            kw_hot = kw_hot + kwin * (tmax >= 30).float()
            window = torch.sigmoid((ph - 0.6) / 0.05) * (1 - post) + post * torch.clamp(1 - gf / 0.5, 0, 1)
            wet = wet + window * torch.sigmoid((d["pr"][:, day] - 2.0) / 0.5)
            eta = eta + (ta + es) * growing
            etp = etp + (tpot + epot) * growing
            stress = stress + (1 - fw) * post * (1 - mat)
            n_post = n_post + post * (1 - mat)
            prev_em, prev_post, prev_gf = em, post, gf
        tech = torch.exp(s["tech"] * (1 - torch.exp(-(d["year"] - 1979) / s["tau"])))
        if self.TECH_FORM == "saturating+linear":
            tech = tech * torch.exp(s["tech_lin"] * (d["year"] - 1979) / 10)
        f_n = torch.exp(-s["c_n"] * torch.nn.functional.softplus((s["n_ref"] - d["n_supply"]) / 10))
        harvested = biomass if self.HARVEST == "biomass" else w_grain     # silage: whole plant
        kernel_ws = kw_stress / (kw_n + 1e-3)                            # mean water stress in the window
        grain = harvested * torch.exp(-s["c_heat"] * heat - s["c_wet"] * wet - s.get("c_ws", 0.0) * kernel_ws)
        yield_t_ha = grain / 100 / dry_matter * tech * f_n
        if record:      # daily green LAI and cumulative water use (field assimilation)
            return yield_t_ha, {"lai_daily": torch.stack(daily, 1), "phys_yield": yield_t_ha,
                                "phys_biomass_t_ha": biomass / 100, "phys_lai_max": lai_max,
                                "phys_eta_mm": eta, "phys_etp_mm": etp, "phys_n_factor": f_n}
        return yield_t_ha, {"phys_yield": yield_t_ha, "phys_biomass_t_ha": biomass / 100,
                            "phys_lai_max": lai_max, "phys_eta_mm": eta, "phys_etp_mm": etp,
                            "phys_water_ratio": eta / (etp + 1e-3), "phys_grainfill_stress":
                            stress / (n_post + 1e-3), "phys_heat": heat, "phys_wet_days": wet, "phys_n_factor": f_n, "phys_tech": tech,
                            **({"phys_kernel_ws": kernel_ws, "phys_kernel_hot": kw_hot}
                               if "stressfeat" in self.VARIANTS else {})}


def fit(model, d, rows, dry_matter):
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ITERATIONS)
    rng = np.random.default_rng(0)
    for it in range(ITERATIONS):
        b = rng.choice(rows, BATCH, replace=False)
        sub = {k: v[b] for k, v in d.items() if v.dim() and len(v) == len(d["obs"])}
        opt.zero_grad()
        pred, _ = model(sub, dry_matter)
        loss = torch.nn.functional.huber_loss(pred, sub["obs"], delta=1.5)
        loss.backward()
        opt.step()
        sched.step()
        if it % 25 == 0 or it == ITERATIONS - 1:
            rmse = torch.sqrt(((pred - sub["obs"]) ** 2).mean()).item()
            print(f"  iteration {it:3d}: RMSE {rmse:5.2f} t/ha (batch)", flush=True)
    return model


def simulate(model, d, dry_matter):
    out = []
    with torch.no_grad():
        for a in range(0, len(d["obs"]), 2000):
            sub = {k: v[a:a + 2000] for k, v in d.items() if v.dim() and len(v) == len(d["obs"])}
            out.append(pd.DataFrame({k: v.numpy() for k, v in model(sub, dry_matter)[1].items()}))
    return pd.concat(out, ignore_index=True)


def booster(setting):
    return HistGradientBoostingRegressor(max_iter=400, min_samples_leaf=40, l2_regularization=1.0,
                                         random_state=0, **setting)


def main():
    crop = sys.argv[1] if len(sys.argv) > 1 else "winter_wheat"
    refit = len(sys.argv) > 2 and sys.argv[2] == "refit"
    dry_matter = GROWTH[crop]["dry_matter"]
    t0 = time.time()
    m, d = load_growth(crop)
    print(f"{crop}: {len(m)} district-years loaded in {time.time() - t0:.0f} s", flush=True)
    path = os.path.join(P, f"growth_model_{crop}_v7{SUFFIX}.json")
    if V8 and not os.path.exists(path):
        import shutil
        if MODIS:                                                   # v9 = v8 physics (MODIS: correction only)
            shutil.copy(os.path.join(P, f"growth_model_{crop}_v7_v8.json"), path)
        elif crop != "potato":                                      # v8 = v7 physics (fitted on <= 2017)
            shutil.copy(os.path.join(P, f"growth_model_{crop}_v7.json"), path)
    model = Growth(None if refit or not os.path.exists(path) else json.load(open(path)))
    train = np.flatnonzero(((m["role"] == "train") & m["yield_t_ha"].notna()).values)
    if refit or not os.path.exists(path):
        sample = np.sort(np.random.default_rng(0).choice(train, min(TRAIN_ROWS, len(train)), replace=False))
        model = fit(model, d, sample, dry_matter)
        with torch.no_grad():
            s = {k: round(v.item(), 4) for k, v in model.settings().items()}
        json.dump(s, open(path, "w"), indent=1)
    with torch.no_grad():
        print("  settings: " + ", ".join(f"{k} {v.item():.3g}" for k, v in model.settings().items()))
    tr = ((m["role"] == "train") & m["yield_t_ha"].notna()).values
    bench_feats = [c for c in features(m, m[tr]) if c != "n_crop_kg_ha"]   # Phase 1 boosting inputs
    sim = simulate(model, d, dry_matter)
    # technology rule chosen by forward validation (fit to 2008, test 2009-2017): the technology
    # level is held at the last training year instead of following the fitted curve further
    with torch.no_grad():
        st = {k: v.item() for k, v in model.settings().items()}
    tech = lambda yr: np.exp(st["tech"] * (1 - np.exp(-(yr - 1979) / st["tau"])))
    last = int(m.loc[tr, "year"].max())
    if TECH_RULE.get(crop, "frozen") == "frozen":
        sim["phys_yield"] = sim["phys_yield"] * tech(np.minimum(m["year"].values, last)) / tech(m["year"].values)
        sim["phys_tech"] = tech(np.minimum(m["year"].values, last))
    print(f"  technology rule: {TECH_RULE.get(crop, 'frozen')} after {last}")
    m = pd.concat([m.reset_index(drop=True), sim], axis=1)
    folds = sorted(m.loc[tr, "fold"].unique())

    def out_of_fold(make, cols, target, back):
        """Year-block cross-validated predictions on the training rows (in t/ha)."""
        pred = np.full(len(m), np.nan)
        for f in folds:
            a, b = tr & (m["fold"] != f).values, tr & (m["fold"] == f).values
            pred[b] = back(make().fit(m.loc[a, cols], target[a]).predict(m.loc[b, cols]), b)
        return pred

    def rmse(p):
        return float(np.sqrt(np.nanmean((p[tr] - m.loc[tr, "yield_t_ha"].values) ** 2)))

    # learned correction: log(observed / simulated) from simulated quantities + monthly weather,
    # without the calendar year (so it cannot extend a trend into new years)
    feats = [c for c in bench_feats + list(sim.columns) if c not in DRIFTING]
    phys = m["phys_yield"].clip(lower=0.5).values
    target = np.log(m["yield_t_ha"].values / phys)
    best = None
    for setting in ({"max_depth": 3, "learning_rate": 0.05}, {"max_depth": 6, "learning_rate": 0.05}):
        oof = out_of_fold(lambda: booster(setting), feats, target, lambda p, b: phys[b] * np.exp(p))
        print(f"  correction {setting}: cross-validated RMSE {rmse(oof):.2f} t/ha", flush=True)
        if best is None or rmse(oof) < best[1]:
            best = (setting, rmse(oof), oof)
    corr = booster(best[0]).fit(m.loc[tr, feats], target[tr])
    m["physics"] = m["phys_yield"]
    m["hybrid"] = phys * np.exp(corr.predict(m[feats]))
    # blend with the Phase 1 boosting model; weight chosen on the same out-of-fold predictions
    from phase1_baselines import BOOST, boost
    y = m["yield_t_ha"].values
    bb = min(BOOST, key=lambda st: rmse(out_of_fold(lambda: booster(st), bench_feats, y, lambda p, b: p)))
    boost_oof = out_of_fold(lambda: booster(bb), bench_feats, y, lambda p, b: p)
    weight = min(np.linspace(0, 1, 11), key=lambda w: rmse(w * best[2] + (1 - w) * boost_oof))
    if crop in FORWARD_WEIGHT:                 # chosen by forward validation instead (see CROP_SETTINGS)
        weight = FORWARD_WEIGHT[crop]
    boost_all = boost(m.loc[tr, bench_feats].values.astype(float), y[tr], bb).predict(
        m[bench_feats].values.astype(float))
    m["blend"] = weight * m["hybrid"] + (1 - weight) * boost_all
    print(f"  blend: weight of the hybrid {weight:.1f} (cross-validated RMSE "
          f"{rmse(weight * best[2] + (1 - weight) * boost_oof):.2f} t/ha; boosting alone "
          f"{rmse(boost_oof):.2f}, hybrid alone {best[1]:.2f})", flush=True)
    keep = ["district", "year", "role", "yield_t_ha", "physics", "hybrid", "blend"] + list(sim.columns)
    m[keep].to_csv(os.path.join(P, f"growth_predictions_{crop}{SUFFIX}.csv"), index=False)

    # comparison on identical district-years with the Phase 1 benchmarks
    bench = pd.read_csv(os.path.join(P, f"wofost_predictions_{crop}.csv"), dtype={"district": str})
    names = ["trend", "boosting", "trend+boosting", "wofost+weather", "physics", "hybrid", "blend"]
    c = bench[["district", "year", "role", "yield_t_ha"] + names[:4]].merge(
        m[["district", "year", "physics", "hybrid", "blend"]], on=["district", "year"])
    groups = [("training (in-sample)", c["role"] == "train"),
              ("test: new years", c["role"] == "test_years"),
              ("test: new regions", c["role"] == "test_regions"),
              ("test: both", c["role"] == "test_both"),
              ("test: drought years", (c["role"] != "train") & c["year"].isin(DROUGHT))]
    rows = []
    print(f"\n{crop}: RMSE t/ha on identical district-years")
    print(f"  {'data':22s}{'n':>6s}" + "".join(f"{n:>16s}" for n in names))
    for label, sel in groups:
        line = f"  {label:22s}{sel.sum():6d}"
        for name in names:
            rmse = float(np.sqrt(np.mean((c.loc[sel, name] - c.loc[sel, "yield_t_ha"]) ** 2)))
            line += f"{rmse:16.2f}"
            rows.append(dict(crop=crop, data=label, n=int(sel.sum()), model=name, rmse=round(rmse, 3)))
        print(line, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(P, f"growth_comparison_{crop}{SUFFIX}.csv"), index=False)
    s = m[m["role"] != "train"]
    print(f"\n  simulated (test rows, median): LAI max {s.phys_lai_max.median():.1f}, biomass "
          f"{s.phys_biomass_t_ha.median():.1f} t/ha, water use {s.phys_eta_mm.median():.0f} mm of "
          f"{s.phys_etp_mm.median():.0f} mm demand; total time {time.time() - t0:.0f} s")


def forward(crop, cut=2008, save=False):
    """Forward validation inside the training data: fit everything on training rows up to `cut`,
    predict the training rows after it (2009-2017), as if forecasting new years. Used to choose,
    without touching the 2018-2025 test years, how the technology level is carried forward:
    'extrapolated' (fitted saturating curve continues) or 'frozen' (held at the last fitted year)."""
    from phase1_baselines import BOOST, boost
    dry_matter = GROWTH[crop]["dry_matter"]
    m, d = load_growth(crop)
    ok = ((m["role"] == "train") & m["yield_t_ha"].notna()).values
    fit_rows, val = ok & (m["year"] <= cut).values, ok & (m["year"] > cut).values
    form = "" if Growth.TECH_FORM == "saturating" else "_techlin"
    path = os.path.join(P, f"growth_model_{crop}_forward{cut}_v7{SUFFIX}{form}.json")
    base = os.path.join(P, f"growth_model_{crop}_forward{cut}_v7{SUFFIX.replace('_modis', '')}{form}.json")
    if MODIS and not os.path.exists(path) and os.path.exists(base):    # MODIS changes no physics
        import shutil
        shutil.copy(base, path)
    if os.path.exists(path):
        model = Growth(json.load(open(path)))
    else:
        rows = np.flatnonzero(fit_rows)
        sample = np.sort(np.random.default_rng(0).choice(rows, min(TRAIN_ROWS, len(rows)), replace=False))
        model = fit(Growth(), d, sample, dry_matter)
        with torch.no_grad():
            json.dump({k: round(v.item(), 4) for k, v in model.settings().items()}, open(path, "w"), indent=1)
    with torch.no_grad():
        st = {k: v.item() for k, v in model.settings().items()}
    bench_feats = [c for c in features(m, m[fit_rows]) if c != "n_crop_kg_ha"]
    sim = simulate(model, d, dry_matter)
    m = pd.concat([m.reset_index(drop=True), sim], axis=1)
    y = m["yield_t_ha"].values
    tech = lambda yr: np.exp(st["tech"] * (1 - np.exp(-(yr - 1979) / st["tau"])))
    frozen = tech(np.minimum(m["year"].values, cut)) / tech(m["year"].values)
    phys = m["phys_yield"].clip(lower=0.5).values
    feats = [c for c in bench_feats + list(sim.columns) if c not in drifting(crop)]
    corr = booster({"max_depth": 6, "learning_rate": 0.05}).fit(
        m.loc[fit_rows, feats], np.log(y[fit_rows] / phys[fit_rows]))
    hybrid = phys * np.exp(corr.predict(m[feats]))
    b = boost(m.loc[fit_rows, bench_feats].values.astype(float), y[fit_rows], BOOST[1]).predict(
        m[bench_feats].values.astype(float))
    models = {"boosting": b, "physics, technology extrapolated": phys,
              "physics, technology frozen": phys * frozen, "hybrid, technology extrapolated": hybrid,
              "hybrid, technology frozen": hybrid * frozen,
              "blend 0.9, extrapolated": 0.9 * hybrid + 0.1 * b,
              "blend 0.9, frozen": 0.9 * hybrid * frozen + 0.1 * b,
              **{f"blend {w:.1f}, extrapolated": w * hybrid + (1 - w) * b for w in (0.2, 0.4, 0.6)}}
    print(f"\n{crop}: forward validation, fitted on training rows up to {cut}, "
          f"scored on {val.sum()} training rows {cut + 1}-2017")
    print(f"  {'model':36s}{'RMSE':>7s}{'bias':>7s}   per year bias")
    rows = []
    for name, p in models.items():
        e = p[val] - y[val]
        per_year = pd.Series(e).groupby(m.loc[val, "year"].values).mean()
        print(f"  {name:36s}{np.sqrt(np.mean(e ** 2)):7.2f}{e.mean():+7.2f}   "
              + " ".join(f"{v:+.1f}" for v in per_year.values), flush=True)
        rows.append(dict(crop=crop, cut=cut, model=name, rmse=round(float(np.sqrt(np.mean(e ** 2))), 3),
                         bias=round(float(e.mean()), 3)))
    pd.DataFrame(rows).to_csv(os.path.join(P, f"growth_forward_{crop}{SUFFIX}{form}.csv"), index=False)
    if save:     # per district-year forward predictions with the crop's final rules (range method)
        w = FORWARD_WEIGHT.get(crop, 0.8)
        h = hybrid * (frozen if TECH_RULE.get(crop, "frozen") == "frozen" else 1.0)
        out = m.loc[val, ["district", "year", "yield_t_ha"]].assign(pred=(w * h + (1 - w) * b)[val], cut=cut)
        out.to_csv(os.path.join(P, f"forward_predictions_{crop}{SUFFIX}_cut{cut}.csv"), index=False)
        print(f"  saved forward predictions (blend weight {w}): {len(out)} rows")


if __name__ == "__main__":
    if "techincorr" in sys.argv:          # let the technology level into the correction (barley test)
        DRIFTING.discard("phys_tech")
        sys.argv.remove("techincorr")
    for arg in [a for a in sys.argv if a.startswith("physvar=")]:   # physics variants (forward tests)
        var = arg.split("=", 1)[1]
        if var == "maize_harvest":          # grain filling until the typical harvest (physiological maturity)
            GROWTH["grain_maize"]["maturity"] = "harvest_typical"
        elif var == "potato_canopy":        # tuber bulking from canopy closure instead of flowering
            GROWTH["potato"]["heading"] = "canopy_closed"
        elif var in ("kernel_water", "drought_heat", "stressfeat", "stressboost", "soil", "lst"):    # v10 variants
            Growth.VARIANTS.add(var)
        SUFFIX = SUFFIX + "_" + var
        sys.argv.remove(arg)
    if MODIS and not V8:
        SUFFIX = SUFFIX + "_modis"
    if "techlin" in sys.argv:             # technology = saturating curve x linear progress (barley test)
        Growth.TECH_FORM = "saturating+linear"
        sys.argv.remove("techlin")
    if "nonitrogen" in sys.argv:          # switch the N effect off (c_n fixed at 0)
        Growth.BASE_LIMITS = {**Growth.BASE_LIMITS, "c_n": (0.0, 1e-9)}   # kept by configure()
        sys.argv.remove("nonitrogen")
        SUFFIX = "_noN"
    if sys.argv[1] in TECH_IN_CORRECTION:
        DRIFTING.discard("phys_tech")
    if len(sys.argv) > 2 and sys.argv[2] == "forward":
        cut = int(sys.argv[3]) if len(sys.argv) > 3 else 2008
        forward(sys.argv[1], cut, save="save" in sys.argv)
    else:
        main()
