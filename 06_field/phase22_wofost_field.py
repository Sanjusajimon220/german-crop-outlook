"""WOFOST per field (Job 2, design 2026-10-22 with amendments 1-3; owner: true per-field simulation, reusable for any field).

Each field runs with its OWN weather (HYRAS 1 km cell: tmin / tmax / humidity / rain; district SARAH-3 radiation and wind)
and its OWN soil (BK50: field capacity, plant-available water, rooting depth). Crop: PCSE Wofost72_WLP_CWB, Winter_wheat_102
with the calibrated TSUM scales (barley = wheat with barley's calibrated development; no winter barley in WOFOST).
Members: Latin hypercube over TSUM1 / TSUM2 (x 0.88-1.12), leaf area per leaf mass SLATB (x 0.8-1.2), sowing (+-7 days around
the observed district sowing) and plant-available water (x 0.8-1.2).
  particle ensemble (C): 1,000 members per field, weekly LAI saved per field (resumable);
  normal WOFOST (E1): one run with all factors 1 and no shift;
  Kalman (E2): 100 members, LAI updated at each measured week before the cut (stochastic EnKF), then run freely.
"""
import datetime as dt
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "02_district_model"))

FD = os.path.join("data", "processed", "field")
P = os.path.join("data", "processed")
OUT = os.path.join("data", "processed", "field_v2", "wofost")
CROP_KEYS = {"wheat": "winter_wheat", "barley": "winter_barley"}
RANGES = {"tsum1_f": (0.88, 1.12), "tsum2_f": (0.88, 1.12), "sla_f": (0.8, 1.2), "water_f": (0.8, 1.2), "sow_shift": (-7.0, 7.0)}
OBS_ERR = 0.5
logging.disable(logging.CRITICAL)


def soil_site(fk_mm, nfk_mm, root_dm, water_f=1.0):
    """BK50 field capacity / plant-available water (mm over the rooting depth) -> WOFOST soil (volume fractions) and site."""
    from pcse.input import DummySoilDataProvider, WOFOST72SiteDataProvider
    depth_mm = max(root_dm, 3.0) * 100.0
    fc = fk_mm / depth_mm
    wp = max(0.01, fc - nfk_mm * water_f / depth_mm)
    soil = DummySoilDataProvider()
    soil.update({"SMFCF": fc, "SMW": wp, "SM0": min(fc + 0.08, 0.55), "RDMSOL": max(root_dm, 3.0) * 10.0})
    site = WOFOST72SiteDataProvider(WAV=(fc - wp) * max(root_dm, 3.0) * 10.0 * 0.9)
    return soil, site


def lhs(n, seed):
    """Latin hypercube: each setting's range cut into n equal strata, one value per stratum, strata shuffled per setting."""
    rng = np.random.default_rng(seed)
    out = {}
    for k, (lo, hi) in RANGES.items():
        u = (rng.permutation(n) + rng.random(n)) / n
        out[k] = lo + u * (hi - lo)
    m = pd.DataFrame(out)
    m["sow_shift"] = np.round(m["sow_shift"]).astype(int)
    return m


def weekly_sample(output, weeks):
    """LAI on each week date from PCSE daily output (list of dicts with 'day' and 'LAI'); 0 where the run has no value."""
    lai = {np.datetime64(o["day"], "D"): (o.get("LAI") or 0.0) for o in output}
    return np.array([float(lai.get(w, 0.0)) for w in weeks.astype("datetime64[D]")])


def enkf_update(ens, y, r, rng):
    """Stochastic EnKF update of one scalar state (LAI) with perturbed observations; LAI kept >= 0."""
    pf = np.var(ens, ddof=1)
    k = pf / (pf + r) if pf + r > 0 else 0.0
    return np.maximum(ens + k * (y + rng.normal(0, np.sqrt(r), len(ens)) - ens), 0.0)


def enkf_run(models, obs_dates, obs, sd, weeks, rng):
    """E2: run every member to each measured date, update its LAI towards the observation (error sd^2 + OBS_ERR^2),
    then run freely to the end; weekly LAI per member. Dates where a member has no crop yet are skipped."""
    for day, y, s in zip(obs_dates, obs, sd):
        for m in models:
            m.run_till(day)
        lai = np.array([np.nan if m.get_variable("LAI") is None else m.get_variable("LAI") for m in models], float)
        if np.isfinite(lai).all():
            for m, v in zip(models, enkf_update(lai, y, s ** 2 + OBS_ERR ** 2, rng)):
                m.set_variable("LAI", float(v))
    out = []
    for m in models:
        m.run_till_terminate()
        out.append(weekly_sample(m.get_output(), weeks))
    return np.array(out)


# ---- weather, model and field runs ----
_C = {}


def _shared():
    if "dist" not in _C:
        w = np.load(os.path.join(P, "weather_daily.npz"))
        _C["dist"] = {k: w[k] for k in ("dates", "rsds", "wind2")}
        _C["dcol"] = {c: i for i, c in enumerate(w["codes"])}
        _C["calib"] = json.load(open(os.path.join(P, "wofost_calibration.json")))
        from pcse.input import YAMLCropDataProvider
        _C["crop"] = YAMLCropDataProvider()
    return _C


def _hyras(year):
    key = ("h", year)
    if key not in _C:
        parts = [np.load(os.path.join(FD, f"hyras_cells_{y}.npz")) for y in (year - 1, year)]
        _C[key] = dict(dates=np.concatenate([p["dates"] for p in parts]),
                       col={c: i for i, c in enumerate(parts[0]["cells"])},
                       **{v: np.concatenate([p[v] for p in parts]) for v in ("tasmin", "tasmax", "hurs", "pr")})
    return _C[key]


def field_weather(cell, district, year, lat):
    """PCSE weather provider for one field: HYRAS cell + district radiation and wind (Penman-Monteith reference ET)."""
    from pcse.base import WeatherDataContainer, WeatherDataProvider
    from pcse.util import reference_ET
    h, c = _hyras(year), _shared()
    k, dk = h["col"][cell], c["dcol"][district]
    dpos = {d: i for i, d in enumerate(c["dist"]["dates"])}

    class W(WeatherDataProvider):
        def __init__(self):
            super().__init__()
            self.latitude, self.longitude, self.elevation = lat, 6.3, 100.0
            self.angstA, self.angstB = 0.212, 0.547
            for i, d in enumerate(h["dates"]):
                j = dpos.get(d)
                if j is None:
                    continue
                day = d.astype(object)
                tmin, tmax = float(h["tasmin"][i, k]), float(h["tasmax"][i, k])
                es = lambda t: 0.6108 * np.exp(17.27 * t / (t + 237.3))
                vap = float(h["hurs"][i, k]) / 100.0 * (es(tmax) + es(tmin)) / 2 * 10.0
                irrad, wind = float(c["dist"]["rsds"][j, dk]) * 86400.0, float(c["dist"]["wind2"][j, dk])
                e0, es0, et0 = reference_ET(day, lat, 100.0, tmin, tmax, irrad, vap, wind, 0.212, 0.547, ETMODEL="PM")
                self._store_WeatherDataContainer(WeatherDataContainer(
                    LAT=lat, LON=6.3, ELEV=100.0, DAY=day, IRRAD=irrad, TMIN=tmin, TMAX=tmax, VAP=vap,
                    RAIN=float(h["pr"][i, k]) / 10.0, WIND=wind, E0=e0 / 10.0, ES0=es0 / 10.0, ET0=et0 / 10.0), day)
    return W()


def make_model(weather, crop_key, sowing, member, soil_row):
    from pcse.base import ParameterProvider
    from pcse.models import Wofost72_WLP_CWB
    from wofost_runner import agromanagement
    c = _shared()
    crop = c["crop"]; crop.set_active_crop("wheat", "Winter_wheat_102")
    t1, t2 = c["calib"][crop_key]
    soil, site = soil_site(soil_row["fk_mm"], soil_row["nfk_mm"], soil_row["root_dm"], member["water_f"])
    p = ParameterProvider(cropdata=crop, soildata=soil, sitedata=site)
    p.set_override("TSUM1", crop["TSUM1"] * t1 * member["tsum1_f"])
    p.set_override("TSUM2", crop["TSUM2"] * t2 * member["tsum2_f"])
    sl = list(crop["SLATB"]); p.set_override("SLATB", [v * member["sla_f"] if i % 2 else v for i, v in enumerate(sl)])
    sow = sowing + dt.timedelta(days=int(member["sow_shift"]))
    return Wofost72_WLP_CWB(p, weather, agromanagement("wheat", "Winter_wheat_102", sow, "maturity", None, max_days=320))


def run_member(weather, crop_key, sowing, member, soil_row, weeks):
    m = make_model(weather, crop_key, sowing, member, soil_row)
    m.run_till_terminate()
    out = m.get_output()
    s = (m.get_summary_output() or [{}])[0]
    return weekly_sample(out, weeks), s.get("DOA"), s.get("DOM")
