"""Run WOFOST (PCSE, model Wofost72_WLP_CWB: water-limited production, classic water balance)
for one district and season, using our daily district weather, soil and observed crop dates.

Used by phase1_wofost.py. Conversions from our data to WOFOST's inputs:
  IRRAD  J/m2/day    = sunlight W/m2 x 86400
  VAP    hPa         = humidity / 100 x mean saturation pressure of Tmax and Tmin x 10
  RAIN   cm/day      = precipitation mm / 10
  WIND   m/s at 2 m  = our wind2
  E0, ES0, ET0 cm/day from PCSE's own Penman-Monteith routine (pcse.util.reference_ET)
  soil: field capacity and wilting point = mean of the top 1 m of the DWD soil maps
"""
import datetime as dt
import logging
import os

import numpy as np
import pandas as pd
from pcse.base import ParameterProvider, WeatherDataContainer, WeatherDataProvider
from pcse.input import DummySoilDataProvider, WOFOST72SiteDataProvider, YAMLCropDataProvider
from pcse.models import Wofost72_Phenology, Wofost72_WLP_CWB
from pcse.util import reference_ET

P = os.path.join("data", "processed")
# crop: (WOFOST crop, variety, end of season: "maturity" or "harvest", output used as yield)
WOFOST_CROPS = {
    "winter_wheat": ("wheat", "Winter_wheat_102", "maturity", "TWSO"),
    "winter_barley": ("wheat", "Winter_wheat_102", "maturity", "TWSO"),   # no winter barley in WOFOST
    "silage_maize": ("maize", "Fodder_maize_nl", "harvest", "TAGP"),       # whole plant is harvested
    "grain_maize": ("maize", "Grain_maize_201", "maturity", "TWSO"),
    "potato": ("potato", "Fontane", "harvest", "TWSO"),
}
ANGSTROM = (0.212, 0.547)       # fitted in phase0_stations.py
ELEVATION = 100.0               # m; elevation not yet included per district
ROOT_DEPTH_CM = 120.0

_cache = {}
logging.disable(logging.CRITICAL)   # PCSE's own log messages are not needed here


def shared():
    """Load the daily weather, soil table and crop parameters once per process."""
    if not _cache:
        w = np.load(os.path.join(P, "weather_daily.npz"))
        _cache["w"] = {k: w[k] for k in ("dates", "codes", "tas", "tasmax", "tasmin", "hurs", "pr",
                                           "rsds", "wind2")}
        _cache["col"] = {c: i for i, c in enumerate(w["codes"])}
        _cache["soil"] = pd.read_csv(os.path.join(P, "soil_districts.csv"),
                                     dtype={"district": str}).set_index("district")
        _cache["crop"] = YAMLCropDataProvider()
        g = np.load(os.path.join(P, "district_grid_soil.npz"))
        _cache["lat"] = district_latitudes()
    return _cache


def district_latitudes():
    import netCDF4
    g = np.load(os.path.join(P, "district_grid_soil.npz"))
    with netCDF4.Dataset(os.path.join("data", "raw", "soil", "AG_SOILINFO_THETAFC.nc")) as d:
        lat = d["lat"][:].ravel()
    idx = g["index"].ravel()
    ok = idx >= 0
    n = len(g["codes"])
    mean = np.bincount(idx[ok], weights=lat[ok], minlength=n) / np.bincount(idx[ok], minlength=n)
    return dict(zip(g["codes"], mean))


class DistrictWeather(WeatherDataProvider):
    """Daily weather of one district, in WOFOST's units."""

    def __init__(self, district, start, end):
        super().__init__()
        c = shared()
        w, k = c["w"], c["col"][district]
        self.latitude, self.longitude, self.elevation = c["lat"][district], 10.0, ELEVATION
        self.angstA, self.angstB = ANGSTROM
        sel = np.flatnonzero((w["dates"] >= np.datetime64(start)) & (w["dates"] <= np.datetime64(end)))
        for i in sel:
            day = w["dates"][i].astype(object)
            tmin, tmax = float(w["tasmin"][i, k]), float(w["tasmax"][i, k])
            es = lambda t: 0.6108 * np.exp(17.27 * t / (t + 237.3))
            vap = float(w["hurs"][i, k]) / 100.0 * (es(tmax) + es(tmin)) / 2 * 10.0   # hPa
            irrad = float(w["rsds"][i, k]) * 86400.0
            wind = float(w["wind2"][i, k])
            e0, es0, et0 = reference_ET(day, self.latitude, self.elevation, tmin, tmax, irrad, vap,
                                        wind, self.angstA, self.angstB, ETMODEL="PM")
            self._store_WeatherDataContainer(WeatherDataContainer(
                LAT=self.latitude, LON=self.longitude, ELEV=self.elevation, DAY=day,
                IRRAD=irrad, TMIN=tmin, TMAX=tmax, VAP=vap, RAIN=float(w["pr"][i, k]) / 10.0,
                WIND=wind, E0=e0 / 10.0, ES0=es0 / 10.0, ET0=et0 / 10.0), day)


def soil_and_site(district):
    s = shared()["soil"].loc[district]
    fc = float(np.mean([s[f"fc_{i}"] for i in range(1, 11)]))
    wp = float(np.mean([s[f"wp_{i}"] for i in range(1, 11)]))
    soil = DummySoilDataProvider()
    soil.update({"SMFCF": fc, "SMW": wp, "SM0": min(fc + 0.08, 0.55), "RDMSOL": ROOT_DEPTH_CM})
    site = WOFOST72SiteDataProvider(WAV=(fc - wp) * ROOT_DEPTH_CM * 0.9)   # soil nearly full at start
    return soil, site


def agromanagement(crop, variety, sowing, end_type, harvest, max_days=330):
    event = {"crop_name": crop, "variety_name": variety, "crop_start_date": sowing,
             "crop_start_type": "sowing", "crop_end_type": end_type, "max_duration": max_days}
    if end_type == "harvest":
        event["crop_end_date"] = harvest
    end = (harvest if end_type == "harvest" else sowing + dt.timedelta(days=max_days))
    return [{sowing: {"CropCalendar": event, "TimedEvents": None, "StateEvents": None}},
            {end + dt.timedelta(days=1): None}]


def run_season(district, crop_key, sowing, harvest, tsum_scale=(1.0, 1.0), phenology_only=False):
    """One season -> dict with yield output (kg/ha dry matter), TAGP, anthesis and maturity day."""
    crop, variety, end_type, out_var = WOFOST_CROPS[crop_key]
    c = shared()["crop"]
    c.set_active_crop(crop, variety)
    soil, site = soil_and_site(district)
    params = ParameterProvider(cropdata=c, soildata=soil, sitedata=site)
    params.set_override("TSUM1", c["TSUM1"] * tsum_scale[0])
    params.set_override("TSUM2", c["TSUM2"] * tsum_scale[1])
    end = harvest if end_type == "harvest" else sowing + dt.timedelta(days=331)
    weather = DistrictWeather(district, sowing - dt.timedelta(days=1), end + dt.timedelta(days=2))
    model_cls = Wofost72_Phenology if phenology_only else Wofost72_WLP_CWB
    model = model_cls(params, weather, agromanagement(crop, variety, sowing, end_type, harvest))
    model.run_till_terminate()
    summary = model.get_summary_output()
    s = summary[0] if summary else {}
    return {"yield": s.get(out_var), "TAGP": s.get("TAGP"), "anthesis": s.get("DOA"),
            "maturity": s.get("DOM"), "LAImax": s.get("LAIMAX")}


if __name__ == "__main__":
    import time
    t = time.perf_counter()
    r = run_season("05358", "winter_wheat", dt.date(2019, 10, 15), dt.date(2020, 7, 31))
    print(f"Dueren winter wheat 2020: {r} in {time.perf_counter() - t:.1f} s")
    t = time.perf_counter()
    r = run_season("05358", "potato", dt.date(2022, 4, 25), dt.date(2022, 9, 20))
    print(f"Dueren potato 2022: {r} in {time.perf_counter() - t:.1f} s")
    t = time.perf_counter()
    r = run_season("05358", "winter_wheat", dt.date(2019, 10, 15), dt.date(2020, 7, 31), phenology_only=True)
    print(f"phenology only: {r} in {time.perf_counter() - t:.2f} s")
