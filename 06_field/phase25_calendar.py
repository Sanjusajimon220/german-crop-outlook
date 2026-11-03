"""Crop calendar per field for the platform's "Season timing" panel (owner request 2026-11-03).

The frozen district crop calendar (phenology_model_<crop>.json, as in the district model) run on each field's own HYRAS
1 km weather cell (tmin / tmax / rain), the district's sowing day and day length - the same development curve the field
model (phase23) uses. Fields of one crop, district and cell share one curve. Winter wheat and winter barley only.
Not a field measurement: Job 6a found no field-level gain in stage dates, so the platform labels it as the regional
calendar. Writes docs/field/<area>_<year>/calendar.json:
  weeks           the series.json Mondays
  stages          per crop the stage names in order
  curves[key]     crop, dev (development at each Monday; null before sowing), thr (stage thresholds of that year),
                  dates (day each stage is reached), sowing, own_weather
  fields[id]      key of the field's curve
Run from D:\\vista: python phase25_calendar.py [AREA YEAR ...]   (default: every area and season)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "02_district_model"))

DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "field")
FD = os.path.join("data", "processed", "field")
CODES = {"wheat": "Winterweichweizen", "barley": "Wintergerste"}      # NRW register crop names (as phase22b)


def stage_dates(dev, thresholds, sowing, names):
    """Day each stage is first reached (dev >= threshold), ISO date; None if not reached within the run."""
    out = {}
    for name, t in zip(names, thresholds):
        hit = np.flatnonzero(dev >= t)
        out[name] = None if not len(hit) else str((sowing + pd.to_timedelta(int(hit[0]), unit="D")).date())
    return out


def weekly_dev(dev, sowing, weeks):
    """Development on each Monday (day 0 = sowing), 1 decimal; None before sowing and beyond the run."""
    day = (weeks.astype("datetime64[D]") - np.datetime64(sowing.date(), "D")).astype(int)
    return [round(float(dev[d]), 1) if 0 <= d < len(dev) else None for d in day]


def calendar_curve(crop, year, district, cell):
    """Daily development from sowing on the cell's weather (phase23 field_ensemble without the growth run)."""
    import phase23_ours_field as OF
    L = OF._load(crop)
    m, d = L["m"], L["d"]
    i = L["row"].get((district, year))
    if i is None:
        return None
    dates, col, w = OF._weather(year)
    sow = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=round(m.at[i, "doy_sowing"]) - 1)
    first = int((np.datetime64(sow.date()) - dates[0]).astype(int))
    n_days = d["tmin"].shape[1]
    sub = {k: d[k][i:i + 1].clone() for k in ("tmin", "tmax", "pr", "daylen", "decade")}
    own = cell in col
    for key, var in (("tmin", "tasmin"), ("tmax", "tasmax"), ("pr", "pr")):
        series = w[var][first:first + n_days, col[cell]] if own else np.array([])
        if len(series) == n_days and np.isfinite(series).all():
            sub[key] = torch.tensor(series, dtype=torch.float32)[None, :]
        else:
            own = False
    with torch.no_grad():
        dev, th = L["cal"]({"tmin": sub["tmin"], "tmax": sub["tmax"], "daylen": sub["daylen"],
                            "rain": torch.cumsum(sub["pr"], 1), "decade": sub["decade"]}, return_dev=True)
    return dict(dev=dev[0].numpy(), thr=th[0].numpy(), sowing=sow, own_weather=own)


def build(area, year):
    from phase2_phenology import CROPS
    import phase23_ours_field as OF
    out_dir = os.path.join(DOCS, f"{area}_{year}")
    weeks = np.array(json.load(open(os.path.join(out_dir, "series.json")))["weeks"], dtype="datetime64[D]")
    fl = pd.read_csv(os.path.join("data", "processed", "field_v2", f"{area}_{year}", "fields_nrw.csv"))
    cell = pd.read_csv(os.path.join(FD, "fields_hyras_cell.csv"))
    cell = cell[cell.VALIDFROM == year].set_index("ID")["cell"]
    dist = pd.read_csv(os.path.join(FD, "fields_district.csv"), dtype={"district": str})
    dist = dist[dist.VALIDFROM == year].drop_duplicates("ID").set_index("ID")["district"]
    out = dict(weeks=[str(w) for w in weeks], stages={}, curves={}, fields={})
    for crop, name in CODES.items():
        out["stages"][crop] = CROPS[OF.CROP_KEYS[crop]][0]
        for fid in fl[fl.crop.str.contains(name, na=False)].field_id:
            if fid not in cell.index or fid not in dist.index:
                continue
            key = f"{crop}_{dist[fid]}_{int(cell[fid])}"
            if key not in out["curves"]:
                c = calendar_curve(crop, year, dist[fid], int(cell[fid]))
                if c is None:
                    continue
                out["curves"][key] = dict(crop=crop, dev=weekly_dev(c["dev"], c["sowing"], weeks),
                                          thr=[round(float(t), 1) for t in c["thr"]],
                                          dates=stage_dates(c["dev"], c["thr"], c["sowing"], out["stages"][crop]),
                                          sowing=str(c["sowing"].date()), own_weather=bool(c["own_weather"]))
            if key in out["curves"]:
                out["fields"][str(int(fid))] = key
    with open(os.path.join(out_dir, "calendar.json"), "w") as f:
        json.dump(out, f, separators=(",", ":"))
    return len(out["curves"]), len(out["fields"])


if __name__ == "__main__":
    os.chdir(r"D:\vista")
    from areas import SEASONS
    a = sys.argv[1:]
    jobs = [(a[i], int(a[i + 1])) for i in range(0, len(a), 2)] or [(ar, y) for ar, ys in SEASONS.items() for y in ys]
    for area, year in jobs:
        print(area, year, "curves / fields:", build(area, year), flush=True)
