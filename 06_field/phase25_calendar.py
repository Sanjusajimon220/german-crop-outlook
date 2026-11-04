"""Crop calendar per field for the platform's "Season timing" panel (owner request 2026-11-03).

The frozen district crop calendar (phenology_model_<crop>.json, as in the district model) run on each field's own HYRAS
1 km weather cell (tmin / tmax / rain), the district's sowing day and day length - the same development curve the field
model (phase23) uses. Fields of one crop, district and cell share one curve. Winter wheat, winter barley, silage and
grain maize: maize and potato removed, see CALENDARS; other crops have none.
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
# fitted district calendar per field crop: NRW register crop name prefix -> (platform key, district model crop)
CALENDARS = [("Winterweichweizen", "wheat", "winter_wheat"), ("Wintergerste", "barley", "winter_barley")]
# maize and potato removed (owner decision 2026-11-04): against the satellite canopy of the study-area fields their DWD-fitted
# calendars ran early (potato canopy closed 2-5 weeks before median LAI >= 2; maize flowering 1-3 weeks before the LAI peak);
# wheat / barley agreed (LAI peak +4 / +14 days after heading). The maize / potato loaders stay for a later satellite-shifted test:
# add ("Silomais", "silage_maize", "silage_maize"), ("Mais (ohne", "grain_maize", "grain_maize"), ("Kartoffeln", "potato", "potato")
MODEL_KEY = {k: m for _, k, m in CALENDARS}
# grain maize district rows end in 2021 (no Destatis district yields after): its fitted calendar runs on the silage maize
# row of the same district-year (same crop, same sowing day)
INPUT_FROM = {"grain_maize": "silage_maize"}


def calendar_crop(name):
    """Platform calendar key for an NRW register crop name, None without a fitted calendar (prefix match:
    'Mais (ohne Silomais)' is grain maize although it contains 'Silomais')."""
    if not isinstance(name, str):
        return None
    for prefix, key, _ in CALENDARS:
        if name.startswith(prefix):
            return key
    return None


def nearest_assign(xy, ref_xy, ref_cells, ref_dists, max_m=1500):
    """Weather cell and district of the nearest reference field (the wheat / barley fields of fields_hyras_cell.csv)
    for fields outside those tables; -1 / None when no reference field lies within max_m."""
    from scipy.spatial import cKDTree
    dist, k = cKDTree(ref_xy).query(xy)
    ok = dist <= max_m
    cells = np.where(ok, np.asarray(ref_cells)[k], -1)
    dists = np.array([d if o else None for d, o in zip(np.asarray(ref_dists, dtype=object)[k], ok)], dtype=object)
    return cells, dists


_L = {}


def _load(crop):
    """District rows, inputs and the fitted calendar of one crop (as phase23_ours_field._load, calendar part only)."""
    if crop not in _L:
        from phase2_growth import load_growth
        from phase2_phenology import CROPS, Calendar
        key = MODEL_KEY[crop]
        m, d = load_growth(key)
        stages, gate, vp, _ = CROPS[key]
        cal = Calendar(len(stages), stages.index(gate), vp,
                       json.load(open(os.path.join("data", "processed", f"phenology_model_{key}.json"))))
        _L[crop] = dict(m=m, d=d, cal=cal, stages=stages,
                        row={(r.district, int(r.year)): i for i, r in m[["district", "year"]].iterrows()})
    return _L[crop]


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
    L = _load(crop)
    src = L
    i = L["row"].get((district, year))
    if i is None and crop in INPUT_FROM:                  # grain maize: district data end 2021 -> same district-year
        src = _load(INPUT_FROM[crop])                     # inputs (sowing day, day length) of silage maize
        i = src["row"].get((district, year))
    if i is None:
        return None
    m, d = src["m"], src["d"]
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
    out_dir = os.path.join(DOCS, f"{area}_{year}")
    weeks = np.array(json.load(open(os.path.join(out_dir, "series.json")))["weeks"], dtype="datetime64[D]")
    fl = pd.read_csv(os.path.join("data", "processed", "field_v2", f"{area}_{year}", "fields_nrw.csv"))
    cell = pd.read_csv(os.path.join(FD, "fields_hyras_cell.csv"))
    cell = cell[cell.VALIDFROM == year].set_index("ID")["cell"]
    dist = pd.read_csv(os.path.join(FD, "fields_district.csv"), dtype={"district": str})
    dref = pd.read_csv(os.path.join(FD, "fields_district.csv"), dtype={"district": str})
    dref = dref[dref.VALIDFROM == year].drop_duplicates("ID").set_index("ID")
    dist = dref["district"]
    out = dict(weeks=[str(w) for w in weeks], stages={}, curves={}, fields={})
    fl = fl.assign(cal=fl["crop"].map(calendar_crop))
    # maize / potato fields are not in the two tables (built for wheat / barley): cell and district of the nearest
    # wheat / barley field within 1.5 km (centroids, UTM 32N)
    miss = fl[fl["cal"].notna() & ~fl["field_id"].isin(cell.index)]["field_id"].values
    if len(miss):
        import geopandas as gpd
        from areas import bounds
        g = gpd.read_file(os.path.join("data", "processed", "fields_rur_allcrops.gpkg"), bbox=bounds(area))
        g = g[(g["VALIDFROM"] == year) & g["ID"].isin(miss)].drop_duplicates("ID").set_index("ID")
        ref = dref[dref.index.isin(cell.index)]
        cen = g.geometry.centroid
        c2, d2 = nearest_assign(np.column_stack([cen.x, cen.y]), ref[["x", "y"]].values, cell[ref.index].values,
                                ref["district"].values)
        ok = c2 >= 0
        cell = pd.concat([cell, pd.Series(c2[ok], index=g.index[ok])])
        dist = pd.concat([dist, pd.Series(d2[ok], index=g.index[ok])])
    for crop in [k for _, k, _ in CALENDARS]:
        out["stages"][crop] = CROPS[MODEL_KEY[crop]][0]
        for fid in fl[fl["cal"] == crop].field_id:
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
