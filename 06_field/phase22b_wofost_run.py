"""Job 2, method C (+ E1): WOFOST per field, 1,000 Latin-hypercube members + the normal run, wheat and barley, 2024-2025.

One task = one field-season; output data/processed/field_v2/wofost/<year>/<crop>_<field>.npz (weekly LAI of the members as
float16, flowering / maturity day of year, the normal run). Existing outputs are skipped, so a stopped run continues.
Runs on 7 processes. Run detached from D:\\vista: python phase22b_wofost_run.py [n_members] ; log wofost/run_log.txt
"""
import datetime as dt
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase22_wofost_field as W

YEARS = (2024, 2025)
AREAS = ("pilot", "selhausen", "north", "south")
CODES = {"wheat": (115, "Winterweichweizen"), "barley": (131, "Wintergerste")}
N_MEMBERS = 1000


def mondays(year):
    return np.array(pd.date_range(f"{year - 1}-10-01", f"{year}-08-31", freq="W-MON").values, dtype="datetime64[D]")


def doy(d, year):
    if d is None:
        return np.nan
    x = (d - dt.date(year, 1, 1)).days
    return x + 1 if x >= 0 else x


def field_tasks():
    from phase2_growth import load_growth
    from pyproj import Transformer
    to_ll = Transformer.from_crs(25832, 4326, always_xy=True)
    cell = pd.read_csv(os.path.join(W.FD, "fields_hyras_cell.csv"))
    dist = pd.read_csv(os.path.join(W.FD, "fields_district.csv"), dtype={"district": str})
    soil = pd.read_csv(os.path.join(W.FD, "fields_soil_bk50.csv"))
    sow = {c: load_growth(W.CROP_KEYS[c])[0] for c in CODES}
    tasks = []
    for year in YEARS:
        for area in AREAS:
            fl = pd.read_csv(os.path.join("data", "processed", "field_v2", f"{area}_{year}", "fields_nrw.csv"))
            geo = json.load(open(os.path.join(r"D:\german-crop-outlook-field\docs\field", f"{area}_{year}", "fields.geojson"), encoding="utf-8"))
            partial = {int(f["properties"]["id"]) for f in geo["features"] if f["properties"].get("partial")}
            for crop, (code, name) in CODES.items():
                ids = fl[fl.crop.str.contains(name, na=False) & (fl.area_ha >= 0.5) & ~fl.field_id.isin(partial)].field_id
                for fid in ids:
                    c = cell[(cell.ID == fid) & (cell.VALIDFROM == year)]
                    d = dist[(dist.ID == fid) & (dist.VALIDFROM == year)]
                    s = soil[(soil.ID == fid) & (soil.VALIDFROM == year)]
                    if c.empty or d.empty or s.empty or s.nfk_mm.iloc[0] <= 0:
                        continue
                    m = sow[crop]
                    r = m[(m.district == d.district.iloc[0]) & (m.year == year)]
                    if r.empty:
                        continue
                    lat = to_ll.transform(d.x.iloc[0], d.y.iloc[0])[1]
                    tasks.append(dict(area=area, year=year, crop=crop, field=int(fid), cell=int(c.cell.iloc[0]),
                                      district=d.district.iloc[0], lat=float(lat),
                                      sowing=(dt.date(year, 1, 1) + dt.timedelta(days=round(r.doy_sowing.iloc[0]) - 1)).isoformat(),
                                      fk_mm=float(s.fk_mm.iloc[0]), nfk_mm=float(s.nfk_mm.iloc[0]), root_dm=float(s.root_dm.iloc[0])))
    return tasks


def out_path(t):
    return os.path.join(W.OUT, str(t["year"]), f"{t['crop']}_{t['field']}.npz")


def run_task(t, n=N_MEMBERS):
    path = out_path(t)
    if os.path.exists(path):
        return "skip"
    weeks = mondays(t["year"])
    w = W.field_weather(t["cell"], t["district"], t["year"], t["lat"])
    sow = dt.date.fromisoformat(t["sowing"])
    soil = {k: t[k] for k in ("fk_mm", "nfk_mm", "root_dm")}
    mem = W.lhs(n, seed=t["field"] % 100000 * 10 + t["year"] % 10)
    lai = np.zeros((n, len(weeks)), np.float16)
    doa, dom = np.full(n, np.nan), np.full(n, np.nan)
    for i, m in enumerate(mem.to_dict("records")):
        try:
            l, a, b = W.run_member(w, W.CROP_KEYS[t["crop"]], sow, m, soil, weeks)
            lai[i], doa[i], dom[i] = l, doy(a, t["year"]), doy(b, t["year"])
        except Exception:
            lai[i] = np.nan
    normal = {"tsum1_f": 1.0, "tsum2_f": 1.0, "sla_f": 1.0, "water_f": 1.0, "sow_shift": 0}
    l0, a0, b0 = W.run_member(w, W.CROP_KEYS[t["crop"]], sow, normal, soil, weeks)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path + ".part.npz", weeks=weeks, lai=lai, doa=doa, dom=dom, members=mem.values, member_cols=np.array(mem.columns),
                        normal_lai=l0, normal_doa=doy(a0, t["year"]), normal_dom=doy(b0, t["year"]), task=json.dumps(t))
    os.replace(path + ".part.npz", path)
    return "done"


def _log(msg):
    os.makedirs(W.OUT, exist_ok=True)
    with open(os.path.join(W.OUT, "run_log.txt"), "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M ") + msg + "\n")


def _work(t):
    try:
        return run_task(t)
    except Exception as e:
        return f"error {t['crop']} {t['field']} {t['year']}: {type(e).__name__}: {str(e)[:150]}"


if __name__ == "__main__":
    os.chdir(r"D:\vista")
    sys.path.insert(0, r"D:\vista")
    tasks = field_tasks()
    os.makedirs(W.OUT, exist_ok=True)
    with open(os.path.join(W.OUT, "tasks.json"), "w") as fh:
        json.dump(tasks, fh)
    todo = [t for t in tasks if not os.path.exists(out_path(t))]
    _log(f"{len(tasks)} field-seasons, {len(todo)} to run, {N_MEMBERS} members each")
    t0, n = time.time(), 0
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 7      # fewer processes = cooler laptop
    with Pool(nproc) as pool:
        for r in pool.imap_unordered(_work, todo, chunksize=1):
            n += 1
            if r.startswith("error"):
                _log(r)
            if n % 25 == 0 or n == len(todo):
                el = time.time() - t0
                _log(f"{n}/{len(todo)} done, {el / 3600:.1f} h, about {el / n * (len(todo) - n) / 3600:.1f} h left")
    _log("WOFOST RUN DONE")
