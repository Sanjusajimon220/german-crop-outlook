"""Job 2, methods B and D: our field crop model PER FIELD (design 2026-10-22 with amendments 1-2).

The field's own HYRAS 1 km cell (tmin / tmax / rain) drives development (frozen crop calendar recomputed for the cell)
and growth; light and reference ET stay from the district (as field model v4). 3,000 members per district and season
as tested on 15 October (phase4 draw; variant B2: dev_f stretched to 0.88-1.12). Soil prior from BK50 (field
plant-available water / district median of the same crop, as v3). Wheat: v7 settings + the satellite-fitted LAI shape;
barley: the district v7 barley settings (no barley LAI shape was fitted).
Fields of the same crop, district and cell share one run. Output per run: data/processed/field_v2/ours/<year>/
<crop>_<district>_<cell>.npz (weekly LAI of the members as float16 on WOFOST's Mondays, cap_f, sowing, maturity day,
own_weather flag). Resumable. Run detached from D:\\vista: python phase23_ours_field.py [processes]
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "02_district_model"))

FD = os.path.join("data", "processed", "field")
P = os.path.join("data", "processed")
OUT = os.path.join("data", "processed", "field_v2", "ours")
CROP_KEYS = {"wheat": "winter_wheat", "barley": "winter_barley"}
CODES = {"wheat": 115, "barley": 131}
DEV_STRETCH = 2.0          # B2: dev_f 0.94-1.06 -> 0.88-1.12


def weekly_from_daily(lai, sowing, weeks):
    """Members x days (day 0 = sowing) -> members x weeks; 0 before sowing and beyond the model horizon."""
    day = (weeks.astype("datetime64[D]") - np.datetime64(pd.Timestamp(sowing).date(), "D")).astype(int)
    ok = (day >= 0) & (day < lai.shape[1])
    out = np.zeros((lai.shape[0], len(weeks)))
    out[:, ok] = lai[:, day[ok]]
    return out


def nfk_ratio(soil, code):
    """Field plant-available water / district median of the fields of the same crop (well mapped fields only)."""
    s = soil[(soil["CODE"] == code) & (soil["share_mapped"] >= 0.5) & (soil["nfk_mm"] > 0)]
    r = s["nfk_mm"] / s.groupby("district")["nfk_mm"].transform("median")
    return pd.Series(r.values, index=pd.MultiIndex.from_arrays([s["ID"], s["VALIDFROM"]]))


def soil_ratios():
    s = pd.read_csv(os.path.join(FD, "fields_soil_bk50.csv"))
    d = pd.read_csv(os.path.join(FD, "fields_district.csv"), dtype={"district": str})[["ID", "VALIDFROM", "district"]]
    s = s.merge(d.drop_duplicates(["ID", "VALIDFROM"]), on=["ID", "VALIDFROM"])
    return {c: nfk_ratio(s, code) for c, code in CODES.items()}


# ---- model runs ----
_C = {}


def _load(crop):
    if crop not in _C:
        from phase2_growth import GROWTH, load_growth
        from phase2_phenology import CROPS, Calendar
        key = CROP_KEYS[crop]
        m, d = load_growth(key)
        s = json.load(open(os.path.join(P, f"growth_model_{key}_v7.json")))
        shape = os.path.join(P, f"growth_lai_shape_{key}.json")
        if os.path.exists(shape):
            s.update(json.load(open(shape)))
        stages, gate, vp, _ = CROPS[key]
        g = GROWTH[key]
        cal = Calendar(len(stages), stages.index(gate), vp, json.load(open(os.path.join(P, f"phenology_model_{key}.json"))))
        _C[crop] = dict(m=m, d=d, settings=s, cal=cal, dry=g["dry_matter"],
                        idx=[stages.index(g[k]) for k in ("emergence", "heading", "maturity")],
                        row={(r.district, int(r.year)): i for i, r in m[["district", "year"]].iterrows()})
    return _C[crop]


def _weather(year):
    if ("w", year) not in _C:
        from phase4_field_assim_v4 import cell_weather
        dates, cells, w = cell_weather(year)
        _C[("w", year)] = (dates, {c: i for i, c in enumerate(cells)}, w)
    return _C[("w", year)]


def enkf_hook(blocks, n, rng):
    """E3: blocks of n members (one block per cut); blocks[j] = {day after sowing: (observed LAI, error variance)}.
    On those days the block's LAI is updated by the stochastic EnKF (phase22 enkf_update); other states unchanged."""
    from phase22_wofost_field import enkf_update

    def hook(day, lai):
        out = None
        for j, obs in enumerate(blocks):
            if day in obs:
                out = lai.clone() if out is None else out
                s = slice(j * n, (j + 1) * n)
                out[s] = torch.tensor(enkf_update(lai[s].numpy().astype(float), *obs[day], rng), dtype=lai.dtype)
        return lai if out is None else out
    return hook


def sowing_of(crop, year, district):
    L = _load(crop)
    i = L["row"].get((district, year))
    return None if i is None else pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=round(L["m"].at[i, "doy_sowing"]) - 1)


def field_ensemble(crop, year, district, cell, n=3000, pert=None, hook=None):
    """Prior ensemble of one crop / district / HYRAS cell: daily LAI (members x days from sowing) and members.
    pert: members to use instead of the district draw (E3); hook: LAI update during the run (E3 Kalman filter)."""
    from phase2_growth import TAU
    from phase4_field_assim import FieldGrowth, draw
    L = _load(crop)
    m, d = L["m"], L["d"]
    i = L["row"].get((district, year))
    if i is None:
        return None
    dates, col, w = _weather(year)
    sow = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=round(m.at[i, "doy_sowing"]) - 1)
    first = int((np.datetime64(sow.date()) - dates[0]).astype(int))
    n_days = d["tmin"].shape[1]
    sub = {k: v[i:i + 1].clone() for k, v in d.items() if v.dim() and len(v) == len(m)}
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
    if pert is None:
        pert = draw(n, seed=int(district) + year)
        pert["dev_f"] = 1 + (pert["dev_f"] - 1) * DEV_STRETCH
    n = len(pert)
    pt ={k: torch.tensor(v.values, dtype=torch.float32) for k, v in pert.items()}
    sub = {k: v.repeat(n, *([1] * (v.dim() - 1))) for k, v in sub.items()}
    devp = dev.repeat(n, 1) * pt["dev_f"][:, None]
    e, h, mt = (th[0, j] for j in L["idx"])
    sub["em"] = torch.sigmoid((devp - e) / TAU)
    sub["ph"] = torch.clamp((devp - e) / (h - e), 0, 1)
    sub["post"] = torch.sigmoid((devp - h) / TAU)
    sub["gf"] = torch.clamp((devp - h) / (mt - h), 0, 1)
    sub["mat"] = torch.sigmoid((devp - mt) / TAU)
    sub["cap"] = sub["cap"] * pt["cap_f"][:, None]
    sub["n_supply"] = pt["n_supply"]
    model = FieldGrowth(L["settings"], pt)
    model.lai_hook = hook
    with torch.no_grad():
        _, out = model(sub, L["dry"], record=True)
    return dict(lai=out["lai_daily"].numpy(), sowing=sow, mature_day=int((dev[0] >= mt).float().argmax()),
                cap_f=pert["cap_f"].values, own_weather=own)


def out_path(k):
    return os.path.join(OUT, str(k["year"]), f"{k['crop']}_{k['district']}_{k['cell']}.npz")


def run_key(k):
    path = out_path(k)
    if os.path.exists(path):
        return "skip"
    from phase22b_wofost_run import mondays
    e = field_ensemble(k["crop"], k["year"], k["district"], k["cell"])
    if e is None:
        return f"error {k}: no district row"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path + ".part.npz", weeks=mondays(k["year"]),
                        lai=weekly_from_daily(e["lai"], e["sowing"], mondays(k["year"])).astype(np.float16),
                        cap_f=e["cap_f"], sowing=str(e["sowing"].date()), mature_day=e["mature_day"],
                        own_weather=e["own_weather"])
    os.replace(path + ".part.npz", path)
    return "done"


def _work(k):
    torch.set_num_threads(1)
    try:
        return run_key(k)
    except Exception as ex:
        return f"error {k}: {type(ex).__name__}: {str(ex)[:150]}"


def _log(msg):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "run_log.txt"), "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M ") + msg + "\n")


def keys():
    t = pd.DataFrame(json.load(open(os.path.join("data", "processed", "field_v2", "wofost", "tasks.json"))))
    return t[["crop", "year", "district", "cell"]].drop_duplicates().to_dict("records")


if __name__ == "__main__":
    os.chdir(r"D:\vista")
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    ks = keys()
    for crop in CROP_KEYS:            # one pool per crop: load_growth configures the crop per process
        todo = [k for k in ks if k["crop"] == crop and not os.path.exists(out_path(k))]
        _log(f"{crop}: {len(todo)} crop / district / cell runs to do")
        t0, n = time.time(), 0
        with Pool(nproc) as pool:
            for r in pool.imap_unordered(_work, todo, chunksize=4):
                n += 1
                if r.startswith("error"):
                    _log(r)
                if n % 100 == 0 or n == len(todo):
                    el = time.time() - t0
                    _log(f"{crop} {n}/{len(todo)} done, {el / 3600:.2f} h, about {el / n * (len(todo) - n) / 3600:.2f} h left")
    _log("OURS RUN DONE")
