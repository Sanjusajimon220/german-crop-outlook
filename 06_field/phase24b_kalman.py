"""Job 2 Kalman lines (amendments 3-4): E2 WOFOST + EnKF, E3 our field model + EnKF; 100 members per field, one run per
cut (1 April, 1 May, 1 June), LAI updated on each measured week before the cut (same weeks as phase24), then free run.
Outputs data/processed/field_v2/job2/enkf/<year>/<crop>_<field>.npz (E2) and enkf_ours/... (E3): lai [cuts x members x
weeks] float16 on WOFOST's Mondays. Resumable. Run detached from D:\\vista: python phase24b_kalman.py e3|e2 [processes]
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

N = 100
OUT = os.path.join("data", "processed", "field_v2", "job2")
DIRS = {"e2": "enkf", "e3": "enkf_ours"}


def out_path(kind, t):
    return os.path.join(OUT, DIRS[kind], str(t["year"]), f"{t['crop']}_{t['field']}.npz")


def seed_of(t):
    return t["field"] % 100000 * 10 + t["year"] % 10 + 7          # differs from the C members' seed


def cut_weeks(t):
    """Measured weeks used before each cut (phase24 split_weeks)."""
    from phase24_job2 import CUTS, split_weeks
    weeks, y = np.array(t["weeks"], dtype="datetime64[D]"), np.array(t["y"], float)
    out = []
    for cut in CUTS:
        used, _ = split_weeks(weeks, np.array(t["share"]), np.array(t["mask"], bool), np.datetime64(f"{t['year']}-{cut}"))
        out.append(np.flatnonzero(used & np.isfinite(y)))
    return weeks, y, out


def e3_members(n, seed, ratio):
    """D's perturbations (phase4 draw, dev_f 0.88-1.12); cap_f from the soil prior when the field's ratio is known."""
    from phase4_field_assim import draw
    from phase23_ours_field import DEV_STRETCH
    from phase24_job2 import SOIL_SD
    p = draw(n, seed=seed)
    p["dev_f"] = 1 + (p["dev_f"] - 1) * DEV_STRETCH
    if ratio is not None and np.isfinite(ratio):
        rng = np.random.default_rng(seed)
        p["cap_f"] = np.clip(np.clip(ratio, 0.5, 1.5) * np.exp(rng.normal(0, SOIL_SD, n)), 0.5, 1.5)
    return p


def run_e3(t):
    import torch
    import phase23_ours_field as OF
    from phase22_wofost_field import OBS_ERR
    weeks, y, used = cut_weeks(t)
    sow = OF.sowing_of(t["crop"], t["year"], t["district"])
    blocks = [{int((weeks[i] - np.datetime64(sow.date(), "D")).astype(int)): (float(y[i]), OBS_ERR ** 2) for i in u} for u in used]
    p = e3_members(N, seed_of(t), t.get("ratio"))
    p3 = pd.concat([p] * len(used), ignore_index=True)
    rng = np.random.default_rng(seed_of(t))
    e = OF.field_ensemble(t["crop"], t["year"], t["district"], t["cell"], pert=p3, hook=OF.enkf_hook(blocks, N, rng))
    return OF.weekly_from_daily(e["lai"], e["sowing"], weeks).reshape(len(used), N, len(weeks))


def run_e2(t):
    import phase22_wofost_field as W
    weeks, y, used = cut_weeks(t)
    w = W.field_weather(t["cell"], t["district"], t["year"], t["lat"])
    sow = dt.date.fromisoformat(t["sowing"])
    soil = {k: t[k] for k in ("fk_mm", "nfk_mm", "root_dm")}
    mem = W.lhs(N, seed=seed_of(t)).to_dict("records")
    rng = np.random.default_rng(seed_of(t))
    out = []
    for u in used:
        models = [W.make_model(w, W.CROP_KEYS[t["crop"]], sow, m, soil) for m in mem]
        dates = [weeks[i].astype(object) for i in u]
        out.append(W.enkf_run(models, dates, y[u], np.zeros(len(u)), weeks, rng))
    return np.array(out)


def work(args):
    kind, t = args
    path = out_path(kind, t)
    if os.path.exists(path):
        return "skip"
    try:
        import torch
        torch.set_num_threads(1)
        lai = (run_e3 if kind == "e3" else run_e2)(t)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path + ".part.npz", lai=lai.astype(np.float16), weeks=np.array(t["weeks"], dtype="datetime64[D]"))
        os.replace(path + ".part.npz", path)
        return "done"
    except Exception as e:
        return f"error {kind} {t['crop']} {t['field']} {t['year']}: {type(e).__name__}: {str(e)[:150]}"


def _log(kind, msg):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{kind}_log.txt"), "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M ") + msg + "\n")


def tasks_with_obs(kind):
    """WOFOST tasks + the field's weekly observations (so workers need no field tables)."""
    from phase16_hrvpp_items import _field_table
    import phase22b_wofost_run as WR
    import phase23_ours_field as OF
    tasks = pd.DataFrame(json.load(open(os.path.join(WR.W.OUT, "tasks.json"))))
    ratio = OF.soil_ratios()
    out = []
    for (area, year), T in tasks.groupby(["area", "year"]):
        todo = [t for t in T.to_dict("records") if not os.path.exists(out_path(kind, t))]
        if not todo:
            continue
        F = _field_table(area, year)
        pos = {int(f): k for k, f in enumerate(F["fids"])}
        for t in todo:
            k = pos.get(t["field"])
            if k is None:
                continue
            t.update(weeks=[str(w) for w in F["weeks"]], y=F["lai"][:, k].tolist(), share=F["meas"][:, k].tolist(),
                     mask=F["mask"][:, k].astype(int).tolist(), ratio=float(ratio[t["crop"]].get((t["field"], year), np.nan)))
            out.append(t)
    return out


if __name__ == "__main__":
    os.chdir(r"D:\vista")
    sys.path.insert(0, r"D:\vista")
    kind, nproc = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1
    todo = tasks_with_obs(kind)
    if kind == "e3":                    # crop by crop: load_growth configures the crop per process
        groups = [[t for t in todo if t["crop"] == c] for c in ("wheat", "barley")]
    else:
        groups = [todo]
    _log(kind, f"{len(todo)} field-seasons to run, {N} members x 3 cuts")
    t0, n = time.time(), 0
    for g in groups:
        with Pool(nproc) as pool:
            for r in pool.imap_unordered(work, [(kind, t) for t in g], chunksize=2):
                n += 1
                if r.startswith("error"):
                    _log(kind, r)
                if n % 100 == 0 or n == len(todo):
                    el = time.time() - t0
                    _log(kind, f"{n}/{len(todo)} done, {el / 3600:.2f} h, about {el / n * (len(todo) - n) / 3600:.2f} h left")
    _log(kind, f"{kind.upper()} RUN DONE")
