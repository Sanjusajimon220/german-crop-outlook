"""Phase 1: the WOFOST benchmark for district yields.

1. Development speed: on a sample of training district-years, scale WOFOST's temperature sums
   TSUM1 (sowing/emergence -> flowering) and TSUM2 (flowering -> maturity) by 0.7-1.3 and keep
   the factors that best reproduce the observed DWD stage dates (wheat/barley: heading + 5 days
   ~ flowering, yellow ripeness ~ maturity; maize: flowering). Potato is not tuned (its
   flowering is rarely observed).
2. Simulate every district-year (water-limited yield, Wofost72_WLP_CWB) with observed sowing
   dates; results are saved as they come (data/processed/wofost_<crop>.csv), so an interrupted
   run continues where it stopped.
3. Link to official yields on training rows only:
      WOFOST          = trend benchmark + c x (WOFOST yield - the district's mean WOFOST yield
                        1999-2017)
      WOFOST + weather = the same plus the ridge weather correction of phase 1
   and score exactly like phase1_baselines.py.

  python phase1_wofost.py calibrate <crop>     step 1
  python phase1_wofost.py simulate <crop>      step 2 (resumable)
  python phase1_wofost.py score                step 3 for all simulated crops
"""
import datetime as dt
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from phase1_baselines import DROUGHT, Trend, scores
from wofost_runner import WOFOST_CROPS, run_season

P = os.path.join("data", "processed")
CALIB = os.path.join(P, "wofost_calibration.json")
STAGES = {"winter_wheat": ("doy_heading", 5, "doy_yellow_ripe"),
          "winter_barley": ("doy_heading", 5, "doy_yellow_ripe"),
          "silage_maize": ("doy_flowering", 0, None),
          "grain_maize": ("doy_flowering", 0, None)}
SCALES = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3)
WORKERS = 6


def as_date(year, doy):
    """Harvest year + day of year (negative = autumn before) -> date."""
    return dt.date(int(year), 1, 1) + dt.timedelta(days=int(round(doy)) - 1)


def table(crop):
    m = pd.read_csv(os.path.join(P, f"master_{crop}.csv"), dtype={"district": str})
    split = pd.read_csv(os.path.join(P, "split.csv"), dtype={"district": str})
    m = m.merge(split, on=["district", "year"])
    for col in ("doy_sowing", "doy_harvest"):          # missing dates -> district median, else national
        m[col] = m[col].fillna(m.groupby("district")[col].transform("median")).fillna(m[col].median())
    return m


def season_args(crop, row, scale=(1.0, 1.0), phenology_only=False):
    sowing = as_date(row.year, row.doy_sowing)
    harvest = as_date(row.year, row.doy_harvest)
    if harvest <= sowing + dt.timedelta(days=60):
        harvest = sowing + dt.timedelta(days=150)
    return (row.district, crop, sowing, harvest, scale, phenology_only)


def _run(args):
    try:
        return run_season(*args)
    except Exception as e:                               # one failed season must not stop the run
        return {"error": str(e)[:200]}


def calibrate(crop):
    if crop not in STAGES:
        print(f"{crop}: development speed not tuned (default WOFOST values)")
        return
    m = table(crop)
    flower_col, offset, ripe_col = STAGES[crop]
    need = [flower_col] + ([ripe_col] if ripe_col else [])
    t = m[(m["role"] == "train") & m[need].notna().all(axis=1)].sample(250, random_state=1)

    def error(scale, which):
        res = pool.map(_run, [season_args(crop, r, scale, True) for r in t.itertuples()])
        key, col, off = (("anthesis", flower_col, offset) if which == 1 else ("maturity", ripe_col, 0))
        err = [(r[key] - as_date(row.year, getattr(row, col) + off)).days
               for r, row in zip(res, t.itertuples()) if r.get(key)]
        return np.sqrt(np.mean(np.square(err))), np.mean(err), len(err)

    with Pool(WORKERS) as pool:
        s1 = {s: error((s, 1.0), 1) for s in SCALES}
        best1 = min(s1, key=lambda s: s1[s][0])
        best2 = 1.0
        if ripe_col:
            s2 = {s: error((best1, s), 2) for s in SCALES}
            best2 = min(s2, key=lambda s: s2[s][0])
    print(f"{crop}: flowering error by TSUM1 scale: " +
          ", ".join(f"{s}: {v[0]:.0f} d" for s, v in s1.items()))
    print(f"  chosen TSUM1 x {best1} (flowering error {s1[best1][0]:.1f} days, bias {s1[best1][1]:+.1f})")
    if ripe_col:
        print(f"  chosen TSUM2 x {best2} (maturity error {s2[best2][0]:.1f} days, bias {s2[best2][1]:+.1f})")
    calib = json.load(open(CALIB)) if os.path.exists(CALIB) else {}
    calib[crop] = [best1, best2]
    json.dump(calib, open(CALIB, "w"), indent=1)


def simulate(crop, chunk=600):
    calib = json.load(open(CALIB)) if os.path.exists(CALIB) else {}
    scale = tuple(calib.get(crop, [1.0, 1.0]))
    out = os.path.join(P, f"wofost_{crop}.csv")
    done = set()
    if os.path.exists(out):
        d = pd.read_csv(out, dtype={"district": str})
        done = set(zip(d["district"], d["year"]))
    m = table(crop)
    todo = [r for r in m.itertuples() if (r.district, r.year) not in done]
    print(f"{crop}: {len(done)} done, {len(todo)} to simulate (TSUM scales {scale})", flush=True)
    start = time.perf_counter()
    with Pool(WORKERS) as pool:
        for i in range(0, len(todo), chunk):
            part = todo[i:i + chunk]
            res = pool.map(_run, [season_args(crop, r, scale) for r in part])
            rows = [{"district": r.district, "year": r.year, "sim_yield": x.get("yield"),
                     "sim_tagp": x.get("TAGP"), "sim_laimax": x.get("LAImax"),
                     "sim_anthesis": x.get("anthesis"), "sim_maturity": x.get("maturity"),
                     "error": x.get("error")} for r, x in zip(part, res)]
            pd.DataFrame(rows).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
            print(f"  {i + len(part)}/{len(todo)} in {time.perf_counter() - start:.0f} s", flush=True)


def score():
    """Score WOFOST and compare it with the statistical benchmarks on identical district-years
    (seasons WOFOST could not simulate, e.g. winter crops sown before the weather data begins,
    are left out for every model)."""
    from phase1_baselines import MODELS, Ridge, features
    rows = []
    for crop in WOFOST_CROPS:
        path = os.path.join(P, f"wofost_{crop}.csv")
        if not os.path.exists(path):
            continue
        sim = pd.read_csv(path, dtype={"district": str}).drop_duplicates(["district", "year"])
        m = table(crop).merge(sim, on=["district", "year"])
        missing = len(table(crop)) - len(m)
        failed = m["error"].notna() | m["sim_yield"].isna()
        m = m[~failed].reset_index(drop=True)
        # WOFOST's deviation from the district's own normal (simulated 1999-2017; no yields needed)
        normal = m[m["year"] <= 2017].groupby("district")["sim_yield"].mean()
        m["sim_anom"] = m["sim_yield"] - m["district"].map(normal).fillna(m["sim_yield"].mean())
        train = m["role"] == "train"
        trend = Trend().fit(m.loc[train, "district"], m.loc[train, "year"].values,
                            m.loc[train, "yield_t_ha"].values)
        m["trend_w"] = trend.predict(m["district"], m["year"].values)
        resid = m["yield_t_ha"] - m["trend_w"]
        c = np.polyfit(m.loc[train, "sim_anom"], resid[train], 1)
        m["wofost"] = m["trend_w"] + np.polyval(c, m["sim_anom"])
        cols = [col for col in features(m, m[train]) if not col.startswith("sim_") and col != "error"]
        ridge = Ridge().fit(m.loc[train, cols + ["sim_anom"]].values.astype(float), resid[train].values, 100)
        m["wofost+weather"] = m["trend_w"] + ridge.predict(m[cols + ["sim_anom"]].values.astype(float))
        stats = pd.read_csv(os.path.join(P, f"predictions_{crop}.csv"), dtype={"district": str})
        m = m.merge(stats[["district", "year"] + list(MODELS)], on=["district", "year"])
        m.to_csv(os.path.join(P, f"wofost_predictions_{crop}.csv"), index=False)

        names = list(MODELS) + ["wofost", "wofost+weather"]
        print(f"\n{crop}: {len(m)} district-years compared ({failed.sum()} not simulable, "
              f"{missing} not yet simulated); observed t/ha per simulated t/ha {c[0] * 1000:.2f}, "
              f"correlation with observed deviation (training) "
              f"{np.corrcoef(m.loc[m.role == 'train', 'sim_anom'], (m.yield_t_ha - m.trend)[m.role == 'train'])[0, 1]:.2f}")
        print(f"  {'data':20s} {'n':>5s}" + "".join(f"{n:>15s}" for n in names) + "   RMSE t/ha")
        groups = [("test: new years", m["role"] == "test_years"),
                  ("test: new regions", m["role"] == "test_regions"),
                  ("test: both", m["role"] == "test_both"),
                  ("test: drought years", (m["role"] != "train") & m["year"].isin(DROUGHT))]
        for label, sel in groups:
            obs, tr = m.loc[sel, "yield_t_ha"].values, m.loc[sel, "trend"].values
            line = f"  {label:20s} {sel.sum():5d}"
            for name in names:
                rmse, rel, skill = scores(obs, m.loc[sel, name].values, tr)
                line += f"{rmse:15.2f}"
                rows.append(dict(crop=crop, data=label, n=int(sel.sum()), model=name,
                                 rmse=round(rmse, 3), rel_rmse=round(rel, 1), weather_skill=round(skill, 3)))
            print(line)
    pd.DataFrame(rows).to_csv(os.path.join(P, "phase1_comparison.csv"), index=False)


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "calibrate":
        calibrate(sys.argv[2])
    elif step == "simulate":
        simulate(sys.argv[2])
    else:
        score()
