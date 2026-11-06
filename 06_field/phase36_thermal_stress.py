"""Thermal STRESS ALERTS (detection, not amounts), pre-registered docs/project_log.md 2026-11-06.

On each usable thermal date (v4 rules, LAI >= 1): thermal stress index TSI = ET_thermal / (Kc_pot x ET0_model),
Kc_pot = Kcb(LAI) + 0.05. ALERT if TSI < 0.7. Truth: tower index from LE_CORR (ruling before results) < 0.7 with the
tower's own ET0. Comparators on the same dates: v2 alert (Ks < 0.9) and a simple rule (10-day rain < 5 mm AND ET0 >= 3).
JUDGED: DE-RuS 2021-2025 (thermal vs tower never looked at). DE-Kli / DE-Geb reported, not deciding.
Run from D:\\vista: python phase36_thermal_stress.py -> data/processed/field_v2/thermal_stress/
"""
import os

import numpy as np
import pandas as pd

OUT = os.path.join("data", "processed", "field_v2", "thermal_stress")
THRESHOLD = 0.7
RUS_YEARS = [2021, 2022, 2023, 2024, 2025]
OTHER = [("dekli", 2018), ("dekli", 2021), ("dekli", 2023), ("dekli", 2025), ("degeb", 2021), ("degeb", 2022),
         ("degeb", 2024)]


# ------------------------------------------------------------------------------------------------- pure functions
def scores(alert, truth):
    alert, truth = np.asarray(alert, bool), np.asarray(truth, bool)
    n_s, n_n = truth.sum(), (~truth).sum()
    hit = alert[truth].mean() if n_s else np.nan
    fa = alert[~truth].mean() if n_n else np.nan
    ba = (hit + (1 - fa)) / 2 if n_s and n_n else np.nan
    return dict(n=int(len(truth)), stressed=int(n_s), alerts=int(alert.sum()), hit_rate=float(hit),
                false_alarm=float(fa), balanced_accuracy=float(ba))


def simple_rule(rain, et0):
    """Rain over the previous 10 days (incl. the day) < 5 mm AND ET0 >= 3 mm that day."""
    r10 = rain.rolling(10, min_periods=10).sum()
    return (r10 < 5.0) & (et0 >= 3.0)


def stress_index(et, lai, et0, crop):
    from phase29_water_v2 import KCB_MID, kcb_from_lai
    kc = kcb_from_lai(np.atleast_1d(lai), KCB_MID[crop]) + 0.05
    out = np.asarray(et, float) / (kc * np.asarray(et0, float))
    return out if np.ndim(et) else float(out[0])


# ------------------------------------------------------------------------------------------------- DE-RuS thermal
def rus_thermal(year):
    """Thermal ET for the DE-RuS v2 field of the year (v4 code path, pixels >= 100 m inside the field)."""
    import glob
    import json
    import rasterio
    from areas import raw_dir
    from fv2 import pixels
    from phase29_water_v2 import FIELDS, SITE, _wind, site_weather
    from phase34_water_v4 import thermal_core
    fid, crop = FIELDS[year]
    px = pixels("selhausen", year)
    j = int(np.where(px["fids"] == fid)[0][0])
    m = px["inv"] == j
    f0 = sorted(glob.glob(os.path.join(raw_dir("selhausen", year), "s2", "*.tif")))[0]
    with rasterio.open(f0) as r:
        tr10, h, w = r.transform, r.height, r.width
    mask = np.zeros((h, w), bool)
    mask[px["rows"][m], px["cols"][m]] = True
    s = json.load(open(os.path.join("D:/german-crop-outlook-field/docs/field", f"selhausen_{year}", "series.json")))
    f = s["fields"][str(fid)]
    days = pd.date_range(f"{year}-03-01", f"{year}-09-30")
    lw = pd.Series([np.nan if v is None else v for v in f["lai"]], index=pd.DatetimeIndex(s["weeks"]))
    lai = lw.reindex(lw.index.union(days)).interpolate(limit_area="inside").reindex(days).fillna(0)
    W = site_weather(days)
    wind = _wind().reindex(days)
    W["u10"] = wind.fillna(wind.median()).values
    df, _ = thermal_core("selhausen", year, mask, 0, 0, tr10, lai, dict(crop=crop), W,
                         dict(lat=SITE["lat"], lon=SITE["lon"], elev=SITE["elev"]))
    return df


def table(site, year, th):
    """One row per usable thermal date with LAI >= 1 and a tower-valid day."""
    from phase34_water_v4 import usable
    from phase35_water_bias import model_season, tower_season
    m, I = model_season(site, year)
    t = tower_season(site, m.index, I["lat"], I["elev"])
    u = usable(th).copy()
    u["date"] = pd.to_datetime(u["date"])
    u = u[u["date"].isin(m.index)]
    d = m.join(t)
    d["simple"] = simple_rule(d["rain_m"], d["et0_m"])
    x = d.loc[u["date"]].assign(et_thermal=u["et_thermal"].values)
    x = x[(x["lai"] >= 1.0) & x["et_corr"].notna() & (x["n"] == 48)]
    crop = I["crop"]
    return pd.DataFrame(dict(site=site, year=year, crop=crop, date=x.index, lai=x["lai"].values,
                             tsi=stress_index(x["et_thermal"].values, x["lai"].values, x["et0_m"].values, crop),
                             tower_si=stress_index(x["et_corr"].values, x["lai"].values, x["et0_t"].values, crop),
                             tower_si_raw=stress_index(x["et_t"].values, x["lai"].values, x["et0_t"].values, crop),
                             ks_v2=x["ks_m"].values, simple=x["simple"].values.astype(bool)))


def run():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for year in RUS_YEARS:
        th = rus_thermal(year)
        th.to_csv(os.path.join(OUT, f"thermal_selhausen_{year}.csv"), index=False)
        rows.append(table("selhausen", year, th).assign(group="judged"))
        print(f"selhausen {year}: usable thermal dates {int(th['et_thermal'].notna().sum()) if 'et_thermal' in th else 0}, "
              f"with LAI >= 1 and tower day {len(rows[-1])}", flush=True)
    for site, year in OTHER:
        th = pd.read_csv(os.path.join("data", "processed", "field_v2", "water_v4", f"thermal_{site}_{year}.csv"))
        rows.append(table(site, year, th).assign(group="not_deciding"))
    T = pd.concat(rows, ignore_index=True)
    T.to_csv(os.path.join(OUT, "dates.csv"), index=False)
    L = []
    for g, d in T.groupby("group", sort=False):
        for truth_col, tag in (("tower_si", "truth LE_CORR"), ("tower_si_raw", "truth LE_F_MDS (secondary)")):
            truth = d[truth_col].values < THRESHOLD
            res = {name: scores(a, truth) for name, a in (("thermal", d["tsi"].values < THRESHOLD),
                                                          ("v2_model", d["ks_v2"].values < 0.9),
                                                          ("simple_rule", d["simple"].values))}
            L.append(f"== {g}, {tag}: {len(d)} dates, stressed {int(truth.sum())}")
            L.append(pd.DataFrame(res).T.round(3).to_string())
            if g == "judged" and truth_col == "tower_si":
                th, v2, sr = (res[k]["balanced_accuracy"] for k in ("thermal", "v2_model", "simple_rule"))
                testable = len(d) >= 15 and truth.sum() >= 3 and (~truth).sum() >= 1
                ok = testable and th >= 0.70 and th > v2 and th > sr
                L.append("RESULT (B): " + ("NOT TESTABLE (too few dates / stressed days)" if not testable else
                                           ("PASS" if ok else "FAIL")) +
                         f" (needs >= 15 dates, >= 3 stressed, thermal BA >= 0.70 and above v2 {v2:.2f} and rule {sr:.2f})")
        L.append(f"   r(TSI, tower index LE_CORR) = {np.corrcoef(d.tsi, d.tower_si)[0, 1]:.2f}")
    txt = "\n".join(L)
    open(os.path.join(OUT, "stress_log.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    run()
