"""Job 6a: crop stage per field vs DWD observations (design and rule fixed 2026-10-22 before any result; one look).

Truth: DWD annual-reporter observations 2021-2025 at stations within 15 km of the pilot, Selhausen and south areas.
Stages: wheat / barley stem extension (15), heading (18), yellow ripeness (21); maize beginning of flowering (5).
Per field: satellite date (green-up / peak / green-down of the weekly LAI curve), district calendar date, HR-VPP date.
Methods (fitted on training seasons only): calendar + k; HR-VPP + k; ours = calendar + w x (satellite - calendar) + k.
Prediction for an observation = median over the fields of that crop within 5 km. Leave one season out over 2021-2024;
PASS per crop and stage if ours has a lower RMSE (days) than both benchmarks.
Run from D:\\vista: python phase21_crop_stage.py -> data/processed/field_v2/crop_stage/ + printed result.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DWD = r"D:\vista\data\raw\dwd_phenology"
CAL = r"D:\vista\data\processed"
OUT = r"D:\vista\data\processed\field_v2\crop_stage"
STATION_KM, FIELD_M, MIN_FIELDS = 15, 5000, 3
SEASONS, TEST_SEASONS = list(range(2021, 2026)), [2021, 2022, 2023, 2024]
STAGES = {  # (crop group, stage): (DWD file crop, DWD phase id, calendar file, calendar column, satellite event, HR-VPP layer)
    ("wheat", "stem extension"): ("Winterweizen", 15, "winter_wheat", "pred_stem_extension", 0, "SOSD"),
    ("wheat", "heading"): ("Winterweizen", 18, "winter_wheat", "pred_heading", 1, "MAXD"),
    ("wheat", "yellow ripeness"): ("Winterweizen", 21, "winter_wheat", "pred_yellow_ripe", 2, "EOSD"),
    ("barley", "stem extension"): ("Wintergerste", 15, "winter_barley", "pred_stem_extension", 0, "SOSD"),
    ("barley", "heading"): ("Wintergerste", 18, "winter_barley", "pred_heading", 1, "MAXD"),
    ("barley", "yellow ripeness"): ("Wintergerste", 21, "winter_barley", "pred_yellow_ripe", 2, "EOSD"),
    ("maize", "flowering"): ("Mais", 5, "silage_maize", "pred_flowering", 1, "MAXD"),
}
W_GRID = np.round(np.arange(0, 1.0001, 0.05), 2)


def doy(date, year):
    """Day of year relative to 1 January of the harvest year (1 = 1 Jan; autumn before it negative: 31 Dec = -1)."""
    d = int((np.datetime64(date, "D") - np.datetime64(f"{year}-01-01")).astype(int))
    return d + 1 if d >= 0 else d


def near(xy, pt, radius):
    return np.flatnonzero(np.hypot(xy[:, 0] - pt[0], xy[:, 1] - pt[1]) <= radius)


def _med_blend(cal, sat, w):
    """Median over each observation's fields of calendar + w x (satellite - calendar); scalars work too."""
    return np.array([np.median(np.asarray(c) + w * (np.asarray(s) - np.asarray(c))) for c, s in zip(cal, sat)])


def fit_weight(cal, sat, obs):
    best = None
    for w in W_GRID:
        r = np.asarray(obs, float) - _med_blend(cal, sat, w)
        k = r.mean()
        rmse = np.sqrt(np.mean((r - k) ** 2))
        if best is None or rmse < best[0] - 1e-9:
            best = (rmse, float(w), float(k))
    return best[1], best[2]


def predict(cal, sat, w, k):
    return _med_blend(cal, sat, w) + k


def _stats(err):
    e = np.asarray(err, float)
    return {"n": int(len(e)), "rmse": float(np.sqrt(np.mean(e ** 2))) if len(e) else np.nan,
            "mae": float(np.mean(np.abs(e))) if len(e) else np.nan, "bias": float(np.mean(e)) if len(e) else np.nan}


def loso_scores(rows):
    """rows: season, obs, cal, sat, hrvpp (scalars or per-field arrays). Errors = prediction - observation."""
    err = {"calendar": [], "hrvpp": [], "ours": []}
    weights = []
    for h in sorted(rows.season.unique()):
        tr, te = rows[rows.season != h], rows[rows.season == h]
        if tr.empty or te.empty:
            continue
        med = lambda col, d: np.array([np.median(np.asarray(v)) for v in d[col]])
        kc = float(np.mean(tr.obs - med("cal", tr)))
        kh = float(np.mean(tr.obs - med("hrvpp", tr)))
        w, k = fit_weight(list(tr.cal), list(tr.sat), tr.obs.values)
        weights.append(w)
        err["calendar"] += list(med("cal", te) + kc - te.obs.values)
        err["hrvpp"] += list(med("hrvpp", te) + kh - te.obs.values)
        err["ours"] += list(predict(list(te.cal), list(te.sat), w, k) - te.obs.values)
    out = {m: _stats(e) for m, e in err.items()}
    out["weights"] = weights
    return out


# ---- real-data run ----
def _stations():
    from pyproj import Transformer
    from areas import AREAS
    st = pd.read_csv(os.path.join(DWD, "PH_Beschreibung_Phaenologie_Stationen_Jahresmelder.txt"), sep=";", encoding="latin-1", skipinitialspace=True)
    st.columns = [c.strip() for c in st.columns]
    x, y = Transformer.from_crs(4326, 25832, always_xy=True).transform(st["geograph.Laenge"].astype(float).values, st["geograph.Breite"].astype(float).values)
    st = st.assign(x=x, y=y, Stations_id=st.Stations_id.astype(int))
    out = []
    for area, (x0, y0) in AREAS.items():
        d = np.hypot(st.x - (x0 + 4000), st.y - (y0 + 4000)) / 1000
        out.append(st[d <= STATION_KM].assign(area=area, km=d[d <= STATION_KM]))
    return pd.concat(out)[["Stations_id", "Stationsname", "x", "y", "area", "km"]]


def _observations(crop_file, phase):
    d = pd.concat([pd.read_csv(f, sep=";", skipinitialspace=True) for f in glob.glob(os.path.join(DWD, f"PH_Jahresmelder_Landwirtschaft_Kulturpflanze_{crop_file}_*.txt"))])
    d.columns = [c.strip() for c in d.columns]
    d = d[(d.Phase_id == phase) & d.Referenzjahr.between(SEASONS[0], SEASONS[-1])].drop_duplicates(["Stations_id", "Referenzjahr", "Phase_id"])
    dates = pd.to_datetime(d.Eintrittsdatum.astype(str), format="%Y%m%d").values.astype("datetime64[D]")
    return d.assign(obs=[doy(t, y) for t, y in zip(dates, d.Referenzjahr)])[["Stations_id", "Referenzjahr", "obs"]]


def _field_dates(area, year):
    """Per field of this area-season: group, centroid, district, satellite dates (green-up, peak, green-down) and HR-VPP dates."""
    import rasterio
    from phase16_hrvpp_items import _field_median_px, _field_table, _st_dates, decode_yydoy, grid_index, season_timing, HRVPP
    F = _field_table(area, year)
    fd = pd.read_csv(os.path.join(CAL, "field", "fields_district.csv"), dtype={"district": str})
    fd = fd[fd.VALIDFROM == year].drop_duplicates("ID").set_index("ID")
    rows = []
    sat = np.full((len(F["fids"]), 3), np.nan)
    for k in range(len(F["fids"])):
        t = season_timing(F["lai"][:, k], F["mask"][:, k])
        if t is not None:
            sat[k] = [doy(F["weeks"][i], year) for i in t]
    hv = {}
    if year <= 2024:
        st = _st_dates(area)
        with rasterio.open(st[sorted(st)[0]]) as s0:
            T, shape, epsg = s0.transform, s0.shape, s0.crs.to_epsg()
        r, c = grid_index(F["x"], F["y"], T, shape, 25832, epsg)
        inside = r >= 0
        layers = {}
        for p in ("SOSD", "MAXD", "EOSD", "QFLAG"):
            fs = glob.glob(os.path.join(HRVPP, area, f"VPP_{year}_*_s1_{p}.tif"))
            with rasterio.open(fs[0]) as a:
                arr = a.read(1)
            v = np.full(len(r), -1, np.int64); v[inside] = arr[r[inside], c[inside]]
            layers[p] = v
        good = layers["QFLAG"] >= 2
        for p in ("SOSD", "MAXD", "EOSD"):
            dd = decode_yydoy(np.maximum(layers[p], 0))
            dvals = np.array([doy(x, year) if not np.isnat(x) else np.nan for x in dd], float)
            hv[p] = _field_median_px(np.where(good & (layers[p] > 0), dvals, np.nan), F["inv"], len(F["fids"]))
    # centroid from the field's own pixels where the district table (cereal fields only) has no entry; district of the
    # nearest listed field (bug fix 2026-10-22: maize fields were dropped)
    n = len(F["fids"])
    cx = np.bincount(F["inv"], F["x"], n) / np.maximum(np.bincount(F["inv"], minlength=n), 1)
    cy = np.bincount(F["inv"], F["y"], n) / np.maximum(np.bincount(F["inv"], minlength=n), 1)
    listed = fd[fd.index.isin(F["fids"])]
    for k, f in enumerate(F["fids"]):
        if f in fd.index:
            x, y, dist = fd.at[f, "x"], fd.at[f, "y"], fd.at[f, "district"]
        elif len(listed):
            x, y = cx[k], cy[k]
            dist = listed.district.values[int(np.argmin(np.hypot(listed.x.values - x, listed.y.values - y)))]
        else:
            continue
        rows.append(dict(area=area, season=year, field=int(f), group=F["groups"][k], x=x, y=y, district=dist,
                         sat0=sat[k, 0], sat1=sat[k, 1], sat2=sat[k, 2], **{p: (hv[p][k] if hv else np.nan) for p in ("SOSD", "MAXD", "EOSD")}))
    return pd.DataFrame(rows)


def run(only=None):
    """only: list of (group, stage) to run (maize after the bug fix; the cereal results stay as first run)."""
    os.makedirs(OUT, exist_ok=True)
    st = _stations()
    st = st[st.area != "north"]
    fields = pd.concat([_field_dates(a, y) for a in ("pilot", "selhausen", "south") for y in SEASONS])
    fields.to_csv(os.path.join(OUT, "field_dates.csv.gz" if only is None else "field_dates_fixed.csv.gz"), index=False)
    print("fields", len(fields), flush=True)
    results, all_rows = [], []
    for (group, stage), (cf, ph, calf, calc, ev, hl) in STAGES.items():
        if only and (group, stage) not in only:
            continue
        cal = pd.read_csv(os.path.join(CAL, f"phenology_predictions_{calf}.csv"), dtype={"district": str}).set_index(["district", "year"])[calc]
        obs = _observations(cf, ph).merge(st, on="Stations_id")
        rows = []
        for o in obs.itertuples():
            fl = fields[(fields.area == o.area) & (fields.season == o.Referenzjahr) & (fields.group == group)]
            idx = near(fl[["x", "y"]].values, (o.x, o.y), FIELD_M)
            fl = fl.iloc[idx]
            cv = np.array([cal.get((d, o.Referenzjahr), np.nan) for d in fl.district], float)
            sv, hvv = fl[f"sat{ev}"].values, fl[hl].values
            ok = np.isfinite(cv) & np.isfinite(sv)
            if ok.sum() < MIN_FIELDS:
                continue
            hok = np.isfinite(hvv) & ok
            rows.append(dict(group=group, stage=stage, station=o.Stations_id, area=o.area, season=o.Referenzjahr, obs=o.obs, n_fields=int(ok.sum()),
                             cal=cv[ok], sat=sv[ok], hrvpp=(hvv[hok] if hok.sum() >= MIN_FIELDS else np.array([np.nan]))))
        R = pd.DataFrame(rows)
        if R.empty:
            print(group, stage, "no usable observations"); continue
        test = R[R.season.isin(TEST_SEASONS) & R.hrvpp.apply(lambda a: np.isfinite(a).all())]
        s = loso_scores(test)
        ok = s["ours"]["rmse"] < s["calendar"]["rmse"] and s["ours"]["rmse"] < s["hrvpp"]["rmse"]
        # 2025: ours vs calendar, fitted on 2021-2024
        tr, te = R[R.season.isin(TEST_SEASONS)], R[R.season == 2025]
        e25 = {}
        if len(te):
            w, k = fit_weight(list(tr.cal), list(tr.sat), tr.obs.values)
            kc = float(np.mean(tr.obs - [np.median(c) for c in tr.cal]))
            e25 = {"n": len(te), "rmse_ours": float(np.sqrt(np.mean((predict(list(te.cal), list(te.sat), w, k) - te.obs) ** 2))),
                   "rmse_calendar": float(np.sqrt(np.mean((np.array([np.median(c) for c in te.cal]) + kc - te.obs) ** 2)))}
        results.append(dict(group=group, stage=stage, n=s["ours"]["n"], rmse_ours=s["ours"]["rmse"], rmse_calendar=s["calendar"]["rmse"],
                            rmse_hrvpp=s["hrvpp"]["rmse"], mae_ours=s["ours"]["mae"], bias_ours=s["ours"]["bias"],
                            weights="/".join(f"{w:.2f}" for w in s["weights"]), result="PASS" if ok else "fail", **{f"y2025_{k}": v for k, v in e25.items()}))
        all_rows.append(R.assign(cal=R.cal.apply(np.median), sat=R.sat.apply(np.median), hrvpp=R.hrvpp.apply(np.median)))
        print(group, stage, results[-1], flush=True)
    res = pd.DataFrame(results)
    tag = "" if only is None else "_" + "_".join(g for g, _ in only)
    res.to_csv(os.path.join(OUT, f"crop_stage_results{tag}.csv"), index=False)
    pd.concat(all_rows).to_csv(os.path.join(OUT, f"crop_stage_observations{tag}.csv"), index=False)
    print(res[["group", "stage", "n", "rmse_ours", "rmse_calendar", "rmse_hrvpp", "weights", "result"]].round(2).to_string(index=False))
    print("CROP STAGE DONE")


if __name__ == "__main__":
    run([("maize", "flowering")] if sys.argv[1:2] == ["maize"] else None)
