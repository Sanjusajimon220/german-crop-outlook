"""Phase 4, step 5: field model v5 on the YieldSAT wheat fields (outside NRW), before any yield look.

Per field (data/raw/yieldsat/wheat_metadata.csv, 188 wheat fields 2016-2022):
  district      VG250 district containing the field centroid -> district model settings and
                district weather for light / ET0 level; district model prediction ("blend")
  sowing        the farmer's sowing date (184 fields; 4 fields: 290 days before harvest)
  weather       own HYRAS 1 km cell (tmin, tmax, rain; phase0_hyras_points.py)
  sunlight      own SARAH-3 cell, scaled to the district radiation level the model was fitted on
                (x district mean / SARAH mean of the season); ET0 = district ET0 x (0.7 x ratio + 0.3)
                x temperature ratio (as v5)
  calendar      frozen crop calendar run on the field's own temperatures and rain
  soil          plant-available water 0-100 cm from the supplied SoilGrids layers (Saxton & Rawls
                2006, gravel removed); soil prior ratio = field PAW / district AMBAV PAW (0-100 cm),
                divided by the median of that ratio over all 188 fields (the two soil sources differ
                in level); then as v3 (log-normal, sd 0.15)
  satellite     our PROSAIL LAI from the supplied Sentinel-2 clips (phase4_yieldsat_s2.py)
  ensemble      3,000 members per field (same draws and settings as v1-v5), weights = soil prior x
                satellite likelihood (tempered, ESS >= 50)
Yield: field yield = district model (blend) x posterior physics yield / prior mean physics yield
       / R, with R = 0.962 = median district-mean ratio of v4 Rur fields 2021-2023 (the typical field
       sits below the prior mean of the widened ensemble; v1 lesson; fixed from Rur, no yields).
Writes data/processed/yieldsat/field_results_v5.csv. NO comparison with yields here
(phase4_yieldsat_score.py, pre-registered in docs/project_log.md).
"""
import json
import os
import sys
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
import zipfile

from phase2_growth import GROWTH, P, TAU, load_growth
from phase2_phenology import CROPS, Calendar
from phase4_field_assim import CROP, FieldGrowth, N_MEMBERS, OBS_ERR, draw, field_settings, wquant
from phase4_field_assim_v3 import logprior_for, weights

YS = os.path.join("data", "processed", "yieldsat")
RAW = os.path.join("data", "raw", "yieldsat")
R_TYPICAL = 0.962
DEPTHS = [(0, 5), (5, 15), (15, 30), (30, 60), (60, 100), (100, 200)]


def saxton_rawls_paw(sand, clay, om):
    """Plant-available water (m3/m3) between -33 and -1500 kPa (Saxton & Rawls 2006);
    sand, clay as fractions, om in %."""
    t1500 = -0.024 * sand + 0.487 * clay + 0.006 * om + 0.005 * sand * om - 0.013 * clay * om + 0.068 * sand * clay + 0.031
    t1500 = t1500 + (0.14 * t1500 - 0.02)
    t33 = -0.251 * sand + 0.195 * clay + 0.011 * om + 0.006 * sand * om - 0.027 * clay * om + 0.452 * sand * clay + 0.299
    t33 = t33 + (1.283 * t33 ** 2 - 0.374 * t33 - 0.015)
    return np.clip(t33 - t1500, 0.0, 0.4)


def soil_paw(meta):
    z = zipfile.ZipFile(os.path.join(RAW, "Germany_raw.zip"))
    names = z.namelist()
    out = {}
    for f in meta["field_shared_name"]:
        v = {}
        for var in ("sand", "clay", "soc", "cfvo"):
            n = [x for x in names if x.startswith(f + "/soil/" + var + "_")][0]
            with rasterio.MemoryFile(z.read(n)) as mf, mf.open() as r:
                a = r.read().astype(float)
            v[var] = np.nanmean(np.where(a <= 0, np.nan, a), axis=(1, 2))[0::2]   # mean bands per depth
        paw = 0.0
        for k, (a, b) in enumerate(DEPTHS):
            top, bot = a, min(b, 100)
            if bot <= top:
                continue
            om = min(v["soc"][k] / 100 * 1.724, 8.0)                 # dg/kg -> % OC -> % OM
            theta = saxton_rawls_paw(v["sand"][k] / 1000, v["clay"][k] / 1000, om)
            paw += theta * (1 - v["cfvo"][k] / 1000) * (bot - top) * 10   # mm
        out[f] = paw
    return pd.Series(out)


def districts(meta):
    krs = gpd.read_file("/vsizip/" + os.path.join("data", "raw", "boundaries", "vg250_12-31.utm32s.shape.ebenen.zip")
                        + "/vg250_ebenen_1231/VG250_KRS.shp", columns=["AGS", "GEN", "GF"])
    krs = krs[krs["GF"] == 4]
    pts = gpd.GeoDataFrame(meta, geometry=gpd.points_from_xy(meta["centroid_longitude_wgs84"],
                                                             meta["centroid_latitude_wgs84"]), crs="EPSG:4326").to_crs(krs.crs)
    j = gpd.sjoin(pts, krs, how="left", predicate="within")
    return j["AGS"].values, j["GEN"].values


def weather_points():
    parts = [np.load(os.path.join(YS, f"hyras_points_{y}.npz"), allow_pickle=True) for y in range(2015, 2023)]
    ids = list(parts[0]["point_ids"])
    dates = np.concatenate([p["dates"] for p in parts])
    w = {v: np.concatenate([p[v] for p in parts]) for v in ("tasmin", "tasmax", "pr")}
    return ids, dates, w


def main():
    meta = pd.read_csv(os.path.join(RAW, "wheat_metadata.csv"))
    meta["district"], meta["district_name"] = districts(meta)
    meta["sowing"] = pd.to_datetime(meta["seeding_date"], format="%d.%m.%Y", errors="coerce")
    meta["harvest"] = pd.to_datetime(meta["harvesting_date"], format="%d.%m.%Y", errors="coerce")
    meta["paw_sg"] = meta["field_shared_name"].map(soil_paw(meta))
    print(meta.groupby(["district", "district_name"]).size().to_string(), flush=True)

    m, d = load_growth(CROP)
    row = {(r.district, r.year): i for i, r in m[["district", "year"]].iterrows()}
    soil = pd.read_csv(os.path.join(P, "soil_districts.csv"), dtype={"district": str}).set_index("district")
    ambav = sum((soil[f"fc_{k}"] - soil[f"wp_{k}"]).clip(lower=0) * 100 for k in range(1, 11))   # mm, 0-100 cm
    meta["soil_ratio_raw"] = meta["paw_sg"] / meta["district"].map(ambav)
    meta["soil_ratio"] = meta["soil_ratio_raw"] / meta["soil_ratio_raw"].median()
    print(f"SoilGrids PAW 0-100 cm: {np.round(meta['paw_sg'].quantile([.1, .5, .9]).values)} mm; "
          f"raw ratio to district median {meta['soil_ratio_raw'].median():.2f}", flush=True)

    pred = pd.read_csv(os.path.join(P, f"growth_predictions_{CROP}.csv"), dtype={"district": str})
    blend = pred.set_index(["district", "year"])["blend"]
    ids, wdates, w = weather_points()
    sar = np.load(os.path.join(P, "weather", "sarah3_germany.npz"))
    wd = np.load(os.path.join(P, "weather_daily.npz"))
    wcodes = {c: i for i, c in enumerate(wd["codes"])}
    obs = pd.read_csv(os.path.join(YS, "retrieval_fields.csv"), parse_dates=["date"])
    obs = obs[~obs["date"].dt.month.isin([12, 1])]
    settings = field_settings()
    stages, gate, vp, horizon = CROPS[CROP]
    g_ = GROWTH[CROP]
    cal = Calendar(len(stages), stages.index(gate), vp, json.load(open(os.path.join(P, f"phenology_model_{CROP}.json"))))
    ie, ih, im = (stages.index(g_[k]) for k in ("emergence", "heading", "maturity"))
    rows = []
    t0 = time.time()
    for k, r in meta.iterrows():
        key = (r["district"], r["year"])
        if key not in row or pd.isna(r["sowing"]):
            print(f"  skipped {r['field_shared_name']}: no district row / sowing date")
            continue
        i = row[key]
        n_days = d["tmin"].shape[1]
        days = np.arange(np.datetime64(r["sowing"].date()), np.datetime64(r["sowing"].date()) + n_days)
        sub = {kk: v[i:i + 1].clone() for kk, v in d.items() if v.dim() and len(v) == len(m)}
        # own weather
        wi = np.searchsorted(wdates, days)
        p = ids.index(r["field_shared_name"])
        for kk, var in (("tmin", "tasmin"), ("tmax", "tasmax"), ("pr", "pr")):
            sub[kk] = torch.tensor(w[var][wi, p], dtype=torch.float32)[None, :]
        # district light / ET0 on the field's own days, then the field's SARAH pattern
        di = np.searchsorted(wd["dates"], days)
        c = wcodes[r["district"]]
        rs_d = wd["rsds"][di, c]
        iy = np.abs(sar["lat"] - r["centroid_latitude_wgs84"]).argmin()
        ix = np.abs(sar["lon"] - r["centroid_longitude_wgs84"]).argmin()
        si = np.searchsorted(sar["dates"], days)
        rs_f = sar["sis"][np.clip(si, 0, len(sar["dates"]) - 1), iy, ix].astype(float)
        level = np.nanmean(rs_d) / np.nanmean(rs_f)                         # keep the calibrated level
        rs = np.where(np.isfinite(rs_f), rs_f * level, rs_d)
        ratio = np.clip(rs / np.maximum(rs_d, 1.0), 0.5, 1.5)
        t_f = (sub["tmin"][0].numpy() + sub["tmax"][0].numpy()) / 2
        t_d = (wd["tasmin"][di, c] + wd["tasmax"][di, c]) / 2
        sub["par"] = torch.tensor(0.5 * rs * 0.0864, dtype=torch.float32)[None, :]
        sub["et0"] = torch.tensor(wd["et0"][di, c] * (0.7 * ratio + 0.3) *
                                  np.clip((t_f + 17.8) / (t_d + 17.8), 0.7, 1.3), dtype=torch.float32)[None, :]
        doy = (days - days.astype("datetime64[Y]")).astype(int) + 1
        decl = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
        lat = np.radians(r["centroid_latitude_wgs84"])
        sub["daylen"] = torch.tensor(24 / np.pi * np.arccos(np.clip(-np.tan(lat) * np.tan(decl), -1, 1)),
                                     dtype=torch.float32)[None, :]
        with torch.no_grad():
            dev, th = cal({"tmin": sub["tmin"], "tmax": sub["tmax"], "daylen": sub["daylen"],
                           "rain": torch.cumsum(sub["pr"], 1), "decade": sub["decade"]}, return_dev=True)
        pert = draw(N_MEMBERS, seed=int(r["district"]) + int(r["year"]))
        pt = {kk: torch.tensor(v.values, dtype=torch.float32) for kk, v in pert.items()}
        sub = {kk: v.repeat(N_MEMBERS, *([1] * (v.dim() - 1))) for kk, v in sub.items()}
        devp = dev.repeat(N_MEMBERS, 1) * pt["dev_f"][:, None]
        e, h, mt = th[0, ie], th[0, ih], th[0, im]
        sub["em"] = torch.sigmoid((devp - e) / TAU)
        sub["ph"] = torch.clamp((devp - e) / (h - e), 0, 1)
        sub["post"] = torch.sigmoid((devp - h) / TAU)
        sub["gf"] = torch.clamp((devp - h) / (mt - h), 0, 1)
        sub["mat"] = torch.sigmoid((devp - mt) / TAU)
        sub["cap"] = sub["cap"] * pt["cap_f"][:, None]
        sub["n_supply"] = pt["n_supply"]
        with torch.no_grad():
            _, out = FieldGrowth(settings, pt)(sub, g_["dry_matter"], record=True)
        tot = pd.concat([pert, pd.DataFrame({kk: v.numpy() for kk, v in out.items() if kk != "lai_daily"})], axis=1)
        mature = int((dev[0] >= mt).float().argmax())
        heading = int((dev[0] >= h).float().argmax())
        lai = out["lai_daily"].numpy()
        g = obs[obs["field"] == r["field_shared_name"]].dropna(subset=["lai", "lai_sd"]).sort_values("date")
        dd = (g["date"] - r["sowing"]).dt.days.values
        ok = (dd >= 0) & (dd <= mature) & (dd < lai.shape[1])
        e_ = {"tot": tot}
        wts = weights(lai, dd[ok], g["lai"].values[ok], g["lai_sd"].values[ok], logprior_for(e_, r["soil_ratio"]))
        rel = tot["phys_yield"].values / tot["phys_yield"].mean() / R_TYPICAL
        b = blend.get(key, np.nan)
        res = dict(field=r["field_shared_name"], district=r["district"], year=r["year"], farm=r["farm_identifier"],
                   n_obs=int(ok.sum()), ess=1 / (wts ** 2).sum(), blend=b, soil_ratio=r["soil_ratio"],
                   paw_sg=r["paw_sg"], v5_yield_t_ha=b * (wts @ rel), v5_p10=b * wquant(rel, wts, 0.1),
                   v5_p90=b * wquant(rel, wts, 0.9), lai_peak_model=wts @ tot["phys_lai_max"].values,
                   eta_mm=wts @ tot["phys_eta_mm"].values, etp_mm=wts @ tot["phys_etp_mm"].values,
                   heading_pred=(r["sowing"] + pd.Timedelta(days=heading)).date(),
                   maturity_pred=(r["sowing"] + pd.Timedelta(days=mature)).date(), harvest_farmer=r["harvest"].date())
        # satellite features for the satellite-index baseline (from observations only)
        t = (g["date"] - (r["sowing"] + pd.Timedelta(days=heading))).dt.days.values.astype(float)
        if (t <= 0).any() and (t >= 45).any():
            dd45 = np.arange(0, 46)
            res["glad_after"] = np.trapezoid(np.interp(dd45, t, g["lai"].values), dd45)
        res["lai_peak_obs"] = np.percentile(g["lai"].values, 90) if len(g) else np.nan
        rows.append(res)
        if k % 20 == 0:
            print(f"  {k + 1}/{len(meta)} fields ({time.time() - t0:.0f} s)", flush=True)
    f = pd.DataFrame(rows)
    f.to_csv(os.path.join(YS, "field_results_v5.csv"), index=False)
    print(f"\n{len(f)} fields; median satellite dates used {f['n_obs'].median():.0f}, median ESS {f['ess'].median():.0f}")
    print(f[["v5_yield_t_ha", "v5_p10", "v5_p90", "blend", "soil_ratio", "lai_peak_model", "eta_mm"]]
          .describe(percentiles=[.1, .5, .9]).round(2).to_string())
    mat_err = (pd.to_datetime(f["maturity_pred"]) - pd.to_datetime(f["harvest_farmer"])).dt.days
    print(f"predicted maturity minus farmer harvest date: median {mat_err.median():.0f} days (10-90 %: "
          f"{mat_err.quantile(.1):.0f} to {mat_err.quantile(.9):.0f})")


if __name__ == "__main__":
    main()
