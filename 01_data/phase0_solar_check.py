"""Which sunlight level is right? DWD pyranometer stations (measured daily global radiation,
open data, data/raw/dwd_solar) vs SARAH-3 (satellite) and vs our district radiation
(weather_daily.npz: HYRAS grid to 2020 - itself built from these stations, so not independent -
and the sunshine-hour estimate from 2021). Period 2015-09-01 - 2026-08-31, daily, all stations
inside Germany box. Run `python phase0_solar_check.py`.
"""
import glob
import io
import os
import sys
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd

RAW = os.path.join("data", "raw", "dwd_solar")


def stations():
    rows = []
    for z in glob.glob(os.path.join(RAW, "tageswerte_ST_*_row.zip")):
        with zipfile.ZipFile(z) as f:
            name = [n for n in f.namelist() if n.startswith("produkt")][0]
            d = pd.read_csv(io.BytesIO(f.read(name)), sep=";", skipinitialspace=True, na_values=[-999])
        d["date"] = pd.to_datetime(d["MESS_DATUM"].astype(str))
        d["obs"] = d["FG_STRAHL"] * 10000 / 86400          # J/cm2 per day -> W/m2 daily mean
        rows.append(d[["STATIONS_ID", "date", "obs"]].dropna())
    obs = pd.concat(rows)
    meta = pd.read_fwf(os.path.join(RAW, "ST_Tageswerte_Beschreibung_Stationen.txt"), skiprows=[1],
                       encoding="latin1", widths=[5, 9, 9, 15, 12, 10, 41, 98], header=0)
    meta.columns = ["id", "von", "bis", "height", "lat", "lon", "name", "state"]
    return obs[obs["date"] >= "2015-09-01"], meta


def main():
    obs, meta = stations()
    sar = np.load(os.path.join("data", "processed", "weather", "sarah3_germany.npz"))
    w = np.load(os.path.join("data", "processed", sys.argv[1] if len(sys.argv) > 1 else "weather_daily.npz"))
    krs = gpd.read_file("/vsizip/" + os.path.join("data", "raw", "boundaries", "vg250_12-31.utm32s.shape.ebenen.zip")
                        + "/vg250_ebenen_1231/VG250_KRS.shp", columns=["AGS", "GF"])
    krs = krs[krs["GF"] == 4]
    m = meta[meta["id"].isin(obs["STATIONS_ID"].unique())].copy()
    pts = gpd.GeoDataFrame(m, geometry=gpd.points_from_xy(m["lon"], m["lat"]), crs="EPSG:4326").to_crs(krs.crs)
    m["district"] = gpd.sjoin(pts, krs, how="left", predicate="within")["AGS"].values
    codes = {c: i for i, c in enumerate(w["codes"])}
    wd = pd.DatetimeIndex(w["dates"])
    sd = pd.DatetimeIndex(sar["dates"])
    out = []
    for r in m.itertuples():
        o = obs[obs["STATIONS_ID"] == r.id].set_index("date")["obs"]
        iy, ix = np.abs(sar["lat"] - r.lat).argmin(), np.abs(sar["lon"] - r.lon).argmin()
        if not (sar["lat"].min() <= r.lat <= sar["lat"].max() and sar["lon"].min() <= r.lon <= sar["lon"].max()):
            continue
        s = pd.Series(sar["sis"][:, iy, ix].astype(float), index=sd)
        dist = pd.Series(w["rsds"][:, codes[r.district]], index=wd) if r.district in codes else pd.Series(dtype=float)
        df = pd.DataFrame({"obs": o, "sarah": s, "district": dist}).dropna(subset=["obs", "sarah"])
        df["id"], df["name"] = r.id, r.name
        out.append(df.reset_index(names="date"))
    d = pd.concat(out)
    d["period"] = np.where(d["date"].dt.year <= 2020, "2015-2020 (district = HYRAS grid)", "2021-2026 (district = estimate)")
    for per, g in d.groupby("period"):
        print(f"\n{per}: {g['id'].nunique()} stations, {len(g)} station-days, measured mean {g['obs'].mean():.1f} W/m2")
        for k in ("sarah", "district"):
            gg = g.dropna(subset=[k])
            e = gg[k] - gg["obs"]
            print(f"  {k:9s}: mean {gg[k].mean():6.1f}, bias {e.mean():+5.1f} W/m2 ({100 * e.mean() / gg['obs'].mean():+.1f} %), "
                  f"RMSE {np.sqrt((e ** 2).mean()):5.1f}, r {gg[k].corr(gg['obs']):.3f}")
    a = d[d["name"].str.contains("Aachen")]
    if len(a):
        print("\nAachen-Orsbach (next to the Rur region):")
        for k in ("sarah", "district"):
            aa = a.dropna(subset=[k])
            print(f"  {k:9s}: bias {(aa[k] - aa['obs']).mean():+5.1f} W/m2, r {aa[k].corr(aa['obs']):.3f} (n {len(aa)})")
    d.to_csv(os.path.join("data", "processed", "weather", "solar_station_check.csv"), index=False)


if __name__ == "__main__":
    main()
