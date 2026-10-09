"""Phase 0: link yields, weather, soil and crop stages into one dataset per crop.

Outputs in data/processed/:
  weather_daily.npz     daily weather 1999-2025 for all 400 districts, every variable
                        complete: tas, tasmax, tasmin (C), hurs (%), pr (mm), rsds (W/m2,
                        HYRAS until 2020, station sunshine after), wind2 (m/s at 2 m),
                        et0 (mm/day, FAO-56 grass reference evaporation)
  master_<crop>.csv     one row per district-year with a yield: yield (t/ha), soil water,
                        crop-stage dates and monthly weather summaries
  quality_report.txt    what is complete and what is missing

Run `python phase0_master.py`.
"""
import os
import zipfile

import numpy as np
import pandas as pd
import shapefile

from weather_districts import BOUNDARIES, inside, utm32

P = os.path.join("data", "processed")
YEARS = range(1979, 2026)
CROPS = {   # crop: (phenology file, winter crop?, stages {name: DWD phase id}, summary months)
    "winter_wheat": ("winter_wheat", True, {"sowing": 10, "emergence": 12, "stem_extension": 15,
                     "heading": 18, "milk_ripe": 19, "yellow_ripe": 21, "harvest": 24}, range(3, 8)),
    "winter_barley": ("winter_barley", True, {"sowing": 10, "emergence": 12, "stem_extension": 15,
                      "heading": 18, "yellow_ripe": 21, "harvest": 24}, range(3, 7)),
    "grain_maize": ("maize", False, {"sowing": 10, "emergence": 12, "tassel": 65,
                    "flowering": 5, "milk_ripe": 19, "dough_ripe": 20, "harvest": 24}, range(5, 11)),
    "silage_maize": ("maize", False, {"sowing": 10, "emergence": 12, "tassel": 65,
                     "flowering": 5, "milk_ripe": 19, "dough_ripe": 20, "harvest": 24}, range(5, 10)),
    "potato": ("potato", False, {"sowing": 10, "emergence": 12, "canopy_closed": 13,
               "flowering": 5, "harvest": 24}, range(4, 10)),
}
NEIGHBOUR_KM = 50.0


def daily_weather(lat):
    """Concatenate the yearly files, fill sunlight after 2020 from stations, add wind and ET0.
    lat: latitude of each district's centre (degrees), in the order of the weather files."""
    path = os.path.join(P, "weather_daily.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    parts = [np.load(os.path.join(P, "weather", f"weather_{y}.npz")) for y in YEARS]
    w = {k: np.concatenate([p[k] for p in parts]) for k in ("dates", "tas", "tasmax", "tasmin",
                                                             "hurs", "pr", "rsds")}
    st = np.load(os.path.join(P, "weather", "stations.npz"))
    assert (st["dates"] == w["dates"]).all() and (st["codes"] == parts[0]["codes"]).all()
    from_station = np.isnan(w["rsds"])
    w["rsds"] = np.where(from_station, st["rsds_angstrom"], w["rsds"]).astype(np.float32)
    w["rsds_from_station"] = from_station.all(1)
    w["wind2"] = (st["wind_10m"] * 4.87 / np.log(67.8 * 10 - 5.42)).astype(np.float32)  # FAO-56 eq. 47
    w["et0"] = fao56_et0(w, st["dates"], lat).astype(np.float32)
    w["codes"] = parts[0]["codes"]
    np.savez(path, **w)
    return w


def fao56_et0(w, dates, lat):
    """FAO-56 Penman-Monteith grass reference evaporation (mm/day) per day and district.
    Uses each district's latitude; air pressure and clear-sky radiation assume sea level."""
    tmax, tmin, tmean = w["tasmax"], w["tasmin"], w["tas"]

    def es(t):
        return 0.6108 * np.exp(17.27 * t / (t + 237.3))
    ea = w["hurs"] / 100.0 * (es(tmax) + es(tmin)) / 2
    vpd = (es(tmax) + es(tmin)) / 2 - ea
    delta = 4098 * es(tmean) / (tmean + 237.3) ** 2
    gamma = 0.0674
    rs = w["rsds"] * 0.0864                                   # W/m2 -> MJ/m2/day
    doy = (dates - dates.astype("datetime64[Y]")).astype(int) + 1
    phi = np.radians(np.asarray(lat))[None, :]
    dr = (1 + 0.033 * np.cos(2 * np.pi * doy / 365))[:, None]
    decl = (0.409 * np.sin(2 * np.pi * doy / 365 - 1.39))[:, None]
    ws = np.arccos(-np.tan(phi) * np.tan(decl))
    ra = (24 * 60 / np.pi) * 0.0820 * dr * (ws * np.sin(phi) * np.sin(decl)
                                            + np.cos(phi) * np.cos(decl) * np.sin(ws))
    rso = 0.75 * ra
    rnl = (4.903e-9 * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2
           * (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0))) * (1.35 * np.clip(rs / rso, 0.3, 1) - 0.35))
    rn = 0.77 * rs - rnl
    u2 = w["wind2"]
    return np.maximum((0.408 * delta * rn + gamma * 900 / (tmean + 273) * u2 * vpd)
                      / (delta + gamma * (1 + 0.34 * u2)), 0)


def district_of_points(lat, lon, codes):
    """Index into `codes` of the district containing each point (-1 if none)."""
    with zipfile.ZipFile(BOUNDARIES) as z:
        stem = "vg250_ebenen_1231/VG250_KRS"
        sf = shapefile.Reader(shp=z.open(stem + ".shp"), shx=z.open(stem + ".shx"),
                              dbf=z.open(stem + ".dbf"), encoding="utf-8")
        records, shapes = sf.records(), sf.shapes()
    x, y = utm32(np.asarray(lat), np.asarray(lon))
    out = np.full(len(x), -1)
    position = {c: k for k, c in enumerate(codes)}
    for rec, shp in zip(records, shapes):
        rec = rec.as_dict()
        if rec.get("GF") != 4 or rec["AGS"] not in position:
            continue
        x0, y0, x1, y1 = shp.bbox
        box = np.flatnonzero((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1))
        if not len(box):
            continue
        pts = np.array(shp.points)
        hit = np.zeros(len(box), bool)
        for start, end in zip(shp.parts, list(shp.parts[1:]) + [len(pts)]):
            hit ^= inside(x[box], y[box], pts[start:end])
        out[box[hit]] = position[rec["AGS"]]
    return out


def district_centres(codes):
    g = np.load(os.path.join(P, "district_grid_soil.npz"))
    import netCDF4
    with netCDF4.Dataset(os.path.join("data", "raw", "soil", "AG_SOILINFO_THETAFC.nc")) as d:
        lat, lon = d["lat"][:].ravel(), d["lon"][:].ravel()
    idx = g["index"].ravel()
    ok = idx >= 0
    n = len(g["codes"])
    count = np.bincount(idx[ok], minlength=n)
    clat = np.bincount(idx[ok], weights=lat[ok], minlength=n) / count
    clon = np.bincount(idx[ok], weights=lon[ok], minlength=n) / count
    order = [list(g["codes"]).index(c) for c in codes]
    return clat[order], clon[order]


def stage_dates(crop, file, winter, stages, codes, clat, clon):
    """Per district and harvest year: median date of each stage over the district's stations,
    else over stations within NEIGHBOUR_KM of the district centre. Returns two DataFrames
    (day of year relative to 1 Jan of the harvest year; autumn sowing is negative) and source."""
    p = pd.read_csv(os.path.join(P, f"phenology_{file}.csv"), encoding="utf-8")
    p = p[p["phase_id"].isin(stages.values())].copy()
    # harvest year: for winter crops, autumn sowing and emergence count for the next year
    autumn = winter & p["phase_id"].isin([stages["sowing"], stages["emergence"]]) & (p["doy"] > 200)
    p["harvest_year"] = p["year"] + autumn.astype(int)
    p["rel_doy"] = np.where(autumn, p["doy"] - 365, p["doy"])
    p = p[p["harvest_year"].isin(YEARS)]
    # spring and summer stages of winter crops cannot occur in autumn: drop reports after day 250
    if winter:
        spring = ~p["phase_id"].isin([stages["sowing"], stages["emergence"]])
        p = p[~(spring & (p["doy"] > 250))]
    # drop implausible reports: more than 45 days from that year's national median
    med = p.groupby(["harvest_year", "phase_id"])["rel_doy"].transform("median")
    p = p[(p["rel_doy"] - med).abs() <= 45]
    st = p.groupby("station")[["lat", "lon"]].first()
    st["district"] = district_of_points(st["lat"].values, st["lon"].values, codes)
    p = p.join(st["district"], on="station")

    rows = []
    name_of = {v: k for k, v in stages.items()}
    by = p.groupby(["harvest_year", "phase_id"])
    for (year, phase), g in by:
        own = g[g["district"] >= 0].groupby("district")["rel_doy"].median()
        sx = (g["lon"].values[None, :] - clon[:, None]) * 111.2 * np.cos(np.radians(clat))[:, None]
        sy = (g["lat"].values[None, :] - clat[:, None]) * 111.2
        near = np.hypot(sx, sy) <= NEIGHBOUR_KM
        for k, code in enumerate(codes):
            if k in own.index:
                rows.append((code, year, name_of[phase], own[k], "district"))
            elif near[k].any():
                rows.append((code, year, name_of[phase], float(np.median(g["rel_doy"].values[near[k]])),
                             "neighbours"))
    t = pd.DataFrame(rows, columns=["district", "year", "stage", "doy", "source"])
    doy = t.pivot_table(index=["district", "year"], columns="stage", values="doy").add_prefix("doy_")
    src = t.pivot_table(index=["district", "year"], columns="stage", values="source",
                        aggfunc="first").add_prefix("src_")
    return doy, src


def monthly_features(w, codes, months, winter):
    """Per district and harvest year: monthly mean temperature, rain, sunlight, ET0, hot days."""
    dates = pd.DatetimeIndex(w["dates"])
    feats = []
    for year in YEARS:
        cols = {}
        for m in months:
            sel = (dates.year == year) & (dates.month == m)
            cols[f"tas_m{m:02d}"] = w["tas"][sel].mean(0)
            cols[f"pr_m{m:02d}"] = w["pr"][sel].sum(0)
            cols[f"rsds_m{m:02d}"] = w["rsds"][sel].mean(0)
            cols[f"et0_m{m:02d}"] = w["et0"][sel].sum(0)
            cols[f"hot_m{m:02d}"] = (w["tasmax"][sel] > 30).sum(0)
        if winter:   # autumn and winter before the harvest year
            sel = ((dates.year == year - 1) & (dates.month >= 10)) | ((dates.year == year) & (dates.month <= 2))
            if sel.any():
                cols["tas_winter"] = w["tas"][sel].mean(0)
                cols["pr_winter"] = w["pr"][sel].sum(0)
                cols["frost_days_winter"] = (w["tasmin"][sel] < -10).sum(0)
        f = pd.DataFrame(cols)
        f["district"], f["year"] = codes, year
        feats.append(f)
    return pd.concat(feats).set_index(["district", "year"])


def main():
    codes = np.load(os.path.join(P, "weather", "weather_1999.npz"))["codes"]
    clat, clon = district_centres(codes)
    w = daily_weather(clat)
    yields = pd.read_csv(os.path.join(P, "yields_long.csv"), dtype={"district": str})   # 1979-2025
    soil = pd.read_csv(os.path.join(P, "soil_districts.csv"), dtype={"district": str}).set_index("district")
    report = [f"daily weather: {len(w['dates'])} days x {len(codes)} districts; "
              f"sunlight from stations on {w['rsds_from_station'].sum()} days (2021-2025); "
              f"missing values: " + ", ".join(f"{k} {np.isnan(w[k]).mean():.2%}" for k in
                                              ("tas", "tasmax", "tasmin", "hurs", "pr", "rsds", "wind2", "et0"))]
    et0_year = pd.Series(w["et0"].mean(1)).groupby(pd.DatetimeIndex(w["dates"]).year).sum()
    report.append(f"Germany-mean reference evaporation (ET0) per year: {et0_year.min():.0f}-"
                  f"{et0_year.max():.0f} mm (2018: {et0_year[2018]:.0f} mm)")

    for crop, (file, winter, stages, months) in CROPS.items():
        y = yields[(yields["crop"] == crop) & yields["district"].isin(codes)]
        y = y.set_index(["district", "year"])[["yield_t_ha", "source"]].rename(columns={"source": "yield_source"})
        y = y.join(soil[["name"]], on="district")
        doy, src = stage_dates(crop, file, winter, stages, codes, clat, clon)
        feats = monthly_features(w, codes, months, winter)
        m = (y.join(soil[["paw_1m_mm", "paw_2m_mm"]], on="district")
              .join(doy).join(src).join(feats))
        # sunlight is the one input whose source changes between years
        m["sunlight_source"] = np.where(m.index.get_level_values("year") <= 2020,
                                        "HYRAS grid", "DWD stations (sunshine hours)")
        m.reset_index().to_csv(os.path.join(P, f"master_{crop}.csv"), index=False)

        lines = [f"\n{crop}: {len(m)} district-years with a yield, {m.index.get_level_values(0).nunique()} "
                 f"districts, {m.index.get_level_values(1).min()}-{m.index.get_level_values(1).max()}"]
        for s in stages:
            col = f"doy_{s}"
            if col in m:
                have = m[col].notna()
                own = (m.get(f"src_{s}") == "district").mean()
                lines.append(f"  stage {s:15s} known for {have.mean():6.1%} (from own district "
                             f"{own:6.1%}), median day {m[col].median():6.0f}")
        lines.append(f"  weather and soil complete for {m[[c for c in feats.columns]].notna().all(axis=1).mean():.1%} "
                     f"of rows")
        report += lines
    text = "\n".join(report)
    print(text)
    open(os.path.join(P, "quality_report.txt"), "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()
