"""Phase 0 addition: nitrogen (N) supply per district and year, as a model input.

Sources
  1. Batool et al. 2022 (Scientific Data 9:612; Zenodo 10.5281/zenodo.6581441): annual soil N
     surplus 1850-2019 on a 5 arcmin grid (kg N per ha of physical area), 16 estimation methods.
     Here: mean and standard deviation of the 16 methods, averaged over each district's grid cells
     (cell centre inside the district; districts without a cell take the nearest cell). 1979-2019.
  2. Destatis GENESIS 42321 (Duengemittelstatistik): domestic sales of mineral N fertiliser,
     t N per marketing year (July-June), Germany (42321-0001) and federal states (42321-0010).
     Marketing year Y/Y+1 feeds the harvest of year Y+1. Expressed as an index: sales relative to
     the state's (or Germany's) mean of the first five marketing years available.

  3. FAOSTAT Inputs/Fertilizers by Nutrient: agricultural use of N in Germany 1961-2024 (t N) and
     use per ha of cropland; identical to Destatis sales where both exist.

Output data/processed/nitrogen_districts.csv: district, year (harvest year), n_surplus_kg_ha (to
2019), n_mineral_kg_ha (state, 1979-2025), n_organic_excess_kg_ha (fixed district term) and
n_supply_kg_ha = mineral + organic excess, a coherent series for all years; and crop-specific
mineral N per ha of the crop (n_wheat_kg_ha, n_cereals_kg_ha, n_maize_kg_ha, n_potato_kg_ha) from
the crop fertilisation maps (4. figshare 10.6084/m9.figshare.25435432), used by the growth model.
"""
import glob
import io
import os
import zipfile

import netCDF4
import numpy as np
import pandas as pd

from phase0_master import district_centres, district_of_points

P = os.path.join("data", "processed")
RAW = os.path.join("data", "raw")
YEARS = range(1979, 2026)


def surplus_districts(codes):
    files = sorted(glob.glob(os.path.join(RAW, "nitrogen", "batool2022", "*grid_area*_method_*.nc")))
    stack = []
    for f in files:
        with netCDF4.Dataset(f) as d:
            lat, lon = d["latitude"][:], d["longitude"][:]
            t = d["time"]
            yrs = np.array([x.year for x in netCDF4.num2date(t[:], t.units, t.calendar)])
            iy = np.flatnonzero(yrs >= 1979)
            ia = np.flatnonzero((lat > 47) & (lat < 55.2))
            io_ = np.flatnonzero((lon > 5.8) & (lon < 15.1))
            v = d["Nsurplus"][iy, ia.min():ia.max() + 1, io_.min():io_.max() + 1]
            stack.append(np.ma.filled(v.astype(float), np.nan))
    stack = np.stack(stack)                                        # methods, years, lat, lon
    years = yrs[iy]
    la, lo = np.meshgrid(lat[ia.min():ia.max() + 1], lon[io_.min():io_.max() + 1], indexing="ij")
    owner = district_of_points(la.ravel(), lo.ravel(), codes)
    mean, sd = np.nanmean(stack, 0), np.nanstd(stack, 0)          # years, lat, lon
    mean, sd = mean.reshape(len(years), -1), sd.reshape(len(years), -1)
    valid = np.isfinite(mean).all(0)
    clat, clon = district_centres(codes)
    rows = []
    for k, code in enumerate(codes):
        cells = np.flatnonzero((owner == k) & valid)
        if len(cells) == 0:                                        # small city: nearest cell
            dist = (la.ravel() - clat[k]) ** 2 + ((lo.ravel() - clon[k]) * np.cos(np.radians(clat[k]))) ** 2
            dist[~valid] = np.inf
            cells = [int(np.argmin(dist))]
        for j, yr in enumerate(years):
            rows.append((code, int(yr), mean[j, cells].mean(), sd[j, cells].mean(), len(cells)))
    print(f"N surplus: {len(files)} methods, years {years.min()}-{years.max()}")
    return pd.DataFrame(rows, columns=["district", "year", "n_surplus_kg_ha", "n_surplus_sd", "cells"])


def sales(table):
    path = os.path.join(RAW, "fertilizer", f"{table}_de_flat.zip")
    if not os.path.exists(path):
        return None
    with zipfile.ZipFile(path) as z:
        d = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), sep=";", encoding="utf-8-sig", dtype=str)
    d = d[d["2_variable_attribute_code"] == "DMI-N"].copy()
    d["harvest_year"] = d["time"].str[:4].astype(int) + 1
    d["t_n"] = pd.to_numeric(d["value"], errors="coerce")
    d["region"] = d["1_variable_attribute_code"]
    return d[["region", "harvest_year", "t_n"]]


def national_mineral_n():
    """Mineral N per ha of cropland in Germany by harvest year (marketing year Y/Y+1 -> Y+1)."""
    with zipfile.ZipFile(os.path.join(RAW, "fertilizer", "faostat_fertilizers_nutrient_europe.zip")) as z:
        name = [n for n in z.namelist() if n.endswith("_Europe.csv")][0]
        d = pd.read_csv(io.BytesIO(z.read(name)), encoding="latin1")
    g = d[(d["Area"] == "Germany") & d["Item"].str.contains("Nutrient nitrogen")].set_index("Element")
    ycols = [c for c in g.columns if c.startswith("Y") and len(c) == 5]
    use = g.loc["Agricultural Use", ycols].astype(float)
    per_ha = g.loc["Use per area of cropland", ycols].astype(float)
    t = pd.DataFrame({"fao_year": [int(c[1:]) for c in ycols], "t_n": use.values,
                      "area_ha": (use / per_ha).values})
    de = sales("42321-0001")
    if de is not None:                       # Destatis replaces FAO where both exist
        dd = de.assign(fao_year=de["harvest_year"] - 1).set_index("fao_year")["t_n"]
        t = t.set_index("fao_year")
        t = t.reindex(sorted(set(t.index) | set(dd.index)))
        t.loc[dd.index, "t_n"] = dd
        t["area_ha"] = t["area_ha"].ffill()
        t = t.reset_index()
    t["year"] = t["fao_year"] + 1
    t["n_mineral_de_kg_ha"] = t["t_n"] / t["area_ha"]
    print(f"national mineral N: harvest years {t.year.min()}-{t.year.max()} (FAOSTAT + Destatis)")
    return t[["year", "n_mineral_de_kg_ha"]]


def state_share(st, nat):
    """How each state's mineral N sales changed relative to Germany's since the first state
    year: state_factor(y) = [state sales(y) / state sales(first)] / [German sales(y) / (first)].
    1 before the state series starts (2017); the level itself comes from the national series."""
    if st is None:
        return pd.DataFrame(columns=["state", "year", "state_factor"])
    s = st.rename(columns={"region": "state", "harvest_year": "year"}).copy()
    tot = s.groupby("year")["t_n"].sum()
    first = s["year"].min()
    s["state_factor"] = (s["t_n"] / s["state"].map(s[s["year"] == first].set_index("state")["t_n"])) /         (s["year"].map(tot) / tot[first])
    s.loc[s["state"].isin(["02", "04", "11", "10"]), "state_factor"] = 1.0   # city states + Saarland: tiny, noisy
    # sales are booked where fertiliser is sold, not where it is spread: smooth over 3 years and
    # trust only half of a state's deviation from the national change
    s = s.sort_values(["state", "year"])
    s["state_factor"] = s.groupby("state")["state_factor"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    s["state_factor"] = 1 + 0.5 * (s["state_factor"] - 1)
    full = pd.MultiIndex.from_product([sorted(s["state"].unique()), range(1979, 2026)],
                                      names=["state", "year"]).to_frame(index=False)
    full = full.merge(s[["state", "year", "state_factor"]], on=["state", "year"], how="left")
    full["state_factor"] = full["state_factor"].fillna(1.0)
    return full

GDAL = os.path.join("C:/", "Program Files", "QGIS 3.36.2", "bin", "gdal_translate.exe")
# crop groups of the fertiliser maps -> our crops whose area forms the denominator
CROP_GROUPS = {"Wheat": ["winter_wheat"],
               "Other Cereals": ["winter_barley", "spring_barley", "rye", "oats", "triticale"],
               "Maize": ["grain_maize"],       # FAO 'maize' = grain maize; silage maize is a fodder crop
               "Roots and tubers": ["potato"]}


def crop_n_rates(codes):
    """Crop-specific mineral N rate (kg N per ha of the crop) per district and year 1979-2019 from
    the 'Fertilizer application rate maps per crop and year' (figshare 10.6084/m9.figshare.25435432,
    CC0; paper: Global Crop-Specific Fertilization Dataset from 1961-2019, Scientific Data 2024).
    The maps hold kg N per ha of grid cell; the district's total N for a crop group (sum over its
    cells x cell area) is divided by the district's area of the matching crops (census years,
    linearly interpolated between them)."""
    import subprocess
    import tempfile
    src = os.path.join(RAW, "nitrogen", "crop_fertilization", "n_maps")
    tmp = os.path.join(tempfile.gettempdir(), "nmap.asc")
    owner, area_cell, rows = None, None, []
    for group in CROP_GROUPS:
        for year in range(1979, 2020):
            subprocess.run([GDAL, "-q", "-of", "AAIGrid", "-projwin", "5.75", "55.25", "15.25", "47.0",
                            os.path.join(src, f"{group}_N_{year}.tiff"), tmp], check=True)
            with open(tmp) as f:
                head = dict(next(f).split() for _ in range(6))
            v = np.loadtxt(tmp, skiprows=6)
            v[~np.isfinite(v) | (v < 0)] = 0.0
            if owner is None:
                cs = float(head["cellsize"])
                lat = float(head["yllcorner"]) + cs * (int(head["nrows"]) - 0.5 - np.arange(int(head["nrows"])))
                lon = float(head["xllcorner"]) + cs * (0.5 + np.arange(int(head["ncols"])))
                la, lo = np.meshgrid(lat, lon, indexing="ij")
                owner = district_of_points(la.ravel(), lo.ravel(), codes)
                area_cell = (cs * 111.32) ** 2 * np.cos(np.radians(la.ravel())) * 100     # ha
            total = np.bincount(owner[owner >= 0], weights=(v.ravel() * area_cell)[owner >= 0],
                                minlength=len(codes))                                   # kg N
            rows += [(c, year, group, total[k]) for k, c in enumerate(codes)]
    n = pd.DataFrame(rows, columns=["district", "year", "group", "n_kg"])
    y = pd.read_csv(os.path.join(P, "yields_long.csv"), dtype={"district": str})
    out = []
    for group, crops in CROP_GROUPS.items():
        a = y[y["crop"].isin(crops)].groupby(["district", "year"])["area_ha"].sum(min_count=1).unstack("district")
        a = a.reindex(range(1979, 2020)).interpolate(limit_direction="both")
        a = a.stack().rename("crop_area_ha").reset_index()
        g = n[n["group"] == group].merge(a, on=["district", "year"], how="left")
        # the maps place crops with their own area maps, which do not match district statistics
        # exactly (district rates scatter 108-377 kg/ha for wheat): rates are formed per state
        g["state"] = g["district"].str[:2]
        st = g.groupby(["state", "year"])[["n_kg", "crop_area_ha"]].sum()
        st = (st["n_kg"] / st["crop_area_ha"].where(st["crop_area_ha"] > 5000)).rename("rate").reset_index()
        g = g.merge(st, on=["state", "year"], how="left")
        out.append(g[["district", "year", "group", "rate"]])
    out = pd.concat(out)
    return out.pivot_table(index=["district", "year"], columns="group", values="rate").reset_index()


def main():
    codes = np.load(os.path.join("data", "processed", "weather", "weather_1999.npz"))["codes"]
    grid = pd.MultiIndex.from_product([list(codes), list(YEARS)], names=["district", "year"]).to_frame(index=False)
    out = grid.merge(surplus_districts(codes), on=["district", "year"], how="left")
    de = sales("42321-0001")
    if de is not None:
        de = de.sort_values("harvest_year")
        base = de["t_n"].head(5).mean()
        out = out.merge(de.assign(n_sales_index_de=de["t_n"] / base)[["harvest_year", "n_sales_index_de"]]
                        .rename(columns={"harvest_year": "year"}), on="year", how="left")
        print(f"Germany N sales: harvest years {de.harvest_year.min()}-{de.harvest_year.max()}")
    st = sales("42321-0010")
    if st is not None and st["harvest_year"].nunique() >= 5:
        st = st.sort_values("harvest_year")
        first = st.groupby("region")["harvest_year"].transform(lambda y: y <= sorted(y.unique())[4])
        base = st[first].groupby("region")["t_n"].mean()
        st["n_sales_index_state"] = st["t_n"] / st["region"].map(base)
        out["state"] = out["district"].str[:2]
        out = out.merge(st.rename(columns={"region": "state", "harvest_year": "year"})
                        [["state", "year", "n_sales_index_state"]], on=["state", "year"], how="left")
        out = out.drop(columns="state")
        print(f"State N sales: harvest years {st.harvest_year.min()}-{st.harvest_year.max()}")
    else:
        print("State N sales: fewer than 5 years in 42321-0010 - download again with all years")
    # Coherent N supply 1979-2025 (version 2): one national mineral-N series for all years
    # (FAOSTAT agricultural use of N, Germany, which equals Destatis domestic sales; Destatis
    # values replace FAO where both exist), adjusted per state by how its sales changed relative
    # to Germany's since 2017 (Destatis 42321-0010),
    # plus a fixed district term: how much the district's N surplus (Batool et al., mean
    # 2010-2019) exceeds its state's mean - mostly manure from livestock.
    nat = national_mineral_n()
    out = out.merge(nat, on="year", how="left")
    out["state"] = out["district"].str[:2]
    share = state_share(st if st is not None and st["harvest_year"].nunique() >= 3 else None, nat)
    out = out.merge(share, on=["state", "year"], how="left")
    out["state_factor"] = out["state_factor"].fillna(1.0)
    out["n_mineral_kg_ha"] = out["n_mineral_de_kg_ha"] * out["state_factor"]
    recent = out[out["year"].between(2010, 2019)].groupby("district")["n_surplus_kg_ha"].mean()
    state_mean = recent.groupby(recent.index.str[:2]).transform("mean")
    out["n_organic_excess_kg_ha"] = out["district"].map(recent - state_mean)
    out["n_supply_kg_ha"] = out["n_mineral_kg_ha"] + out["n_organic_excess_kg_ha"]
    out = out.drop(columns="state")
    # crop-specific mineral N (version 3, used by the growth model): state rates 1979-2019 from the
    # crop fertilisation maps; missing states -> national median of the year; 2020-2025 = the
    # 2019 rate x change of national mineral N use since 2019 (FAOSTAT/Destatis, only source)
    rates = crop_n_rates(list(codes))
    names = {"Wheat": "n_wheat_kg_ha", "Other Cereals": "n_cereals_kg_ha", "Maize": "n_maize_kg_ha",
             "Roots and tubers": "n_potato_kg_ha"}
    out = out.merge(rates.rename(columns=names), on=["district", "year"], how="left")
    nat_idx = out.groupby("year")["n_mineral_de_kg_ha"].first()
    for col in names.values():
        out[col] = out[col].fillna(out.groupby("year")[col].transform("median"))
        last = out["district"].map(out[out["year"] == 2019].set_index("district")[col])
        later = out["year"] > 2019
        out.loc[later, col] = last[later] * out.loc[later, "year"].map(nat_idx) / nat_idx[2019]
    out.to_csv(os.path.join(P, "nitrogen_districts.csv"), index=False)
    print("crop N (median kg/ha) " + "; ".join(
        f"{c}: " + ", ".join(f"{y} {out[out.year == y][c].median():.0f}" for y in (1979, 2000, 2019, 2023))
        for c in names.values()))
    s2 = out.groupby("year")["n_supply_kg_ha"].median()
    print("median district N supply (kg/ha): " + ", ".join(f"{y} {s2[y]:.0f}" for y in
                                                            (1979, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2017, 2019, 2020, 2021, 2022, 2023, 2024, 2025)))
    s = out.groupby("year")["n_surplus_kg_ha"].median()
    print("median district N surplus (kg/ha): " + ", ".join(f"{y} {s[y]:.0f}" for y in (1979, 1990, 2000, 2010, 2019)))
    print(f"wrote nitrogen_districts.csv: {out['district'].nunique()} districts, {len(out)} rows")


if __name__ == "__main__":
    main()
