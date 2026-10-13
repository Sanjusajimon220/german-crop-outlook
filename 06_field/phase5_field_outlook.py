"""Field Outlook v1 (phase 2 pilot): winter wheat, every field in the Rur region, seasons 2021-2025 (2026 when
its Sentinel-2 data are in). Design from the lessons of the field tests (project log 2026-10-06/07):
the DISTRICT model gives the yield level, the SATELLITE gives the pattern between fields.

Per field and season:
  satellite      LAI from our PROSAIL retrieval (retrieval_115_<year>.csv); heading from the district crop
                 calendar; glad_after = green leaf area duration 0-45 days after heading (LAI x days),
                 lai_peak, lai_heading, stay_green_days, ccc_heading (phase4_field_features.features)
  yield          district v12 forecast at the typical harvest date (ranges_v12_final, lead 0: median and 80 %
                 range) x relative factor:
                   rel = 1 + B x (glad / area-weighted district mean glad - 1), clipped 0.7-1.3
                 B = 0.41: slope of relative yield on relative glad within district-years of the YieldSAT
                 fields (162 fields, 13 district-years, r 0.40; research data, CC BY-NC-ND - used here for a
                 research pilot only, to be re-estimated on patchCROP / farm data for any product).
                 rel is rescaled so the area-weighted district mean equals the district forecast exactly.
                 Field 80 % range = district range x rel, widened by the field-level spread (sd 0.12 of
                 relative yield not explained by the index).
  water          field model v4 (own 1 km weather and BK50 soil, satellite-weighted ensemble): season water
                 demand (potential transpiration + soil evaporation, mm), water use (actual, mm), supply ratio
  quality        number of clear satellite dates; 'index' = which satellite signal set the pattern
Outputs data/processed/field_outlook/field_outlook_wheat_<year>.csv and .geojson (WGS84, simplified 5 m).
Run `python phase5_field_outlook.py 2021 2022 2023 2024 2025`.
"""
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

from phase2_growth import P
from phase4_field_assim import observations
from phase4_field_features import district_dates, features

B, CLIP, SD_FIELD, Z = 0.41, (0.7, 1.3), 0.12, 1.2816
FD, OUT = os.path.join(P, "field"), os.path.join(P, "field_outlook")


def anchor(year):
    r = pd.read_csv(os.path.join(P, "ranges_v12_final", "ranges_winter_wheat_lead00.csv"), dtype={"region": str})
    r = r[(r["level"] == "district") & (r["year"] == year)].set_index("region")
    return r[["median", "low80", "high80", "obs"]]


def main(years):
    os.makedirs(OUT, exist_ok=True)
    for y in years:
        dates = district_dates(y)
        obs = observations(y).sort_values("date")
        rows = []
        for (fid, dist), g in obs.groupby(["field_id", "district"]):
            if dist in dates:
                rows.append({"field_id": fid, "district": dist, "n_dates": len(g),
                             "heading": dates[dist][0].date(), "maturity": dates[dist][1].date(),
                             **{k: v for k, v in features(g, dates[dist][0]).items() if k != "n_dates"}})
        f = pd.DataFrame(rows)
        geo = gpd.read_file(os.path.join(P, "s2", f"fields_115_{y}.geojson"))
        geo = geo.rename(columns={"ID": "field_id", "AREA_HA": "area_ha", "INNER_HA": "inner_ha"})
        area = geo.set_index("field_id")["area_ha"] if "area_ha" in geo else None
        f["area_ha"] = f["field_id"].map(area)
        v4p = os.path.join(FD, f"field_results_v4_winter_wheat_{y}.csv")
        if os.path.exists(v4p):
            v4 = pd.read_csv(v4p)[["field_id", "phys_etp_mm", "phys_eta_mm", "phys_lai_max", "area_ha"]]
            f = f.merge(v4.rename(columns={"area_ha": "area_v4"}), on="field_id", how="left")
            f["area_ha"] = f["area_ha"].fillna(f["area_v4"])
            f = f.drop(columns="area_v4").rename(columns={"phys_etp_mm": "water_demand_mm", "phys_eta_mm": "water_use_mm",
                                                         "phys_lai_max": "model_lai_max"})
            f["water_supply"] = f["water_use_mm"] / f["water_demand_mm"]
        # satellite index: glad_after, else lai_peak (scaled to glad units by the district ratio)
        f["index"] = np.where(f["glad_after"].notna(), "glad_after", np.where(f["lai_peak"].notna(), "lai_peak", "none"))
        w = f["area_ha"].fillna(1.0)
        rel = pd.Series(np.nan, index=f.index)
        for col in ("glad_after", "lai_peak"):
            m = f[col].notna()
            mean = (f.loc[m, col] * w[m]).groupby(f.loc[m, "district"]).sum() / w[m].groupby(f.loc[m, "district"]).sum()
            r = (f[col] / f["district"].map(mean)).where(rel.isna())
            rel = rel.fillna(r)
        f["rel_index"] = rel
        f["rel"] = np.clip(1 + B * (f["rel_index"].fillna(1.0) - 1), *CLIP)
        f["rel"] = f["rel"] / f["district"].map((f["rel"] * w).groupby(f["district"]).sum() / w.groupby(f["district"]).sum())
        a = anchor(y)
        f["district_forecast"], f["district_official"] = f["district"].map(a["median"]), f["district"].map(a["obs"])
        f["yield_t_ha"] = f["district_forecast"] * f["rel"]
        f["yield_official_level"] = f["district_official"] * f["rel"]     # after the season: same pattern, official level
        half_lo = f["district_forecast"] - f["district"].map(a["low80"])
        half_hi = f["district"].map(a["high80"]) - f["district_forecast"]
        extra = Z * SD_FIELD * f["district_forecast"]
        f["yield_low80"] = f["yield_t_ha"] - np.sqrt((half_lo * f["rel"]) ** 2 + extra ** 2)
        f["yield_high80"] = f["yield_t_ha"] + np.sqrt((half_hi * f["rel"]) ** 2 + extra ** 2)
        f["season"] = y
        f.to_csv(os.path.join(OUT, f"field_outlook_wheat_{y}.csv"), index=False)
        g = geo[["field_id", "geometry"]].merge(f, on="field_id")
        g["geometry"] = g.geometry.simplify(5)
        keep = ["field_id", "district", "area_ha", "yield_t_ha", "yield_official_level", "yield_low80", "yield_high80", "rel", "glad_after",
                "lai_peak", "n_dates", "water_demand_mm", "water_use_mm", "water_supply", "index", "geometry"]
        g = g[[c for c in keep if c in g.columns]].to_crs(4326)
        for c in g.columns:
            if g[c].dtype.kind == "f":
                g[c] = g[c].round(3)
        g.to_file(os.path.join(OUT, f"field_outlook_wheat_{y}.geojson"), driver="GeoJSON", COORDINATE_PRECISION=6)
        # checks
        chk = f.dropna(subset=["yield_t_ha", "area_ha"]).groupby("district").apply(
            lambda d: np.average(d["yield_t_ha"], weights=d["area_ha"]) - d["district_forecast"].iloc[0], include_groups=False)
        print(f"{y}: {len(f)} fields, {f['area_ha'].sum():.0f} ha; index glad {100 * (f['index'] == 'glad_after').mean():.0f} %, "
              f"lai_peak {100 * (f['index'] == 'lai_peak').mean():.0f} %, none {100 * (f['index'] == 'none').mean():.0f} %; "
              f"yield 10/50/90 % {np.round(f['yield_t_ha'].quantile([.1, .5, .9]).values, 2)}; "
              f"max |district mean - forecast| {chk.abs().max():.4f}; "
              f"water supply median {f['water_supply'].median() if 'water_supply' in f else float('nan'):.2f}", flush=True)


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]])
