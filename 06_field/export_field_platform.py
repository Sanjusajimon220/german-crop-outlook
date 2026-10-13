"""Field Outlook v1 -> compact web data for the field map (platform/fields/data/).

Per season one JSON: {"cols": [...], "rows": [[props..., ring coordinates]], "districts": {...}} with outlines
simplified to 8 m and coordinates rounded to 5 decimals (about 1 m), largest polygon part only.
"""
import json
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

P = os.path.join("data", "processed")
SRC, OUT = os.path.join(P, "field_outlook"), os.path.join("platform", "fields", "data")
COLS = ["field_id", "district", "area_ha", "yield_t_ha", "yield_low80", "yield_high80", "rel", "glad_after",
        "lai_peak", "n_dates", "water_demand_mm", "water_use_mm", "water_supply", "district_forecast", "district_official", "yield_official_level", "prev_crop", "prev2_crop", "organic"]


def r(v, d):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), d)


def txt(v):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else str(v)


def main(years):
    os.makedirs(OUT, exist_ok=True)
    names = json.load(open(os.path.join("platform", "data", "meta.json"), encoding="utf-8"))["names"]   # clean umlauts
    k = gpd.read_file(os.path.join(P, "field", "krs_nrw.geojson")).to_crs(25832)
    for y in years:
        f = pd.read_csv(os.path.join(SRC, f"field_outlook_wheat_{y}.csv"), dtype={"district": str})
        g = gpd.read_file(os.path.join(P, "s2", f"fields_115_{y}.geojson")).rename(columns={"ID": "field_id"})
        g = g[["field_id", "geometry"]].to_crs(25832)
        g["geometry"] = g.geometry.simplify(8)
        g = g.merge(f, on="field_id").to_crs(4326)
        rows = []
        for _, x in g.iterrows():
            geom = x.geometry
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type != "Polygon":        # MultiPolygon / GeometryCollection: largest polygon part
                parts = [p for p in getattr(geom, "geoms", []) if p.geom_type == "Polygon"]
                if not parts:
                    continue
                geom = max(parts, key=lambda p: p.area)
            ring = [[round(a, 5), round(b, 5)] for a, b in geom.exterior.coords]
            rows.append([int(x["field_id"]), x["district"], r(x["area_ha"], 2), r(x["yield_t_ha"], 2), r(x["yield_low80"], 2),
                         r(x["yield_high80"], 2), r(x["rel"], 3), r(x.get("glad_after"), 1), r(x.get("lai_peak"), 2),
                         int(x["n_dates"]), r(x.get("water_demand_mm"), 0), r(x.get("water_use_mm"), 0),
                         r(x.get("water_supply"), 3), r(x["district_forecast"], 2), r(x["district_official"], 2), r(x["yield_official_level"], 2),
                         txt(x.get("prev_crop")), txt(x.get("prev2_crop")), bool(x.get("organic") is True or str(x.get("organic")).lower() == "true"), ring])
        used = sorted(set(f["district"]))
        kd = k[k["ARS"].isin(used)].dissolve("ARS").reset_index()
        kd["geometry"] = kd.geometry.simplify(200)
        kd = kd.to_crs(4326)
        dist = {}
        for _, x in kd.iterrows():
            polys = x.geometry.geoms if x.geometry.geom_type == "MultiPolygon" else [x.geometry]
            dist[x["ARS"]] = {"name": names.get(x["ARS"], x["GEN"]), "rings": [[[round(a, 4), round(b, 4)] for a, b in p.exterior.coords] for p in polys]}
        json.dump({"season": y, "cols": COLS + ["ring"], "rows": rows, "districts": dist},
                  open(os.path.join(OUT, f"wheat_{y}.json"), "w"), separators=(",", ":"))
        print(y, len(rows), "fields", round(os.path.getsize(os.path.join(OUT, f"wheat_{y}.json")) / 1e6, 2), "MB", flush=True)


def seasons():
    have = sorted(int(f[6:10]) for f in os.listdir(OUT) if f.startswith("wheat_") and f.endswith(".json"))
    json.dump(have, open(os.path.join(OUT, "seasons.json"), "w"))
    print("seasons:", have)


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]])
    seasons()
