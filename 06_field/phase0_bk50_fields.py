"""Field soil from the NRW soil map BK50 (Geologischer Dienst NRW, (c) dl-de/by-2-0).

Raw: data/raw/soil/bk50/BK50_GPKG/BK50_A.gpkg (layer BK50, EPSG:25832, all NRW) + method PDFs and
the column legend (Tabellen_Erlaeuterungen/LEGENDE_SPALTEN_DBF.xlsx).
Preprocessed: data/processed/field/fields_soil_bk50.csv - one row per field (all crops and years
in fields_rur.gpkg), soil values averaged over the soil units the field covers, weighted by the
area of each piece (numbers) or taken from the largest piece (classes):
  nfk_mm        plant-available water to the reference depth (nutzbare Feldkapazitaet)
  fk_mm         field capacity;  root_dm  reference / effective rooting depth (TIEFE)
  gw_depth_dm   mean groundwater depth (99 = no groundwater)
  cap_rise_mm_d capillary rise from groundwater into the root zone (mm/day)
  gw_stage      groundwater stage (0 none .. 6 very high); waterlog_stage  (Staunaesse, 0 .. 5)
  soil_value    Bodenwertzahl (mean of the range; soil quality score used for land taxation)
  water_supply  water supply class for arable crops (WAS_AC); texture  topsoil texture (KA5)
  share_mapped  share of the field area covered by BK50 (towns / water have no soil unit)
Run `python phase0_bk50_fields.py`.
"""
import os

import geopandas as gpd
import numpy as np
import pandas as pd

P = os.path.join("data", "processed")
BK = os.path.join("data", "raw", "soil", "bk50", "BK50_GPKG", "BK50_A.gpkg")
NUM = {"NFK": "nfk_mm", "FK": "fk_mm", "TIEFE": "root_dm", "GW_TIEFE": "gw_depth_dm", "KAP": "cap_rise_mm_d",
       "BWZ_M": "soil_value", "GW_STUFE": "gw_stage", "SW_STUFE": "waterlog_stage"}
CLS = {"WAS_AC": "water_supply", "SBDA": "texture", "TYP_TEXT": "soil_type"}


def main():
    fields = gpd.read_file(os.path.join(P, "fields_rur.gpkg"), layer="fields",
                           columns=["ID", "VALIDFROM", "CODE", "AREA_HA"])
    soil = gpd.read_file(BK, layer="BK50", bbox=tuple(fields.total_bounds), columns=list(NUM) + list(CLS))
    for c in NUM:
        soil[c] = pd.to_numeric(soil[c], errors="coerce")
    print(f"{len(fields)} field records, {len(soil)} soil units in the region", flush=True)
    # the same outline often repeats over the years: intersect each distinct outline once
    fields["wkb"] = fields.geometry.to_wkb()
    shapes = fields.drop_duplicates("wkb")[["wkb", "geometry"]].reset_index(drop=True)
    shapes["shape_id"] = np.arange(len(shapes))
    pieces = gpd.overlay(shapes[["shape_id", "geometry"]], soil, how="intersection", keep_geom_type=True)
    pieces["a"] = pieces.geometry.area
    total = shapes.set_index("shape_id").geometry.area
    rows = []
    for sid, g in pieces.groupby("shape_id"):
        r = {"shape_id": sid, "share_mapped": g["a"].sum() / total[sid]}
        for c, name in NUM.items():
            ok = g[c].notna()
            r[name] = np.average(g.loc[ok, c], weights=g.loc[ok, "a"]) if ok.any() else np.nan
        top = g.loc[g["a"].idxmax()]
        r.update({name: top[c] for c, name in CLS.items()})
        rows.append(r)
    s = shapes[["wkb", "shape_id"]].merge(pd.DataFrame(rows), on="shape_id", how="left")
    out = fields.drop(columns="geometry").merge(s, on="wkb").drop(columns=["wkb", "shape_id"])
    out.to_csv(os.path.join(P, "field", "fields_soil_bk50.csv"), index=False)
    w = out[out["CODE"] == 115]
    print(f"{len(shapes)} distinct outlines; wheat field records {len(w)}; mapped share median "
          f"{w['share_mapped'].median():.2f}")
    print(w[list(NUM.values())].describe(percentiles=[0.1, 0.5, 0.9]).round(1).to_string())
    print(w["water_supply"].value_counts().head(8).to_string())


if __name__ == "__main__":
    main()
