"""Field v2, step 1 (project log 2026-10-14): crop and exact field boundary on the 10 m Sentinel-2 grid, pilot cell
38_706 (EPSG:25832 x 304000-312000, y 5648000-5656000), season 2024. Two SEPARATE versions:

A  NRW field records (fields_rur_allcrops.gpkg): every field of 2024, all crops. For every 10 m pixel the exact share
   of its area inside the field (1 m sub-grid), headland flag (pixel centre within 20 m of the boundary). Fields keep
   their registered outline: no inward or outward buffer. Check: sum of shares x 100 m2 = registered area.
B  DLR CropTypes 2024 alone: crop class per 10 m pixel, grouped like the NRW crop groups; pseudo-fields = connected
   areas (8-neighbourhood) of one group, >= 0.3 ha.
Comparison: per NRW field the DLR group covering most of it (share-weighted) and the agreement share; agreement by
crop group; area totals.
Outputs data/processed/field_v2/pilot_2024/: pixels_nrw.csv.gz, fields_nrw.csv, pixels_dlr.csv.gz,
pseudofields_dlr.csv, compare_by_field.csv, compare_by_crop.csv, grid_info.json
"""
import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from scipy import ndimage

from phase5_field_history import group

P = os.path.join("data", "processed")
OUT = os.path.join(P, "field_v2", "pilot_2024")
X0, Y0, X1, Y1 = 304000, 5648000, 312000, 5656000
RES, SUB, HEAD = 10, 10, 20
YEAR = 2024
DLR_GROUP = {11: "wheat", 21: "wheat", 12: "barley", 22: "barley", 13: "other cereal", 14: "other cereal",
             23: "other cereal", 30: "maize", 40: "legume", 41: "legume", 42: "legume", 50: "potato",
             60: "sugar beet", 71: "rapeseed", 83: "grass / fodder", 84: "grass / fodder"}
TRANSFORM = rasterio.transform.from_origin(X0, Y1, RES, RES)
NR, NC = (Y1 - Y0) // RES, (X1 - X0) // RES


def version_a():
    f = gpd.read_file(os.path.join(P, "fields_rur_allcrops.gpkg"), bbox=(X0, Y0, X1, Y1))
    f = f[f["VALIDFROM"] == YEAR].copy()
    c = f.geometry.centroid
    f = f[c.x.between(X0, X1) & c.y.between(Y0, Y1)]                 # fields whose centre lies in the cell
    f["crop_group"] = f["CODE_TXT"].map(group)
    rows, frows = [], []
    for _, x in f.iterrows():
        g = x.geometry
        xmin, ymin, xmax, ymax = g.bounds
        c0, c1 = int(np.floor((xmin - X0) / RES)), int(np.ceil((xmax - X0) / RES))
        r0, r1 = int(np.floor((Y1 - ymax) / RES)), int(np.ceil((Y1 - ymin) / RES))
        h, w = r1 - r0, c1 - c0
        tr = rasterio.transform.from_origin(X0 + c0 * RES, Y1 - r0 * RES, RES, RES)
        fine = rasterize([(g, 1)], out_shape=(h * SUB, w * SUB), transform=tr * rasterio.Affine.scale(1 / SUB), fill=0, dtype="uint8")
        share = fine.reshape(h, SUB, w, SUB).mean(axis=(1, 3))
        inner = g.buffer(-HEAD)
        core = rasterize([(inner, 1)], out_shape=(h, w), transform=tr, fill=0, dtype="uint8") if not inner.is_empty else np.zeros((h, w), "uint8")
        rr, cc = np.nonzero(share > 0)
        rows.append(pd.DataFrame({"row": rr + r0, "col": cc + c0, "field_id": int(x["ID"]), "share": share[rr, cc].round(3),
                                  "headland": core[rr, cc] == 0}))
        frows.append(dict(field_id=int(x["ID"]), crop=x["CODE_TXT"], crop_group=x["crop_group"], area_ha=x["AREA_HA"],
                          area_from_pixels_ha=share.sum() * RES * RES / 1e4, pixels=len(rr),
                          pixels_edge=int(((share > 0) & (share < 1)).sum()), pixels_headland=int(((share > 0) & (core == 0)).sum())))
    px, fl = pd.concat(rows, ignore_index=True), pd.DataFrame(frows)
    px = px[(px["row"].between(0, NR - 1)) & (px["col"].between(0, NC - 1))]
    fl["area_diff_pct"] = 100 * (fl["area_from_pixels_ha"] / fl["area_ha"] - 1)
    return px, fl


def version_b():
    src = rasterio.open(os.path.join("data", "raw", "croptypes_dlr", f"CROPTYPES_DE_P1Y_{YEAR}_V02.tif"))
    win = from_bounds(X0, Y0, X1, Y1, transform=src.transform)
    a = src.read(1, window=win, out_shape=(NR, NC))
    off = (src.window_transform(win).c - X0, src.window_transform(win).f - Y1)
    grp = np.vectorize(lambda v: DLR_GROUP.get(int(v), "other" if v > 0 else "none"))(a)
    names = sorted(set(grp.ravel()) - {"none"})
    lab_all = np.zeros(a.shape, "int32")
    pf, n0 = [], 0
    for nm in names:
        lab, n = ndimage.label(grp == nm, structure=np.ones((3, 3)))
        sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
        keep = np.flatnonzero(sizes * RES * RES / 1e4 >= 0.3) + 1
        for k in keep:
            n0 += 1
            lab_all[lab == k] = n0
            pf.append(dict(pseudo_id=n0, crop_group=nm, area_ha=sizes[k - 1] * RES * RES / 1e4))
    rr, cc = np.nonzero(a > 0)
    px = pd.DataFrame({"row": rr, "col": cc, "dlr_code": a[rr, cc], "dlr_group": grp[rr, cc], "pseudo_id": lab_all[rr, cc]})
    return px, pd.DataFrame(pf), off


def main():
    os.makedirs(OUT, exist_ok=True)
    pa, fa = version_a()
    pb, pfb, off = version_b()
    m = pa.merge(pb[["row", "col", "dlr_group"]], on=["row", "col"], how="left")
    m["dlr_group"] = m["dlr_group"].fillna("none")
    agree = []
    w = m.groupby(["field_id", "dlr_group"])["share"].sum().reset_index().sort_values("share")
    top = w.drop_duplicates("field_id", keep="last").set_index("field_id")
    tot = m.groupby("field_id")["share"].sum()
    agree = [dict(field_id=fid, dlr_majority=r["dlr_group"], dlr_majority_share=r["share"] / tot[fid]) for fid, r in top.iterrows()]
    cmp = fa.merge(pd.DataFrame(agree), on="field_id")
    cmp["same_group"] = cmp["crop_group"] == cmp["dlr_majority"]
    m["nrw_group"] = m["field_id"].map(fa.set_index("field_id")["crop_group"])
    same = (m["share"] * (m["dlr_group"] == m["nrw_group"])).groupby(m["field_id"]).sum() / m.groupby("field_id")["share"].sum()
    cmp["share_same"] = cmp["field_id"].map(same)
    by = cmp.groupby("crop_group").agg(fields=("field_id", "size"), area_ha=("area_ha", "sum"),
                                       majority_agrees_pct=("same_group", lambda s: 100 * s.mean()),
                                       pixel_share_agrees_pct=("share_same", lambda s: 100 * s.mean())).sort_values("area_ha", ascending=False)
    area_dlr = pb.groupby("dlr_group").size() * RES * RES / 1e4
    by["dlr_area_ha"] = by.index.map(area_dlr).fillna(0)
    pa.to_csv(os.path.join(OUT, "pixels_nrw.csv.gz"), index=False)
    fa.to_csv(os.path.join(OUT, "fields_nrw.csv"), index=False)
    pb.to_csv(os.path.join(OUT, "pixels_dlr.csv.gz"), index=False)
    pfb.to_csv(os.path.join(OUT, "pseudofields_dlr.csv"), index=False)
    cmp.to_csv(os.path.join(OUT, "compare_by_field.csv"), index=False)
    by.round(1).to_csv(os.path.join(OUT, "compare_by_crop.csv"))
    json.dump(dict(crs="EPSG:25832", origin_x=X0, origin_y=Y1, res=RES, rows=NR, cols=NC, season=YEAR,
                   dlr_grid_offset_m=[round(o, 2) for o in off]), open(os.path.join(OUT, "grid_info.json"), "w"), indent=1)
    print(f"A NRW: {len(fa)} fields, {fa['area_ha'].sum():.0f} ha registered; pixels touching fields {len(pa)} "
          f"(edge {fa['pixels_edge'].sum()}, headland {fa['pixels_headland'].sum()}); area check: median |diff| "
          f"{fa['area_diff_pct'].abs().median():.2f} %, max {fa['area_diff_pct'].abs().max():.2f} %")
    print(f"B DLR: {len(pb)} crop pixels, {len(pfb)} pseudo-fields >= 0.3 ha; DLR grid offset vs ours {off} m")
    print(f"agreement: majority group same for {100 * cmp['same_group'].mean():.1f} % of NRW fields "
          f"({100 * np.average(cmp['same_group'], weights=cmp['area_ha']):.1f} % of area)")
    print(by.round(1).to_string())


if __name__ == "__main__":
    main()
