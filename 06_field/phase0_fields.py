"""Phase 0: field outlines and crops for North Rhine-Westphalia from the EU subsidy records.

Source: Landwirtschaftskammer NRW via OpenGeodata.NRW, "Beantragte und als foerderfaehig
festgestellte Teilschlaege" (current year: LWK-TSCHLAG, earlier years: LWK-TSCHLAG-HIST),
Shapefiles in UTM zone 32 (EPSG:25832), Datenlizenz Deutschland - Namensnennung 2.0.
Every field ("Teilschlag") a farmer declared for subsidies, with its crop code.

Steps:
  1. read every field of every year (attributes; geometry only as a bounding box)
  2. keep our five crops (codes below)
  3. place each field in its district (field centre tested against the district outlines)
  4. write data/processed/fields_nrw.csv: field id, year, crop, area, centre, district,
     organic flag; and the full outlines of the Rur test region to
     data/processed/fields_rur.gpkg (GeoPackage, opens in QGIS)
  5. compare the summed field area per district, crop and year with the official crop area
     (Duden et al. 2024 data) as a consistency check

Run `python phase0_fields.py`.
"""
import glob
import os
import subprocess
import zipfile

import numpy as np
import pandas as pd
import shapefile

from weather_districts import BOUNDARIES, inside

RAW = os.path.join("data", "raw", "fields_nrw")
P = os.path.join("data", "processed")
CROP_CODES = {115: "winter_wheat", 131: "winter_barley", 411: "silage_maize",
              171: "grain_maize", 602: "potato"}
OGR = r"C:\Program Files\QGIS 3.36.2\bin\ogr2ogr.exe"
RUR_REGION = {"05358": "Dueren", "05334": "Aachen (Staedteregion)", "05366": "Euskirchen",
              "05370": "Heinsberg", "05362": "Rhein-Erft-Kreis"}


def readers():
    """Yield a shapefile.Reader for every shapefile inside the NRW zip files."""
    for path in sorted(glob.glob(os.path.join(RAW, "LWK-TSCHLAG*.zip"))):
        z = zipfile.ZipFile(path)
        for stem in sorted(n[:-4] for n in z.namelist() if n.lower().endswith(".shp")):
            yield os.path.basename(path), shapefile.Reader(
                shp=z.open(stem + ".shp"), shx=z.open(stem + ".shx"), dbf=z.open(stem + ".dbf"),
                encoding="latin-1")


def nrw_districts():
    """District outlines of NRW (code starts with 05), in the same projection as the fields."""
    with zipfile.ZipFile(BOUNDARIES) as z:
        stem = "vg250_ebenen_1231/VG250_KRS"
        sf = shapefile.Reader(shp=z.open(stem + ".shp"), shx=z.open(stem + ".shx"),
                              dbf=z.open(stem + ".dbf"), encoding="utf-8")
        out = []
        for rec, shp in zip(sf.records(), sf.shapes()):
            rec = rec.as_dict()
            if rec["AGS"].startswith("05") and rec.get("GF") == 4:
                pts = np.array(shp.points)
                parts = list(shp.parts) + [len(pts)]
                out.append((rec["AGS"], shp.bbox, [pts[a:b] for a, b in zip(parts[:-1], parts[1:])]))
        return out


def assign_districts(x, y, districts):
    code = np.full(len(x), "", dtype="<U5")
    for ags, (x0, y0, x1, y1), rings in districts:
        box = np.flatnonzero((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1) & (code == ""))
        if not len(box):
            continue
        hit = np.zeros(len(box), bool)
        for ring in rings:
            hit ^= inside(x[box], y[box], ring)
        code[box[hit]] = ags
    return code


def our_fields():
    """Yield (order number, source, record, shape) for every field of our crops, in a fixed order."""
    k = 0
    for source, r in readers():
        names = [f[0] for f in r.fields[1:]]
        for sr in r.iterShapeRecords():
            rec = dict(zip(names, sr.record))
            if rec.get("CODE") in CROP_CODES:
                yield k, source, rec, sr.shape
                k += 1


def main():
    # pass 1: attributes and field centres (from the bounding box) only, to keep memory low
    rows = []
    for k, source, rec, shape in our_fields():
        x0, y0, x1, y1 = shape.bbox
        rows.append((k, rec["ID"], int(rec.get("VALIDFROM") or 0), CROP_CODES[rec["CODE"]],
                     rec["AREA_HA"], (x0 + x1) / 2, (y0 + y1) / 2,
                     str(rec.get("ORGANICFAR")).strip().lower() == "true", source))
    f = pd.DataFrame(rows, columns=["k", "field_id", "year", "crop", "area_ha", "x", "y", "organic", "source"])
    f["district"] = assign_districts(f["x"].values, f["y"].values, nrw_districts())
    f = f.drop_duplicates(["field_id", "year"])
    f.drop(columns="k").to_csv(os.path.join(P, "fields_nrw.csv"), index=False)
    print(f"saved {len(f)} field-years of our crops to data/processed/fields_nrw.csv "
          f"({(f['district'] == '').mean():.1%} without district)")
    print(f.pivot_table(index="year", columns="crop", values="field_id", aggfunc="count").fillna(0).astype(int))

    # pass 2: full outlines for the Rur test region, cut with GDAL (ogr2ogr from QGIS), which is
    # much faster than reading 3.7 million outlines in Python. Everything inside the region's
    # bounding box is kept; the district of each field is in fields_nrw.csv (join on ID + year).
    rur = [d for d in nrw_districts() if d[0] in RUR_REGION]
    box = [str(v) for v in (min(d[1][0] for d in rur), min(d[1][1] for d in rur),
                            max(d[1][2] for d in rur), max(d[1][3] for d in rur))]
    out = os.path.join(P, "fields_rur.gpkg")
    if os.path.exists(out):
        os.remove(out)
    env = dict(os.environ, SHAPE_ENCODING="LATIN1")
    codes = ",".join(str(c) for c in CROP_CODES)
    for path in sorted(glob.glob(os.path.join(RAW, "LWK-TSCHLAG*.zip"))):
        layer = [n[:-4] for n in zipfile.ZipFile(path).namelist() if n.endswith(".shp")][0]
        src = "/vsizip/" + os.path.abspath(path).replace("\\", "/") + "/" + layer + ".shp"
        sql = f"SELECT ID, VALIDFROM, CODE, CODE_TXT, AREA_HA, ORGANICFAR FROM {layer} WHERE CODE IN ({codes})"
        cmd = [OGR] + (["-update", "-append"] if os.path.exists(out) else ["-f", "GPKG"])
        cmd += [out, src, "-nln", "fields", "-sql", sql, "-spat"] + box
        subprocess.run(cmd, env=env, check=True, capture_output=True)
    print(f"Rur region outlines written to {out}")

    # consistency: field area vs official crop area per district and year
    area = f.groupby(["district", "year", "crop"])["area_ha"].sum().reset_index()
    off = pd.read_csv(os.path.join("data", "raw", "yields_1979_2021", "Final_data.csv"),
                      dtype={"district_no": str})
    names = {"ww": "winter_wheat", "wb": "winter_barley", "silage_maize": "silage_maize",
             "grain_maize": "grain_maize", "potat_tot": "potato"}
    off = off[(off["measure"] == "area") & off["var"].isin(names)]
    off = off.assign(crop=off["var"].map(names)).rename(columns={"district_no": "district", "value": "official_ha"})
    cmp_ = area.merge(off[["district", "year", "crop", "official_ha"]], on=["district", "year", "crop"])
    if len(cmp_):
        cmp_["ratio"] = cmp_["area_ha"] / cmp_["official_ha"]
        print("\nfield area / official crop area (median per crop):")
        print(cmp_.groupby("crop")["ratio"].median().round(2).to_string())


if __name__ == "__main__":
    main()
