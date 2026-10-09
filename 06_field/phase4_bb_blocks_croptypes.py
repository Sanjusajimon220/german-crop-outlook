"""F1 (Brandenburg): crop of every arable field block per year 2017-2024 from DLR CropTypes.

Field blocks: DFBK Brandenburg (data/raw/brandenburg/DFBK_FB.shp, category AL = arable, EPSG:25833).
Crops: DLR CropTypes v2, 10 m, Germany, one Cloud-Optimised GeoTIFF per year (EPSG:32632, CC BY 4.0;
Gessner et al. 2025), streamed - only the Brandenburg window is read, in 20 km tiles.
Per block and year: main crop class, its share of the block's pixels (purity), and the shares of
winter wheat / winter barley / maize / potato / rapeseed. Single-crop blocks (purity >= 0.9) can
later be treated like fields (multi-year Brandenburg field pilot).
Writes data/processed/brandenburg/blocks_croptypes.csv. Run `python phase4_bb_blocks_croptypes.py`.
"""
import os
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds

URL = "https://download.geoservice.dlr.de/CROPTYPES/files/CROPTYPES_DE_P1Y_{y}_V02/CROPTYPES_DE_P1Y_{y}_V02.tif"
OUT = os.path.join("data", "processed", "brandenburg")
CLASSES = {11: "winter_wheat", 12: "winter_barley", 30: "maize", 50: "potato", 71: "rapeseed"}
TILE = 20000.0


def main():
    os.makedirs(OUT, exist_ok=True)
    b = gpd.read_file(os.path.join("data", "raw", "brandenburg", "DFBK_FB.shp"))
    b = b[b["HBN_KAT"] == "AL"].reset_index(drop=True)
    rows = []
    for year in range(2017, 2025):
        t0 = time.time()
        with rasterio.open(URL.format(y=year)) as src:
            bb = b.to_crs(src.crs)
            bb["k"] = np.arange(len(bb))
            cx, cy = bb.geometry.centroid.x.values, bb.geometry.centroid.y.values
            counts = np.zeros((len(bb), 256), np.int64)
            x0, y0, x1, y1 = bb.total_bounds
            for x in np.arange(x0, x1, TILE):
                for y in np.arange(y0, y1, TILE):
                    sub = bb[(cx >= x) & (cx < x + TILE) & (cy >= y) & (cy < y + TILE)]   # blocks by centroid
                    if sub.empty:
                        continue
                    bx0, by0, bx1, by1 = sub.total_bounds
                    win = from_bounds(bx0, by0, bx1, by1, src.transform).round_offsets().round_lengths()
                    a = src.read(1, window=win)
                    if a.size == 0:
                        continue
                    lab = rasterize(((g, k) for g, k in zip(sub.geometry, sub["k"])), out_shape=a.shape,
                                    transform=src.window_transform(win), fill=-1, dtype="int32")
                    ok = lab >= 0
                    np.add.at(counts, (lab[ok], a[ok].astype(np.int64)), 1)
        tot = counts.sum(1)
        main_cls = counts.argmax(1)
        purity = np.where(tot > 0, counts.max(1) / np.maximum(tot, 1), np.nan)
        d = pd.DataFrame({"FB_ID": b["FB_ID"], "year": year, "pixels": tot, "main_class": main_cls,
                          "purity": purity.round(3), "area_ha": b["FL_NETTO"], "kreis": b["KREIS_NR"]})
        for c, name in CLASSES.items():
            d[f"share_{name}"] = np.where(tot > 0, counts[:, c] / np.maximum(tot, 1), np.nan).round(3)
        rows.append(d)
        pd.concat(rows).to_csv(os.path.join(OUT, "blocks_croptypes.csv"), index=False)
        print(f"{year}: {np.sum(tot > 0)} blocks with pixels; winter wheat main crop in "
              f"{np.sum(main_cls == 11)} blocks ({np.sum((main_cls == 11) & (purity >= 0.9))} with purity >= 0.9); "
              f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
