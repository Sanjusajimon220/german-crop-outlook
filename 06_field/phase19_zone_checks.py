"""Module A plausibility checks (written down 2026-10-21 before results; reported, no pass/fail).

Per field: weak minus strong zone mean of (a) BK50 plant-available water (nFK, mm) and (b) terrain from NRW DGM1: slope and
topographic wetness index (TWI = ln(a / tan slope), a = upslope area per contour width; D8 flow on the sink-filled 10 m DEM).
Expected: weak zones more often on low water capacity, slopes / hilltops or wet hollows.
Run from D:\\vista: python phase19_zone_checks.py -> data/processed/field_v2/zones/zone_checks*.csv + printed summary.
"""
import glob
import heapq
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DGM = r"D:\vista\data\raw\dgm1_nrw"
BK = os.path.join("data", "raw", "soil", "bk50", "BK50_GPKG", "BK50_A.gpkg")
OUT = r"D:\vista\data\processed\field_v2\zones"
NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def slope_deg(dem, cell):
    gy, gx = np.gradient(dem, cell)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def fill_sinks(dem, eps=1e-4):
    """Priority-flood (Barnes et al. 2014) with a tiny gradient so that flats drain; NaN cells act as outlets."""
    h, w = dem.shape
    z = np.where(np.isfinite(dem), dem, -1e9)
    out = z.copy()
    seen = np.zeros(dem.shape, bool)
    heap = []
    for r in range(h):
        for c in (0, w - 1):
            heapq.heappush(heap, (z[r, c], r, c)); seen[r, c] = True
    for c in range(1, w - 1):
        for r in (0, h - 1):
            heapq.heappush(heap, (z[r, c], r, c)); seen[r, c] = True
    nan_r, nan_c = np.nonzero(~np.isfinite(dem))
    for r, c in zip(nan_r, nan_c):
        if not seen[r, c]:
            heapq.heappush(heap, (z[r, c], r, c)); seen[r, c] = True
    while heap:
        e, r, c = heapq.heappop(heap)
        for dr, dc in NB:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc]:
                seen[rr, cc] = True
                out[rr, cc] = max(z[rr, cc], e + eps)
                heapq.heappush(heap, (out[rr, cc], rr, cc))
    return np.where(np.isfinite(dem), out, np.nan)


def flow_accumulation(dem):
    """Number of cells draining through each cell (D8 steepest descent on the filled DEM; itself included)."""
    f = fill_sinks(dem)
    h, w = f.shape
    z = np.where(np.isfinite(f), f, -1e9)
    pad = np.pad(z, 1, constant_values=np.inf)
    best, down = np.zeros(f.shape), np.full(f.shape, -1, np.int64)
    idx = np.arange(h * w).reshape(h, w)
    for dr, dc in NB:
        drop = (z - pad[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]) / np.hypot(dr, dc)
        tgt = np.clip(np.arange(h)[:, None] + dr, 0, h - 1) * w + np.clip(np.arange(w)[None, :] + dc, 0, w - 1)
        take = drop > best
        best[take], down[take] = drop[take], tgt[take]
    acc = np.ones(h * w)
    d = down.ravel()
    for k in np.argsort(-z.ravel(), kind="stable"):
        if d[k] >= 0:
            acc[d[k]] += acc[k]
    return acc.reshape(h, w)


def twi(dem, cell):
    a = flow_accumulation(dem) * cell                          # upslope area per unit contour width (m)
    t = np.tan(np.radians(np.maximum(slope_deg(np.where(np.isfinite(dem), dem, np.nanmean(dem)), cell), 0.06)))
    return np.log(a / t)


def zone_contrast(zone, var, field):
    df = pd.DataFrame({"zone": zone, "v": var, "field": field}).dropna()
    m = df[df.zone.isin([1, 3])].groupby(["field", "zone"]).v.mean().unstack()
    m = m.dropna()
    return pd.DataFrame({"weak": m[1], "strong": m[3], "weak_minus_strong": m[1] - m[3]})


# ---- real-data run ----
def _dem10(x0, y0, ring=1000, size=8000):
    """10 m DEM (block mean of DGM1) over the area plus the ring; rows from north."""
    import rasterio
    n = (size + 2 * ring) // 10
    dem = np.full((n, n), np.nan)
    for e in range((x0 - ring) // 1000, (x0 + size + ring) // 1000):
        for nn in range((y0 - ring) // 1000, (y0 + size + ring) // 1000):
            fs = glob.glob(os.path.join(DGM, f"dgm1_32_{e}_{nn}_1_nw_*.tif"))
            if not fs:
                continue
            with rasterio.open(sorted(fs)[-1]) as src:
                a = src.read(1).astype(float)
                if src.nodata is not None:
                    a[a == src.nodata] = np.nan
                a = a.reshape(100, 10, 100, 10).mean(axis=(1, 3))
            r0 = ((y0 + size + ring) - (nn + 1) * 1000) // 10
            c0 = (e * 1000 - (x0 - ring)) // 10
            dem[r0:r0 + 100, c0:c0 + 100] = a
    return dem


def run():
    import geopandas as gpd
    import rasterio.features
    from rasterio.transform import from_origin
    from areas import AREAS, SIZE, out_dir
    rows, summ = [], []
    for area, (x0, y0) in AREAS.items():
        zone = np.load(os.path.join(OUT, f"zones_{area}_2025.npy")).ravel()
        g = pd.read_csv(os.path.join(out_dir(area, 2025), "pixels_nrw.csv.gz"))
        g = g.sort_values("share").drop_duplicates(["row", "col"], keep="last")
        field = np.full(800 * 800, -1, np.int64)
        field[g.row.values * 800 + g.col.values] = g.field_id.values
        dem = _dem10(x0, y0)
        sl, tw = slope_deg(dem, 10.0), twi(dem, 10.0)
        inner = (slice(100, 900), slice(100, 900))
        soil = gpd.read_file(BK, layer="BK50", bbox=(x0, y0, x0 + SIZE, y0 + SIZE), columns=["NFK"])
        nfk = rasterio.features.rasterize(((geom, v) for geom, v in zip(soil.geometry, soil.NFK) if v is not None and v == v),
                                          out_shape=(800, 800), transform=from_origin(x0, y0 + SIZE, 10, 10), fill=np.nan, dtype="float64")
        keep = (zone > 0) & (field >= 0)
        for name, v in (("nfk_mm", nfk.ravel()), ("slope_deg", sl[inner].ravel()), ("twi", tw[inner].ravel()),
                        ("elev_dev_m", (dem[inner] - np.nanmean(dem[inner])).ravel())):
            c = zone_contrast(zone[keep], v[keep], field[keep]).assign(area=area, var=name)
            rows.append(c.reset_index())
        print(area, "done", flush=True)
    t = pd.concat(rows)
    tested = pd.read_csv(os.path.join(OUT, "zones_test.csv"))
    flat = tested.groupby(["area", "field"]).spread.median().reset_index()
    t = t.merge(flat, on=["area", "field"], how="left")
    t.to_csv(os.path.join(OUT, "zone_checks.csv"), index=False)
    for name, g in t.groupby("var"):
        for label, gg in (("all fields", g), ("fields with zone spread >= 5 %", g[g.spread >= 0.05])):
            d = gg.weak_minus_strong
            summ.append(dict(var=name, fields=label, n=len(d), weak_lower=(d < 0).mean(), weak_higher=(d > 0).mean(),
                             median_diff=d.median(), q25=d.quantile(.25), q75=d.quantile(.75)))
    s = pd.DataFrame(summ).round(3)
    s.to_csv(os.path.join(OUT, "zone_checks_summary.csv"), index=False)
    print(s.to_string(index=False))
    print("ZONE CHECKS DONE")


if __name__ == "__main__":
    run()
