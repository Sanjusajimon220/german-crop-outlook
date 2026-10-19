"""Static platform files for the field level (plan 2026-10-17, Task 3): boundaries, weekly field series with alerts
and season labels, per-field pixel arrays, PNG overlays, Sentinel-2 true colour, district summary. Writes docs/field/.
"""
import numpy as np

ARABLE = {"wheat", "barley", "other cereal", "rapeseed", "maize", "potato", "sugar beet"}
SPRING = {"maize", "potato", "sugar beet"}
SEASON_LAI = 0.5
PARTIAL_SHARE = 0.99


def partial(share_inside):
    """Field straddles the area edge: less than 99 % of its registered area inside the 8 x 8 km area."""
    return bool(share_inside < PARTIAL_SHARE)


def field_mean(vals, w, inv, n_fields):
    """Share-weighted mean of pixel values per field; vals [weeks x pixels], NaN ignored."""
    ok = np.isfinite(vals)
    ww = np.where(ok, w[None, :], 0.0)
    out = np.full((vals.shape[0], n_fields), np.nan)
    for t in range(vals.shape[0]):
        sw = np.bincount(inv, weights=ww[t], minlength=n_fields)
        s = np.bincount(inv, weights=np.where(ok[t], vals[t], 0.0) * ww[t], minlength=n_fields)
        out[t] = np.where(sw > 0, s / np.where(sw > 0, sw, 1), np.nan)
    return out


def alerts(field, unc, median):
    """Per week: 1 = below the median of the same crop by more than the uncertainty, 2 weeks in a row;
    2 = falling while the median rises, with the divergence beyond the uncertainty, 2 weeks in a row; else 0."""
    f, u, m = (np.asarray(x, float) for x in (field, unc, median))
    below = (m - f) > u
    div = np.zeros(len(f), bool)
    div[1:] = ((f[1:] - f[:-1]) < 0) & ((m[1:] - m[:-1]) > 0) & (((m[1:] - m[:-1]) - (f[1:] - f[:-1])) > u[1:])
    out = []
    for t in range(len(f)):
        if t > 0 and below[t] and below[t - 1]:
            out.append(1)
        elif t > 0 and div[t] and div[t - 1]:
            out.append(2)
        else:
            out.append(0)
    return out


def season_mask(lai, group):
    """1 = the registered crop is in the field, 0 = before sowing (previous crop / bare) or after harvest.
    Winter crops: from the autumn minimum (sowing); spring crops: from the minimum before the first rise above 0.5 in
    the second half of the season; both until the last week above 0.5."""
    x = np.asarray(lai, float)
    n = len(x)
    if not np.isfinite(x).any():
        return [0] * n
    xx = np.where(np.isfinite(x), x, np.inf)
    if group in SPRING:
        half = n // 2
        rise = [i for i in range(half, n) if np.isfinite(x[i]) and x[i] > SEASON_LAI]
        stop = rise[0] if rise else n
        start = int(np.argmin(xx[:stop])) if stop > 0 else 0
    else:
        start = int(np.argmin(xx[: max(1, n // 2)]))
    above = [i for i in range(n) if np.isfinite(x[i]) and x[i] > SEASON_LAI]
    end = above[-1] if above else start
    return [1 if start <= i <= end else 0 for i in range(n)]


# ---- export (needs Task 2 weekly_pixels_V2.npz) ----
import gzip
import json
import os
import sys

import geopandas as gpd
import pandas as pd
import rasterio
from matplotlib import colormaps
from PIL import Image
from shapely.geometry import box

from areas import AREAS, bounds, out_dir, raw_dir
from fv2_weekly import OUTFILE

DOCS = os.path.join("D:/german-crop-outlook-field", "docs", "field")
LAI_SCALE, NODATA = 50, 255                                   # pixel blocks: uint8 = LAI x 50 (0.02 steps), 255 = none
YEARS = range(2021, 2026)


def _palette_png(grid, vmax, path, cmap="YlGn"):
    """grid [800 x 800] float (NaN = transparent) -> 8-bit palette PNG, 254 colours + transparent index 255."""
    q = np.where(np.isfinite(grid), np.clip(grid / vmax, 0, 1) * 253, 255).astype(np.uint8)
    lut = (colormaps[cmap](np.linspace(0, 1, 254))[:, :3] * 255).astype(np.uint8)
    pal = np.vstack([lut, np.zeros((2, 3), np.uint8)]).ravel().tolist()
    im = Image.fromarray(q, "P")
    im.putpalette(pal)
    im.save(path, optimize=True, transparency=255)


def _rgb_jpg(tif, path):
    with rasterio.open(tif) as r:
        a = r.read([3, 2, 1]).astype(np.float32)
    rgb = (np.clip(np.moveaxis(a, 0, -1) / 2000.0, 0, 1) ** 0.85 * 255).astype(np.uint8)
    Image.fromarray(rgb).save(path, quality=82, optimize=True)


def _rnd(v, d):
    return [None if not np.isfinite(x) else round(float(x), d) for x in v]


def export(area, year):
    out = os.path.join(DOCS, f"{area}_{year}")
    if os.path.exists(os.path.join(out, ".complete")):           # written last: a stopped export is redone
        return
    for sub in ("lai", "ndvi", "rgb"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    z = np.load(os.path.join(out_dir(area, year), OUTFILE))
    weeks = [str(w) for w in z["weeks"]]
    rows, cols = z["rows"], z["cols"]
    px = pd.read_csv(os.path.join(out_dir(area, year), "pixels_nrw.csv.gz"))
    fl = pd.read_csv(os.path.join(out_dir(area, year), "fields_nrw.csv")).set_index("field_id")
    mainf = px.sort_values("share").drop_duplicates(["row", "col"], keep="last").set_index(["row", "col"])
    key = pd.MultiIndex.from_arrays([rows, cols])
    fid, share = mainf["field_id"].reindex(key).values, np.nan_to_num(mainf["share"].reindex(key).values)
    fids, inv = np.unique(fid, return_inverse=True)
    F = len(fids)
    # boundaries (WGS84) with partial flag and district
    x0, y0, x1, y1 = bounds(area)
    g = gpd.read_file(os.path.join("data", "processed", "fields_rur_allcrops.gpkg"), bbox=(x0, y0, x1, y1))
    g = g[(g["VALIDFROM"] == year) & g["ID"].isin(fids)].copy()
    g["share_in"] = g.geometry.intersection(box(x0, y0, x1, y1)).area / g.geometry.area
    dist = pd.read_csv(os.path.join("data", "processed", "field", "fields_district.csv"), dtype={"district": str})
    dist = dist[dist["VALIDFROM"] == year].drop_duplicates("ID").set_index("ID")["district"]
    g = g.assign(id=g["ID"].astype(int), crop=g["CODE_TXT"], group=g["ID"].map(fl["crop_group"]).fillna("other"),
                 ha=g["AREA_HA"].round(2), partial=g["share_in"].map(partial), district=g["ID"].map(dist))
    g = g[["id", "crop", "group", "ha", "partial", "district", "geometry"]]
    g["geometry"] = g.geometry.simplify(1.0)
    g.to_crs(4326).to_file(os.path.join(out, "fields.geojson"), driver="GeoJSON", COORDINATE_PRECISION=6)
    group = pd.Series(fids).map(fl["crop_group"]).fillna("other").values
    # field series: share-weighted means; uncertainty = weighted mean of pixel values; source = majority by share
    fs = {}
    for t in ("lai", "ndvi"):
        fs[t] = field_mean(z[t], share, inv, F)
        fs[f"{t}_unc"] = field_mean(z[f"{t}_unc"], share, inv, F)
        srcw = np.stack([np.stack([np.bincount(inv, weights=share * (z[f"{t}_src"][w] == k), minlength=F) for k in range(5)], 1)
                         for w in range(len(weeks))])
        fs[f"{t}_src"] = srcw.argmax(2)
    bbox = {}
    for k in range(F):                                        # pixel window of each field: [row0, row1, col0, col1]
        p = np.flatnonzero(inv == k)
        bbox[k] = [int(rows[p].min()), int(rows[p].max()) + 1, int(cols[p].min()), int(cols[p].max()) + 1]
    series = {"weeks": weeks, "fields": {}}
    med = {gr: np.nanmedian(np.where(group[None, :] == gr, fs["lai"], np.nan), axis=1) for gr in set(group)}
    for k, f in enumerate(fids):
        lai = fs["lai"][:, k]
        season = season_mask(lai, group[k])
        alert = alerts(lai, fs["lai_unc"][:, k], med[group[k]]) if group[k] in ARABLE else [0] * len(weeks)
        alert = [a if s else 0 for a, s in zip(alert, season)]
        series["fields"][str(int(f))] = {"lai": _rnd(lai, 2), "ndvi": _rnd(fs["ndvi"][:, k], 3), "lai_unc": _rnd(fs["lai_unc"][:, k], 2),
                                         "ndvi_unc": _rnd(fs["ndvi_unc"][:, k], 3), "src": fs["lai_src"][:, k].tolist(),
                                         "ndvi_src": fs["ndvi_src"][:, k].tolist(), "season": season, "alert": alert,
                                         "bb": bbox[k], "group": group[k]}
    json.dump(series, open(os.path.join(out, "series.json"), "w"), separators=(",", ":"))
    # pixel blocks: one gzip block per field (weeks x its pixels, uint8), index with byte offsets
    index, blobs, off = {}, [], 0
    for k, f in enumerate(fids):
        p = np.flatnonzero(inv == k)
        q = np.where(np.isfinite(z["lai"][:, p]), np.clip(np.round(z["lai"][:, p] * LAI_SCALE), 0, 254), NODATA).astype(np.uint8)
        b = gzip.compress(q.tobytes(), 6)
        index[str(int(f))] = {"o": off, "l": len(b), "n": int(len(p)), "r": rows[p].tolist(), "c": cols[p].tolist(),
                              "w": np.round(share[p], 3).tolist()}
        blobs.append(b)
        off += len(b)
    open(os.path.join(out, "px.bin"), "wb").write(b"".join(blobs))
    json.dump({"weeks": len(weeks), "scale": LAI_SCALE, "nodata": NODATA, "fields": index}, open(os.path.join(out, "px.json"), "w"),
              separators=(",", ":"))
    # weekly overlays (800 x 800, transparent outside fields)
    for w, wd in enumerate(weeks):
        for t, vmax in (("lai", 5.0), ("ndvi", 1.0)):
            grid = np.full((800, 800), np.nan, np.float32)
            grid[rows, cols] = z[t][w]
            _palette_png(grid, vmax, os.path.join(out, t, f"{wd}.png"))
    # Sentinel-2 true colour for every date used by the retrieval
    s2d = [str(d) for d in np.load(os.path.join(out_dir(area, year), "s2_pixels_fv2.npz"))["dates"]]
    for d in s2d:
        _rgb_jpg(os.path.join(raw_dir(area, year), "s2", f"openEO_{d}Z.tif"), os.path.join(out, "rgb", f"{d}.jpg"))
    json.dump(s2d, open(os.path.join(out, "rgb.json"), "w"))
    open(os.path.join(out, ".complete"), "w").close()
    print(f"{area} {year}: {F} fields exported", flush=True)


def write_index():
    idx = {"areas": {}}
    for a in AREAS:
        x0, y0, x1, y1 = bounds(a)
        w, s, e, n = gpd.GeoSeries([box(x0, y0, x1, y1)], crs=25832).to_crs(4326).total_bounds
        seasons = [y for y in YEARS if os.path.exists(os.path.join(DOCS, f"{a}_{y}", "series.json"))]
        ul, ur, ll = gpd.GeoSeries(gpd.points_from_xy([x0, x1, x0], [y1, y1, y0]), crs=25832).to_crs(4326)
        idx["areas"][a] = {"bounds_wgs84": [round(float(v), 6) for v in (w, s, e, n)], "bounds_utm32": [x0, y0, x1, y1],
                           "corners_wgs84": [[round(pt.x, 7), round(pt.y, 7)] for pt in (ul, ur, ll)],
                           "seasons": seasons}
    json.dump(idx, open(os.path.join(DOCS, "index.json"), "w"), indent=1)


def write_districts():
    out = {}
    for a in AREAS:
        for y in YEARS:
            p = os.path.join(DOCS, f"{a}_{y}")
            if not os.path.exists(os.path.join(p, "series.json")):
                continue
            s = json.load(open(os.path.join(p, "series.json")))
            g = gpd.read_file(os.path.join(p, "fields.geojson"))[["id", "group", "district"]].dropna()
            yo = out.setdefault(str(y), {})
            for d, gd in g.groupby("district"):
                for gr, gg in gd.groupby("group"):
                    lai = [[np.nan if v is None else v for v in s["fields"][str(i)]["lai"]] for i in gg["id"] if str(i) in s["fields"]]
                    if not lai:
                        continue
                    e = yo.setdefault(d, {"weeks": s["weeks"], "groups": {}})
                    e["groups"].setdefault(gr, {"n_fields": 0, "lai": []})
                    e["groups"][gr]["n_fields"] += len(lai)
                    e["groups"][gr]["lai"].extend(lai)
    for y in out.values():
        for e in y.values():
            for v in e["groups"].values():
                v["median_lai"] = _rnd(np.nanmedian(np.array(v.pop("lai"), float), axis=0), 2)
    json.dump(out, open(os.path.join(DOCS, "districts.json"), "w"), separators=(",", ":"))


if __name__ == "__main__" and sys.argv[1:] == ["all"]:
    for a in AREAS:
        for y in YEARS:
            export(a, y)
    write_index()
    write_districts()
