"""HR-VPP benchmark item 4: CGLS LAI 300 m v2 level check (pre-registered 2026-10-19; one look, results logged first).

For every CGLS 300 m pixel lying >= 90 % inside a study area, our weekly 10 m green LAI is averaged over all its 10 m pixels
with a value, at the week nearest the CGLS 10-day date (<= 3 days). Kept if >= 50 % of the 300 m area has our values and the
CGLS value is valid (QFLAG 0, LAI not 255). Reported per season and month: bias (ours - CGLS), RMSE, Pearson r; also with our
MEASURED weeks only. A level check: CGLS is effective LAI of mixed 300 m land, ours green LAI of 10 m pixels.
Run from D:\vista: python phase14_lai300_check.py  -> data/processed/field_v2/benchmark/lai300_*.csv + log.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from shapely.geometry import Polygon, box
from shapely.ops import transform as shp_transform

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CDSE = r"D:\vista\data\raw\hrvpp_portal\lai300\cdse"
OUT = r"D:\vista\data\processed\field_v2\benchmark"
INSIDE, COVER, MAX_DAYS, MEASURED = 0.90, 0.50, 3, 1


def pixel_index(lon, lat, t):
    """Row / column of the 300 m cell (affine t, north-up) holding each point."""
    c = np.floor((lon - t.c) / t.a).astype(int)
    r = np.floor((lat - t.f) / t.e).astype(int)
    return r, c


def aggregate(r, c, v, expected, inside):
    """Mean of our 10 m values per 300 m cell; keep cells >= INSIDE in the area and >= COVER covered by values."""
    ok = np.isfinite(v)
    df = pd.DataFrame({"r": r[ok], "c": c[ok], "v": v[ok]}).groupby(["r", "c"])["v"].agg(["mean", "size"])
    df.columns = ["ours", "n"]
    keys = list(df.index)
    df["share"] = [df.loc[k, "n"] / expected[k] for k in keys]
    df["inside"] = [inside.get(k, 0.0) for k in keys]
    return df[(df.inside >= INSIDE) & (df.share >= COVER)]


def nearest_week(weeks, day):
    d = np.abs((weeks - day).astype(int))
    k = int(np.argmin(d))
    return k if d[k] <= MAX_DAYS else -1


def cell_geometry(t, shape, area_box):
    """Expected number of 10 m pixels and the share inside the area for every 300 m cell (EPSG:25832 areas)."""
    to_utm = Transformer.from_crs(4326, 25832, always_xy=True).transform
    expected, inside = {}, {}
    for r in range(shape[0]):
        for c in range(shape[1]):
            x0, y0 = t * (c, r); x1, y1 = t * (c + 1, r + 1)
            p = shp_transform(to_utm, Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))
            if not p.intersects(area_box):
                continue
            expected[(r, c)] = p.area / 100.0
            inside[(r, c)] = p.intersection(area_box).area / p.area
    return expected, inside


def stats(df):
    e = df.ours - df.cgls
    return pd.Series({"n": len(df), "bias": e.mean(), "rmse": np.sqrt((e ** 2).mean()),
                      "r": df.ours.corr(df.cgls) if len(df) > 2 else np.nan, "ours_mean": df.ours.mean(), "cgls_mean": df.cgls.mean()})


def run(tag=""):
    from areas import AREAS, bounds, out_dir
    from fv2_weekly import OUTFILE
    os.makedirs(OUT, exist_ok=True)
    days = sorted({os.path.basename(f)[:8] for f in glob.glob(os.path.join(CDSE, "*_lai.tif"))})
    with rasterio.open(os.path.join(CDSE, f"{days[0]}_lai.tif")) as d:
        t, shape = d.transform, d.shape
    to_ll = Transformer.from_crs(25832, 4326, always_xy=True)
    rows = []
    for area in AREAS:
        x0, y0, x1, y1 = bounds(area)
        expected, inside = cell_geometry(t, shape, box(x0, y0, x1, y1))
        for year in range(2021, 2026):
            g = json.load(open(os.path.join(out_dir(area, year), "grid_info.json")))
            z = np.load(os.path.join(out_dir(area, year), OUTFILE))
            weeks, pr, pc = z["weeks"].astype("datetime64[D]"), z["rows"], z["cols"]
            lai, src = z["lai"], z["lai_src"]
            lon, lat = to_ll.transform(g["origin_x"] + (pc + 0.5) * g["res"], g["origin_y"] - (pr + 0.5) * g["res"])
            r, c = pixel_index(np.asarray(lon), np.asarray(lat), t)
            for day in days:
                dd = np.datetime64(f"{day[:4]}-{day[4:6]}-{day[6:]}")
                w = nearest_week(weeks, dd)
                if w < 0:
                    continue
                with rasterio.open(os.path.join(CDSE, f"{day}_lai.tif")) as a, rasterio.open(os.path.join(CDSE, f"{day}_qflag.tif")) as q:
                    cg, qf, sc = a.read(1), q.read(1), a.scales[0]
                for mode, vals in (("all", lai[w]), ("measured", np.where(src[w] == MEASURED, lai[w], np.nan))):
                    agg = aggregate(r, c, vals.astype(float), expected, inside)
                    for (rr, cc), x in agg.iterrows():
                        if cg[rr, cc] == 255 or qf[rr, cc] != 0:
                            continue
                        rows.append(dict(area=area, season=year, date=str(dd), month=int(day[4:6]), mode=mode, row=rr, col=cc,
                                         ours=x.ours, cgls=cg[rr, cc] * sc, share=x.share, inside=x.inside))
        print(area, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"lai300_pairs{tag}.csv.gz"), index=False)
    out = []
    for mode in ("all", "measured"):
        m = df[df["mode"] == mode]
        out.append(stats(m).rename(f"{mode} | pooled"))
        for s, g in m.groupby("season"): out.append(stats(g).rename(f"{mode} | season {s}"))
        for mo, g in m.groupby("month"): out.append(stats(g).rename(f"{mode} | month {mo:02d}"))
        for a, g in m.groupby("area"): out.append(stats(g).rename(f"{mode} | area {a}"))
    res = pd.DataFrame(out).round(3)
    res.to_csv(os.path.join(OUT, f"lai300_summary{tag}.csv"))
    print(res.to_string())
    print(f"mean 10 m coverage share of kept cells: {df.share.mean():.2f}; cells per area-date: {df[df['mode'] == 'all'].groupby(['area', 'date']).size().median():.0f}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "")      # e.g. _after_fix2: a later run never overwrites the first
