"""FIELD level: within-field relative yield maps from Sentinel-2 (pre-registered 2026-10-09, project log).

Data: YieldSAT Germany_raw.zip (CC BY-NC-ND, research only): per field 10 m yield map (mean, outlier-
masked), Sentinel-2 L2A clips + SCL masks and DEM layers on the same grid.
Step 1 (extract): per wheat field (map quality Good / Average) build a pixel table: relative yield (pixel /
field median) and relative satellite / terrain features -> data/processed/yieldsat/relative_pixels.csv.gz.
Step 2 (evaluate): B0 flat, B1 June NDVI slope, M1 ridge, M2 boosting; leave-one-farm-out and
leave-one-year-out; per-field r, RMSE, tercile zone agreement -> relative_maps_scores.csv.
"""
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import binary_erosion

ZIP = os.path.join("data", "raw", "yieldsat", "Germany_raw.zip")
OUT = os.path.join("data", "processed", "yieldsat")
META = os.path.join("data", "raw", "yieldsat", "wheat_metadata.csv")
MONTHS = (4, 5, 6, 7)


def read(z, name):
    with rasterio.MemoryFile(z.read(name)) as mf, mf.open() as r:
        return r.read().astype(np.float32), r.nodata


def field_pixels(z, names, field, year):
    p = field + "/"
    ym, nd = read(z, p + "yield_masks/mean_scaled_yield_masked_regional_statistical_outlier.tif")
    y = ym[0]
    valid = np.isfinite(y) & (y > 0) & ((y != nd) if nd is not None else True)
    valid = binary_erosion(valid, iterations=2)
    if valid.sum() < 200:
        return None
    med = np.median(y[valid])
    feats = {}
    stacks = {m: {"ndvi": [], "ndre": []} for m in MONTHS}
    for n in sorted(x for x in names if x.startswith(p + "s2_images/")):
        d = n.rsplit("_", 1)[-1][:8]
        if int(d[:4]) != year or int(d[4:6]) not in MONTHS:
            continue
        a, _ = read(z, n)
        sname = n.replace("s2_images/S2_L2A_", "scl_masks/S2_L2A_SCL_")
        if sname not in names:
            continue
        scl, _ = read(z, sname)
        if a.shape[1:] != y.shape or scl.shape[1:] != y.shape:
            continue
        clear = np.isin(scl[0], (4, 5))
        if (clear & valid).sum() < 0.5 * valid.sum():
            continue
        off = 1000.0 if (int(d) >= 20220125 and np.nanmedian(a[1][clear]) > 1000) else 0.0
        b = (a - off) / 10000.0
        red, re1, nir, n8a = b[3], b[4], b[7], b[8]
        ndvi = (nir - red) / (nir + red + 1e-6)
        ndre = (n8a - re1) / (n8a + re1 + 1e-6)
        ndvi[~clear], ndre[~clear] = np.nan, np.nan
        stacks[int(d[4:6])]["ndvi"].append(ndvi)
        stacks[int(d[4:6])]["ndre"].append(ndre)
    allv = []
    for m in MONTHS:
        for k in ("ndvi", "ndre"):
            if stacks[m][k]:
                v = np.nanmedian(np.stack(stacks[m][k]), 0)
                fm = np.nanmedian(v[valid])
                rel = v / fm if np.isfinite(fm) and fm > 0.05 else np.full(y.shape, np.nan)
                if k == "ndvi":
                    allv.append(v)
            else:
                rel = np.full(y.shape, np.nan)
            feats[f"{k}_m{m:02d}"] = rel
    if allv:
        mx = np.nanmax(np.stack(allv), 0)
        feats["ndvi_max"] = mx / np.nanmedian(mx[valid])
    else:
        feats["ndvi_max"] = np.full(y.shape, np.nan)
    feats["gld"] = np.nansum(np.stack([feats["ndvi_m06"], feats["ndvi_m07"]]), 0) / 2
    for n in [x for x in names if x.startswith(p + "dem/")]:
        key = os.path.basename(n).split("-")[0]
        a, ndd = read(z, n)
        if a.shape[1:] != y.shape:
            continue
        v = a[0]
        feats[f"dem_{key}"] = v - np.nanmean(v[valid]) if key == "dem" else v
    rows = {"rel_yield": np.clip(y[valid] / med, 0.3, 1.7)}
    for k, v in feats.items():
        rows[k] = v[valid]
    df = pd.DataFrame(rows)
    df["field"], df["year"] = field, year
    return df


def extract():
    meta = pd.read_csv(META)
    meta = meta[meta["yieldmap_quality"].isin(["Good", "Average"])]
    z = zipfile.ZipFile(ZIP)
    names = set(z.namelist())
    parts = []
    for _, r in meta.iterrows():
        f = r["field_shared_name"]
        if f + "/yield_masks/mean_scaled_yield_masked_regional_statistical_outlier.tif" not in names:
            continue
        df = field_pixels(z, names, f, int(r["year"]))
        if df is not None:
            df["farm"] = r["farm_identifier"]
            parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    os.makedirs(OUT, exist_ok=True)
    out.to_csv(os.path.join(OUT, "relative_pixels.csv.gz"), index=False)
    print(f"{out['field'].nunique()} fields, {len(out)} pixels, farms {out['farm'].nunique()}", flush=True)


def evaluate():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    d = pd.read_csv(os.path.join(OUT, "relative_pixels.csv.gz"))
    feats = [c for c in d.columns if c not in ("rel_yield", "field", "year", "farm")]
    for c in feats:                      # missing satellite month = field average (1.0); terrain = 0 / as is
        d[c] = d[c].fillna(0.0 if c.startswith("dem_") else 1.0)
    rng = np.random.default_rng(0)
    res = []
    for scheme, key in (("leave-one-farm-out", "farm"), ("leave-one-year-out", "year")):
        for k in sorted(d[key].unique()):
            tr, te = d[d[key] != k], d[d[key] == k]
            if te.empty:
                continue
            trs = tr.groupby("field", group_keys=False).apply(lambda g: g.sample(min(300, len(g)), random_state=0))
            slope = np.polyfit(trs["ndvi_m06"] - 1, trs["rel_yield"] - 1, 1)[0]
            m1 = Ridge(alpha=1.0).fit(trs[feats], trs["rel_yield"])
            m2 = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=4, random_state=0).fit(trs[feats], trs["rel_yield"])
            preds = {"B0 flat": np.ones(len(te)), "B1 June NDVI": 1 + slope * (te["ndvi_m06"] - 1),
                     "M1 ridge": m1.predict(te[feats]), "M2 boosting": m2.predict(te[feats])}
            for name, pr in preds.items():
                t = te.assign(pred=pr)
                for f, g in t.groupby("field"):
                    r = np.corrcoef(g["pred"], g["rel_yield"])[0, 1] if g["pred"].std() > 0 else 0.0
                    qt = pd.qcut(g["rel_yield"].rank(method="first"), 3, labels=False)
                    qp = pd.qcut(g["pred"].rank(method="first"), 3, labels=False) if g["pred"].std() > 0 else pd.Series(1, index=g.index)
                    res.append(dict(scheme=scheme, fold=k, model=name, field=f, farm=g["farm"].iloc[0], r=r,
                                    rmse=np.sqrt(((g["pred"] - g["rel_yield"]) ** 2).mean()), zone=(qt == qp).mean()))
    s = pd.DataFrame(res)
    s.to_csv(os.path.join(OUT, "relative_maps_scores.csv"), index=False)
    tab = s.groupby(["scheme", "model"]).agg(fields=("field", "nunique"), median_r=("r", "median"),
                                            median_rmse=("rmse", "median"), median_zone=("zone", "median")).round(3)
    print(tab.to_string())
    lofo = s[(s["scheme"] == "leave-one-farm-out")]
    print(lofo.groupby(["farm", "model"])["r"].median().unstack().round(2).to_string())


if __name__ == "__main__":
    if "evaluate" not in sys.argv:
        extract()
    evaluate()
