"""Field v2 pipeline for any area and season (project log 2026-10-16). Same methods as the pilot 2024 scripts:

retrieval_s2      phase7_step2_lai (SCL 4/5, 2-pixel buffer around 3/8/9/10/11, Dec/Jan dropped) + fix 1 (scene >= 20 %
                  clear, or the pixel's main field >= 50 % clear) -> s2_pixels_fv2.npz (own name: the pilot 2024 test files stay untouched)
retrieval_landsat Landsat 8/9 C2 L2, QA_PIXEL clear bit and no cloud / shadow / cirrus / snow, 1-pixel (30 m) buffer, same
                  scene / field rule; OLI network (prosail_net_landsat.pt); 30 m value given to each 10 m pixel inside it
                  -> landsat_pixels.npz
setup             pixels, main field, crop, folds, radar (radar_tools.field_radar, both orbits), weather / season context
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import rasterio
import torch
from scipy.ndimage import binary_dilation

from areas import out_dir, raw_dir
from phase4_prosail import BANDS as S2_BANDS, MODEL as S2_MODEL, Net, TARGETS
from phase7_step3_s1proxy import CROPS
from radar_tools import field_radar

P = os.path.join("data", "processed")
L_BANDS = ["blue", "green", "red", "nir08", "swir16", "swir22"]
ORBITS = {"asc": "s1", "desc": "s1_desc"}
RADAR = ["VV", "VH", "RAT", "RVI"]


def pixels(area, year):
    out = out_dir(area, year)
    px = pd.read_csv(os.path.join(out, "pixels_nrw.csv.gz"))
    fl = pd.read_csv(os.path.join(out, "fields_nrw.csv"))
    up = px[["row", "col"]].drop_duplicates().sort_values(["row", "col"]).reset_index(drop=True)
    rows, cols = up["row"].values, up["col"].values
    mainf = px.sort_values("share").drop_duplicates(["row", "col"], keep="last").set_index(["row", "col"])
    key = pd.MultiIndex.from_arrays([rows, cols])
    fid, share = mainf["field_id"].reindex(key).values, mainf["share"].reindex(key).values
    fids, inv = np.unique(fid, return_inverse=True)
    grp = pd.Series(fid).map(fl.set_index("field_id")["crop_group"]).fillna("other").values
    crop = np.array([CROPS.index(g) if g in CROPS else CROPS.index("other") for g in grp])
    return dict(rows=rows, cols=cols, fid=fid, share=share, fids=fids, inv=inv, grp=grp, crop=crop, fl=fl)


def _net(path):
    ck = torch.load(path, weights_only=False)
    net = Net(len(ck["xm"]), len(TARGETS))
    net.load_state_dict(ck["state"])
    return net, ck


def _predict(net, ck, x):
    with torch.no_grad():
        mu, lv = net(torch.tensor((x - ck["xm"]) / ck["xs"], dtype=torch.float32))
    return mu.numpy() * ck["ys"] + ck["ym"], np.exp(0.5 * lv.numpy()) * ck["ys"]


def _keep(c, pinv, pshare, ftot):
    """Scene >= 20 % clear: all clear pixels; otherwise clear pixels of fields >= 50 % clear (fix 1)."""
    if c.mean() >= 0.2:
        return c
    fclear = np.bincount(pinv, weights=pshare * c, minlength=len(ftot)) / ftot
    return c & (fclear[pinv] >= 0.5)


def retrieval_s2(area, year, px):
    rows, cols = px["rows"], px["cols"]
    pinv, pshare = px["inv"], np.nan_to_num(px["share"])
    ftot = np.bincount(pinv, weights=pshare)
    net, ck = _net(S2_MODEL)
    dates, out = [], {k: [] for k in ("lai", "lai_sd", "ndvi")}
    for f in sorted(glob.glob(os.path.join(raw_dir(area, year), "s2", "*.tif"))):
        d = pd.Timestamp(os.path.basename(f)[7:17])
        if d.month in (12, 1):
            continue
        with rasterio.open(f) as r:
            a = r.read()
        bad = binary_dilation(np.isin(a[10], (3, 8, 9, 10, 11)), iterations=2)
        c = (np.isin(a[10], (4, 5)) & ~bad & np.all(np.isfinite(a[:10]), axis=0))[rows, cols]
        c = _keep(c, pinv, pshare, ftot)
        if not c.any():
            continue
        refl = a[:10, rows, cols].T / 10000.0
        sza, vza = a[11, rows, cols], a[12, rows, cols]
        raa = np.abs(a[13, rows, cols] - a[14, rows, cols]) % 360
        raa = np.where(raa > 180, 360 - raa, raa)
        x = np.concatenate([refl, np.cos(np.radians(np.stack([sza, vza, raa], 1)))], 1).astype(np.float32)
        ok = c & np.all(np.isfinite(x), axis=1)
        lai, sd = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
        pred, psd = _predict(net, ck, x[ok])
        lai[ok], sd[ok] = np.clip(pred[:, 0], 0, None), psd[:, 0]
        ndvi = np.where(ok, (refl[:, 6] - refl[:, 2]) / (refl[:, 6] + refl[:, 2] + 1e-6), np.nan)
        dates.append(d)
        for k, v in (("lai", lai), ("lai_sd", sd), ("ndvi", ndvi)):
            out[k].append(v.astype(np.float32))
    np.savez_compressed(os.path.join(out_dir(area, year), "s2_pixels_fv2.npz"), dates=np.array(dates, dtype="datetime64[D]"),
                        rows=rows, cols=cols, **{k: np.stack(v) for k, v in out.items()})
    return len(dates)


def retrieval_landsat(area, year, px):
    rows, cols = px["rows"], px["cols"]
    pinv, pshare = px["inv"], np.nan_to_num(px["share"])
    ftot = np.bincount(pinv, weights=pshare)
    net, ck = _net(os.path.join(P, "prosail_net_landsat.pt"))
    folder = os.path.join(raw_dir(area, year), "landsat")
    sc = pd.read_csv(os.path.join(folder, "scenes.csv"))
    r30, c30 = (rows * 10) // 30, (cols * 10) // 30
    inside = (r30 < 266) & (c30 < 266)
    r30, c30 = np.minimum(r30, 265), np.minimum(c30, 265)
    by_date = {}
    for _, s in sc.iterrows():
        d = pd.Timestamp(s["date"])
        if d.month in (12, 1):
            continue
        with rasterio.open(os.path.join(folder, s["file"])) as r:
            a = r.read().astype(np.float32)
        qa = a[6].astype(np.uint16)
        valid = (qa > 0) & ((qa >> 6) & 1).astype(bool) & ~(((qa >> 1) & 1) | ((qa >> 2) & 1) | ((qa >> 3) & 1) | ((qa >> 4) & 1)
                                                            | ((qa >> 5) & 1)).astype(bool)
        bad = binary_dilation((qa > 0) & ~valid, iterations=1)
        clear = valid & ~bad & np.all(a[:6] > 0, axis=0)
        refl = a[:6] * 0.0000275 - 0.2
        c = clear[r30, c30] & inside
        if d in by_date:                                          # second row of the same overpass: fill gaps only
            prev = by_date[d]
            c = c & ~prev["c"]
        x = np.concatenate([refl[:, r30, c30].T, np.cos(np.radians(np.tile([90 - s["sun_elevation"], 0.0, 90.0],
                                                                             (len(rows), 1))))], 1).astype(np.float32)
        ok = c & np.all(np.isfinite(x), axis=1) & np.all((x[:, :6] > 0) & (x[:, :6] < 1), axis=1)
        lai = np.full(len(rows), np.nan, np.float32)
        if ok.any():
            pred, _ = _predict(net, ck, x[ok])
            lai[ok] = np.clip(pred[:, 0], 0, None)
        ndvi = np.where(ok, (x[:, 3] - x[:, 2]) / (x[:, 3] + x[:, 2] + 1e-6), np.nan).astype(np.float32)
        if d in by_date:
            prev = by_date[d]
            m = ok & ~prev["c"]
            prev["lai"][m], prev["ndvi"][m], prev["c"] = lai[m], ndvi[m], prev["c"] | ok
        else:
            by_date[d] = dict(lai=lai, ndvi=ndvi, c=ok, platform=s["platform"])
    dates, lai, ndvi, plat = [], [], [], []
    for d in sorted(by_date):
        b = by_date[d]
        keep = _keep(b["c"], pinv, pshare, ftot)
        if not keep.any():
            continue
        dates.append(d); plat.append(b["platform"])
        lai.append(np.where(keep, b["lai"], np.nan)); ndvi.append(np.where(keep, b["ndvi"], np.nan))
    np.savez_compressed(os.path.join(out_dir(area, year), "landsat_pixels.npz"), dates=np.array(dates, dtype="datetime64[D]"),
                        rows=rows, cols=cols, lai=np.stack(lai).astype(np.float32), ndvi=np.stack(ndvi).astype(np.float32),
                        platform=np.array(plat))
    return len(dates)


def district_of(area, year, px):
    d = pd.read_csv(os.path.join(P, "field", "fields_district.csv"), dtype={"district": str})
    d = d[(d["VALIDFROM"] == year) & d["ID"].isin(px["fids"])]
    return d["district"].mode().iloc[0]


def context(dates, year, dist):
    """Season day, rain and days since the district's model heading (as phase7_step3_s1proxy.context)."""
    from phase4_field_features import district_dates
    w = np.load(os.path.join(P, "weather_daily_v8.npz"))
    wd = pd.DatetimeIndex(w["dates"])
    pr = pd.Series(w["pr"][:, list(w["codes"]).index(dist)], index=wd)
    try:
        head = district_dates(year)[dist][0]
    except (KeyError, IndexError):
        head = pd.Timestamp(f"{year}-05-20")                         # season without district model: typical heading
    start = pd.Timestamp(f"{year - 1}-09-01")
    return pd.DataFrame({"doy": (dates - start).days, "since_heading": (dates - head).days}, index=dates), pr.cumsum()


def radar(area, year, px):
    return {o: field_radar(os.path.join(raw_dir(area, year), f), px["rows"], px["cols"], px["inv"]) for o, f in ORBITS.items()}
