"""YieldSAT wheat fields: Sentinel-2 field means and our PROSAIL LAI (no credits needed).

Source: YieldSAT (yieldsat.github.io, CC BY-NC-ND 4.0, research only), data/raw/yieldsat/
Germany_raw.zip: per field ~40 Sentinel-2 L2A clips (12 bands, uint16, 10 m) + SCL masks + yield map.
Per field and date: mean reflectance over the pixels that are inside the yield map (valid yield
pixel) AND clear (SCL 4 vegetation / 5 bare soil); dates with < 50 % of the field clear are
dropped. Processing baseline 04.00 (from 25 Jan 2022) adds +1000 to L2A values: removed when the
clip is from that period and its clear-pixel blue band is above 1000 (checked per clip).
Sun / view angles are not in the clips: sun zenith from date, latitude and overpass time (10:30
local solar time), view zenith and relative azimuth = the monthly medians of our Rur Sentinel-2
extraction (approximation; the network's uncertainty covers part of it).
Writes data/processed/yieldsat/s2_fields.csv and retrieval_fields.csv (LAI etc. per field-date).
"""
import json
import os
import zipfile

import numpy as np
import pandas as pd
import rasterio
import torch

from phase4_prosail import BANDS as PB, MODEL, TARGETS, Net, inputs

ZIP = os.path.join("data", "raw", "yieldsat", "Germany_raw.zip")
OUT = os.path.join("data", "processed", "yieldsat")
ORDER = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]


def read(z, name):
    with rasterio.MemoryFile(z.read(name)) as mf, mf.open() as r:
        return r.read()


def sun_zenith(date, lat):
    doy = date.dayofyear
    decl = np.radians(23.44) * np.sin(2 * np.pi * (284 + doy) / 365)
    hour_angle = np.radians(15 * (10.5 - 12))
    lat = np.radians(lat)
    cos_z = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    return np.degrees(np.arccos(cos_z))


def rur_angles():
    a = pd.read_csv(os.path.join("data", "processed", "s2", "s2_115_2021.csv"),
                    usecols=["date", "viewZenithMean", "sunAzimuthAngles", "viewAzimuthMean"]).dropna()
    raa = np.abs(a["sunAzimuthAngles"] - a["viewAzimuthMean"]) % 360
    a["raa"] = np.where(raa > 180, 360 - raa, raa)
    a["month"] = pd.to_datetime(a["date"]).dt.month
    g = a.groupby("month")[["viewZenithMean", "raa"]].median().reindex(range(1, 13))
    return g.interpolate(limit_direction="both")     # September: no Rur image -> neighbouring months


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = pd.read_csv(os.path.join("data", "raw", "yieldsat", "wheat_metadata.csv")).set_index("field_shared_name")
    z = zipfile.ZipFile(ZIP)
    names = z.namelist()
    rows = []
    for k, field in enumerate(meta.index):
        files = [n for n in names if n.startswith(field + "/")]
        ymask = read(z, [n for n in files if "yield_masks/mean" in n][0])[0] >= 0
        for img in sorted(n for n in files if "/s2_images/" in n):
            date = pd.Timestamp(img[-12:-4])
            scl = read(z, img.replace("s2_images/S2_L2A_", "scl_masks/S2_L2A_SCL_"))[0]
            a = read(z, img).astype(np.float32)
            if a.shape[1:] != ymask.shape:
                continue
            ok = ymask & np.isin(scl, (4, 5))
            if ok.sum() < 0.5 * ymask.sum():
                continue
            v = a[:, ok].mean(1)
            offset = 1000.0 if date >= pd.Timestamp("2022-01-25") and v[1] > 1000 else 0.0
            r = {"field": field, "date": date, "clear_share": ok.sum() / ymask.sum(), "offset": offset}
            r.update({b: (x - offset) / 10000.0 for b, x in zip(ORDER, v)})
            rows.append(r)
        if k % 20 == 0:
            print(f"  {k + 1}/{len(meta)} fields, {len(rows)} clear field-dates", flush=True)
    s = pd.DataFrame(rows)
    ang = rur_angles()
    lat = s["field"].map(meta["centroid_latitude_wgs84"])
    s["sza"] = [sun_zenith(d, la) for d, la in zip(s["date"], lat)]
    s["vza"] = s["date"].dt.month.map(ang["viewZenithMean"])
    s["raa"] = s["date"].dt.month.map(ang["raa"])
    s.to_csv(os.path.join(OUT, "s2_fields.csv"), index=False)
    ck = torch.load(MODEL, weights_only=False)
    net = Net(len(PB) + 3, len(TARGETS))
    net.load_state_dict(ck["state"])
    with torch.no_grad():
        mu, lv = net(torch.tensor((inputs(s) - ck["xm"]) / ck["xs"]))
    pred, sd = mu.numpy() * ck["ys"] + ck["ym"], np.exp(0.5 * lv.numpy()) * ck["ys"]
    for j, t in enumerate(TARGETS):
        s[t], s[f"{t}_sd"] = pred[:, j], sd[:, j]
    s["lai"] = s["lai"].clip(lower=0)
    s[["field", "date", "clear_share", "offset"] + [c for t in TARGETS for c in (t, f"{t}_sd")]].to_csv(
        os.path.join(OUT, "retrieval_fields.csv"), index=False)
    print(f"{len(s)} clear field-dates for {s['field'].nunique()} fields; median per field "
          f"{s.groupby('field').size().median():.0f}; offset removed on {int((s['offset'] > 0).sum())} clips")
    print(s.assign(month=s["date"].dt.month).groupby("month")[["lai", "lai_sd", "cab"]].median().round(2).to_string())


if __name__ == "__main__":
    main()
