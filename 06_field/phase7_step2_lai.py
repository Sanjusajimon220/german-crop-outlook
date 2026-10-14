"""Field v2 step 2 (pre-registered 2026-10-14): Sentinel-2 green LAI and NDVI per 10 m pixel, pilot cell 38_706, 2024.

Per S2 date (one GeoTIFF, bands B02..B12, SCL, 4 angle bands; same 800 x 800 grid as step 1):
  clear = SCL 4 / 5 and at least 2 pixels away from SCL 3 (shadow), 8, 9 (cloud), 10 (cirrus), 11 (snow);
  December / January dropped; a date is kept if >= 20 % of field pixels are clear.
  PROSAIL network (prosail_net.pt, unchanged) -> lai, cab, ccc, cbrown with sd; NDVI from B04 / B08.
Only pixels touching an NRW field (step 1). Output data/processed/field_v2/pilot_2024/s2_pixels.npz
(dates, rows, cols, lai, lai_sd, ndvi, ccc as [dates x pixels], NaN = not clear).
Check 2d: per wheat field, area-weighted pixel LAI (step-1 shares) vs the existing field-mean retrieval
(retrieval_115_2024.csv, from inward-shrunk outlines) on the same dates.
"""
import glob
import os

import numpy as np
import pandas as pd
import rasterio
import torch
from scipy.ndimage import binary_dilation

from phase4_prosail import BANDS, MODEL, TARGETS, Net

P = os.path.join("data", "processed")
RAW = os.path.join("data", "raw", "field_v2", "pilot_2024", "s2")
OUT = os.path.join(P, "field_v2", "pilot_2024")
BAD = (3, 8, 9, 10, 11)


def main():
    px = pd.read_csv(os.path.join(OUT, "pixels_nrw.csv.gz"))
    upix = px[["row", "col"]].drop_duplicates().sort_values(["row", "col"]).reset_index(drop=True)
    rows, cols = upix["row"].values, upix["col"].values
    ck = torch.load(MODEL, weights_only=False)
    net = Net(len(BANDS) + 3, len(TARGETS))
    net.load_state_dict(ck["state"])
    dates, out = [], {k: [] for k in ("lai", "lai_sd", "ndvi", "ccc")}
    for f in sorted(glob.glob(os.path.join(RAW, "*.tif"))):
        d = pd.Timestamp(os.path.basename(f)[7:17])
        with rasterio.open(f) as r:
            a = r.read()
        scl = a[10]
        bad = binary_dilation(np.isin(scl, BAD), iterations=2)
        clear = np.isin(scl, (4, 5)) & ~bad & np.all(np.isfinite(a[:10]), axis=0)
        c = clear[rows, cols]
        if d.month in (12, 1) or c.mean() < 0.2:
            print(f"{d.date()}: clear {100 * c.mean():.0f} % of field pixels - skipped", flush=True)
            continue
        refl = a[:10, rows, cols].T / 10000.0
        sza, vza = a[11, rows, cols], a[12, rows, cols]
        raa = np.abs(a[13, rows, cols] - a[14, rows, cols]) % 360
        raa = np.where(raa > 180, 360 - raa, raa)
        x = np.concatenate([refl, np.cos(np.radians(np.stack([sza, vza, raa], 1)))], 1).astype(np.float32)
        ok = c & np.all(np.isfinite(x), axis=1)
        lai, sd, ccc = np.full(len(rows), np.nan), np.full(len(rows), np.nan), np.full(len(rows), np.nan)
        with torch.no_grad():
            mu, lv = net(torch.tensor((x[ok] - ck["xm"]) / ck["xs"], dtype=torch.float32))
        pred, psd = mu.numpy() * ck["ys"] + ck["ym"], np.exp(0.5 * lv.numpy()) * ck["ys"]
        lai[ok], sd[ok], ccc[ok] = np.clip(pred[:, 0], 0, None), psd[:, 0], pred[:, 2]
        ndvi = np.where(ok, (refl[:, 6] - refl[:, 2]) / (refl[:, 6] + refl[:, 2] + 1e-6), np.nan)
        dates.append(d)
        for k, v in (("lai", lai), ("lai_sd", sd), ("ndvi", ndvi), ("ccc", ccc)):
            out[k].append(v.astype(np.float32))
        print(f"{d.date()}: clear {100 * c.mean():.0f} %, median LAI {np.nanmedian(lai):.2f}, NDVI {np.nanmedian(ndvi):.2f}", flush=True)
    np.savez_compressed(os.path.join(OUT, "s2_pixels.npz"), dates=np.array(dates, dtype="datetime64[D]"), rows=rows,
                        cols=cols, **{k: np.stack(v) for k, v in out.items()})
    # check 2d: wheat fields, pixel-based field mean vs the existing field-mean retrieval
    fl = pd.read_csv(os.path.join(OUT, "fields_nrw.csv"))
    wheat = set(fl.loc[fl["crop"].str.contains("Winterweichweizen", na=False), "field_id"])
    pw = px[px["field_id"].isin(wheat)].merge(upix.reset_index().rename(columns={"index": "pi"}), on=["row", "col"])
    lai = np.stack(out["lai"])
    recs = []
    for k, d in enumerate(dates):
        v = lai[k, pw["pi"].values]
        m = np.isfinite(v)
        g = pd.DataFrame({"field_id": pw["field_id"].values[m], "w": pw["share"].values[m], "v": v[m]})
        if g.empty:
            continue
        s = g.groupby("field_id").apply(lambda z: pd.Series({"lai_px": np.average(z["v"], weights=z["w"]), "wsum": z["w"].sum()}),
                                        include_groups=False)
        s["date"] = d
        recs.append(s.reset_index())
    pxm = pd.concat(recs)
    pxm = pxm.merge(pw.groupby("field_id")["share"].sum().rename("wtot"), on="field_id")
    pxm = pxm[pxm["wsum"] >= 0.5 * pxm["wtot"]]                  # at least half the field clear
    ref = pd.read_csv(os.path.join(P, "s2", "retrieval_115_2024.csv"))
    ref["date"] = pd.to_datetime(ref["date"]).dt.tz_localize(None).dt.normalize()
    j = pxm.merge(ref[["field_id", "date", "lai"]], on=["field_id", "date"])
    print(f"\ncheck 2d: {j['field_id'].nunique()} wheat fields, {len(j)} field-dates; pixel-based vs field-mean retrieval: "
          f"r {j['lai_px'].corr(j['lai']):.3f}, mean diff {(j['lai_px'] - j['lai']).mean():+.3f}, "
          f"RMSE {np.sqrt(((j['lai_px'] - j['lai']) ** 2).mean()):.3f} LAI")
    j.to_csv(os.path.join(OUT, "check2d_fieldmeans.csv"), index=False)


if __name__ == "__main__":
    main()
