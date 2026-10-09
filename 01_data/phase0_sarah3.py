"""Satellite sunlight: CM SAF SARAH-3 daily surface solar radiation (SIS, W/m2, 0.05 deg).

Citation: Pfeifroth, U. et al. (2023): Surface Radiation Data Set - Heliosat (SARAH) - Edition 3,
CM SAF, DOI 10.5676/EUM_SAF_CM/SARAH/V003 (CDR to 2020, interim record 2021 on).
Input: the order as delivered (data/raw/sarah3/*.tar with daily NetCDF files, or loose *.nc),
read directly from the archives - nothing is unpacked to disk. Each daily file covers the whole
Meteosat disk; only two boxes are kept:
  Rur box      lat 50.2-51.4, lon 5.7-7.4 (field model)        -> data/processed/field/sarah3_rur.npz
  Germany box  lat 47.2-55.1, lon 5.8-15.1 (district model)    -> data/processed/weather/sarah3_germany.npz
(dates, lat, lon, sis[days, lat, lon]; Germany as float16 to keep it small). The large raw files
are only deleted with `--delete-raw`, after the owner agrees.
Run `python phase0_sarah3.py` (add --delete-raw to remove the processed archives afterwards).
"""
import glob
import os
import sys
import tarfile

import netCDF4
import numpy as np

RAW = os.path.join("data", "raw", "sarah3")
BOXES = {"rur": ((50.2, 51.4), (5.7, 7.4), os.path.join("data", "processed", "field", "sarah3_rur.npz"), np.float32),
         "germany": ((47.2, 55.1), (5.8, 15.1), os.path.join("data", "processed", "weather", "sarah3_germany.npz"),
                     np.float16)}


def daily_files():
    """Yield (name, bytes) for every daily NetCDF file in the archives / folder."""
    for path in sorted(glob.glob(os.path.join(RAW, "**", "*"), recursive=True)):
        if path.endswith((".tar", ".tar.gz", ".tgz")):
            with tarfile.open(path) as t:
                for m in t:
                    if m.isfile() and m.name.endswith(".nc"):
                        yield os.path.basename(m.name), t.extractfile(m).read()
        elif path.endswith(".nc"):
            with open(path, "rb") as f:
                yield os.path.basename(path), f.read()


def main():
    out = {k: {"dates": [], "sis": []} for k in BOXES}
    idx = {}
    n = 0
    for name, data in daily_files():
        with netCDF4.Dataset(name, memory=data) as d:
            lat, lon = d["lat"][:], d["lon"][:]
            if not idx:
                for k, ((a, b), (c, e), _, _) in BOXES.items():
                    idx[k] = (np.flatnonzero((lat >= a) & (lat <= b)), np.flatnonzero((lon >= c) & (lon <= e)))
            t = netCDF4.num2date(d["time"][:], d["time"].units)
            sis = d["SIS"]
            for k, (iy, ix) in idx.items():
                v = sis[:, iy.min():iy.max() + 1, ix.min():ix.max() + 1].astype(np.float32).filled(np.nan)
                for j, tt in enumerate(t):
                    out[k]["dates"].append(np.datetime64(f"{tt.year:04d}-{tt.month:02d}-{tt.day:02d}"))
                    out[k]["sis"].append(v[j])
        n += 1
        if n % 500 == 0:
            print(f"  {n} daily files read", flush=True)
    for k, ((a, b), (c, e), path, dtype) in BOXES.items():
        dates = np.array(out[k]["dates"])
        o = np.argsort(dates)
        dates, keep = np.unique(dates[o], return_index=True)
        sis = np.stack(out[k]["sis"])[o][keep].astype(dtype)
        iy, ix = idx[k]
        np.savez_compressed(path, dates=dates, lat=lat[iy.min():iy.max() + 1], lon=lon[ix.min():ix.max() + 1], sis=sis)
        print(f"{k}: {len(dates)} days ({dates.min()} - {dates.max()}), grid {sis.shape[1]} x {sis.shape[2]}, "
              f"mean {np.nanmean(sis.astype(np.float32)):.0f} W/m2, missing days in range "
              f"{(dates.max() - dates.min()).astype(int) + 1 - len(dates)} -> {path}")
    if "--delete-raw" in sys.argv:
        for p in glob.glob(os.path.join(RAW, "**", "*"), recursive=True):
            if p.endswith((".tar", ".tar.gz", ".tgz", ".nc")):
                os.remove(p)
        print("raw archives deleted")


if __name__ == "__main__":
    main()
