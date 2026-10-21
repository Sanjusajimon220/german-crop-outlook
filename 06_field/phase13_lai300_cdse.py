r"""LAI 300 m v2 from the Copernicus Data Space (CDSE) eodata S3 (owner OK 2026-10-20; CLMS portal down 2026-10-21).

Reads only our box (BOX, EPSG:4326 native grid, no resampling) from each 10-daily cloud-optimised GeoTIFF: layers lai, rmse,
qflag, nobs, 2020-10..2025-08 (177 dates; most consolidated RT per date). S3 keys from D:\vista\credentials\cdse_s3.json, never printed.
Output data/raw/hrvpp_portal/lai300/cdse/<YYYYMMDD>_<layer>.tif; resumable. `test` = first date only.
"""
import json
import os
import sys
import time

import rasterio
import requests
from rasterio.windows import from_bounds

OUT = r"D:\vista\data\raw\hrvpp_portal\lai300\cdse"
KEYS = r"D:\vista\credentials\cdse_s3.json"
BOX = (5.963, 50.589, 6.693, 51.101)                                    # W S E N
LAYERS = ("lai", "rmse", "qflag", "nobs")
STAC = "https://stac.dataspace.copernicus.eu/v1/search"


def items():
    body = {"collections": ["clms_lai_global_300m_10daily_v2_cog"], "bbox": list(BOX),
            "datetime": "2020-10-01T00:00:00Z/2025-08-31T23:59:59Z", "limit": 200}
    r = requests.post(STAC, json=body, timeout=120); r.raise_for_status()
    best = {}                                                          # one per date: the most consolidated (highest RTn)
    for f in r.json()["features"]:
        day, rt = f["id"].split("_")[3][:8], int(f["id"].split("_")[2].split("-RT")[1])
        if day not in best or rt > best[day][0]:
            best[day] = (rt, f)
    return [best[d][1] for d in sorted(best)]


def env():
    k = json.load(open(KEYS, encoding="utf-8-sig"))
    os.environ["AWS_ACCESS_KEY_ID"], os.environ["AWS_SECRET_ACCESS_KEY"] = k["access_key"], k["secret_key"]   # this process only
    return rasterio.Env(AWS_S3_ENDPOINT="eodata.dataspace.copernicus.eu", AWS_HTTPS="YES", AWS_VIRTUAL_HOSTING="FALSE",
                        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR")


def clip(href, dest):
    with rasterio.open("/vsis3/" + href.removeprefix("s3://")) as src:
        win = from_bounds(*BOX, src.transform).round_offsets().round_lengths()
        data = src.read(window=win)
        prof = {"driver": "GTiff", "dtype": data.dtype, "count": data.shape[0], "height": data.shape[1], "width": data.shape[2],
                "crs": src.crs, "transform": src.window_transform(win), "nodata": src.nodata, "compress": "deflate"}
        tags, scales, offsets = src.tags(), src.scales, src.offsets
    with rasterio.open(dest + ".part", "w", **prof) as dst:
        dst.write(data); dst.update_tags(**tags); dst.scales = scales; dst.offsets = offsets
    os.replace(dest + ".part", dest)
    return data


def main(test=False):
    os.makedirs(OUT, exist_ok=True)
    feats = items()
    print(f"{len(feats)} dates")
    n = 0
    with env():
        for f in feats:
            day = f["id"].split("_")[3][:8]
            for layer in LAYERS:
                dest = os.path.join(OUT, f"{day}_{layer}.tif")
                if os.path.exists(dest): continue
                for attempt in range(3):
                    try:
                        d = clip(f["assets"][layer]["href"], dest); break
                    except Exception as e:
                        print(f"{day} {layer}: {type(e).__name__} (attempt {attempt + 1})"); time.sleep(10)
                if test: print(day, layer, d.shape, d.dtype, "min", d.min(), "max", d.max())
            n += 1
            if test: return
            if n % 20 == 0: print(f"{n} dates done", flush=True)
    print("LAI300 CDSE DONE")


if __name__ == "__main__":
    main(test=sys.argv[1:2] == ["test"])
