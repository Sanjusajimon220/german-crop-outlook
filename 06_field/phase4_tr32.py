"""Phase 4, step 2c: check satellite LAI against measured (destructive) green LAI - TR32 fields.

Ground truth: Reichenau et al. 2020 (ESSD 12, 2333; data DOI 10.5880/TR32DB.39, CC BY 4.0),
data/raw/tr32. Winter wheat / barley field-seasons with LAI after Sentinel-2A started (July 2015).

1. `python phase4_tr32.py extract`: for each field-season, a 20 m circle around every sampling point
   (GPS, UTM 32N), merged per field; one openEO job for Jun 2015 - Aug 2017 with the same bands,
   angles and cloud mask (SCL 4/5) as the main field extraction -> data/processed/tr32/s2_tr32.csv
2. `python phase4_tr32.py compare`: our PROSAIL network and ESA SL2P on those spectra, matched to
   ground dates (satellite within +-3 days; nearest date) -> tr32/lai_matchups.csv + summary.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

P = os.path.join("data", "processed")
OUT = os.path.join(P, "tr32")
RAW = os.path.join("data", "raw", "tr32")
FIELDS = ["SEF01WW15", "SEF03WW15", "SEF05WW15", "SEF01WB16", "SEF05WW16", "SEF06WB16", "SEF06WB17",
          "MEF01WW17"]
SITE_FILE = {"SE": "selhausen/vegetation_SE.csv", "ME": "merzenhausen/vegetation_ME.csv"}


def ground():
    v = pd.concat([pd.read_csv(os.path.join(RAW, f), sep=";", skiprows=[1], na_values="NA", low_memory=False)
                   for f in SITE_FILE.values()], ignore_index=True)
    return v[v["dataset"].isin(FIELDS)]


def geometries():
    from pyproj import Transformer
    from shapely.geometry import MultiPoint, mapping
    from shapely.ops import transform
    v = ground().dropna(subset=["UTM_easting", "UTM_northing"])
    to_wgs = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True).transform
    feats = []
    for name, g in v.groupby("dataset"):
        pts = MultiPoint(list(zip(g["UTM_easting"], g["UTM_northing"])))
        geom = transform(to_wgs, pts.buffer(20).simplify(1))
        feats.append({"type": "Feature", "properties": {"dataset": name, "points": len(g)},
                      "geometry": mapping(geom)})
    return {"type": "FeatureCollection", "features": feats}


def extract():
    import openeo
    from phase4_s2_fields import ANGLES, BANDS
    os.makedirs(OUT, exist_ok=True)
    gj = geometries()
    json.dump(gj, open(os.path.join(OUT, "tr32_fields.geojson"), "w"))
    print(f"{len(gj['features'])} field-seasons:", [f["properties"]["dataset"] for f in gj["features"]])
    con = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()
    old = [j for j in con.list_jobs() if j.get("title") == "TR32 LAI check" and
           j.get("status") in ("finished", "running", "queued")]
    if old:     # reuse the job already on the server
        job = con.job(old[0]["id"])
    else:
        cube = con.load_collection("SENTINEL2_L2A", temporal_extent=["2015-06-23", "2017-08-31"],
                                   bands=BANDS + ANGLES + ["SCL"], max_cloud_cover=80)
        scl = cube.band("SCL")
        cube = cube.filter_bands(BANDS + ANGLES).mask(~((scl == 4) | (scl == 5)))
        job = cube.aggregate_spatial(geometries=gj, reducer="mean").create_job(title="TR32 LAI check",
                                                                               out_format="CSV")
        job.start()
    job.start_and_wait() if job.status() == "created" else None
    import time
    while job.status() in ("queued", "running"):
        time.sleep(60)
    if job.status() != "finished":
        raise RuntimeError(f"job {job.status()}")
    folder = os.path.join(OUT, "job")
    job.get_results().download_files(folder)
    df = pd.concat([pd.read_csv(os.path.join(folder, f)) for f in os.listdir(folder) if f.endswith(".csv")])
    df["dataset"] = df["feature_index"].map({i: f["properties"]["dataset"] for i, f in enumerate(gj["features"])})
    df.to_csv(os.path.join(OUT, "s2_tr32.csv"), index=False)
    print(f"{len(df)} rows, {df.dropna(subset=['B04']).shape[0]} cloud-free; "
          f"cpu {job.describe().get('usage', {}).get('cpu')}")


def compare():
    import torch
    from phase4_prosail import BANDS, MODEL, TARGETS, Net, inputs
    from phase4_sl2p import BANDS as SB, SL2P
    s = pd.read_csv(os.path.join(OUT, "s2_tr32.csv")).dropna(subset=BANDS)
    s["date"] = pd.to_datetime(s["date"]).dt.tz_localize(None).dt.normalize()
    s = s.rename(columns={"sunZenithAngles": "sza", "viewZenithMean": "vza"})
    raa = np.abs(s["sunAzimuthAngles"] - s["viewAzimuthMean"]) % 360
    s["raa"] = np.where(raa > 180, 360 - raa, raa)
    s[BANDS] = s[BANDS] / 10000.0
    ck = torch.load(MODEL, weights_only=False)
    net = Net(len(BANDS) + 3, len(TARGETS))
    net.load_state_dict(ck["state"])
    with torch.no_grad():
        mu, lv = net(torch.tensor((inputs(s) - ck["xm"]) / ck["xs"]))
    s["ours"] = np.clip(mu.numpy()[:, 0] * ck["ys"][0] + ck["ym"][0], 0, None)
    s["ours_sd"] = np.exp(0.5 * lv.numpy()[:, 0]) * ck["ys"][0]
    x = np.column_stack([s[SB].values, np.cos(np.radians(s["vza"])), np.cos(np.radians(s["sza"])),
                         np.cos(np.radians(s["sunAzimuthAngles"] - s["viewAzimuthMean"]))])
    s["sl2p"], _ = SL2P("LAI")(x)

    g = ground().dropna(subset=["LAI_green"])
    g["bbch"] = pd.to_numeric(g["bbch"], errors="coerce")
    g = g.groupby(["dataset", "land_use", "date"]).agg(lai=("LAI_green", "mean"), lai_sd=("LAI_green", "std"),
                                                       n=("LAI_green", "size"), bbch=("bbch", "median")).reset_index()
    g["date"] = pd.to_datetime(g["date"])
    rows = []
    for r in g.itertuples():
        c = s[s["dataset"] == r.dataset]
        if c.empty:
            continue
        dd = (c["date"] - r.date).dt.days
        if dd.abs().min() <= 3:              # satellite within 3 days: take that date
            b = c.loc[dd.abs().idxmin()]
            sat = dict(how="same week", days_apart=int(dd.abs().min()), ours=b["ours"], sl2p=b["sl2p"])
        else:                                # otherwise interpolate between the dates before and after,
            before, after = c[dd < 0], c[dd > 0]     # if they are at most 3 weeks apart
            if before.empty or after.empty:
                continue
            a, b = before.loc[before["date"].idxmax()], after.loc[after["date"].idxmin()]
            gap = (b["date"] - a["date"]).days
            if gap > 21:
                continue
            w = (r.date - a["date"]).days / gap
            sat = dict(how="interpolated", days_apart=gap, ours=(1 - w) * a["ours"] + w * b["ours"],
                       sl2p=(1 - w) * a["sl2p"] + w * b["sl2p"])
        rows.append(dict(dataset=r.dataset, crop=r.land_use, date=r.date.date(), bbch=r.bbch, ground=r.lai,
                         ground_sd=r.lai_sd, n=r.n, **sat))
    m = pd.DataFrame(rows)
    m.to_csv(os.path.join(OUT, "lai_matchups.csv"), index=False)
    print(m.round(2).to_string(index=False))
    for name in ("ours", "sl2p"):
        d = (m[name] - m["ground"]).dropna()
        for how, dh in d.groupby(m.loc[d.index, "how"]):
            print(f"  {name} {how}: n {len(dh)}, mean diff {dh.mean():+.2f}, RMSE {np.sqrt((dh ** 2).mean()):.2f}")
        print(f"{name:5s}: n {len(d)}, mean diff {d.mean():+.2f}, RMSE {np.sqrt((d ** 2).mean()):.2f}, "
              f"r {np.corrcoef(m.loc[d.index, name], m.loc[d.index, 'ground'])[0, 1]:.2f}")
    hi = m[m["ground"] >= 3]
    print(f"ground LAI >= 3 (n {len(hi)}): ours {(hi['ours'] - hi['ground']).mean():+.2f}, "
          f"SL2P {(hi['sl2p'] - hi['ground']).mean():+.2f}")


if __name__ == "__main__":
    extract() if sys.argv[1] == "extract" else compare()
