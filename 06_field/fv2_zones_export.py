"""Module A zones for the platform (owner OK 2026-10-21): per area, for the latest season's field boundaries.

docs/field/<area>_2025/zones.png  : 800 x 800 grey PNG, 0 = no zone, 85 / 170 / 255 = weak / middle / strong (5-season zones)
docs/field/<area>_2025/zones.json : per field id: held (seasons the zone order held), tests, spread (median zone-mean spread),
                                     seasons used, and weak minus strong slope (deg), TWI and nFK (mm) where available.
Run from D:\\vista after phase18_zones.py and phase19_zone_checks.py: python fv2_zones_export.py
"""
import json
import os
import time

import numpy as np
import pandas as pd
from PIL import Image

ZONES = r"D:\vista\data\processed\field_v2\zones"
DOCS = r"D:\german-crop-outlook-field\docs\field"
SEASON = 2025
VARS = {"slope_deg": "slope", "twi": "twi", "nfk_mm": "nfk"}


def zone_png_values(z):
    return (np.clip(z, 0, 3).astype(np.int16) * 85).astype(np.uint8)


def field_meta(tests, checks, area):
    t = tests[tests.area == area]
    c = checks[checks.area == area]
    out = {}
    for f, g in t.groupby("field"):
        s = g.score.dropna()
        sp = g.spread.median()
        m = {"held": int((s >= 0.5).sum()), "tests": int(len(s)), "spread": round(float(sp), 3) if np.isfinite(sp) else None,
             "seasons": int(g.n_seasons.max())}
        for _, r in c[c.field == f].iterrows():
            if r["var"] in VARS and np.isfinite(r.weak_minus_strong):
                m[VARS[r["var"]]] = round(float(r.weak_minus_strong), 2)
        out[str(int(f))] = m
    return out


def run():
    tests = pd.read_csv(os.path.join(ZONES, "zones_test.csv"))
    checks = pd.read_csv(os.path.join(ZONES, "zone_checks.csv"))
    for area in ["pilot", "selhausen", "north", "south"]:
        d = os.path.join(DOCS, f"{area}_{SEASON}")
        Image.fromarray(zone_png_values(np.load(os.path.join(ZONES, f"zones_{area}_{SEASON}.npy"))), "L").save(os.path.join(d, "zones.png"), optimize=True)
        meta = field_meta(tests, checks, area)
        with open(os.path.join(d, "zones.json"), "w", encoding="utf-8") as f:
            json.dump({"season": SEASON, "seasons": "2021-2025", "fields": meta}, f, separators=(",", ":"), allow_nan=False)
        print(area, len(meta), "fields", os.path.getsize(os.path.join(d, "zones.png")) // 1024, "KB png")
    idx_path = os.path.join(DOCS, "index.json")                       # new data version: browsers fetch the new field files
    idx = json.load(open(idx_path, encoding="utf-8"))
    idx["version"] = time.strftime("%Y%m%d%H%M%S")
    with open(idx_path + ".part", "w", encoding="utf-8") as f:
        json.dump(idx, f, indent=1)
    os.replace(idx_path + ".part", idx_path)


if __name__ == "__main__":
    run()
