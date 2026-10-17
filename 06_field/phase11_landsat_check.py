"""Landsat vs Sentinel-2 cross-check on near-same-day pairs (pre-registered docs/project_log.md 2026-10-16).
Run `python phase11_landsat_check.py AREA YEAR [AREA YEAR ...]` -> data/processed/field_v2/landsat_check.csv"""
import os
import sys

import numpy as np
import pandas as pd

import fv2
from areas import out_dir


def field_means(v, px):
    ok = np.isfinite(v)
    w = np.nan_to_num(px["share"])
    F = len(px["fids"])
    tot = np.bincount(px["inv"], weights=w, minlength=F)
    sw = np.bincount(px["inv"], weights=w * ok, minlength=F)
    m = np.bincount(px["inv"], weights=np.where(ok, v, 0) * w, minlength=F) / np.where(sw > 0, sw, np.nan)
    return np.where(sw >= 0.5 * tot, m, np.nan)


def main(pairs):
    rows = []
    for area, year in pairs:
        px = fv2.pixels(area, year)
        s = np.load(os.path.join(out_dir(area, year), "s2_pixels_fv2.npz"))
        l = np.load(os.path.join(out_dir(area, year), "landsat_pixels.npz"))
        sd, ld = pd.DatetimeIndex(s["dates"]), pd.DatetimeIndex(l["dates"])
        for j, d in enumerate(ld):
            k = int(np.argmin(np.abs((sd - d).days)))
            if abs((sd[k] - d).days) > 1:
                continue
            for t in ("lai", "ndvi"):
                a, b = field_means(s[t][k], px), field_means(l[t][j], px)
                ok = np.isfinite(a) & np.isfinite(b)
                rows.append(pd.DataFrame({"area": area, "year": year, "date": d.date(), "target": t, "field_id": px["fids"][ok],
                                          "crop": pd.Series(px["fids"][ok]).map(px["fl"].set_index("field_id")["crop_group"]).values,
                                          "s2": a[ok], "landsat": b[ok], "platform": l["platform"][j]}))
    r = pd.concat(rows, ignore_index=True)
    r.to_csv(os.path.join("data", "processed", "field_v2", "landsat_check.csv"), index=False)
    for (t, ar, y), g in r.groupby(["target", "area", "year"]):
        e = g.landsat - g.s2
        print(f"{t:4s} {ar} {y}: pairs {g.date.nunique()} dates / {len(g)} field-dates; bias {e.mean():+.3f}, RMSE {np.sqrt((e ** 2).mean()):.3f}, "
              f"r {g.s2.corr(g.landsat):.3f}")
    for t, g in r.groupby("target"):
        e = g.landsat - g.s2
        slope, icpt = np.polyfit(g.landsat, g.s2, 1)
        print(f"ALL {t}: n {len(g)}, bias {e.mean():+.3f}, RMSE {np.sqrt((e ** 2).mean()):.3f}, r {g.s2.corr(g.landsat):.3f}; "
              f"fit S2 = {slope:.3f} x Landsat {icpt:+.3f}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main([(a[i], int(a[i + 1])) for i in range(0, len(a), 2)])
