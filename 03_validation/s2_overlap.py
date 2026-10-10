"""Overlap check 2023: Sentinel-2 district x crop greenness sampled on the CLMS map vs the DLR map (pass rule in
docs/project_log.md, 2026-10-10 13:30)."""
import json
import os
import sys

import numpy as np
import pandas as pd

from s2_features import dekads, feats

S2 = os.path.join("data", "processed", "s2_district")
CELLS = {"wheat": [("winter_wheat", 12), ("winter_wheat", 6)], "barley": [("winter_barley", 8)],
         "maize": [("silage_maize", 6), ("silage_maize", 4)]}


def load(tag, year):
    smp = pd.read_csv(os.path.join(S2, f"sample_{tag}.csv"), dtype={"AGS": str})
    ts = pd.read_csv(os.path.join(S2, f"sample_{tag}", "timeseries.csv")).rename(columns={"band_unnamed": "ndvi"}).dropna(subset=["ndvi"])
    return {(x["AGS"], x["crop"].replace("potatoes", "potato")): dekads(ts[ts["feature_index"] == i], year)
            for i, x in smp.iterrows() if x["cells"] >= 5}


def main():
    year, states = 2023, sys.argv[1:] or ["16", "03"]
    meta = json.load(open(os.path.join("platform", "data", "meta.json")))["crops"]
    a, b = {}, {}
    for st in states:
        a.update(load(f"{st}_{year}_n25_cheap", year))
        b.update(load(f"{st}_{year}_n25_cheap_dlr", year))
    ok_all = True
    for crop in ("wheat", "barley", "maize", "potato"):
        keys = [k for k in a if k in b and k[1] == crop]
        p = pd.concat([pd.DataFrame({"clms": a[k], "dlr": b[k]}).dropna() for k in keys]) if keys else pd.DataFrame()
        if p.empty:
            continue
        r, diff = p.corr().iloc[0, 1], (p["dlr"] - p["clms"]).mean()
        fr = []
        for c, lead in CELLS.get(crop, []):
            doy = meta[c]["harvest_doy"] - 7 * lead
            f = pd.DataFrame([{**{"c_" + n: v for n, v in (feats(a[k], doy) or {}).items()},
                               **{"d_" + n: v for n, v in (feats(b[k], doy) or {}).items()}} for k in keys]).dropna()
            fr += [f["c_" + n].corr(f["d_" + n]) for n in ("peak", "last3", "integral")]
        ok = r >= 0.90 and abs(diff) <= 0.03 and (not fr or min(fr) >= 0.85)
        if crop != "potato":
            ok_all &= ok
        print(f"{crop:7s} districts {len(keys):3d} pairs {len(p):4d}  r {r:.3f}  mean diff {diff:+.3f}  "
              f"feature r min {min(fr) if fr else float('nan'):.3f}  {'PASS' if ok else 'FAIL'}")
    print("OVERLAP", "PASS" if ok_all else "FAIL")


if __name__ == "__main__":
    main()
