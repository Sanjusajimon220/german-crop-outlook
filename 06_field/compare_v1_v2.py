"""Paired confirmation test v1 vs v2 (pre-registered docs/project_log.md 2026-10-15): identical hidden pixel-dates."""
import json
import os
import sys

import numpy as np
import pandas as pd

OUT = os.path.join("data", "processed", "field_v2", "pilot_2024")
NP = "nophys" in sys.argv                                       # satellite-only: v1 = fusion2, v2 = fusion6
SFX = "_nophys" if NP else ""
MAIN6 = ["wheat", "barley", "sugar beet", "potato", "maize", "rapeseed"]


def load(tag, npz):
    d = pd.DatetimeIndex(np.load(os.path.join(OUT, npz))["dates"])
    e = pd.read_csv(os.path.join(OUT, f"{tag}_check.csv.gz"))
    e["date"] = d[e["date_i"].values]
    return e.drop(columns="date_i")


def main():
    v1, v2 = load("fusion2" if NP else "fusion3", "s2_pixels.npz"), load("fusion6" if NP else "fusion5", "s2_pixels_v2.npz")
    j = v1.merge(v2[["target", "pixel", "date", "live", "reanalysis", "gap"]], on=["target", "pixel", "date"], suffixes=("_v1", "_v2"))
    print(f"paired hidden values: {len(j)} (v1 {len(v1)}, v2 {len(v2)})")
    j["gap_class"] = pd.cut(j["gap_v1"], [0, 15, 30, 400], labels=["<=15 d", "16-30 d", ">30 d"])
    rm = lambda v: float(np.sqrt(np.mean(np.square(v))))
    rows, choice = [], {}
    for (t, gc), g in j.groupby(["target", "gap_class"], observed=True):
        for m in ("reanalysis", "live"):
            r1, r2 = rm(g[f"{m}_v1"] - g.obs), rm(g[f"{m}_v2"] - g.obs)
            cr = [(c, rm(h[f"{m}_v1"] - h.obs), rm(h[f"{m}_v2"] - h.obs)) for c in MAIN6 if len(h := g[g.crop == c])]
            better = sum(b < a for _, a, b in cr)
            use = "v2" if r2 < r1 and better >= 4 else "v1"
            rows.append(dict(target=t, gap=gc, method=m, n=len(g), line=rm(g.interp - g.obs), v1=r1, v2=r2,
                             crops_v2_better=f"{better}/{len(cr)}", wheat_v1=dict((c, a) for c, a, _ in cr).get("wheat"),
                             wheat_v2=dict((c, b) for c, _, b in cr).get("wheat"), use=use))
            choice[f"{t}|{gc}|{m}"] = {"use": use, "rmse": r2 if use == "v2" else r1}
    s = pd.DataFrame(rows).round(3)
    s.to_csv(os.path.join(OUT, f"compare_v1_v2{SFX}.csv"), index=False)
    json.dump(choice, open(os.path.join(OUT, f"choice_v1_v2{SFX}.json"), "w"), indent=1)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
