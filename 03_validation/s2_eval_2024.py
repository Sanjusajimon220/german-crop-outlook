"""v11.1 item 1: frozen S2 correction on 2024, the first year not used to fit it (pre-registered
docs/project_log.md 2026-10-10 13:45). Features 2024 from DLR-labelled cells (s2_features_cells.csv,
source dlr_label). One look, no refit. Success: district RMSE better in >= 3 of 5 cells and on the mean change.
"""
import json
import os

import numpy as np
import pandas as pd

from build_v11_1 import F, HELD, apply

P = os.path.join("data", "processed")


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e)))) if len(e) else np.nan


def main():
    model = json.load(open(os.path.join(P, "v11_1", "model.json")))
    s2 = pd.read_csv(os.path.join(P, "s2_features_cells.csv"), dtype={"district": str, "state": str})
    s2 = s2[(s2["year"] == 2024) & (s2["source"] == "dlr_label")]
    tr = pd.read_csv(os.path.join(P, "benchmarks", "truth_official.csv"), dtype={"region": str})
    tr = tr[tr["year"] == 2024].set_index(["crop", "region"])["yield_t_ha"]
    y = pd.read_csv(os.path.join(P, "yields_long.csv"), dtype={"district": str})
    rows = []
    for key, m in model["cells"].items():
        crop, lead = m["crop"], m["lead"]
        v = pd.read_csv(os.path.join(P, "ranges_v10_final", f"ranges_{crop}_lead{lead:02d}.csv"), dtype={"region": str})
        v = v[(v["level"] == "district") & (v["year"] == 2024)].rename(columns={"region": "district"})
        d = v.merge(s2[(s2["crop"] == crop) & (s2["lead"] == lead)], on="district", how="left").dropna(subset=["obs"])
        ok = d[F].notna().all(axis=1)
        d["new"] = d["median"]
        d.loc[ok, "new"] = d.loc[ok, "median"] + apply(m, d[ok])
        known, held = ok & ~d["district"].str[:2].isin(HELD), ok & d["district"].str[:2].isin(HELD)
        r = dict(cell=key, n=int(known.sum()), n_held=int(held.sum()),
                 dist_v10=rmse(d.loc[known, "median"] - d.loc[known, "obs"]), dist_v111=rmse(d.loc[known, "new"] - d.loc[known, "obs"]),
                 held_v10=rmse(d.loc[held, "median"] - d.loc[held, "obs"]), held_v111=rmse(d.loc[held, "new"] - d.loc[held, "obs"]))
        r["change_pct"] = 100 * (r["dist_v111"] / r["dist_v10"] - 1)
        a = y[(y["crop"] == crop) & (y["area_ha"] > 0)].sort_values("year").groupby("district")["area_ha"].last()
        d["a"] = d["district"].map(a)
        dd = d.dropna(subset=["a"])
        for lvl, k in (("state", dd["district"].str[:2]), ("nat", pd.Series("DE", index=dd.index))):
            g = dd.assign(k=k).groupby("k").apply(lambda x: pd.Series({c: np.average(x[c], weights=x["a"]) for c in ("median", "new")}))
            g["truth"] = [tr.get((crop, s), np.nan) for s in g.index]
            g = g.dropna()
            r[f"{lvl}_v10"], r[f"{lvl}_v111"] = rmse(g["median"] - g["truth"]), rmse(g["new"] - g["truth"])
        rows.append(r)
    o = pd.DataFrame(rows)
    o.to_csv(os.path.join(P, "s2_eval_2024.csv"), index=False)
    print(o.round(3).to_string(index=False))
    better = int((o["change_pct"] < 0).sum())
    print(f"cells better: {better}/5, mean change {o['change_pct'].mean():+.1f} % ->",
          "SUCCESS" if better >= 3 and o["change_pct"].mean() < 0 else "NOT MET")


if __name__ == "__main__":
    main()
