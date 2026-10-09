"""v11 ranges for the test years 2018-2025 (+ 2026 national/state/district), method chosen on training years.

Same normal-error method as ranges_adaptive.py (weather sd from the scenarios + year-wide sd + local sd);
static or adaptive per crop as decided in the project log (adaptive: barley, grain maize; potato as decided
when its result is in). Error history available at year t: v8 rolling forward errors 1991-2008, v9 forward
lead-0 errors 2009-2017, v11 lead-0 errors of test years <= t-1 (year part) / <= t-2 (local part).
Output per level: median, 10 / 90 % bounds, observed (district) -> data/processed/ranges_v11/ranges_<crop>_leadNN.csv
"""
import os
import sys

import numpy as np
import pandas as pd

from phase5_benchmarks import census_area
from ranges_adaptive import F, P, Z, history

ADAPTIVE = {"winter_barley", "grain_maize"}
TAG = next((a for a in sys.argv[1:] if a.startswith("_v")), "_v11")
OUT = os.path.join(P, f"ranges{TAG}_final")


def main(crops):
    os.makedirs(OUT, exist_ok=True)
    y = pd.read_csv(os.path.join(P, "yields_long.csv"), dtype={"district": str})
    for crop in crops:
        adaptive = crop in ADAPTIVE or (crop == "potato" and "potato_adaptive" in sys.argv)
        ym, h = history(crop)
        e0 = pd.read_csv(os.path.join(F, f"forecast_{crop}{TAG}_lead00.csv"), dtype={"district": str})
        q0 = e0.groupby(["district", "year"]).agg(med=("blend", "median"), obs=("obs", "first")).reset_index().dropna()
        t_err = pd.DataFrame({"district": q0["district"], "year": q0["year"], "e": q0["med"] - q0["obs"]})
        t_err = t_err[t_err["year"] >= 2018]
        h = pd.concat([h[["district", "year", "e"]], t_err], ignore_index=True)
        ym = h.groupby("year")["e"].mean()
        h["loc"] = h["e"] - h["year"].map(ym)
        yc = y[y["crop"] == crop].copy()
        yc["area"] = census_area(yc)
        area = yc.set_index(["district", "year"])["area"]
        for lead in (12, 8, 6, 4, 2, 0):
            f = pd.read_csv(os.path.join(F, f"forecast_{crop}{TAG}_lead{lead:02d}.csv"), dtype={"district": str})
            q = f.groupby(["district", "year"])["blend"].quantile([0.1, 0.5, 0.9]).unstack().reset_index()
            q = q.merge(f.groupby(["district", "year"])[["obs", "role"]].first().reset_index(), on=["district", "year"])
            q["wsd"] = (q[0.9] - q[0.1]) / (2 * Z)
            rows = []
            for t, g in q.groupby("year"):
                yh, lh = ym[ym.index < t], h[h["year"] <= t - 2]["loc"]
                ysd, lsd = yh.std(ddof=1), lh.std()
                if adaptive:
                    ry = ym[(ym.index >= t - 5) & (ym.index <= t - 1)]
                    rl = h[(h["year"] >= t - 6) & (h["year"] <= t - 2)]["loc"]
                    ysd, lsd = max(ysd, np.sqrt((ry ** 2).mean())), max(lsd, np.sqrt((rl ** 2).mean()))
                sd = np.sqrt(g["wsd"] ** 2 + ysd ** 2 + lsd ** 2)
                for (_, r), s in zip(g.iterrows(), sd):
                    rows.append(dict(level="district", region=r["district"], year=t, role=r["role"], obs=r["obs"],
                                     median=r[0.5], low80=r[0.5] - Z * s, high80=r[0.5] + Z * s))
                a = np.array([area.get((d, t), np.nan) for d in g["district"]])
                ok = np.isfinite(a) & (a > 0)
                g, a = g[ok], a[ok]
                for level, keys in (("national", np.array(["DE"] * len(g))), ("state", g["district"].str[:2].values)):
                    for k in np.unique(keys):
                        s, aa = g[keys == k], a[keys == k]
                        wt = aa / aa.sum()
                        med = float((wt * s[0.5]).sum())
                        sdn = np.sqrt(float((wt * s["wsd"]).sum()) ** 2 + ysd ** 2 + lsd ** 2 * float((wt ** 2).sum()))
                        rows.append(dict(level=level, region=k, year=t, role="", obs=np.nan, median=med,
                                         low80=med - Z * sdn, high80=med + Z * sdn))
            pd.DataFrame(rows).to_csv(os.path.join(OUT, f"ranges_{crop}_lead{lead:02d}.csv"), index=False)
        print(crop, "adaptive" if adaptive else "static", "ranges written", flush=True)


if __name__ == "__main__":
    main([a for a in sys.argv[1:] if not a.endswith("_adaptive") and not a.startswith("_v")] or
         ["winter_wheat", "winter_barley", "grain_maize", "silage_maize", "potato"])
