"""v10 test look, district part: RMSE of the median forecast per lead, v9 vs v10, test years 2018-2025
(test_years = known districts, test_both = held-out states)."""
import os
import numpy as np
import pandas as pd
F = os.path.join("data", "processed", "forecast")
rows = []
for crop in ("winter_wheat", "winter_barley", "grain_maize", "silage_maize", "potato"):
    for lead in (12, 8, 6, 4, 2, 0):
        out = {}
        for tag in ("_v9", "_v10"):
            f = pd.read_csv(os.path.join(F, f"forecast_{crop}{tag}_lead{lead:02d}.csv"), dtype={"district": str})
            q = f.groupby(["district", "year", "role"]).agg(med=("blend", "median"), obs=("obs", "first")).reset_index().dropna(subset=["obs"])
            out[tag] = q
        for role in ("test_years", "test_both"):
            r = {t: q[q["role"] == role] for t, q in out.items()}
            rm = {t: float(np.sqrt(((q["med"] - q["obs"]) ** 2).mean())) for t, q in r.items()}
            rows.append(dict(crop=crop, lead=lead, role=role, n=len(r["_v10"]), rmse_v9=rm["_v9"], rmse_v10=rm["_v10"],
                             change_pct=100 * (rm["_v10"] / rm["_v9"] - 1),
                             bias_v9=float((r["_v9"]["med"] - r["_v9"]["obs"]).mean()),
                             bias_v10=float((r["_v10"]["med"] - r["_v10"]["obs"]).mean())))
s = pd.DataFrame(rows).round(3)
s.to_csv(os.path.join("data", "processed", "v10_district_score.csv"), index=False)
with pd.option_context("display.width", 200, "display.max_rows", 100):
    print(s.to_string(index=False))
