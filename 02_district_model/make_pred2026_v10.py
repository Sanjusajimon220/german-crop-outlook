"""v10 predictions for 2026 (frozen 2026-10-09 for the clean check against official 2026 district yields,
spring 2027). Base: v9 2026 predictions (check2026/pred2026_<crop>_v9.csv; barley from the refit-2023 model,
pred2026_winter_barley_v9_r2023.csv), minus the v10 corrections for 2026 (district offsets from errors
<= 2024; grain maize national update). State means as phase2_predict2026.aggregate (latest census area)."""
import os
import pandas as pd
from phase2_predict2026 import aggregate, OUT
F = os.path.join("data", "processed", "forecast")
for crop in ("winter_wheat", "winter_barley", "grain_maize", "silage_maize", "potato"):
    src = "pred2026_winter_barley_v9_r2023.csv" if crop == "winter_barley" else f"pred2026_{crop}_v9.csv"
    p = pd.read_csv(os.path.join(OUT, src), dtype={"district": str})
    c = pd.read_csv(os.path.join(F, f"v10_corrections_{crop}.csv"), dtype={"district": str})
    c = c[c["year"] == 2026].set_index("district")
    corr = p["district"].map(c["offset"]).fillna(0) + (c["national"].iloc[0] if len(c) else 0)
    for col in ("blend",):
        p[col] = p[col] - corr
    p.to_csv(os.path.join(OUT, f"pred2026_{crop}_v10.csv"), index=False)
    aggregate(crop, "_v10")
