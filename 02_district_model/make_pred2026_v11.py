"""v11 2026 predictions (frozen 2026-10-09 for the spring-2027 comparison with v10): v10 2026 district
predictions scaled by fused / ours with the latest 2026 MARS bulletin, where fusion was adopted (wheat, grain
maize; late phase weights from fusion_train.csv). State means as phase2_predict2026.aggregate."""
import os
import numpy as np
import pandas as pd
import phase5_benchmarks as pb
from phase2_predict2026 import OUT, aggregate
from build_v11 import phase
fus = pd.read_csv(os.path.join(pb.P, "fusion_train.csv"))
fus = fus[fus["adopt"]].set_index(["crop", "month"])["w_ours"]
mars = pb.official(pb.truth())
mars = mars[mars["source"].str.startswith("mars") & (mars["year"] == 2026)]
y = pd.read_csv(os.path.join(pb.P, "yields_long.csv"), dtype={"district": str})
for crop in pb.CROPS:
    p = pd.read_csv(os.path.join(OUT, f"pred2026_{crop}_v10.csv"), dtype={"district": str})
    b = mars[mars["crop"] == crop].sort_values("date")
    note = "no fusion"
    if len(b) and (crop, phase(b.iloc[-1]["date"].month)) in fus.index:
        last = b.iloc[-1]
        yc = y[(y["crop"] == crop) & (y["area_ha"] > 0)].sort_values("year")
        area = p["district"].map(yc.groupby("district")["area_ha"].last())
        ok = area.notna()
        ours = float(np.average(p.loc[ok, "blend"], weights=area[ok]))
        w = float(fus.loc[(crop, phase(last["date"].month))])
        fused = w * ours + (1 - w) * float(last["value"])
        p["blend"] = p["blend"] * fused / ours
        note = f"MARS {last['date'].date()} {last['value']:.2f}, w {w}, national {ours:.2f} -> {fused:.2f}"
    p.to_csv(os.path.join(OUT, f"pred2026_{crop}_v11.csv"), index=False)
    aggregate(crop, "_v11")
    print(crop, note, flush=True)
