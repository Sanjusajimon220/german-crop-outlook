"""Production wheat assimilation with the adopted timing variant (ageing_choice.json) + yield level report
(pre-registered docs/project_log.md 2026-10-15: report only)."""
import json
import os

import numpy as np
import pandas as pd

import phase9_assim_wheat as A
import phase9b_assim_ageing as G

v = json.load(open(os.path.join(A.OUT, "ageing_choice.json")))["variant"]
f = A.wheat_fields()
obs = A.observations()
f = f[f["field_id"].isin(obs["field_id"])]
ens = G.ensembles(A.YEAR, f, heading=v in ("B1", "B3"), wide=v in ("B2", "B3"))
A.run(f, obs, ens, A.soil_ratio(), "B")
r = pd.read_csv(os.path.join(A.OUT, "assim_wheat_2024.csv"), dtype={"district": str})
# raw posterior physics yield (not scaled to the district model): rel * ensemble prior mean
pm = {d: ens[d]["prior_mean"] for d in ens}
r["raw_phys_yield"] = r["rel"] * r["district"].map(pm)
off = {"05358": 7.18, "05370": 7.07}
for d, g in r.groupby("district"):
    print(f"{d}: official 2024 {off[d]:.2f} | scaled to district model {np.average(g.yield_t_ha, weights=g.area_ha):.2f} | "
          f"raw posterior physics {np.average(g.raw_phys_yield, weights=g.area_ha):.2f} t/ha (variant {v})")
r.to_csv(os.path.join(A.OUT, "assim_wheat_2024.csv"), index=False)
