"""Phase 4, step 4: field results summed to districts vs official district yields (frozen plan,
docs/project_log.md 2026-10-06). Field yields are averaged per district weighted by field area
("fields + satellite") and compared with the official yield and with the district model alone.
Run `python phase4_field_compare.py 2021 2022 ...` after phase4_field_assim.py run <year>.
"""
import os
import sys

import numpy as np
import pandas as pd

P = os.path.join("data", "processed")
CORE = {"05334": "Aachen", "05358": "Dueren", "05362": "Rhein-Erft", "05366": "Euskirchen", "05370": "Heinsberg"}


def main(years):
    pred = pd.read_csv(os.path.join(P, "growth_predictions_winter_wheat.csv"), dtype={"district": str})
    rows = []
    for y in years:
        f = pd.read_csv(os.path.join(P, "field", f"field_results_winter_wheat_{y}.csv"), dtype={"district": str})
        g = f.groupby("district").apply(lambda g: pd.Series({
            "fields": len(g), "area_ha": g["area_ha"].sum(),
            "fields_satellite": np.average(g["yield_t_ha"], weights=g["area_ha"])}), include_groups=False)
        g = g.reset_index().assign(year=y)
        rows.append(g.merge(pred[pred["year"] == y][["district", "yield_t_ha", "blend"]], on="district"))
    c = pd.concat(rows, ignore_index=True).rename(columns={"yield_t_ha": "official", "blend": "district_model"})
    c = c[c["fields"] >= 100]
    c["name"] = c["district"].map(CORE)
    c.round(2).to_csv(os.path.join(P, "field", "district_comparison_winter_wheat.csv"), index=False)
    print(c.round(2).to_string(index=False))

    def score(s, label):
        out = {}
        for k in ("district_model", "fields_satellite"):
            e = s[k] - s["official"]
            out[k] = f"RMSE {np.sqrt((e ** 2).mean()):.2f}, mean error {e.mean():+.2f}"
        print(f"{label:28s} n {len(s):3d} | district model: {out['district_model']} | "
              f"fields + satellite: {out['fields_satellite']}")
    print()
    core = c[c["name"].notna()]
    score(core, "5 core Rur districts")
    for y, s in core.groupby("year"):
        score(s, f"  {y}")
    score(c, "all districts >= 100 fields")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]])
