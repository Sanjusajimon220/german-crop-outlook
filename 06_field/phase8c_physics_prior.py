"""Step 1a (pre-registered 2026-10-15): model-expected daily LAI for every WHEAT field of the pilot, 2024, from the
field model v4 PRIOR ensemble (no satellite data): field -> 1 km HYRAS cell -> the district's weather group ->
3,000-member ensemble daily LAI from sowing -> member mean. Output field_v2/pilot_2024/prior_lai_wheat.csv.gz
(field_id, date, prior_lai).
"""
import os

import numpy as np
import pandas as pd

import phase4_field_assim_v4 as v4
from phase2_growth import P

OUT = os.path.join(P, "field_v2", "pilot_2024")
YEAR = 2024


def main():
    fl = pd.read_csv(os.path.join(OUT, "fields_nrw.csv"))
    wheat = fl.loc[fl["crop"].str.contains("Winterweichweizen", na=False), "field_id"]
    cell = pd.read_csv(os.path.join(P, "field", "fields_hyras_cell.csv"))
    cell = cell[(cell["VALIDFROM"] == YEAR) & (cell["CODE"] == 115)].set_index("ID")["cell"]
    dist = pd.read_csv(os.path.join(P, "field", "fields_district.csv"), dtype={"district": str})
    dist = dist[(dist["VALIDFROM"] == YEAR) & (dist["CODE"] == 115)].set_index("ID")["district"]
    f = pd.DataFrame({"field_id": wheat.values})
    f["cell"], f["district"] = f["field_id"].map(cell), f["field_id"].map(dist)
    f = f.dropna().astype({"cell": int})
    print(f"wheat fields with weather cell and district: {len(f)} of {len(wheat)}; districts {f['district'].value_counts().to_dict()}", flush=True)
    ens = v4.ensembles(YEAR, f)
    rows = []
    for _, x in f.iterrows():
        e = ens.get(x["district"])
        if e is None or x["cell"] not in e["cell_group"]:
            continue
        g = e["groups"][e["cell_group"][x["cell"]]]
        lai = g["lai"].mean(0)                                   # member mean, days from sowing
        dates = pd.date_range(g["sowing"], periods=len(lai), freq="D")
        rows.append(pd.DataFrame({"field_id": x["field_id"], "date": dates, "prior_lai": lai.round(3)}))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(os.path.join(OUT, "prior_lai_wheat.csv.gz"), index=False)
    m = out.assign(month=out["date"].dt.strftime("%Y-%m")).groupby("month")["prior_lai"].median().round(2)
    print(f"{out['field_id'].nunique()} fields; median prior LAI by month: {m.to_dict()}")


if __name__ == "__main__":
    main()
