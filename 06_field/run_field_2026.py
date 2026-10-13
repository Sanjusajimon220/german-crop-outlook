"""Field Outlook 2026 (current season): run the steps that need the 2026 district rows.

The district tables hold 2026 only as 'extra' rows (no yield yet, phase2_predict2026.extra_rows) and the 2026
weather is weather_daily_v8_2026.npz, so: run with VISTA_WEATHER=v8_2026 VISTA_MODIS=1.
  1. LAI retrieval for the 2026 Sentinel-2 field table (phase4_prosail.apply)
  2. field model v4 for 2026 (water use / demand per field). v4 rescales its field yields to the district
     model's 'blend' of the year; for 2026 that is the frozen v9 2026 prediction (pred2026_winter_wheat_v9.csv),
     added in memory only. The Field Outlook re-anchors yields to v12 anyway.
  3. Field Outlook 2026 (phase5_field_outlook.main), anchored to the frozen v12 2026 forecast at lead 0.
"""
import os

import pandas as pd

import phase2_phenology as phen
from phase2_growth import P
from phase2_predict2026 import extra_rows

CROP = "winter_wheat"
phen.EXTRA[CROP] = extra_rows(CROP)

_read = pd.read_csv


def read_with_2026(path, *a, **k):
    d = _read(path, *a, **k)
    if str(path).endswith(f"growth_predictions_{CROP}.csv"):
        p = _read(os.path.join(P, "check2026", f"pred2026_{CROP}_v9.csv"), dtype={"district": str})
        d = pd.concat([d, p[["district", "blend"]].assign(year=2026)], ignore_index=True)
    return d


pd.read_csv = read_with_2026

if __name__ == "__main__":
    import phase4_prosail
    import phase4_field_assim_v4 as v4
    import phase5_field_outlook as fo
    if not os.path.exists(os.path.join(P, "s2", "retrieval_115_2026.csv")):
        phase4_prosail.apply(115, 2026)
    if not os.path.exists(os.path.join(P, "field", "field_results_v4_winter_wheat_2026.csv")):
        try:
            v4.run(2026)
        except FileNotFoundError as e:          # only the final v3 comparison print needs older files
            print("v4 2026 saved; comparison skipped:", e)
    fo.main([2026])
