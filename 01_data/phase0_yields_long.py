"""Phase 0: one long yield table 1979-2025 from two official sources.

  1979-2021  Duden, Nacke & Offermann (2024), Scientific Data 11:95, "German yield and area
             data for 11 crops from 1979 to 2021 at a harmonized spatial resolution of 397
             districts", OpenAgrar doi:10.3220/DATA20231117103252-0 (file Final_data.csv).
             Already harmonised to the district boundaries of 2020 and checked for outliers by
             the authors; licence: Datenlizenz Deutschland - Namensnennung 2.0 (attribution).
  2022-2025  Regionaldatenbank table 41241-01-03-4 (our yields.csv), same boundaries.

In 1999-2021 both sources agree to within 0.05 t/ha for 99-100 % of values (checked).
Additional rule: grain maize below 2 t/ha is dropped as implausible (one value, Hagen 2013).
Eisenach (16056), merged into Wartburgkreis in 2021, has no weather district and is dropped.

Output data/processed/yields_long.csv: district, year, crop, yield_t_ha, area_ha, source.
Run `python phase0_yields_long.py`.
"""
import os

import numpy as np
import pandas as pd

P = os.path.join("data", "processed")
LONG = os.path.join("data", "raw", "yields_1979_2021", "Final_data.csv")
NAMES = {"ww": "winter_wheat", "wb": "winter_barley", "silage_maize": "silage_maize",
         "grain_maize": "grain_maize", "potat_tot": "potato", "rye": "rye", "sb": "spring_barley",
         "oats": "oats", "triticale": "triticale", "sugarbeet": "sugar_beet", "wrape": "rapeseed"}


def main():
    d = pd.read_csv(LONG, dtype={"district_no": str})
    d = d[d["var"].isin(NAMES) & (d["outlier"] == 0)]
    wide = d.pivot_table(index=["district_no", "year", "var"], columns="measure", values="value").reset_index()
    old = pd.DataFrame({"district": wide["district_no"], "year": wide["year"],
                        "crop": wide["var"].map(NAMES), "yield_t_ha": wide["yield"],
                        "area_ha": wide.get("area"), "source": "Duden et al. 2024"})
    old = old[old["yield_t_ha"].notna()]
    old = old[~((old["crop"] == "grain_maize") & (old["yield_t_ha"] < 2.0))]

    new = pd.read_csv(os.path.join(P, "yields.csv"), dtype={"district": str})
    new = new[new["year"] >= 2022].assign(area_ha=np.nan, source="Regionaldatenbank 41241-01-03-4")
    new = new[["district", "year", "crop", "yield_t_ha", "area_ha", "source"]]

    codes = set(np.load(os.path.join(P, "weather_daily.npz"))["codes"])
    both = pd.concat([old, new])
    both = both[both["district"].isin(codes)].sort_values(["crop", "district", "year"])
    both.to_csv(os.path.join(P, "yields_long.csv"), index=False)

    print(f"saved {len(both)} yields to data/processed/yields_long.csv")
    print(f"  {'crop':14s} {'values':>7s} {'districts':>9s} {'1979-1998':>10s} {'1999-2021':>10s} {'2022-2025':>10s}")
    for crop, g in both.groupby("crop"):
        print(f"  {crop:14s} {len(g):7d} {g['district'].nunique():9d} {(g.year < 1999).sum():10d} "
              f"{g.year.between(1999, 2021).sum():10d} {(g.year >= 2022).sum():10d}")


if __name__ == "__main__":
    main()
