"""Phase 0: official district yields (Regionaldatenbank table 41241-01-03-4) -> clean table.

Input  data/raw/yields/41241-01-03-4_flat.csv   (flat-file CSV from regionalstatistik.de)
Output data/processed/yields.csv with one row per district, year and crop:
       district code, district name, year, crop, yield in t/ha
and a printed overview: how much data each crop has, how it matches the 400 districts of
the weather data, and the national yield trend.

Run `python phase0_yields.py`.
"""
import os

import numpy as np
import pandas as pd

RAW = os.path.join("data", "raw", "yields", "41241-01-03-4_flat.csv")
OUT = os.path.join("data", "processed", "yields.csv")
WEATHER_GRID = os.path.join("data", "processed", "district_grid.npz")
CROPS = {"Winterweizen": "winter_wheat", "Wintergerste": "winter_barley",
         "Kartoffeln": "potato", "Silomais": "silage_maize"}
OTHER = {"Roggen und Wintermenggetreide": "rye", "Sommergerste": "spring_barley",
         "Hafer": "oats", "Triticale": "triticale", "Zuckerrüben": "sugar_beet",
         "Winterraps": "rapeseed"}


def main():
    raw = pd.read_csv(RAW, sep=";", encoding="latin-1", dtype=str)
    t = pd.DataFrame({
        "district": raw["1_variable_attribute_code"].str.strip(),
        "name": raw["1_variable_attribute_label"].str.strip(),
        "year": raw["time"].astype(int),
        "crop": raw["2_variable_attribute_label"].str.strip().map({**CROPS, **OTHER}),
        "flag": raw["value"].str.strip(),
    })
    # "-" nothing grown, "/" withheld (too uncertain), "." unknown, "x" not applicable
    t["yield_t_ha"] = pd.to_numeric(t["flag"], errors="coerce") / 10.0     # dt/ha -> t/ha
    print(f"{len(t)} values; flags: " + ", ".join(
        f"'{k}' {v}" for k, v in t.loc[t["yield_t_ha"].isna(), "flag"].value_counts().items()))

    districts = t[t["district"].str.len() == 5]          # 2 digits = federal state, DG = Germany
    weather_codes = set(np.load(WEATHER_GRID)["codes"])
    in_weather = districts["district"].isin(weather_codes)
    old = districts.loc[~in_weather & districts["yield_t_ha"].notna(), "district"].unique()
    print(f"district codes in the table: {districts['district'].nunique()}, "
          f"of which in today's 400 weather districts: {districts.loc[in_weather, 'district'].nunique()}; "
          f"{len(old)} former districts (abolished in reforms) have values but no weather yet")

    keep = districts[districts["yield_t_ha"].notna()].drop(columns="flag")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    keep.to_csv(OUT, index=False)
    print(f"saved {len(keep)} district-year-crop yields to {OUT}\n")

    print(f"{'crop':14s} {'values':>7s} {'districts':>9s} {'years':>10s} "
          f"{'1999-2003':>10s} {'2021-2025':>10s} {'trend t/ha/yr':>14s} {'2018 vs trend':>14s}")
    national = t[t["district"] == "DG"]
    for crop in list(CROPS.values()) + list(OTHER.values()):
        k = keep[keep["crop"] == crop]
        nat = national[(national["crop"] == crop)].dropna(subset=["yield_t_ha"]).sort_values("year")
        slope, inter = np.polyfit(nat["year"], nat["yield_t_ha"], 1)
        y18 = nat.loc[nat["year"] == 2018, "yield_t_ha"]
        dev = (y18.iloc[0] / (slope * 2018 + inter) - 1) * 100 if len(y18) else np.nan
        early = nat.loc[nat["year"].between(1999, 2003), "yield_t_ha"].mean()
        late = nat.loc[nat["year"].between(2021, 2025), "yield_t_ha"].mean()
        mark = "" if crop in CROPS.values() else "   (not modelled)"
        print(f"{crop:14s} {len(k):7d} {k['district'].nunique():9d} "
              f"{k['year'].min()}-{k['year'].max()} {early:10.1f} {late:10.1f} "
              f"{slope:+14.3f} {dev:+13.0f}%{mark}")


if __name__ == "__main__":
    main()
