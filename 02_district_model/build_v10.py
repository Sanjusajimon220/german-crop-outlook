"""Assemble the v10 in-season forecasts (pre-registered 2026-10-09; no test result used).

Base: v9 in-season forecasts data/processed/forecast/forecast_<crop>_v9_leadNN.csv (39 weather scenarios
per district-year). Barley: refit schedule (2018-2020 v9, 2021-2023 _v9_r2020, 2024-2025 _v9_r2023).
Error history (out-of-sample only): v9 forward errors 2009-2017 (fit <= 2008, MODIS) and, for test years,
the v9/v10-base end-of-season error (lead-0 median of the scenarios minus the official yield).
Correction for district d in year t: offset = sum of d's errors of years <= t-2 with the year-wide part
removed / (n + 3); grain maize additionally the national update = mean of the yearly mean errors of the
three latest years <= t-1. Every scenario value of 'blend' is shifted by the correction.
Writes data/processed/forecast/forecast_<crop>_v10_leadNN.csv and v10_corrections_<crop>.csv.
"""
import os

import numpy as np
import pandas as pd

P = os.path.join("data", "processed")
F = os.path.join(P, "forecast")
LEADS = (12, 8, 6, 4, 2, 0)
K = 3
FWD = {"potato": "forward_predictions_potato_potato_canopy_modis_cut2008.csv"}


def base_file(crop, lead, year):
    tag = "_v9"
    if crop == "winter_barley":
        tag = "_v9" if year <= 2020 else "_v9_r2020" if year <= 2023 else "_v9_r2023"
    return os.path.join(F, f"forecast_{crop}{tag}_lead{lead:02d}.csv")


def load_base(crop, lead):
    parts = []
    for path in sorted({base_file(crop, lead, y) for y in range(2018, 2026)}):
        f = pd.read_csv(path, dtype={"district": str})
        years = [y for y in range(2018, 2026) if base_file(crop, lead, y) == path]
        parts.append(f[f["year"].isin(years)])
    return pd.concat(parts, ignore_index=True)


def main(crops=("winter_wheat", "winter_barley", "grain_maize", "silage_maize", "potato")):
    for crop in crops:
        fwd = pd.read_csv(os.path.join(P, FWD.get(crop, f"forward_predictions_{crop}_modis_cut2008.csv")),
                          dtype={"district": str}).dropna(subset=["yield_t_ha", "pred"])
        hist = pd.DataFrame({"district": fwd["district"], "year": fwd["year"], "e": fwd["pred"] - fwd["yield_t_ha"]})
        end = load_base(crop, 0)
        med = end.groupby(["district", "year"]).agg(pred=("blend", "median"), obs=("obs", "first")).reset_index()
        med = med.dropna(subset=["obs"])
        hist = pd.concat([hist, pd.DataFrame({"district": med["district"], "year": med["year"],
                                               "e": med["pred"] - med["obs"]})], ignore_index=True)
        ye = hist.groupby("year")["e"].mean()
        hist["e_loc"] = hist["e"] - hist["year"].map(ye)
        corr = []
        for t in range(2018, 2027):             # 2026: frozen prediction for the spring-2027 check
            h = hist[hist["year"] <= t - 2]
            off = h.groupby("district")["e_loc"].sum() / (h.groupby("district")["e_loc"].count() + K)
            nat = ye[ye.index <= t - 1].iloc[-3:].mean() if crop == "grain_maize" else 0.0
            corr.append(pd.DataFrame({"district": off.index, "year": t, "offset": off.values, "national": nat}))
        corr = pd.concat(corr, ignore_index=True)
        corr.to_csv(os.path.join(F, f"v10_corrections_{crop}.csv"), index=False)
        for lead in LEADS:
            b = load_base(crop, lead).merge(corr, on=["district", "year"], how="left")
            b["offset"] = b["offset"].fillna(0.0)
            b["national"] = b["national"].fillna(0.0 if crop != "grain_maize" else
                                                 b.groupby("year")["national"].transform("max").fillna(0.0))
            b["blend"] = b["blend"] - b["offset"] - b["national"]
            b.drop(columns=["offset", "national"]).to_csv(os.path.join(F, f"forecast_{crop}_v10_lead{lead:02d}.csv"), index=False)
        print(f"{crop}: corrections for {corr['district'].nunique()} districts, mean |offset| "
              f"{corr['offset'].abs().mean():.2f} t/ha" + (f", national 2018-2025: {corr.groupby('year')['national'].first().round(2).tolist()}"
                                                          if crop == "grain_maize" else ""), flush=True)


if __name__ == "__main__":
    import sys
    main(tuple(sys.argv[1:]) or ("winter_wheat", "winter_barley", "grain_maize", "silage_maize", "potato"))
