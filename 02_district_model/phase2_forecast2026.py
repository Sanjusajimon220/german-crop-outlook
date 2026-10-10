"""v11.1 item 4: v10 IN-SEASON forecasts for 2026 (pre-registered 2026-10-10 13:45; no 2026 yield used).

For each forecast date (lead 12/8/6/4/2/0 weeks before the typical harvest) the 2026 weather up to the date is
kept (weather_daily_v8_2026.npz) and the rest of the season is replaced by each of the 39 training seasons
1979-2017 (same scenario set as the v10 test forecasts). Stage dates are all predicted by the frozen crop
calendar (2026 observations not published), MODIS greenness only up to the date, nitrogen = 2025 level
(as phase2_predict2026). Model = the v9 final model (barley: refit-2023 model) minus the v10 corrections for
2026 (district offsets, grain maize national update; forecast/v10_corrections_<crop>.csv).
Run with VISTA_WEATHER=v8_2026 VISTA_MODIS=1 (+ VISTA_REFIT=2023 for barley).
Output data/processed/check2026/inseason2026_<crop>_leadNN.csv (district x scenario, v10 blend).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

import phase2_phenology as phen
from phase0_master import monthly_features
from phase2_forecast import SCENARIOS, adjust, fit_final, spliced
from phase2_growth import FORWARD_WEIGHT, GROWTH, MODIS, N_COLUMN, P, load_growth, simulate
from phase2_predict2026 import OUT, extra_rows

F = os.path.join(P, "forecast")


def main(crop, leads):
    phen.EXTRA[crop] = extra_rows(crop)
    weight = FORWARD_WEIGHT.get(crop, 0.8)
    model, corr, bst, feats, bench, _ = fit_final(crop)
    harvest = int(pd.read_csv(os.path.join(P, f"master_{crop}.csv"))["doy_harvest"].median())
    c = pd.read_csv(os.path.join(F, f"v10_corrections_{crop}.csv"), dtype={"district": str})
    c = c[c["year"] == 2026].set_index("district")
    nat = float(c["national"].iloc[0]) if len(c) else 0.0
    nit = pd.read_csv(os.path.join(P, "nitrogen_districts.csv"), dtype={"district": str})
    n25 = nit[nit["year"] == 2025].set_index("district")[N_COLUMN[crop]]
    from phase0_modis_features import features as modis_features, series
    ms = series() if MODIS else None
    w0 = np.load(os.path.join(P, phen.WEATHER_FILE))
    w0 = {k: w0[k] for k in w0.files}
    stages, gate, vp, _ = phen.CROPS[crop]
    cal = phen.Calendar(len(stages), stages.index(gate), vp, json.load(open(os.path.join(P, f"phenology_model_{crop}.json"))))
    for lead in leads:
        path = os.path.join(OUT, f"inseason2026_{crop}_lead{lead:02d}.csv")
        if os.path.exists(path):
            continue
        cutoff = harvest - 7 * lead
        print(f"{crop} lead {lead} (day {cutoff})", flush=True)
        rows, t0 = [], time.time()
        for s in SCENARIOS:
            phen.WEATHER = spliced(w0, cutoff, s, [2026])
            m, d = load_growth(crop)
            sel = (m["year"] == 2026).values
            n26 = m.loc[sel, "district"].map(n25).fillna(n25.median()).values
            sub = {k: v[sel] for k, v in d.items() if v.dim() and len(v) == len(m)}
            sub["n_supply"] = torch.tensor(n26, dtype=torch.float32)
            mt = m[sel].reset_index(drop=True)
            mt["n_crop_kg_ha"] = n26
            sim = adjust(simulate(model, sub, GROWTH[crop]["dry_matter"]), model, mt["year"].values, crop)
            months = sorted({int(x[-2:]) for x in mt.columns if x.startswith("tas_m")})
            mf = monthly_features(phen.WEATHER, phen.WEATHER["codes"], months, "tas_winter" in mt.columns)
            mf = mf.reindex(pd.MultiIndex.from_frame(mt[["district", "year"]]))
            for col in mf.columns:
                if col in mt.columns:
                    mt[col] = mf[col].values
            with torch.no_grad():
                days = cal(sub).numpy()
            for k, st in enumerate(stages):
                if f"doy_{st}" in mt.columns:
                    mt[f"doy_{st}"] = mt["doy_sowing"] + days[:, k]
            tr = m[m["role"] == "train"]
            if "doy_harvest" in mt.columns:
                mt["doy_harvest"] = mt["district"].map(tr.groupby("district")["doy_harvest"].median())
            if MODIS:
                mf2 = modis_features(crop, cutoff, ms).set_index(["district", "year"])
                mf2 = mf2.reindex(pd.MultiIndex.from_frame(mt[["district", "year"]]))
                for col in mf2.columns:
                    if col in mt.columns:
                        mt[col] = mf2[col].values
            mm = pd.concat([mt, sim], axis=1)
            phys = mm["phys_yield"].clip(lower=0.5).values
            hybrid = phys * np.exp(corr.predict(mm[feats]))
            b = bst.predict(mm[bench].values.astype(float))
            blend = weight * hybrid + (1 - weight) * b
            v10 = blend - mt["district"].map(c["offset"]).fillna(0).values - nat
            rows.append(pd.DataFrame({"district": mt["district"], "scenario": s, "blend_v9": blend, "blend": v10,
                                      "water_ratio": sim["phys_water_ratio"].values}))
        phen.WEATHER = None
        pd.concat(rows, ignore_index=True).to_csv(path, index=False)
        print(f"  saved {path} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], [int(x) for x in sys.argv[2:]] or [12, 8, 6, 4, 2, 0])
