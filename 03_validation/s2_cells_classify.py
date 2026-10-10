"""v11.1 item 2: crop classification from per-cell Sentinel-2 NDVI and district crop features (pre-registered
docs/project_log.md 2026-10-10 13:58).

Input  s2_district/cells_<state>.csv (cell positions, DLR labels 2024 / 2023) and cells_<state>_<year>/timeseries.csv
Steps  1. per-cell dekad NDVI (s2_features.dekads)  2. gradient boosting on dekad NDVI + peak / peak dekad,
       trained on 2024 cells (DLR 2024 labels)  3. validation A leave-state-out 2024, B 2024 -> 2023 (16, 03)
       4. district x crop curves: 2024 from DLR labels, 2025/2026 from predictions (p >= 0.6), >= 5 cells
       5. v11.1 features (s2_features.feats) at every lead
Output data/processed/s2_cells_validation.csv, s2_features_cells.csv (same columns as s2_features.csv + source)
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

from s2_features import LEADS, dekads, feats

P = os.path.join("data", "processed")
S2 = os.path.join(P, "s2_district")
CLS = {11: "wheat", 12: "barley", 30: "maize", 50: "potato"}
DEK = list(range(100, 275, 10))          # dekad-end doy grid for the classifier (Apr-Sep)
MODEL_CROPS = {"wheat": ["winter_wheat"], "barley": ["winter_barley"], "maize": ["grain_maize", "silage_maize"],
               "potato": ["potato"]}
MIN_CELLS, P_MIN = 5, 0.6


def cell_series(state, year):
    f = os.path.join(S2, f"cells_{state}_{year}", "timeseries.csv")
    if not os.path.exists(f):
        return None, None
    c = pd.read_csv(os.path.join(S2, f"cells_{state}.csv"), dtype={"AGS": str})
    ts = pd.read_csv(f).rename(columns={"band_unnamed": "ndvi"}).dropna(subset=["ndvi"])
    ser = {i: dekads(g, year) for i, g in ts.groupby("feature_index")}
    return c, ser


def xfeat(s):
    if s is None or len(s) < 6:
        return None
    v = np.interp(DEK, s.index, s.values, left=np.nan, right=np.nan)
    return list(v) + [s.max(), float(s.idxmax())]


def label(v):
    return CLS.get(int(v), "other") if pd.notna(v) and int(v) > 0 else None


def table(year, states, label_col=None):
    rows = []
    for st in states:
        c, ser = cell_series(st, year)
        if c is None:
            continue
        for i, r in c.iterrows():
            x = xfeat(ser.get(i))
            if x is None:
                continue
            rows.append(dict(state=st, district=r["AGS"], cell=i, y=label(r[label_col]) if label_col and label_col in c else None,
                             **{f"f{k}": v for k, v in enumerate(x)}))
    return pd.DataFrame(rows), {st: cell_series(st, year)[1] for st in states}


def district_features(tab, ser, year, crop_col, meta):
    out = []
    for (st, ags), g in tab.groupby(["state", "district"]):
        for crop in ("wheat", "barley", "maize", "potato"):
            ids = g.loc[g[crop_col] == crop, "cell"].tolist()
            if len(ids) < MIN_CELLS:
                continue
            cur = pd.concat([ser[st][i] for i in ids], axis=1).mean(axis=1).sort_index()
            for mc in MODEL_CROPS[crop]:
                for lead in LEADS:
                    fe = feats(cur, meta[mc]["harvest_doy"] - 7 * lead)
                    if fe:
                        out.append(dict(crop=mc, district=ags, state=st, year=year, lead=lead, cells=len(ids), **fe))
    return pd.DataFrame(out)


def main():
    meta = json.load(open(os.path.join("platform", "data", "meta.json")))["crops"]
    states = [f"{i:02d}" for i in range(1, 17)]
    tr, ser24 = table(2024, states, "label2024")
    tr = tr.dropna(subset=["y"])
    X = [c for c in tr.columns if c.startswith("f")]
    model = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=0)
    val = []
    # A: leave-state-out 2024
    pa = pd.Series(index=tr.index, dtype=object)
    for st in tr["state"].unique():
        m = model().fit(tr.loc[tr["state"] != st, X], tr.loc[tr["state"] != st, "y"])
        pa[tr["state"] == st] = m.predict(tr.loc[tr["state"] == st, X])
    for crop in ("wheat", "barley", "maize", "potato"):
        val.append(dict(check="A leave-state-out 2024", crop=crop, n=int((tr["y"] == crop).sum()),
                        f1=f1_score(tr["y"] == crop, pa == crop)))
    final = model().fit(tr[X], tr["y"])
    # B: temporal transfer 2024 -> 2023 (states 16, 03)
    t23, ser23 = table(2023, ["16", "03"], "label2023")
    feat_r = {}
    if len(t23):
        t23 = t23.dropna(subset=["y"])
        t23["pred"] = final.predict(t23[X])
        prob = final.predict_proba(t23[X])
        t23["pred_p"] = np.where(prob.max(axis=1) >= P_MIN, t23["pred"], "unsure")
        for crop in ("wheat", "barley", "maize", "potato"):
            val.append(dict(check="B 2024 -> 2023 (16, 03)", crop=crop, n=int((t23["y"] == crop).sum()),
                            f1=f1_score(t23["y"] == crop, t23["pred"] == crop)))
        a = district_features(t23, ser23, 2023, "y", meta)
        b = district_features(t23, ser23, 2023, "pred_p", meta)
        j = a.merge(b, on=["crop", "district", "lead"], suffixes=("_lab", "_cls"))
        for mc, g in j.groupby("crop"):
            feat_r[mc] = min(g[f"{n}_lab"].corr(g[f"{n}_cls"]) for n in ("peak", "last3", "integral"))
            val.append(dict(check="B features labelled vs classified", crop=mc, n=len(g), f1=np.nan, feature_r_min=feat_r[mc]))
    v = pd.DataFrame(val)
    v.to_csv(os.path.join(P, "s2_cells_validation.csv"), index=False)
    print(v.round(3).to_string(index=False))
    passed = {c for c in ("wheat", "barley", "maize")
              if (v[(v["check"].str.startswith("B 2024")) & (v["crop"] == c)]["f1"] >= 0.85).all()
              and all(feat_r.get(mc, 0) >= 0.90 for mc in MODEL_CROPS[c])}
    print("PASS (classification usable for 2025/2026):", sorted(passed))
    out = [district_features(tr, ser24, 2024, "y", meta).assign(source="dlr_label")]
    for yr in (2025, 2026):
        t, ser = table(yr, states)
        if t.empty:
            continue
        prob = final.predict_proba(t[X])
        t["pred"] = np.where(prob.max(axis=1) >= P_MIN, final.classes_[prob.argmax(axis=1)], "unsure")
        t["pred"] = t["pred"].where(t["pred"].isin(passed), "unsure")
        out.append(district_features(t, ser, yr, "pred", meta).assign(source="classified"))
    d = pd.concat(out, ignore_index=True)
    for c in ("peak", "last3", "integral"):
        d[c + "_anom"] = d[c] - d.groupby(["crop", "state", "year", "lead"])[c].transform("mean")
    d.to_csv(os.path.join(P, "s2_features_cells.csv"), index=False)
    print(d.groupby(["crop", "year"]).size().unstack())


if __name__ == "__main__":
    main()
