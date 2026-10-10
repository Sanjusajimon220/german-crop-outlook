"""v12 A evaluation, exactly as pre-registered (docs/project_log.md 2026-10-10 18:28). Run ONCE when the large
DLR-based samples 2019-2023 are complete.

1. features: district curve = mean over the crop's cells (>= 10 cells), s2_features.feats at each lead, state
   anomalies; noise check: same features from two random halves of the cells, r between halves.
2. A1 district correction (v11.1 model and rule) - leave-one-year-out 2019-2023, held-out states not in fitting.
3. A2 state correction: state residual (area-weighted) ~ area-weighted mean district features; LOYO.
Outputs data/processed/s2_big_features.csv, s2_big_eval.csv, v12/model_s2.json (adopted cells, fitted 2019-2023).
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from s2_features import LEADS, dekads, feats

P = os.path.join("data", "processed")
S2 = os.path.join(P, "s2_district")
F = ["peak", "last3", "integral", "peak_anom", "last3_anom", "integral_anom"]
HELD = {"01", "08", "12"}
MODEL_CROPS = {"wheat": ["winter_wheat"], "barley": ["winter_barley"], "maize": ["grain_maize", "silage_maize"],
               "potato": ["potato"]}
MIN_CELLS = 10


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e)))) if len(e) else np.nan


def build_features(meta):
    rows = []
    rng = np.random.default_rng(12)
    for f in sorted(glob.glob(os.path.join(S2, "big_*_*.csv"))):
        st, yr = os.path.basename(f)[4:-4].split("_")
        tsf = os.path.join(S2, f"big_{st}_{yr}", "timeseries.csv")
        if not os.path.exists(tsf):
            continue
        yr = int(yr)
        c = pd.read_csv(f, dtype={"AGS": str})
        ts = pd.read_csv(tsf).rename(columns={"band_unnamed": "ndvi"}).dropna(subset=["ndvi"])
        ser = {i: dekads(g, yr) for i, g in ts.groupby("feature_index")}
        for (ags, crop), g in c.groupby(["AGS", "crop"]):
            ids = [i for i in g.index if i in ser and len(ser[i]) >= 4]
            if len(ids) < MIN_CELLS:
                continue
            half = rng.permutation(ids)
            curves = {"all": ids, "h1": half[: len(ids) // 2], "h2": half[len(ids) // 2:]}
            cur = {k: pd.concat([ser[i] for i in v], axis=1).mean(axis=1).sort_index() for k, v in curves.items()}
            for mc in MODEL_CROPS[crop]:
                for lead in LEADS:
                    doy = meta[mc]["harvest_doy"] - 7 * lead
                    fe = {k: feats(v, doy) for k, v in cur.items()}
                    if fe["all"] and fe["h1"] and fe["h2"]:
                        rows.append(dict(crop=mc, district=ags, state=st, year=yr, lead=lead, cells=len(ids), **fe["all"],
                                         **{f"h1_{n}": v for n, v in fe["h1"].items()}, **{f"h2_{n}": v for n, v in fe["h2"].items()}))
    d = pd.DataFrame(rows)
    for c in ("peak", "last3", "integral"):
        d[c + "_anom"] = d[c] - d.groupby(["crop", "state", "year", "lead"])[c].transform("mean")
    d.to_csv(os.path.join(P, "s2_big_features.csv"), index=False)
    return d


def fit(X, y):
    sc = StandardScaler().fit(X)
    m = Ridge(alpha=10).fit(sc.transform(X), y)
    return dict(mean=sc.mean_.tolist(), scale=sc.scale_.tolist(), coef=m.coef_.tolist(), intercept=float(m.intercept_))


def pred(mod, X):
    return 0.5 * (((np.asarray(X, float) - mod["mean"]) / mod["scale"]) @ np.array(mod["coef"]) + mod["intercept"])


def main():
    meta = json.load(open(os.path.join("platform", "data", "meta.json")))["crops"]
    s2 = build_features(meta)
    y = pd.read_csv(os.path.join(P, "yields_long.csv"), dtype={"district": str})
    tr = pd.read_csv(os.path.join(P, "benchmarks", "truth_official.csv"), dtype={"region": str}).set_index(["crop", "region", "year"])["yield_t_ha"]
    out, models = [], {}
    for (crop, lead), g in s2.groupby(["crop", "lead"]):
        f = os.path.join(P, "ranges_v10_final", f"ranges_{crop}_lead{lead:02d}.csv")
        if not os.path.exists(f):
            continue
        v = pd.read_csv(f, dtype={"region": str})
        v = v[v["level"] == "district"].rename(columns={"region": "district"})[["district", "year", "obs", "median"]]
        d = g.merge(v, on=["district", "year"]).dropna(subset=["obs", "median"] + F)
        years = sorted(d["year"].unique())
        if len(years) < 4:
            continue
        noise = min(d[f"h1_{n}"].corr(d[f"h2_{n}"]) for n in ("peak", "last3", "integral"))
        d["res"] = d["obs"] - d["median"]
        a = y[(y["crop"] == crop) & (y["area_ha"] > 0)].sort_values("year").groupby("district")["area_ha"].last()
        d["a"] = d["district"].map(a)
        known = ~d["state"].isin(HELD)
        # A1 district
        d["c1"] = np.nan
        for t in years:
            trn = d[(d["year"] != t) & known]
            d.loc[d["year"] == t, "c1"] = pred(fit(trn[F], trn["res"]), d.loc[d["year"] == t, F])
        d["n1"] = d["median"] + d["c1"]
        # A2 state: area-weighted state means
        da = d.dropna(subset=["a"])
        sg = da.groupby(["state", "year"]).apply(lambda x: pd.Series(
            {**{c: np.average(x[c], weights=x["a"]) for c in F + ["median"]}})).reset_index()
        sg["truth"] = [tr.get((crop, s, t), np.nan) for s, t in zip(sg["state"], sg["year"])]
        sg = sg.dropna(subset=["truth"])
        sg["res"] = sg["truth"] - sg["median"]
        sg["c2"] = np.nan
        for t in years:
            trn = sg[sg["year"] != t]
            if len(trn) >= 10:
                sg.loc[sg["year"] == t, "c2"] = pred(fit(trn[F], trn["res"]), sg.loc[sg["year"] == t, F])
        sg["n2"] = sg["median"] + sg["c2"].fillna(0)
        r = dict(crop=crop, lead=lead, n=int(known.sum()), years=len(years), noise_r=noise)
        per1 = [rmse(d.loc[known & (d["year"] == t), "n1"] - d.loc[known & (d["year"] == t), "obs"]) <
                rmse(d.loc[known & (d["year"] == t), "median"] - d.loc[known & (d["year"] == t), "obs"]) for t in years]
        r["d_old"], r["d_new"] = rmse(d.loc[known, "median"] - d.loc[known, "obs"]), rmse(d.loc[known, "n1"] - d.loc[known, "obs"])
        r["d_gain"], r["d_years"] = 100 * (1 - r["d_new"] / r["d_old"]), int(sum(per1))
        h = ~known
        r["held_old"], r["held_new"] = rmse(d.loc[h, "median"] - d.loc[h, "obs"]), rmse(d.loc[h, "n1"] - d.loc[h, "obs"])
        # state / national effect of A1
        ds = da.assign(n1=d.loc[da.index, "n1"]).groupby(["state", "year"]).apply(
            lambda x: pd.Series({"m": np.average(x["median"], weights=x["a"]), "n": np.average(x["n1"], weights=x["a"])})).reset_index()
        ds["truth"] = [tr.get((crop, s, t), np.nan) for s, t in zip(ds["state"], ds["year"])]
        ds = ds.dropna()
        r["s_old"], r["s_new1"] = rmse(ds["m"] - ds["truth"]), rmse(ds["n"] - ds["truth"])
        nat = da.assign(n1=d.loc[da.index, "n1"]).groupby("year").apply(
            lambda x: pd.Series({"m": np.average(x["median"], weights=x["a"]), "n": np.average(x["n1"], weights=x["a"])}))
        nat["truth"] = [tr.get((crop, "DE", t), np.nan) for t in nat.index]
        nat = nat.dropna()
        r["nat_old"], r["nat_new1"] = rmse(nat["m"] - nat["truth"]), rmse(nat["n"] - nat["truth"])
        r["adopt_A1"] = bool(noise >= 0.85 and r["d_gain"] >= 3 and r["d_years"] >= len(years) - 1 and
                             (np.isnan(r["held_old"]) or r["held_new"] <= r["held_old"]) and
                             r["s_new1"] <= r["s_old"] and r["nat_new1"] <= r["nat_old"])
        # A2 effect
        per2 = [rmse(sg.loc[sg["year"] == t, "n2"] - sg.loc[sg["year"] == t, "truth"]) <
                rmse(sg.loc[sg["year"] == t, "median"] - sg.loc[sg["year"] == t, "truth"]) for t in years]
        r["s2_old"], r["s2_new"] = rmse(sg["median"] - sg["truth"]), rmse(sg["n2"] - sg["truth"])
        r["s_gain2"], r["s_years2"] = 100 * (1 - r["s2_new"] / r["s2_old"]), int(sum(per2))
        sg["a"] = sg["state"].map(da.groupby("state")["a"].sum())
        nat2 = sg.groupby("year").apply(lambda x: pd.Series({"m": np.average(x["median"], weights=x["a"]),
                                                             "n": np.average(x["n2"], weights=x["a"])}))
        nat2["truth"] = [tr.get((crop, "DE", t), np.nan) for t in nat2.index]
        nat2 = nat2.dropna()
        r["nat_new2"] = rmse(nat2["n"] - nat2["truth"])
        r["adopt_A2"] = bool(not r["adopt_A1"] and noise >= 0.85 and r["s_gain2"] >= 3 and r["s_years2"] >= len(years) - 1
                             and r["nat_new2"] <= rmse(nat2["m"] - nat2["truth"]))
        key = f"{crop}_lead{lead:02d}"
        if r["adopt_A1"]:
            models[key] = dict(level="district", crop=crop, lead=lead, **fit(d.loc[known, F], d.loc[known, "res"]))
        elif r["adopt_A2"]:
            models[key] = dict(level="state", crop=crop, lead=lead, **fit(sg[F], sg["res"]))
        out.append(r)
    o = pd.DataFrame(out)
    o.to_csv(os.path.join(P, "s2_big_eval.csv"), index=False)
    os.makedirs(os.path.join(P, "v12"), exist_ok=True)
    json.dump(dict(features=F, shrink=0.5, alpha=10, cells=models), open(os.path.join(P, "v12", "model_s2.json"), "w"), indent=1)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(o.round(3).to_string(index=False))
    print("adopted:", {k: v["level"] for k, v in models.items()})


if __name__ == "__main__":
    main()
