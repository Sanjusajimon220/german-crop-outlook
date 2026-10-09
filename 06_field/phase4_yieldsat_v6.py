"""Phase 4, step 6: field hybrid v6 on YieldSAT - the district recipe at field scale.

Same idea as the district model: physics (v5 field run, frozen) + a small learned correction +
a blend with a data model. Only 188 fields (6 farms, 7 years, 124 fields on one farm), so the
learned parts are kept small (ridge regression on standardised features, alpha chosen by inner
cross-validation) and everything is validated out-of-sample:
  LOYO  leave one harvest year out (7 folds)
  LOFO  leave one farm out (6 folds; the big farm out = learn from 64 fields only)
Inner CV for alpha / blend weight uses only the training part of each fold (nested).

Features per field (no yield inside):
  physics   v5 field yield relative to the district model (v5 / blend), district model (blend),
            water ratio (actual / potential ET), model LAI peak
  satellite green leaf duration after heading (0-45 d), LAI peak (90th pct), LAI at heading,
            canopy chlorophyll and leaf chlorophyll around heading (+-10 d), number of dates
  soil      SoilGrids plant-available water 0-100 cm (relative)
  weather   rain Oct-Mar, rain Apr-Jun, mean Tmax in June, hot days (Tmax > 30 C) heading-7..+21,
            SARAH sunlight heading..+45 (field cell)
Models:
  M0 district model (blend)          M1 satellite index (yield ~ green leaf duration, linear)
  M2 hybrid: v5 x exp(ridge correction on all features)
  M3 direct ridge: yield ~ all features
  M4 blend of M2 and M1 (weight from inner CV)
Truth: YieldSAT field yield, 15 % -> 14 % moisture. NOTE: YieldSAT was used once for the frozen v5
test; these are cross-validated estimates, not a fresh pre-registered test.
Writes data/processed/yieldsat/v6_features.csv, v6_predictions.csv.
"""
import os

import numpy as np
import pandas as pd

YS = os.path.join("data", "processed", "yieldsat")
FEATS = ["rel_v5", "blend", "water_ratio", "lai_peak_model", "glad_after", "lai_peak_obs", "lai_heading",
         "ccc_heading", "cab_heading", "n_obs", "paw_rel", "rain_winter", "rain_spring", "tmax_june",
         "hot_days", "sun_after_heading"]


def features():
    f = pd.read_csv(os.path.join(YS, "field_results_v5.csv"), dtype={"district": str})
    meta = pd.read_csv(os.path.join("data", "raw", "yieldsat", "wheat_metadata.csv"))
    f = f.merge(meta[["field_shared_name", "yield_ground_truth", "farm_identifier", "centroid_latitude_wgs84",
                      "centroid_longitude_wgs84"]], left_on="field", right_on="field_shared_name")
    f["truth"] = f["yield_ground_truth"] * 0.85 / 0.86
    f["rel_v5"] = f["v5_yield_t_ha"] / f["blend"]
    f["water_ratio"] = f["eta_mm"] / f["etp_mm"]
    f["paw_rel"] = f["paw_sg"] / f["paw_sg"].median()
    f["heading"] = pd.to_datetime(f["heading_pred"])
    r = pd.read_csv(os.path.join(YS, "retrieval_fields.csv"), parse_dates=["date"]).dropna(subset=["lai"])
    parts = [np.load(os.path.join(YS, f"hyras_points_{y}.npz"), allow_pickle=True) for y in range(2015, 2023)]
    ids = list(parts[0]["point_ids"])
    wd = np.concatenate([p["dates"] for p in parts])
    w = {v: np.concatenate([p[v] for p in parts]) for v in ("tasmax", "pr")}
    sar = np.load(os.path.join("data", "processed", "weather", "sarah3_germany.npz"))
    rows = []
    for x in f.itertuples():
        g = r[r["field"] == x.field].sort_values("date")
        t = (g["date"] - x.heading).dt.days.values.astype(float)
        o = {"field": x.field}
        if len(g) and (t <= 0).any() and (t >= 0).any():
            o["lai_heading"] = np.interp(0, t, g["lai"].values)
        near = np.abs(t) <= 10
        if near.any():
            o["ccc_heading"], o["cab_heading"] = g["ccc"].values[near].mean(), g["cab"].values[near].mean()
        p = ids.index(x.field)
        y = x.year
        sel = lambda a, b: (wd >= np.datetime64(a)) & (wd <= np.datetime64(b))
        o["rain_winter"] = w["pr"][sel(f"{y - 1}-10-01", f"{y}-03-31"), p].sum()
        o["rain_spring"] = w["pr"][sel(f"{y}-04-01", f"{y}-06-30"), p].sum()
        o["tmax_june"] = w["tasmax"][sel(f"{y}-06-01", f"{y}-06-30"), p].mean()
        h = np.datetime64(x.heading.date())
        o["hot_days"] = (w["tasmax"][(wd >= h - 7) & (wd <= h + 21), p] > 30).sum()
        iy = np.abs(sar["lat"] - x.centroid_latitude_wgs84).argmin()
        ix = np.abs(sar["lon"] - x.centroid_longitude_wgs84).argmin()
        k = (sar["dates"] >= h) & (sar["dates"] <= h + 45)
        o["sun_after_heading"] = np.nanmean(sar["sis"][k, iy, ix].astype(float))
        rows.append(o)
    f = f.drop(columns=[c for c in ("lai_heading",) if c in f]).merge(pd.DataFrame(rows), on="field")
    f["dy"] = f["district"] + "_" + f["year"].astype(str)
    f.to_csv(os.path.join(YS, "v6_features.csv"), index=False)
    return f


def ridge_fit(X, y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    coef = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return lambda Xn: (Xn - mu) / sd @ coef + y.mean()


def prep(train, test, cols):
    med = train[cols].median()
    return train[cols].fillna(med).values.astype(float), test[cols].fillna(med).values.astype(float)


ALPHAS = [1, 3, 10, 30, 100, 300]


def model_preds(train, test):
    """All models fitted on `train` only; alpha and blend weight by inner leave-one-year-out."""
    out = {"M0 district model": test["blend"].values}
    # M1 satellite index
    tr = train.dropna(subset=["glad_after"])
    a, b = np.polyfit(tr["glad_after"], tr["truth"], 1)
    m1 = np.where(test["glad_after"].notna(), a * test["glad_after"] + b, test["blend"])
    out["M1 satellite index"] = m1

    def fit_m(train_, test_, alpha, kind):
        Xtr, Xte = prep(train_, test_, FEATS)
        if kind == "hybrid":
            t = np.log(train_["truth"].values / train_["v5_yield_t_ha"].values)
            return test_["v5_yield_t_ha"].values * np.exp(ridge_fit(Xtr, t, alpha)(Xte))
        return ridge_fit(Xtr, train_["truth"].values, alpha)(Xte)

    def inner(kind):
        yrs = train["year"].unique()
        best, err = None, np.inf
        for al in ALPHAS:
            e = []
            for yr in yrs:
                itr, ite = train[train["year"] != yr], train[train["year"] == yr]
                e.append(((fit_m(itr, ite, al, kind) - ite["truth"].values) ** 2).sum())
            if sum(e) < err:
                best, err = al, sum(e)
        return best

    for kind, name in (("hybrid", "M2 hybrid (v5 x correction)"), ("direct", "M3 direct ridge")):
        out[name] = fit_m(train, test, inner(kind), kind)
    # M4: blend of M2 and M1, weight by inner LOYO
    al = inner("hybrid")
    best_w, best_e = 0.5, np.inf
    for wgt in np.linspace(0, 1, 11):
        e = 0.0
        for yr in train["year"].unique():
            itr, ite = train[train["year"] != yr], train[train["year"] == yr]
            trg = itr.dropna(subset=["glad_after"])
            aa, bb = np.polyfit(trg["glad_after"], trg["truth"], 1)
            s = np.where(ite["glad_after"].notna(), aa * ite["glad_after"] + bb, ite["blend"])
            e += (((wgt * fit_m(itr, ite, al, "hybrid") + (1 - wgt) * s) - ite["truth"].values) ** 2).sum()
        if e < best_e:
            best_w, best_e = wgt, e
    out["M4 blend M2+M1"] = best_w * out["M2 hybrid (v5 x correction)"] + (1 - best_w) * m1
    return out, best_w


def evaluate(f, scheme):
    key = "year" if scheme == "LOYO" else "farm_identifier"
    preds = {}
    weights = []
    for k in f[key].unique():
        tr, te = f[f[key] != k], f[f[key] == k]
        out, wgt = model_preds(tr, te)
        weights.append(wgt)
        for name, p in out.items():
            preds.setdefault(name, pd.Series(index=f.index, dtype=float)).loc[te.index] = p
    return pd.DataFrame(preds), weights


def report(f, P, label):
    print(f"\n{label}")
    rng = np.random.default_rng(0)
    for subset, s in (("all 188", f.index), ("2018-2022", f.index[f["year"] >= 2018])):
        print(f"  {subset}:")
        for name in P.columns:
            e = P.loc[s, name] - f.loc[s, "truth"]
            wr = [np.corrcoef(P.loc[g.index, name], g["truth"])[0, 1] for _, g in f.loc[s].groupby("dy")
                  if len(g) >= 4 and P.loc[g.index, name].std() > 0]
            print(f"    {name:30s} RMSE {np.sqrt((e ** 2).mean()):.2f}  bias {e.mean():+.2f}  r "
                  f"{P.loc[s, name].corr(f.loc[s, 'truth']):.2f}  within district-year r "
                  f"{(np.mean(wr) if wr else np.nan):.2f}")
    best = min(P.columns[1:], key=lambda c: ((P[c] - f["truth"]) ** 2).mean())
    dys = f["dy"].unique()
    for base in ("M1 satellite index", "M0 district model"):
        d = []
        for _ in range(2000):
            idx = np.concatenate([f.index[f["dy"] == k] for k in rng.choice(dys, len(dys))])
            d.append(np.sqrt(((P.loc[idx, best] - f.loc[idx, "truth"]) ** 2).mean())
                     - np.sqrt(((P.loc[idx, base] - f.loc[idx, "truth"]) ** 2).mean()))
        print(f"  best ({best}) minus {base}: 90 % interval {np.percentile(d, 5):+.2f} .. {np.percentile(d, 95):+.2f}")


def main():
    f = features()
    print(f"{len(f)} fields; features available:",
          {c: int(f[c].notna().sum()) for c in FEATS if f[c].notna().sum() < len(f)})
    out = []
    for scheme in ("LOYO", "LOFO"):
        P, w = evaluate(f, scheme)
        report(f, P, f"{scheme} (blend weight of M2 in M4 per fold: {np.round(w, 1)})")
        out.append(P.add_prefix(f"{scheme}: "))
    pd.concat([f[["field", "year", "farm_identifier", "district", "truth"]]] + out, axis=1).to_csv(
        os.path.join(YS, "v6_predictions.csv"), index=False)


if __name__ == "__main__":
    main()
