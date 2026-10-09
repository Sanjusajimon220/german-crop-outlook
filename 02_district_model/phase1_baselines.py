"""Phase 1: statistical benchmarks for district yields.

Three benchmarks, all fitted on the training data only (split.csv), with settings chosen by
leaving out one block of training years at a time:

  trend          district's average level + national linear trend over the years. No weather.
                 Any useful model must beat this. Districts never seen in training (test
                 regions) get the national trend only.
  weather        ridge regression on everything in the master file except the district's
                 identity: monthly weather, winter weather, soil water, crop-stage dates, year.
                 Can be applied anywhere, also in regions it has not seen.
  trend+weather  the trend benchmark plus a ridge regression of its errors on the same
                 features: uses both the district's history and the season's weather.
  boosting       gradient boosting (many small decision trees) on the same features, without the
                 district's identity: can learn thresholds such as heat at flowering.
  trend+boosting the trend benchmark plus gradient boosting of its errors.

Scores: RMSE (t/ha), relative RMSE (% of the mean yield), and the "weather skill":
the correlation between the predicted and the observed deviation from the trend benchmark.
The trend benchmark has a weather skill of 0 by construction.

Outputs: data/processed/phase1_results.csv, data/processed/predictions_<crop>.csv
Run `python phase1_baselines.py`.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

P = os.path.join("data", "processed")
CROPS = ("winter_wheat", "winter_barley", "silage_maize", "grain_maize", "potato")
ROLES = ("train_cv", "test_years", "test_regions", "test_both")
DROUGHT = (2018, 2019, 2022)
LAMBDAS = (0.1, 1, 3, 10, 30, 100, 300, 1000, 3000)
BOOST = ({"max_depth": 3, "learning_rate": 0.05}, {"max_depth": 5, "learning_rate": 0.05},
         {"max_depth": None, "learning_rate": 0.03})        # settings tried in cross-validation
MODELS = ("trend", "weather", "trend+weather", "boosting", "trend+boosting")


def features(m, train):
    """Every master column except identity and provenance; columns missing in more than half
    of the training rows are left out (potato flowering and harvest dates)."""
    skip = {"district", "year", "name", "yield_t_ha", "role", "fold", "sunlight_source", "yield_source"}
    cols = [c for c in m.columns if c not in skip and not c.startswith("src_")
            and (c.startswith("modis_") or train[c].notna().mean() >= 0.5)]   # satellite: 2000+ only
    return cols + ["year"]


class Ridge:
    """Linear regression with an L2 penalty; inputs standardised with training statistics."""

    def fit(self, x, y, lam):
        self.mu, self.sd = np.nanmean(x, 0), np.nanstd(x, 0) + 1e-9
        z = np.nan_to_num((x - self.mu) / self.sd)            # missing values -> training mean
        self.y0 = y.mean()
        a = z.T @ z + lam * len(z) * 1e-3 * np.eye(z.shape[1])
        self.w = np.linalg.solve(a, z.T @ (y - self.y0))
        return self

    def predict(self, x):
        return np.nan_to_num((x - self.mu) / self.sd) @ self.w + self.y0


class Trend:
    """National linear trend over years + each district's mean offset (training districts)."""

    def fit(self, district, year, y):
        self.b, self.a = np.polyfit(year, y, 1)
        resid = pd.Series(y - (self.a + self.b * year)).groupby(np.asarray(district)).mean()
        self.offset = resid.to_dict()
        return self

    def predict(self, district, year):
        return self.a + self.b * year + np.array([self.offset.get(d, 0.0) for d in district])


def boost(x, y, setting):
    return HistGradientBoostingRegressor(max_iter=400, min_samples_leaf=40, l2_regularization=1.0,
                                         random_state=0, **setting).fit(x, y)


def fit_predict(train, other, cols, lam):
    """Fit all benchmarks on `train` and predict `other`. lam holds each model's setting."""
    trend = Trend().fit(train["district"], train["year"].values, train["yield_t_ha"].values)
    t_train = trend.predict(train["district"], train["year"].values)
    t_other = trend.predict(other["district"], other["year"].values)
    xw, xo = train[cols].values.astype(float), other[cols].values.astype(float)
    weather = Ridge().fit(xw, train["yield_t_ha"].values, lam["weather"])
    residual = Ridge().fit(xw, train["yield_t_ha"].values - t_train, lam["trend+weather"])
    out = {"trend": t_other, "weather": weather.predict(xo),
           "trend+weather": t_other + residual.predict(xo)}
    if "boosting" in lam:
        out["boosting"] = boost(xw, train["yield_t_ha"].values, lam["boosting"]).predict(xo)
        out["trend+boosting"] = t_other + boost(xw, train["yield_t_ha"].values - t_train,
                                                lam["trend+boosting"]).predict(xo)
    return out


def cross_validate(train, cols):
    """Choose each benchmark's setting (ridge penalty, boosting tree depth) by leaving out one
    year block at a time; returns the chosen settings and the out-of-fold predictions."""
    folds = sorted(train["fold"].unique())

    def cv_error(name, setting):
        err = []
        for f in folds:
            tr, va = train[train["fold"] != f], train[train["fold"] == f]
            if "boost" in name:
                lam = {"weather": 1, "trend+weather": 1, "boosting": setting, "trend+boosting": setting}
            else:
                lam = {"weather": setting, "trend+weather": setting}
            p = fit_predict(tr, va, cols, lam)[name]
            err.append((p - va["yield_t_ha"].values) ** 2)
        return np.sqrt(np.concatenate(err).mean())

    best = {}
    for name in ("weather", "trend+weather"):
        best[name] = min(LAMBDAS, key=lambda lam: cv_error(name, lam))
    for name in ("boosting", "trend+boosting"):
        best[name] = min(BOOST, key=lambda st: cv_error(name, st))
    oof = {n: np.zeros(len(train)) for n in MODELS}
    for f in folds:
        tr, va = train["fold"] != f, train["fold"] == f
        p = fit_predict(train[tr], train[va], cols, best)
        for n in MODELS:
            oof[n][va.values] = p[n]
    return best, oof


def scores(obs, pred, trend):
    rmse = np.sqrt(np.mean((pred - obs) ** 2))
    dev_p, dev_o = pred - trend, obs - trend
    skill = np.corrcoef(dev_p, dev_o)[0, 1] if np.std(dev_p) > 1e-9 else 0.0
    return rmse, 100 * rmse / obs.mean(), skill


def main():
    split = pd.read_csv(os.path.join(P, "split.csv"), dtype={"district": str})
    results, names = [], MODELS
    for crop in (sys.argv[1:] or CROPS):
        m = pd.read_csv(os.path.join(P, f"master_{crop}.csv"), dtype={"district": str})
        m = m.merge(split, on=["district", "year"])
        train = m[m["role"] == "train"].reset_index(drop=True)
        cols = features(m, train)
        lam, oof = cross_validate(train, cols)
        final = fit_predict(train, m, cols, lam)          # fitted on all training data
        out = m[["district", "year", "role", "yield_t_ha"]].copy()
        for n in names:
            out[n] = final[n]
            out.loc[out["role"] == "train", n] = oof[n]  # training rows: out-of-fold predictions
        out.to_csv(os.path.join(P, f"predictions_{crop}.csv"), index=False)

        print(f"\n{crop}: {len(cols)} features; mean yield {train['yield_t_ha'].mean():.2f} t/ha")
        print(f"  {'data':24s} {'n':>5s}" + "".join(f"{n:>16s}" for n in names) + "   RMSE t/ha (%)")
        groups = [("training (out-of-fold)", out["role"] == "train"),
                  ("test: new years", out["role"] == "test_years"),
                  ("test: new regions", out["role"] == "test_regions"),
                  ("test: both", out["role"] == "test_both"),
                  ("test: drought years", (out["role"] != "train") & out["year"].isin(DROUGHT))]
        for label, sel in groups:
            obs, tr = out.loc[sel, "yield_t_ha"].values, out.loc[sel, "trend"].values
            line = f"  {label:24s} {sel.sum():5d}"
            for n in names:
                rmse, rel, skill = scores(obs, out.loc[sel, n].values, tr)
                line += f"{rmse:9.2f} ({rel:4.1f}%)"
                results.append(dict(crop=crop, data=label, n=int(sel.sum()), model=n,
                                    rmse=round(rmse, 3), rel_rmse=round(rel, 1), weather_skill=round(skill, 3)))
            print(line)
    path = os.path.join(P, "phase1_results.csv")         # keep results of crops not rerun now
    new = pd.DataFrame(results)
    if os.path.exists(path):
        old = pd.read_csv(path)
        new = pd.concat([old[~old["crop"].isin(new["crop"])], new])
    new.to_csv(path, index=False)


if __name__ == "__main__":
    main()
