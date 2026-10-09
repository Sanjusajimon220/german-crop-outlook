"""v10 priority 2, method choice for honest ranges (training data only; pre-registered 2026-10-08).

Build from forward errors of models fitted <= 2002 (years 2003-2008), score on forward errors of models
fitted <= 2008 (years 2009-2017, true weather = end of season). Error = official - predicted (t/ha).
  A  pooled district errors, drawn independently per district (current method)
  B  year-wide part + local rest: year part ~ bias + sd_year * sqrt(1 + 1/n) * Student-t(n - 1)
     (n = number of build years), local rest from the empirical pool
  C  B plus a state-year part (drawn from the empirical state-year pool)
Joint draws (K per test year): one year draw shared by all districts (B, C), so state and national
ranges come from the same draws as the district ranges.
Scores: district 80 % coverage (and worst year), CRPS, PIT share in the outer 10 % tails; state and
national (area-weighted) 80 % coverage and CRPS; Brier score of P(yield < district's previous 5-year
official mean) against the base rate of the build years.
Writes data/processed/range_method_v10.csv.
"""
import os

import numpy as np
import pandas as pd

from phase5_benchmarks import census_area

P = os.path.join("data", "processed")
CROPS = ("winter_wheat", "winter_barley", "grain_maize", "silage_maize", "potato")
K = 1000


def crps(draws, y):
    """Sample CRPS for each row: mean|X - y| - 0.5 mean|X - X'| (sorted-sample formula)."""
    x = np.sort(draws, axis=1)
    n = x.shape[1]
    a = np.abs(x - y[:, None]).mean(1)
    i = np.arange(1, n + 1)
    b = ((2 * i - n - 1) * x).sum(1) / (n * n)
    return a - b


def parts(e, year, state):
    d = pd.DataFrame({"e": e, "year": year, "state": state})
    ym = d.groupby("year")["e"].transform("mean")
    sym = d.groupby(["year", "state"])["e"].transform("mean")
    return dict(year_means=d.groupby("year")["e"].mean().values,
                sy=(sym - ym).values, local_b=(d["e"] - ym).values, local_c=(d["e"] - sym).values)


def draws_for(method, pool, n_dist, states, rng):
    """K x n_dist error draws for one test year."""
    if method == "A":
        return rng.choice(pool["all"], size=(K, n_dist))
    ym = pool["year_means"]
    n = len(ym)
    yr = ym.mean() + ym.std(ddof=1) * np.sqrt(1 + 1 / n) * rng.standard_t(n - 1, size=(K, 1))
    if method == "B":
        return yr + rng.choice(pool["local_b"], size=(K, n_dist))
    us = np.unique(states)
    sy = rng.choice(pool["sy"], size=(K, len(us)))
    idx = np.searchsorted(us, states)
    return yr + sy[:, idx] + rng.choice(pool["local_c"], size=(K, n_dist))


def main():
    rng = np.random.default_rng(0)
    y = pd.read_csv(os.path.join(P, "yields_long.csv"), dtype={"district": str})
    rows = []
    for crop in CROPS:
        yc = y[y["crop"] == crop].copy()
        yc["area"] = census_area(yc)
        hist = yc.set_index(["district", "year"])["yield_t_ha"]
        area = yc.set_index(["district", "year"])["area"]
        b = pd.read_csv(os.path.join(P, f"forward_predictions_{crop}_cut2002.csv"), dtype={"district": str})
        b = b[(b["year"] <= 2008)].dropna(subset=["yield_t_ha", "pred"])
        eb = (b["yield_t_ha"] - b["pred"]).values
        pool = parts(eb, b["year"].values, b["district"].str[:2].values) | {"all": eb}

        def below(df):
            prev = np.array([np.nanmean([hist.get((d, yr - k), np.nan) for k in range(1, 6)])
                             for d, yr in zip(df["district"], df["year"])])
            return prev
        thr_b = below(b)
        base = np.nanmean(b["yield_t_ha"].values < thr_b)
        t = pd.read_csv(os.path.join(P, f"forward_predictions_{crop}_cut2008.csv"), dtype={"district": str})
        t = t.dropna(subset=["yield_t_ha", "pred"]).reset_index(drop=True)
        t["state"] = t["district"].str[:2]
        t["area"] = [area.get((d, yr), np.nan) for d, yr in zip(t["district"], t["year"])]
        t["thr"] = below(t)
        for method in ("A", "B", "C"):
            dist, agg = [], []
            for yr, g in t.groupby("year"):
                g = g.reset_index(drop=True)
                dr = g["pred"].values[None, :] + draws_for(method, pool, len(g), g["state"].values, rng)
                obs = g["yield_t_ha"].values
                q10, q90 = np.percentile(dr, [10, 90], axis=0)
                pit = (dr < obs[None, :]).mean(0)
                c = crps(dr.T, obs)
                ev = obs < g["thr"].values
                p_below = (dr < g["thr"].values[None, :]).mean(0)
                ok = np.isfinite(g["thr"].values)
                dist.append(pd.DataFrame(dict(year=yr, inside=(obs >= q10) & (obs <= q90), pit=pit, crps=c,
                                              brier=np.where(ok, (p_below - ev) ** 2, np.nan),
                                              brier_base=np.where(ok, (base - ev) ** 2, np.nan))))
                g["w"] = g["area"].fillna(0)
                for level, keys in (("national", np.zeros(len(g), int)), ("state", g["state"].values)):
                    for key in np.unique(keys):
                        s = (keys == key) & (g["w"].values > 0)
                        if s.sum() < 3:
                            continue
                        w = g["w"].values[s] / g["w"].values[s].sum()
                        ad = dr[:, s] @ w
                        ao = float(obs[s] @ w)
                        lo, hi = np.percentile(ad, [10, 90])
                        agg.append(dict(level=level, inside=lo <= ao <= hi, crps=crps(ad[None, :], np.array([ao]))[0],
                                        rel_crps=crps(ad[None, :], np.array([ao]))[0] / ao))
            dd, ag = pd.concat(dist), pd.DataFrame(agg)
            row = dict(crop=crop, method=method, n=len(dd), coverage80=dd["inside"].mean(),
                       worst_year_cov=dd.groupby("year")["inside"].mean().min(), crps=dd["crps"].mean(),
                       pit_tails=((dd["pit"] < 0.1) | (dd["pit"] > 0.9)).mean(),
                       brier=np.nanmean(dd["brier"]), brier_base=np.nanmean(dd["brier_base"]))
            for level in ("state", "national"):
                a = ag[ag["level"] == level]
                row[f"{level}_cov80"] = a["inside"].mean()
                row[f"{level}_crps"] = a["crps"].mean()
            rows.append(row)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    out = pd.DataFrame(rows).round(3)
    out.to_csv(os.path.join(P, "range_method_v10.csv"), index=False)
    with pd.option_context("display.width", 250):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
