"""Phase 4, step 2b: independent check of our LAI with ESA's SL2P (SNAP Biophysical Processor).

SL2P is ESA's standard Sentinel-2 LAI algorithm: a small neural network (11 inputs -> 5 tanh
neurons -> 1 output) that ESA trained on its own PROSAIL simulations with its own assumptions
(generic vegetation, not cereals). Official weights from ESA SNAP (data/raw/sl2p, version 2_1,
S2A network; source github.com/senbox-org/s2tbx). Inputs: B03 B04 B05 B06 B07 B8A B11 B12
(reflectance 0-1), cos(view zenith), cos(sun zenith), cos(relative azimuth).

Steps (as in SNAP):
  1. input check: spectrum inside ESA's definition domain (training range); otherwise flagged
  2. normalise each input to -1..1, layer 1 (tanh), layer 2 (linear), denormalise
  3. output check: slightly below 0 / above max -> clipped; far outside -> invalid
The code is first verified against ESA's own test cases (must give the same numbers).

Run `python phase4_sl2p.py 115 2021`: computes SL2P LAI and canopy chlorophyll (CCC = LAI x Cab)
for all fields and dates, compares with our PROSAIL retrieval (retrieval_<code>_<year>.csv) and
writes data/processed/s2/sl2p_<code>_<year>.csv and sl2p_comparison_<code>_<year>.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

P = os.path.join("data", "processed")
S2 = os.path.join(P, "s2")
AUX = os.path.join("data", "raw", "sl2p")
BANDS = ["B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12"]


class SL2P:
    def __init__(self, var):
        f = lambda name: np.loadtxt(os.path.join(AUX, var, f"{var}_{name}"), delimiter=",", ndmin=2)
        self.norm = f("Normalisation")                 # min, max per input (11 rows)
        self.denorm = f("Denormalisation")[0]
        self.w1, self.b1 = f("Weights_Layer1_Neurons"), f("Weights_Layer1_Bias").ravel()
        self.w2, self.b2 = f("Weights_Layer2_Neurons").ravel(), f("Weights_Layer2_Bias").ravel()[0]
        self.tol, self.lo, self.hi = f("ExtremeCases")[0]
        self.dom = f("DefinitionDomain_MinMax")        # band min / max of the training spectra
        self.grid = {tuple(r) for r in f("DefinitionDomain_Grid").astype(int)}
        self.test = f("TestCases")

    def raw(self, x):
        lo, hi = self.norm[:, 0], self.norm[:, 1]
        z = 2 * (x - lo) / (hi - lo) - 1
        h = np.tanh(z @ self.w1.T + self.b1)
        y = h @ self.w2 + self.b2
        return 0.5 * (y + 1) * (self.denorm[1] - self.denorm[0]) + self.denorm[0]

    def in_domain(self, x):
        lo, hi = self.dom
        cell = np.floor(10 * (x[:, :8] - lo) / (hi - lo)) + 1
        ok = np.isfinite(cell).all(1)
        return np.array([o and tuple(c.astype(int)) in self.grid for o, c in zip(ok, cell)])

    def __call__(self, x):
        y = self.raw(x)
        y = np.where((y < self.lo) & (y >= self.lo + self.tol), self.lo, y)
        y = np.where((y > self.hi) & (y <= self.hi - self.tol), self.hi, y)
        bad = (y < self.lo + self.tol) | (y > self.hi - self.tol)
        return np.where(bad, np.nan, y), self.in_domain(x)


def verify(net, var):
    """Our code must reproduce ESA's test cases, including the ones ESA marks invalid (NaN)."""
    t = net.test
    y, _ = net(t[:, :11])
    ok = np.isfinite(t[:, 11])
    err = np.abs(y[ok] - t[ok, 11])
    same_invalid = (np.isnan(y) == ~ok).mean()
    print(f"{var}: ESA test cases {len(t)} ({(~ok).sum()} invalid), max difference {err.max():.5f}, "
          f"same valid/invalid decision {100 * same_invalid:.0f} %")
    assert err.max() < 1e-3 * max(1, np.abs(t[ok, 11]).max())


def main(code, year):
    nets = {v: SL2P(v) for v in ("LAI", "LAI_Cab")}
    for v, n in nets.items():
        verify(n, v)
    df = pd.read_csv(os.path.join(S2, f"s2_{code}_{year}.csv")).dropna(subset=BANDS)
    raa = np.radians(df["sunAzimuthAngles"] - df["viewAzimuthMean"])
    x = np.column_stack([df[BANDS].values / 10000.0, np.cos(np.radians(df["viewZenithMean"])),
                         np.cos(np.radians(df["sunZenithAngles"])), np.cos(raa)])
    df["sl2p_lai"], df["in_domain"] = nets["LAI"](x)
    df["sl2p_ccc"], _ = nets["LAI_Cab"](x)
    s = df[["date", "field_id", "sl2p_lai", "sl2p_ccc", "in_domain"]]
    s.to_csv(os.path.join(S2, f"sl2p_{code}_{year}.csv"), index=False)

    ours = pd.read_csv(os.path.join(S2, f"retrieval_{code}_{year}.csv"))
    m = ours.merge(s, on=["date", "field_id"])
    m["month"] = pd.to_datetime(m["date"]).dt.strftime("%Y-%m")
    print(f"\n{len(m)} field-dates; inside ESA's definition domain: {100 * m['in_domain'].mean():.0f} %; "
          f"SL2P LAI invalid: {100 * m['sl2p_lai'].isna().mean():.1f} %")
    rows = []
    for name, g in [("all", m), ("in domain", m[m["in_domain"]])] + list(m.groupby("month")):
        for var, a, b in (("LAI", "lai", "sl2p_lai"), ("CCC", "ccc", "sl2p_ccc")):
            gg = g[[a, b]].dropna()
            if len(gg) < 50:
                continue
            d = gg[a] - gg[b]
            rows.append(dict(group=name, var=var, n=len(gg), ours_median=gg[a].median(),
                             sl2p_median=gg[b].median(), mean_diff=d.mean(),
                             rmsd=np.sqrt((d ** 2).mean()), r=np.corrcoef(gg[a], gg[b])[0, 1]))
    t = pd.DataFrame(rows).round(3)
    t.to_csv(os.path.join(S2, f"sl2p_comparison_{code}_{year}.csv"), index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
