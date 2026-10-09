"""Phase 4, step 2: physics-based Sentinel-2 retrieval (PROSAIL + neural network).

1. Simulate: 60,000 virtual cereal canopies with PROSAIL (PROSPECT-D leaf model + 4SAIL canopy
   model). Crop properties are drawn from ranges typical of German cereals (below); sun and view
   angles are drawn from the real Sentinel-2 field observations (data/processed/s2), so the
   simulated geometry matches what the satellite sees over the Rur region.
2. Convert each simulated spectrum (400-2500 nm) to the 10 Sentinel-2 bands with Gaussian band
   responses (centre wavelength and width of each band) and add measurement noise
   (3 % relative + 0.005 absolute), so the network learns to cope with real-data noise.
3. Train a small neural network: bands + angles -> LAI, leaf chlorophyll (Cab), canopy chlorophyll
   (CCC = LAI x Cab, closely linked to canopy nitrogen) and brown pigment (senescence). It outputs
   a mean and a standard deviation for each, so every retrieval carries its own uncertainty.
4. Check on 10,000 simulated canopies the network never saw (accuracy and whether the stated
   uncertainty is honest), then apply to the field tables: data/processed/s2/s2_<code>_<year>.csv
   -> data/processed/s2/retrieval_<code>_<year>.csv.

Note: LAI here is the effective LAI of a turbid canopy (as in ESA's SL2P); leaf clumping in row
crops and the green/brown split are refined in the assimilation step.
Run `python phase4_prosail.py train` then `python phase4_prosail.py apply 115 2021`.
"""
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import prosail
import torch

P = os.path.join("data", "processed")
S2 = os.path.join(P, "s2")
MODEL = os.path.join(P, "prosail_net.pt")
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
# Sentinel-2A band centres and widths (nm, full width at half maximum)
CENTRE = [492, 560, 665, 704, 740, 783, 833, 865, 1614, 2202]
FWHM = [66, 36, 31, 15, 15, 20, 106, 21, 91, 175]
TARGETS = ["lai", "cab", "ccc", "cbrown"]
torch.manual_seed(0)


def band_weights():
    wl = np.arange(400, 2501)
    w = np.stack([np.exp(-0.5 * ((wl - c) / (f / 2.355)) ** 2) for c, f in zip(CENTRE, FWHM)])
    return w / w.sum(1, keepdims=True)


def observed_angles(n, rng):
    files = sorted(glob.glob(os.path.join(S2, "s2_115_*_part*.csv")))[:4]
    a = pd.concat([pd.read_csv(f, usecols=["sunZenithAngles", "viewZenithMean", "sunAzimuthAngles",
                                            "viewAzimuthMean", "B04"]) for f in files]).dropna()
    a = a.drop_duplicates().sample(n, replace=True, random_state=1)
    raa = np.abs(a["sunAzimuthAngles"].values - a["viewAzimuthMean"].values) % 360
    raa = np.where(raa > 180, 360 - raa, raa)
    return a["sunZenithAngles"].values, a["viewZenithMean"].values, raa


def sample_canopies(n, rng):
    """Crop properties typical of cereals (ranges after SL2P and cereal field studies)."""
    def tnorm(mu, sd, lo, hi):
        x = rng.normal(mu, sd, n)
        bad = (x < lo) | (x > hi)
        while bad.any():
            x[bad] = rng.normal(mu, sd, bad.sum())
            bad = (x < lo) | (x > hi)
        return x
    lai = np.where(rng.random(n) < 0.25, rng.uniform(0, 1, n), tnorm(3.5, 2.0, 0, 8))
    senescent = rng.random(n) < 0.3
    return pd.DataFrame({
        "n": rng.uniform(1.2, 2.2, n),
        "cab": tnorm(45, 15, 10, 85),
        "cbrown": np.where(senescent, rng.uniform(0.1, 1.5, n), rng.uniform(0, 0.1, n)),
        "cw": rng.uniform(0.005, 0.04, n),
        "cm": rng.uniform(0.003, 0.011, n),
        "lai": lai,
        "ala": tnorm(60, 10, 35, 80),            # cereals: fairly upright leaves
        "hspot": rng.uniform(0.01, 0.5, n),
        "psoil": rng.uniform(0, 1, n),           # dry (1) to wet (0) soil
        "rsoil": rng.uniform(0.5, 1.5, n),       # soil brightness
    })


def simulate(n=70000, seed=0):
    rng = np.random.default_rng(seed)
    c = sample_canopies(n, rng)
    c["sza"], c["vza"], c["raa"] = observed_angles(n, rng)
    w = band_weights()
    refl = np.zeros((n, len(BANDS)))
    t0 = time.time()
    for i, r in enumerate(c.itertuples()):
        spec = prosail.run_prosail(n=r.n, cab=r.cab, car=r.cab / 4.5, cbrown=r.cbrown, cw=r.cw, cm=r.cm,
                                   lai=r.lai, lidfa=r.ala, hspot=r.hspot, tts=r.sza, tto=r.vza, psi=r.raa,
                                   typelidf=2, lidfb=0, rsoil=r.rsoil, psoil=r.psoil, factor="SDR")
        refl[i] = w @ spec
        if i % 10000 == 0:
            print(f"  simulated {i} canopies ({time.time() - t0:.0f} s)", flush=True)
    noisy = refl * (1 + rng.normal(0, 0.03, refl.shape)) + rng.normal(0, 0.005, refl.shape)
    for k, b in enumerate(BANDS):
        c[b] = np.clip(noisy[:, k], 0, 1)
    c["ccc"] = c["lai"] * c["cab"]
    return c


def inputs(df):
    """Network inputs: reflectances (0-1) and cosines of the angles."""
    x = df[BANDS].values.astype(np.float32)
    ang = np.cos(np.radians(df[["sza", "vza", "raa"]].values)).astype(np.float32)
    return np.concatenate([x, ang], 1)


class Net(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Linear(n_in, 128), torch.nn.Tanh(),
                                        torch.nn.Linear(128, 128), torch.nn.Tanh())
        self.mean, self.logvar = torch.nn.Linear(128, n_out), torch.nn.Linear(128, n_out)

    def forward(self, x):
        h = self.body(x)
        return self.mean(h), self.logvar(h)


def train():
    path = os.path.join(P, "prosail_simulations.csv")
    if os.path.exists(path):
        sims = pd.read_csv(path)
    else:
        print("simulating canopies with PROSAIL", flush=True)
        sims = simulate()
        sims.to_csv(path, index=False)
    x, y = inputs(sims), sims[TARGETS].values.astype(np.float32)
    xm, xs, ym, ys = x.mean(0), x.std(0), y.mean(0), y.std(0)
    xt, yt = torch.tensor((x - xm) / xs), torch.tensor((y - ym) / ys)
    test = np.arange(len(x)) >= len(x) - 10000
    tr = np.flatnonzero(~test)
    net = Net(x.shape[1], y.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    for epoch in range(60):
        perm = np.random.default_rng(epoch).permutation(tr)
        for a in range(0, len(perm), 512):
            b = perm[a:a + 512]
            mu, lv = net(xt[b])
            loss = (0.5 * (lv + (yt[b] - mu) ** 2 / lv.exp())).mean()     # Gaussian likelihood
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 10 == 9:
            print(f"  epoch {epoch + 1}: loss {loss.item():.3f}", flush=True)
    torch.save({"state": net.state_dict(), "xm": xm, "xs": xs, "ym": ym, "ys": ys}, MODEL)
    with torch.no_grad():
        mu, lv = net(xt[test])
    pred = mu.numpy() * ys + ym
    sd = np.exp(0.5 * lv.numpy()) * ys
    true = y[test]
    print("\nunseen simulated canopies (10,000):")
    for k, t in enumerate(TARGETS):
        err = pred[:, k] - true[:, k]
        inside = np.abs(err) <= 1.645 * sd[:, k]          # 90 % interval
        r2 = 1 - (err ** 2).mean() / true[:, k].var()
        print(f"  {t:7s} RMSE {np.sqrt((err ** 2).mean()):7.3f}  R2 {r2:.3f}  "
              f"90 % range holds truth {100 * inside.mean():.0f} %")


def apply(code, year):
    ck = torch.load(MODEL, weights_only=False)
    net = Net(len(BANDS) + 3, len(TARGETS))
    net.load_state_dict(ck["state"])
    df = pd.read_csv(os.path.join(S2, f"s2_{code}_{year}.csv")).dropna(subset=BANDS)
    df = df.rename(columns={"sunZenithAngles": "sza", "viewZenithMean": "vza"})
    raa = np.abs(df["sunAzimuthAngles"] - df["viewAzimuthMean"]) % 360
    df["raa"] = np.where(raa > 180, 360 - raa, raa)
    df[BANDS] = df[BANDS] / 10000.0
    x = torch.tensor((inputs(df) - ck["xm"]) / ck["xs"])
    with torch.no_grad():
        mu, lv = net(x)
    pred, sd = mu.numpy() * ck["ys"] + ck["ym"], np.exp(0.5 * lv.numpy()) * ck["ys"]
    for k, t in enumerate(TARGETS):
        df[t], df[f"{t}_sd"] = pred[:, k], sd[:, k]
    df["lai"] = df["lai"].clip(lower=0)
    out = df[["date", "field_id", "area_ha", "inner_ha"] + [c for t in TARGETS for c in (t, f"{t}_sd")]]
    out.to_csv(os.path.join(S2, f"retrieval_{code}_{year}.csv"), index=False)
    out = out.assign(month=pd.to_datetime(out["date"]).dt.strftime("%Y-%m"))
    print(out.groupby("month")[["lai", "lai_sd", "cab", "ccc"]].median().round(2).to_string())


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train()
    else:
        apply(int(sys.argv[2]), int(sys.argv[3]))
