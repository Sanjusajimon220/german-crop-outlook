"""Landsat 8/9 OLI leaf retrieval network (pre-registered docs/project_log.md 2026-10-16).

Same method as the Sentinel-2 network (phase4_prosail.py): 70,000 PROSAIL canopies (same crop ranges), Gaussian band
responses, 3 % + 0.005 noise, network bands + angle cosines -> LAI, Cab, CCC, Cbrown with sd. Differences:
  bands  OLI blue / green / red / nir08 / swir16 / swir22 (USGS band edges -> centre, width in nm below)
  angles sun zenith from the real Landsat scenes of our areas (scenes.csv), view zenith 0-7.5 deg (OLI is near nadir),
         relative azimuth 0-180 deg
Check on 10,000 unseen simulated canopies; output data/processed/prosail_net_landsat.pt.
"""
import glob
import os
import time

import numpy as np
import pandas as pd
import prosail
import torch

from phase4_prosail import Net, TARGETS, sample_canopies

P = os.path.join("data", "processed")
MODEL = os.path.join(P, "prosail_net_landsat.pt")
BANDS = ["blue", "green", "red", "nir08", "swir16", "swir22"]
CENTRE = [482, 561.5, 654.5, 865, 1608.5, 2200.5]           # OLI band 2-7 (USGS edges 452-512 ... 2107-2294 nm)
FWHM = [60, 57, 37, 28, 85, 187]
torch.manual_seed(0)


def band_weights():
    wl = np.arange(400, 2501)
    w = np.stack([np.exp(-0.5 * ((wl - c) / (f / 2.355)) ** 2) for c, f in zip(CENTRE, FWHM)])
    return w / w.sum(1, keepdims=True)


def angles(n, rng):
    sc = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join("data", "raw", "field_v2", "*", "landsat", "scenes.csv"))])
    sza = 90 - sc["sun_elevation"].dropna().values
    return rng.choice(sza, n), rng.uniform(0, 7.5, n), rng.uniform(0, 180, n)


def simulate(n=70000, seed=0):
    rng = np.random.default_rng(seed)
    c = sample_canopies(n, rng)
    c["sza"], c["vza"], c["raa"] = angles(n, rng)
    w = band_weights()
    refl = np.zeros((n, len(BANDS)))
    t0 = time.time()
    for i, r in enumerate(c.itertuples()):
        spec = prosail.run_prosail(n=r.n, cab=r.cab, car=r.cab / 4.5, cbrown=r.cbrown, cw=r.cw, cm=r.cm, lai=r.lai, lidfa=r.ala,
                                   hspot=r.hspot, tts=r.sza, tto=r.vza, psi=r.raa, typelidf=2, lidfb=0, rsoil=r.rsoil,
                                   psoil=r.psoil, factor="SDR")
        refl[i] = w @ spec
        if i % 10000 == 0:
            print(f"  simulated {i} canopies ({time.time() - t0:.0f} s)", flush=True)
    noisy = refl * (1 + rng.normal(0, 0.03, refl.shape)) + rng.normal(0, 0.005, refl.shape)
    for k, b in enumerate(BANDS):
        c[b] = np.clip(noisy[:, k], 0, 1)
    c["ccc"] = c["lai"] * c["cab"]
    return c


def inputs(df):
    x = df[BANDS].values.astype(np.float32)
    ang = np.cos(np.radians(df[["sza", "vza", "raa"]].values)).astype(np.float32)
    return np.concatenate([x, ang], 1)


def main():
    path = os.path.join(P, "prosail_simulations_landsat.csv")
    sims = pd.read_csv(path) if os.path.exists(path) else simulate()
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
            loss = (0.5 * (lv + (yt[b] - mu) ** 2 / lv.exp())).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    torch.save({"state": net.state_dict(), "xm": xm, "xs": xs, "ym": ym, "ys": ys, "bands": BANDS}, MODEL)
    with torch.no_grad():
        mu, lv = net(xt[test])
    pred, sd, true = mu.numpy() * ys + ym, np.exp(0.5 * lv.numpy()) * ys, y[test]
    print("unseen simulated canopies (10,000), Landsat network:")
    for k, t in enumerate(TARGETS):
        err = pred[:, k] - true[:, k]
        print(f"  {t:7s} RMSE {np.sqrt((err ** 2).mean()):7.3f}  R2 {1 - (err ** 2).mean() / true[:, k].var():.3f}  "
              f"90 % range holds truth {100 * (np.abs(err) <= 1.645 * sd[:, k]).mean():.0f} %")


if __name__ == "__main__":
    main()
