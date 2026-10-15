"""Radar tools for field-level fusion (pre-registered docs/project_log.md 2026-10-15).

quegan()         multi-temporal speckle filter (Quegan & Yu 2001) on a stack of linear sigma0 images:
                 J_t(x) = M_t(x) * (1/N) * sum_i I_i(x) / M_i(x)
                 M_t = local mean (window x window) of date t; the sum over all N dates estimates each pixel's
                 stable texture relative to its neighbourhood. Speckle is reduced by using TIME, so edges and small
                 fields keep their 10 m structure (a plain spatial filter would blur them).
field_radar()    per orbit: filtered linear VV / VH -> mean over each field's pixels per date -> rolling mean over
                 the last 3 dates -> features VV, VH, VH-VV (dB) and RVI = 4 VH / (VV + VH).
"""
import glob
import os

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import uniform_filter


def load_stack(folder):
    """All GeoTIFFs of one radar orbit -> dates, VV and VH stacks [T x rows x cols] (linear, NaN = no data)."""
    files = sorted(glob.glob(os.path.join(folder, "*.tif")))
    dates, vv, vh = [], [], []
    for f in files:
        with rasterio.open(f) as r:
            a = r.read().astype(np.float32)
        a[a <= 0] = np.nan
        dates.append(pd.Timestamp(os.path.basename(f)[7:17]))
        vv.append(a[0]); vh.append(a[1])
    return pd.DatetimeIndex(dates), np.stack(vv), np.stack(vh)


def local_mean(img, window):
    """Mean over a window x window neighbourhood, ignoring missing pixels."""
    ok = np.isfinite(img)
    num = uniform_filter(np.where(ok, img, 0), window)
    den = uniform_filter(ok.astype(np.float32), window)
    return np.where(den > 0, num / np.maximum(den, 1e-6), np.nan)


def quegan(stack, window=5):
    """Multi-temporal speckle filter. stack: [T x rows x cols] linear intensities."""
    M = np.stack([local_mean(img, window) for img in stack])          # smooth part of each date
    ratio = np.nanmean(stack / M, axis=0)                              # each pixel's stable texture over time
    return M * ratio[None, :, :]                                       # today's smooth level x stable texture


def field_radar(folder, rows, cols, field_index, window=5):
    """Field-mean radar features of one orbit. field_index: field number per pixel (0..F-1)."""
    dates, vv, vh = load_stack(folder)
    F = field_index.max() + 1
    feats = {}
    lin = {}
    for name, st in (("VV", vv), ("VH", vh)):
        filt = quegan(st, window)[:, rows, cols]                       # [T x pixels]
        ok = np.isfinite(filt)
        s = np.stack([np.bincount(field_index, weights=np.where(ok[t], filt[t], 0), minlength=F) for t in range(len(dates))])
        n = np.stack([np.bincount(field_index, weights=ok[t].astype(float), minlength=F) for t in range(len(dates))])
        m = np.where(n > 0, s / np.maximum(n, 1), np.nan)               # field mean per date
        lin[name] = pd.DataFrame(m).rolling(3, min_periods=1).mean().values
    feats["VV"] = 10 * np.log10(lin["VV"])
    feats["VH"] = 10 * np.log10(lin["VH"])
    feats["RAT"] = 10 * np.log10(lin["VH"] / lin["VV"])
    feats["RVI"] = 4 * lin["VH"] / (lin["VV"] + lin["VH"])
    return dates, feats
