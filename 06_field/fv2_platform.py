"""Static platform files for the field level (plan 2026-10-17, Task 3): boundaries, weekly field series with alerts
and season labels, per-field pixel arrays, PNG overlays, Sentinel-2 true colour, district summary. Writes docs/field/.
"""
import numpy as np

ARABLE = {"wheat", "barley", "other cereal", "rapeseed", "maize", "potato", "sugar beet"}
SPRING = {"maize", "potato", "sugar beet"}
SEASON_LAI = 0.5
PARTIAL_SHARE = 0.99


def partial(share_inside):
    """Field straddles the area edge: less than 99 % of its registered area inside the 8 x 8 km area."""
    return bool(share_inside < PARTIAL_SHARE)


def field_mean(vals, w, inv, n_fields):
    """Share-weighted mean of pixel values per field; vals [weeks x pixels], NaN ignored."""
    ok = np.isfinite(vals)
    ww = np.where(ok, w[None, :], 0.0)
    out = np.full((vals.shape[0], n_fields), np.nan)
    for t in range(vals.shape[0]):
        sw = np.bincount(inv, weights=ww[t], minlength=n_fields)
        s = np.bincount(inv, weights=np.where(ok[t], vals[t], 0.0) * ww[t], minlength=n_fields)
        out[t] = np.where(sw > 0, s / np.where(sw > 0, sw, 1), np.nan)
    return out


def alerts(field, unc, median):
    """Per week: 1 = below the median of the same crop by more than the uncertainty, 2 weeks in a row;
    2 = falling while the median rises, with the divergence beyond the uncertainty, 2 weeks in a row; else 0."""
    f, u, m = (np.asarray(x, float) for x in (field, unc, median))
    below = (m - f) > u
    div = np.zeros(len(f), bool)
    div[1:] = ((f[1:] - f[:-1]) < 0) & ((m[1:] - m[:-1]) > 0) & (((m[1:] - m[:-1]) - (f[1:] - f[:-1])) > u[1:])
    out = []
    for t in range(len(f)):
        if t > 0 and below[t] and below[t - 1]:
            out.append(1)
        elif t > 0 and div[t] and div[t - 1]:
            out.append(2)
        else:
            out.append(0)
    return out


def season_mask(lai, group):
    """1 = the registered crop is in the field, 0 = before sowing (previous crop / bare) or after harvest.
    Winter crops: from the autumn minimum (sowing); spring crops: from the minimum before the first rise above 0.5 in
    the second half of the season; both until the last week above 0.5."""
    x = np.asarray(lai, float)
    n = len(x)
    if not np.isfinite(x).any():
        return [0] * n
    xx = np.where(np.isfinite(x), x, np.inf)
    if group in SPRING:
        half = n // 2
        rise = [i for i in range(half, n) if np.isfinite(x[i]) and x[i] > SEASON_LAI]
        stop = rise[0] if rise else n
        start = int(np.argmin(xx[:stop])) if stop > 0 else 0
    else:
        start = int(np.argmin(xx[: max(1, n // 2)]))
    above = [i for i in range(n) if np.isfinite(x[i]) and x[i] > SEASON_LAI]
    end = above[-1] if above else start
    return [1 if start <= i <= end else 0 for i in range(n)]
