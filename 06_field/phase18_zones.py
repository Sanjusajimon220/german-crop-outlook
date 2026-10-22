"""Module A: zones of consistent crop growth (design and rule fixed 2026-10-21 before any result; one look).

Per pixel and season: in-season sum of weekly green LAI, relative to its field's mean that season. Zones per field (latest
season's boundaries): tertiles of the multi-season mean, headland excluded, patches < 0.1 ha merged. Test: leave one season
out; per field Spearman of zone rank vs held-out zone means; PASS if >= 0.5 in >= 70 % of fields in every held-out season.
Run from D:\\vista: python phase18_zones.py -> data/processed/field_v2/zones/ (test tables, zone grids) + printed result.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = r"D:\vista\data\processed\field_v2\zones"
DOCS = r"D:\german-crop-outlook-field\docs\field"
SEASONS = list(range(2021, 2026))
GRID = 800
MIN_PATCH = 10                                                   # 0.1 ha of 10 m pixels
MIN_HA, MIN_SEASONS, PASS_RHO, PASS_SHARE, FLAT = 2.0, 3, 0.5, 0.70, 0.05


def season_sum(lai, mask, min_cover=0.8):
    """Sum of weekly LAI over each pixel's in-season weeks; NaN if fewer than min_cover of those weeks have a value."""
    ok = np.isfinite(lai) & mask
    n_in = mask.sum(0)
    s = np.where(ok, lai, 0.0).sum(0)
    return np.where((n_in > 0) & (ok.sum(0) >= min_cover * n_in), s, np.nan)


def relative_growth(s, field, w):
    """Pixel value / weighted mean of its field (NaN pixels ignored)."""
    ok = np.isfinite(s)
    n = int(field.max()) + 1
    num = np.bincount(field[ok], (s * w)[ok], n)
    den = np.bincount(field[ok], w[ok], n)
    mean = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    return s / mean[field]


def tertile_zones(v, field):
    """1 / 2 / 3 = weakest / middle / strongest third of each field's pixels (by rank; NaN -> 0)."""
    z = np.zeros(len(v), np.int8)
    for f in np.unique(field):
        idx = np.flatnonzero((field == f) & np.isfinite(v))
        if len(idx) < 3:
            continue
        r = pd.Series(v[idx]).rank(method="first").values
        z[idx] = np.ceil(3 * r / len(idx)).astype(np.int8)
    return z


def merge_small_patches(grid, min_pixels=MIN_PATCH):
    """Connected patches (4-neighbour) of a zone smaller than min_pixels take the most common neighbouring zone."""
    g = grid.copy()
    for _ in range(3):
        changed = False
        for zval in (1, 2, 3):
            lab, n = ndimage.label(g == zval)
            if not n:
                continue
            sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
            for k in np.flatnonzero(sizes < min_pixels) + 1:
                m = lab == k
                ring = ndimage.binary_dilation(m) & ~m & (g > 0)
                if ring.any():
                    vals, cnt = np.unique(g[ring], return_counts=True)
                    g[m] = vals[np.argmax(cnt)]
                    changed = True
        if not changed:
            break
    return g


def zone_order_score(zones, held):
    """Spearman of zone rank (1-3) vs the held-out mean of each zone; NaN unless all three zones have values."""
    means = [np.nanmean(held[zones == k]) if np.isfinite(held[zones == k]).any() else np.nan for k in (1, 2, 3)]
    if not np.all(np.isfinite(means)):
        return np.nan
    return float(pd.Series([1, 2, 3]).corr(pd.Series(means).rank(), method="spearman"))


# ---- real-data run ----
def _season_grid(area, year):
    """Per grid pixel (row * 800 + col): relative growth, peak LAI, arable flag, field id / group of that season."""
    from areas import out_dir
    from fv2_platform import ARABLE, field_mean, season_mask
    d = out_dir(area, year)
    z = dict(np.load(os.path.join(d, "weekly_pixels_V2.npz")))
    px = pd.read_csv(os.path.join(d, "pixels_nrw.csv.gz"))
    mainf = px.sort_values("share").drop_duplicates(["row", "col"], keep="last").set_index(["row", "col"])
    key = pd.MultiIndex.from_arrays([z["rows"], z["cols"]])
    fid = mainf["field_id"].reindex(key).values
    share = np.nan_to_num(mainf["share"].reindex(key).values)
    fl = pd.read_csv(os.path.join(d, "fields_nrw.csv")).set_index("field_id")
    ok = np.isfinite(fid)
    fids, inv = np.unique(fid[ok].astype(np.int64), return_inverse=True)
    groups = fl["crop_group"].reindex(fids).fillna("other").values
    lai = z["lai"][:, ok]
    fm = field_mean(lai, share[ok], inv, len(fids))
    fmask = np.array([season_mask(fm[:, k], groups[k]) for k in range(len(fids))], bool).T      # weeks x fields
    pmask = fmask[:, inv]
    s = season_sum(lai, pmask)
    rel = relative_growth(s, inv, np.maximum(share[ok], 1e-6))
    peak = np.where(pmask, lai, np.nan)
    peak = np.where(np.isfinite(peak).any(0), np.nanmax(np.where(np.isfinite(peak), peak, -np.inf), 0), np.nan)
    g = np.full(GRID * GRID, np.nan); P = np.full(GRID * GRID, np.nan)
    A = np.zeros(GRID * GRID, bool); F = np.full(GRID * GRID, -1, np.int64)
    gk = z["rows"][ok] * GRID + z["cols"][ok]
    g[gk], P[gk], F[gk] = rel, peak, fids[inv]
    A[gk] = np.isin(groups[inv], list(ARABLE)) & np.isfinite(rel)
    SUM = np.full(GRID * GRID, np.nan); SUM[gk] = s                  # raw season sum (used by the per-field classes)
    PG = np.full(GRID * GRID, "", dtype=object); PG[gk] = groups[inv]
    return dict(rel=g, peak=P, arable=A, field=F, group={int(f): gr for f, gr in zip(fids, groups)}, sum=SUM, pgroup=PG)


def run():
    os.makedirs(OUT, exist_ok=True)
    from areas import out_dir
    rows = []
    for area in ["pilot", "selhausen", "north", "south"]:
        S = {y: _season_grid(area, y) for y in SEASONS}
        last = SEASONS[-1]
        geo = json.load(open(os.path.join(DOCS, f"{area}_{last}", "fields.geojson"), encoding="utf-8"))
        props = {int(f["properties"]["id"]): f["properties"] for f in geo["features"]}
        px = pd.read_csv(os.path.join(out_dir(area, last), "pixels_nrw.csv.gz"))
        head = set((px.row[px.headland] * GRID + px.col[px.headland]).tolist())
        F = S[last]["field"]
        zone_all = np.zeros(GRID * GRID, np.int8)
        for f, p in props.items():
            if p.get("partial") or (p.get("ha") or 0) < MIN_HA:
                continue
            P = np.flatnonzero(F == f)
            P = np.array([k for k in P if k not in head], np.int64)
            if len(P) < 3 * MIN_PATCH:
                continue
            counted = [y for y in SEASONS if S[y]["arable"][P].mean() >= 0.5]
            if len(counted) < MIN_SEASONS:
                continue
            r0, c0 = P // GRID, P % GRID

            def zones_from(seasons):
                m = np.nanmean(np.vstack([S[y]["rel"][P] for y in seasons]), 0)
                z = tertile_zones(m, np.zeros(len(P), np.int64))
                grid = np.zeros((r0.max() - r0.min() + 1, c0.max() - c0.min() + 1), np.int8)
                grid[r0 - r0.min(), c0 - c0.min()] = z
                grid = merge_small_patches(grid)
                return grid[r0 - r0.min(), c0 - c0.min()]

            for h in counted:
                others = [y for y in counted if y != h]
                if len(others) < MIN_SEASONS - 1:
                    continue
                z = zones_from(others)
                held = S[h]["rel"][P]
                score = zone_order_score(z, held)
                means = [np.nanmean(held[z == k]) for k in (1, 2, 3)]
                pk = S[h]["peak"][P]
                crop_h = S[h]["group"].get(int(np.bincount(np.maximum(S[h]["field"][P], 0)).argmax()), "?") if (S[h]["field"][P] >= 0).any() else "?"
                rows.append(dict(area=area, field=f, ha=p.get("ha"), held_out=h, crop=crop_h, n_seasons=len(counted), score=score,
                                 spread=np.nanmax(means) - np.nanmin(means) if np.isfinite(means).all() else np.nan,
                                 peak_gap=np.nanmean(pk[z == 3]) - np.nanmean(pk[z == 1]) if ((z == 3).any() and (z == 1).any()) else np.nan))
            zone_all[P] = zones_from(counted)
        np.save(os.path.join(OUT, f"zones_{area}_{last}.npy"), zone_all.reshape(GRID, GRID))
        print(area, "fields tested", len({r['field'] for r in rows if r['area'] == area}), flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "zones_test.csv"), index=False)
    t = t.dropna(subset=["score"])
    per = t.groupby("held_out").agg(fields=("score", "size"), share_ok=("score", lambda s: (s >= PASS_RHO).mean()))
    print("Share of fields with zone order kept (Spearman >= 0.5), per held-out season:\n", per.round(3).to_string())
    ok = bool((per.share_ok >= PASS_SHARE).all())
    print(f"RESULT Module A: {'PASS' if ok else 'FAIL'} (rule: >= {PASS_SHARE:.0%} of fields in every held-out season)")
    print("by area:\n", t.groupby("area").score.apply(lambda s: (s >= PASS_RHO).mean()).round(3).to_string())
    print("by crop in the held-out season:\n", t.groupby("crop").score.agg(["size", lambda s: (s >= PASS_RHO).mean()]).round(3).to_string())
    print(f"flat fields (zone spread < {FLAT:.0%}): {(t.spread < FLAT).mean():.1%}; median spread {t.spread.median():.3f}; "
          f"median peak LAI gap strong - weak {t.peak_gap.median():.2f}")
    print("ZONES DONE")


if __name__ == "__main__":
    run()
