"""Weekly SL2P vs our weekly LAI (descriptive comparison, project log 2026-11-03; no pass/fail rule, no claim).

For every area-season the raw Sentinel-2 scenes are read again with exactly the pixels, cloud mask, fix 1 and Dec/Jan
rule of fv2.retrieval_s2, and ESA's SL2P LAI network (phase4_sl2p.SL2P, SNAP v2_1) is run on the same clear pixels.
Three weekly series on the published Mondays:
  sl2p       SL2P, line only: measured within 3 days, else a straight line between clear images whatever the gap,
             after the last clear image the last value
  ours_line  our clear LAI (fix 2 flags dropped, as in production) with the same line-only rule (like for like)
  ours       the published weekly series (weekly_pixels_V2: line up to 30 days, radar model beyond)
The radar model was trained on our LAI, so it is not applied to SL2P.

Run from D:\\vista: `python phase26_sl2p_weekly.py run [processes]` (resumable), then `... charts`.
Writes data/processed/field_v2/sl2p_weekly/: weekly_<area>_<year>.csv (median and middle 50 % per crop and week),
pairs_<area>_<year>.csv (pixel-level differences by crop, month and source), sample_<area>_<year>.npz, charts.
"""
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

MEASURED_DAYS = 3
SL2P_BANDS = [1, 2, 3, 4, 5, 7, 8, 9]          # B03 B04 B05 B06 B07 B8A B11 B12 in fv2's B02..B12 band order
CROPS = ["wheat", "barley", "maize", "rapeseed", "sugar beet", "potato"]
SRC_NAMES = {1: "measured", 2: "line", 3: "radar", 4: "after last image"}
OUT = os.path.join("data", "processed", "field_v2", "sl2p_weekly")
AREAS = ["pilot", "selhausen", "north", "south"]
YEARS = [2021, 2022, 2023, 2024, 2025]


def _encode(cells):
    """Integer code per row of 8 domain cells (1..11 inside ESA's grid); -1 when any cell is invalid or far outside."""
    cells = np.asarray(cells, float)
    bad = ~np.isfinite(cells).all(1) | (cells < 0).any(1) | (cells > 15).any(1)
    c = np.where(bad[:, None], 0, cells).astype(np.int64)
    code = (c * (16 ** np.arange(8))).sum(1)
    return np.where(bad, -1, code)


def domain_codes(x, lo, hi):
    """Vectorised form of phase4_sl2p.SL2P.in_domain: the definition-domain cell of each spectrum, as one integer."""
    return _encode(np.floor(10 * (x[:, :8] - lo) / (hi - lo)) + 1)


def line_weekly(days, vals, weeks):
    """Line-only weekly series. days (D,) ascending clear-image days, vals (D, P) with NaN = not clear, weeks (W,).
    Returns (values (W, P), measured (W, P) bool)."""
    D, P = vals.shape
    fin = np.isfinite(vals)
    ar = np.arange(D)[:, None]
    ff = np.maximum.accumulate(np.where(fin, ar, -1), axis=0)                         # last clear index up to j
    bf = np.minimum.accumulate(np.where(fin, ar, D)[::-1], axis=0)[::-1]              # first clear index from j
    out = np.full((len(weeks), P), np.nan, np.float32)
    meas = np.zeros((len(weeks), P), bool)
    cols = np.arange(P)
    for w, wd in enumerate(weeks):
        for j in np.argsort(np.abs(days - wd), kind="stable"):                        # nearest clear within 3 days
            if abs(days[j] - wd) > MEASURED_DAYS:
                break
            take = ~meas[w] & fin[j]
            out[w, take], meas[w, take] = vals[j, take], True
        kl, kr = np.searchsorted(days, wd, "left"), np.searchsorted(days, wd, "right")
        prv = ff[kl - 1] if kl > 0 else np.full(P, -1)
        nxt = bf[kr] if kr < D else np.full(P, D)
        rest = ~meas[w] & (prv >= 0)
        both = rest & (nxt < D)
        a, b = prv[both], nxt[both]
        va, vb = vals[a, cols[both]], vals[b, cols[both]]
        out[w, both] = va + (wd - days[a]) / (days[b] - days[a]) * (vb - va)
        last = rest & (nxt >= D)
        out[w, last] = vals[prv[last], cols[last]]
    return out, meas


def pair_stats(a, b):
    """a - b over values present in both: n, mean difference, RMSD, correlation."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    k = np.isfinite(a) & np.isfinite(b)
    if k.sum() < 3:
        return dict(n=int(k.sum()), mean_diff=np.nan, rmsd=np.nan, r=np.nan)
    d = a[k] - b[k]
    return dict(n=int(k.sum()), mean_diff=float(d.mean()), rmsd=float(np.sqrt((d ** 2).mean())),
                r=float(np.corrcoef(a[k], b[k])[0, 1]))


# ---- full area-season run (reads the raw scenes; needs fv2 setup outputs) ----

def sl2p_clear(area, year):
    """SL2P LAI on the clear pixels of every kept scene, same rules as fv2.retrieval_s2; cached as s2_pixels_sl2p.npz."""
    import rasterio
    from scipy.ndimage import binary_dilation
    import fv2
    from areas import out_dir, raw_dir
    from phase4_sl2p import SL2P
    path = os.path.join(out_dir(area, year), "s2_pixels_sl2p.npz")
    if os.path.exists(path):
        try:
            return dict(np.load(path))
        except Exception:                                   # half-written by a stopped run: compute again
            pass
    net = SL2P("LAI")
    grid = _encode(np.array(sorted(net.grid)))
    px = fv2.pixels(area, year)
    rows, cols = px["rows"], px["cols"]
    pinv, pshare = px["inv"], np.nan_to_num(px["share"])
    ftot = np.bincount(pinv, weights=pshare)
    dates, lai, dom = [], [], []
    for f in sorted(glob.glob(os.path.join(raw_dir(area, year), "s2", "*.tif"))):
        d = pd.Timestamp(os.path.basename(f)[7:17])
        if d.month in (12, 1):
            continue
        with rasterio.open(f) as r:
            a = r.read()
        bad = binary_dilation(np.isin(a[10], (3, 8, 9, 10, 11)), iterations=2)
        c = (np.isin(a[10], (4, 5)) & ~bad & np.all(np.isfinite(a[:10]), axis=0))[rows, cols]
        c = fv2._keep(c, pinv, pshare, ftot)
        if not c.any():
            continue
        refl = a[:10, rows, cols].T / 10000.0
        sza, vza = a[11, rows, cols], a[12, rows, cols]
        raa = np.abs(a[13, rows, cols] - a[14, rows, cols]) % 360
        raa = np.where(raa > 180, 360 - raa, raa)
        ang = np.cos(np.radians(np.stack([sza, vza, raa], 1)))
        ok = c & np.all(np.isfinite(np.concatenate([refl, ang], 1)), axis=1)      # the pixels our network used
        x = np.column_stack([refl[ok][:, SL2P_BANDS], ang[ok][:, 1], ang[ok][:, 0], ang[ok][:, 2]])   # vza, sza, raa
        y = net.raw(x)
        y = np.where((y < net.lo) & (y >= net.lo + net.tol), net.lo, y)
        y = np.where((y > net.hi) & (y <= net.hi - net.tol), net.hi, y)
        y = np.where((y < net.lo + net.tol) | (y > net.hi - net.tol), np.nan, y)
        v, m = np.full(len(rows), np.nan, np.float32), np.zeros(len(rows), bool)
        v[ok], m[ok] = y, np.isin(domain_codes(x, net.dom[0], net.dom[1]), grid)
        dates.append(d)
        lai.append(v)
        dom.append(m)
    res = dict(dates=np.array(dates, dtype="datetime64[D]"), rows=rows, cols=cols, lai=np.stack(lai), in_domain=np.stack(dom))
    np.savez_compressed(path + ".part.npz", **res)          # written whole, then renamed
    os.replace(path + ".part.npz", path)
    return res


def run_one(job):
    try:
        return _run_one(job)
    except Exception as e:                                  # one failed area-season does not stop the others
        return f"{job[0]} {job[1]}: FAILED {type(e).__name__}: {e}"


def _run_one(job):
    area, year = job
    import fv2
    from areas import out_dir
    done = os.path.join(OUT, f"pairs_{area}_{year}.csv")
    if os.path.exists(done):
        return f"{area} {year}: done before"
    s = sl2p_clear(area, year)
    o = np.load(os.path.join(out_dir(area, year), "s2_pixels_fv2.npz"))
    if not (np.array_equal(o["dates"], s["dates"]) and np.array_equal(o["rows"], s["rows"])):
        raise ValueError(f"{area} {year}: SL2P dates / pixels differ from our retrieval")
    flags = np.load(os.path.join("data", "processed", "field_v2", "benchmark", f"fix2_flags_lai_{area}_{year}.npz"))["flags"]
    ours_clear = np.where(flags, np.nan, o["lai"])
    wk = np.load(os.path.join(out_dir(area, year), "weekly_pixels_V2.npz"))
    if not np.array_equal(wk["rows"], s["rows"]):
        raise ValueError(f"{area} {year}: weekly pixels differ")
    day = lambda d: d.astype("datetime64[D]").astype(np.int64)
    sday, wday = day(s["dates"]), day(wk["weeks"])
    sl, sl_meas = line_weekly(sday, s["lai"], wday)
    ol, _ = line_weekly(sday, ours_clear, wday)
    op, src = wk["lai"], wk["lai_src"]
    grp = fv2.pixels(area, year)["grp"]
    month = pd.DatetimeIndex(wk["weeks"]).month.values
    series = {"sl2p": sl, "ours_line": ol, "ours": op}
    rows = []
    for w, wdate in enumerate(pd.DatetimeIndex(wk["weeks"])):
        for crop in CROPS + ["all crops"]:
            k = np.ones(len(grp), bool) if crop == "all crops" else grp == crop
            for name, a in series.items():
                v = a[w, k]
                v = v[np.isfinite(v)]
                if len(v) < 50:
                    continue
                q = np.percentile(v, [25, 50, 75])
                rows.append(dict(area=area, season=year, week=wdate.date(), crop=crop, series=name, n=len(v),
                                 q25=q[0], median=q[1], q75=q[2]))
    pd.DataFrame(rows).round(4).to_csv(os.path.join(OUT, f"weekly_{area}_{year}.csv"), index=False)
    prs = []
    for crop in CROPS + ["all crops"]:
        k = np.ones(len(grp), bool) if crop == "all crops" else grp == crop
        groups = [("all weeks", np.ones_like(src, bool))]
        groups += [(f"month {m:02d}", np.repeat((month == m)[:, None], len(grp), 1)) for m in np.unique(month)]
        groups += [(f"source {n}", src == c) for c, n in SRC_NAMES.items()]
        groups += [("SL2P measured", sl_meas), ("SL2P gap-filled", ~sl_meas)]
        for gname, g in groups:
            sel = g[:, k]
            for pair, a in (("ours_line - sl2p", ol), ("ours - sl2p", op)):
                st = pair_stats(a[:, k][sel], sl[:, k][sel])
                prs.append(dict(area=area, season=year, crop=crop, group=gname, pair=pair, **st))
    dshare = float(np.nanmean(np.where(np.isfinite(s["lai"]), s["in_domain"], np.nan)))
    inval = float(np.mean(np.isnan(s["lai"][np.isfinite(ours_clear) | np.isfinite(o["lai"])])))
    prs.append(dict(area=area, season=year, crop="all crops", group="clear: SL2P in domain share", pair="", n=0,
                    mean_diff=dshare, rmsd=np.nan, r=np.nan))
    prs.append(dict(area=area, season=year, crop="all crops", group="clear: SL2P invalid share", pair="", n=0,
                    mean_diff=inval, rmsd=np.nan, r=np.nan))
    rng = np.random.default_rng(0)
    ww, pp = np.nonzero(np.isfinite(sl) & np.isfinite(op))
    pick = rng.choice(len(ww), min(200_000, len(ww)), replace=False)
    np.savez_compressed(os.path.join(OUT, f"sample_{area}_{year}.npz"), sl2p=sl[ww[pick], pp[pick]],
                        ours_line=ol[ww[pick], pp[pick]], ours=op[ww[pick], pp[pick]], src=src[ww[pick], pp[pick]],
                        month=month[ww[pick]], crop=grp[pp[pick]].astype(str))
    pd.DataFrame(prs).round(4).to_csv(done, index=False)
    return f"{area} {year}: {len(s['dates'])} clear dates, SL2P in domain {100 * dshare:.0f} %, invalid {100 * inval:.1f} %"


def run(nproc):
    os.makedirs(OUT, exist_ok=True)
    jobs = [(a, y) for a in AREAS for y in YEARS]
    log = open(os.path.join(OUT, "run_log.txt"), "a")
    with Pool(nproc) as pool:
        for msg in pool.imap_unordered(run_one, jobs):
            line = f"{pd.Timestamp.now():%Y-%m-%d %H:%M} {msg}"
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    elif sys.argv[1] == "charts":
        from phase26_sl2p_charts import charts
        charts(OUT)
