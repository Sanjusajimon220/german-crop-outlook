"""Weekly canopy chlorophyll (CCC), leaf chlorophyll (Cab) and brown pigment (Cbrown) per field (project log 2026-11-04).

The PROSAIL network (phase4_prosail, prosail_net.pt, unchanged) gives four outputs per clear pixel; the field pipeline kept
only green LAI. Here the same raw scenes, pixels, cloud mask, fix 1 and Dec/Jan rule as fv2.retrieval_s2 are read again and
the other three outputs are kept (the re-computed LAI must equal s2_pixels_fv2 or the run stops). Fix 2 flags (bad clear
images, found on LAI) are applied to all three.
Weekly rule (no radar model for these traits): measured = clear value within 3 days (uncertainty = network sd), otherwise a
straight line between clear values whatever the gap, after the last clear image the last value held. The typical error of
the gap values is measured by hiding 30 % of the clear dates (uncertainty_traits.json).

Run from D:\\vista: `python phase27_traits.py retrieve [processes]` (resumable), `... errors`, `... export [processes]`.
"""
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

TRAITS = ["cab", "ccc", "cbrown"]
NET_ORDER = ["lai", "cab", "ccc", "cbrown"]            # phase4_prosail.TARGETS
SRC_MEASURED, SRC_LINE, SRC_AFTER = 1, 2, 4            # same codes as fv2_weekly (the app labels them)
MEASURED_DAYS, HIDE_SHARE, HIDE_SEED = 3, 0.3, 70
ROUND = {"cab": 1, "ccc": 0, "cbrown": 2}
AREAS = ["pilot", "selhausen", "north", "south"]
YEARS = [2021, 2022, 2023, 2024, 2025]
FV2 = os.path.join("data", "processed", "field_v2")
UNC = os.path.join(FV2, "uncertainty_traits.json")
TRAIT_FILE = "s2_pixels_traits.npz"


def trait_weekly(days, vals, weeks):
    """days (D,) ascending clear days, vals (D, P) NaN = not clear, weeks (W,). Returns value, source and gap (W, P):
    gap = days between the two clear values (line) or days since the last clear value (after), 0 when measured."""
    D, P = vals.shape
    fin = np.isfinite(vals)
    ar = np.arange(D)[:, None]
    ff = np.maximum.accumulate(np.where(fin, ar, -1), axis=0)
    bf = np.minimum.accumulate(np.where(fin, ar, D)[::-1], axis=0)[::-1]
    out = np.full((len(weeks), P), np.nan, np.float32)
    src = np.zeros((len(weeks), P), np.int8)
    gap = np.zeros((len(weeks), P), np.int16)
    cols = np.arange(P)
    for w, wd in enumerate(weeks):
        for j in np.argsort(np.abs(days - wd), kind="stable"):
            if abs(days[j] - wd) > MEASURED_DAYS:
                break
            take = (src[w] == 0) & fin[j]
            out[w, take], src[w, take] = vals[j, take], SRC_MEASURED
        kl, kr = np.searchsorted(days, wd, "left"), np.searchsorted(days, wd, "right")
        prv = ff[kl - 1] if kl > 0 else np.full(P, -1)
        nxt = bf[kr] if kr < D else np.full(P, D)
        rest = (src[w] == 0) & (prv >= 0)
        both, last = rest & (nxt < D), rest & (nxt >= D)
        a, b = prv[both], nxt[both]
        va, vb = vals[a, cols[both]], vals[b, cols[both]]
        out[w, both] = va + (wd - days[a]) / (days[b] - days[a]) * (vb - va)
        src[w, both], gap[w, both] = SRC_LINE, days[b] - days[a]
        out[w, last] = vals[prv[last], cols[last]]
        src[w, last], gap[w, last] = SRC_AFTER, wd - days[prv[last]]
    return out, src, gap


def week_uncertainty(src, gap, sd, unc):
    """Typical error of each weekly value: the network sd when measured, the gap-class error otherwise."""
    u = np.full(src.shape, np.nan, np.float32)
    m = src == SRC_MEASURED
    u[m] = sd[m]
    ln, af = src == SRC_LINE, src == SRC_AFTER
    u[ln & (gap <= 15)] = unc["line_le15"]
    u[ln & (gap > 15) & (gap <= 30)] = unc["line_16_30"]
    u[ln & (gap > 30)] = unc["line_gt30"]
    u[af & (gap <= 30)] = unc["after_le30"]
    u[af & (gap > 30)] = unc["after_gt30"]
    return u


def gap_errors(days, vals, hide):
    """Hidden-date test for one trait: per hidden date, the line between the kept neighbours and the held previous kept
    value against the hidden clear value. Returns rows {kind, cls, n, sse} for every class (n may be 0)."""
    kept = np.isfinite(vals) & ~hide[:, None]
    D = len(days)
    ar = np.arange(D)[:, None]
    ff = np.maximum.accumulate(np.where(kept, ar, -1), axis=0)
    bf = np.minimum.accumulate(np.where(kept, ar, D)[::-1], axis=0)[::-1]
    acc = {(k, c): [0, 0.0] for k, cs in (("line", ("le15", "16_30", "gt30")), ("after", ("le30", "gt30"))) for c in cs}
    for j in np.flatnonzero(hide):
        truth = vals[j]
        prv = ff[j - 1] if j > 0 else np.full(vals.shape[1], -1)
        nxt = bf[j + 1] if j + 1 < D else np.full(vals.shape[1], D)
        ok = np.isfinite(truth) & (prv >= 0)
        cols = np.flatnonzero(ok)
        a = prv[cols]
        held = vals[a, cols]
        since = days[j] - days[a]
        for c, sel in (("le30", since <= 30), ("gt30", since > 30)):
            acc[("after", c)][0] += int(sel.sum())
            acc[("after", c)][1] += float(((held[sel] - truth[cols][sel]) ** 2).sum())
        bb = nxt[cols]
        two = bb < D
        c2, a2, b2 = cols[two], a[two], bb[two]
        g = days[b2] - days[a2]
        line = vals[a2, c2] + (days[j] - days[a2]) / g * (vals[b2, c2] - vals[a2, c2])
        e2 = (line - truth[c2]) ** 2
        for c, sel in (("le15", g <= 15), ("16_30", (g > 15) & (g <= 30)), ("gt30", g > 30)):
            acc[("line", c)][0] += int(sel.sum())
            acc[("line", c)][1] += float(e2[sel].sum())
    return [dict(kind=k, cls=c, n=n, sse=s) for (k, c), (n, s) in acc.items()]


def hidden_dates(n_dates, seed=HIDE_SEED, share=HIDE_SHARE):
    """30 % of the clear dates hidden, never the first or the last."""
    rng = np.random.default_rng(seed)
    mid = np.arange(1, n_dates - 1)
    hide = np.zeros(n_dates, bool)
    hide[rng.choice(mid, int(round(share * n_dates)), replace=False)] = True
    return hide


# ---- full area-season runs (read the raw scenes; need the fv2 setup outputs) ----

def retrieve_one(job):
    area, year = job
    try:
        return _retrieve(area, year)
    except Exception as e:                                  # one failed area-season does not stop the others
        return f"{area} {year}: FAILED {type(e).__name__}: {e}"


def _retrieve(area, year):
    import rasterio
    from scipy.ndimage import binary_dilation
    import fv2
    from areas import out_dir, raw_dir
    path = os.path.join(out_dir(area, year), TRAIT_FILE)
    if os.path.exists(path):
        return f"{area} {year}: done before"
    px = fv2.pixels(area, year)
    rows, cols = px["rows"], px["cols"]
    pinv, pshare = px["inv"], np.nan_to_num(px["share"])
    ftot = np.bincount(pinv, weights=pshare)
    net, ck = fv2._net(fv2.S2_MODEL)
    ref = np.load(os.path.join(out_dir(area, year), "s2_pixels_fv2.npz"))
    if not np.array_equal(ref["rows"], rows):
        raise ValueError("pixels differ from s2_pixels_fv2")
    dates, out = [], {k: [] for k in [t for t in NET_ORDER] + [f"{t}_sd" for t in TRAITS]}
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
        x = np.concatenate([refl, np.cos(np.radians(np.stack([sza, vza, raa], 1)))], 1).astype(np.float32)
        ok = c & np.all(np.isfinite(x), axis=1)
        pred, psd = fv2._predict(net, ck, x[ok])
        dates.append(d)
        for k, t in enumerate(NET_ORDER):
            v = np.full(len(rows), np.nan, np.float32)
            v[ok] = np.clip(pred[:, k], 0, None)
            out[t].append(v)
            if t in TRAITS:
                s = np.full(len(rows), np.nan, np.float32)
                s[ok] = psd[:, k]
                out[f"{t}_sd"].append(s)
    dates = np.array(dates, dtype="datetime64[D]")
    lai = np.stack(out.pop("lai"))
    if not np.array_equal(dates, ref["dates"]):
        raise ValueError("clear dates differ from s2_pixels_fv2")
    same_nan = np.array_equal(np.isnan(lai), np.isnan(ref["lai"]))
    diff = float(np.nanmax(np.abs(lai - ref["lai"])))
    if not same_nan or diff >= 1e-4:
        raise ValueError(f"re-computed LAI differs from s2_pixels_fv2 (same clear pixels {same_nan}, max diff {diff:.2e})")
    res = {k: np.stack(v) for k, v in out.items()}
    np.savez_compressed(path + ".part.npz", dates=dates, rows=rows, cols=cols, **res)
    os.replace(path + ".part.npz", path)
    return f"{area} {year}: {len(dates)} clear dates, LAI check max diff {diff:.1e}"


def _flagged(area, year, x):
    flags = np.load(os.path.join(FV2, "benchmark", f"fix2_flags_lai_{area}_{year}.npz"))["flags"]
    return np.where(flags, np.nan, x)


def errors():
    """Hidden-date error of the gap values, pooled over all area-seasons -> uncertainty_traits.json."""
    from areas import out_dir
    rows = []
    for a in AREAS:
        for y in YEARS:
            z = np.load(os.path.join(out_dir(a, y), TRAIT_FILE))
            days = z["dates"].astype(np.int64)
            hide = hidden_dates(len(days))
            for t in TRAITS:
                for r in gap_errors(days, _flagged(a, y, z[t]), hide):
                    rows.append(dict(area=a, season=y, trait=t, **r))
            print(f"{a} {y} errors done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(FV2, "traits_gap_errors.csv"), index=False)
    g = df.groupby(["trait", "kind", "cls"])[["n", "sse"]].sum()
    unc = {t: {f"{k}_{c}": round(float(np.sqrt(g.loc[(t, k, c), "sse"] / max(g.loc[(t, k, c), "n"], 1))), 3)
               for k, cs in (("line", ("le15", "16_30", "gt30")), ("after", ("le30", "gt30"))) for c in cs} for t in TRAITS}
    unc["n"] = {t: {f"{k}_{c}": int(g.loc[(t, k, c), "n"]) for (tt, k, c) in g.index if tt == t} for t in TRAITS}
    json.dump(unc, open(UNC, "w"), indent=1)
    print(json.dumps(unc, indent=1))


def export_one(job):
    area, year = job
    try:
        return _export(area, year)
    except Exception as e:
        return f"{area} {year}: FAILED {type(e).__name__}: {e}"


def _export(area, year):
    import fv2
    from areas import out_dir
    from fv2_platform import DOCS, field_mean
    unc = json.load(open(UNC))
    z = np.load(os.path.join(out_dir(area, year), TRAIT_FILE))
    wk = np.load(os.path.join(out_dir(area, year), "weekly_pixels_V2.npz"))
    if not np.array_equal(wk["rows"], z["rows"]):
        raise ValueError("weekly pixels differ")
    px = fv2.pixels(area, year)
    share, inv, fids = np.nan_to_num(px["share"]), px["inv"], px["fids"]
    F = len(fids)
    days, weeks = z["dates"].astype(np.int64), wk["weeks"].astype("datetime64[D]").astype(np.int64)
    fields = {str(int(f)): {} for f in fids}
    for t in TRAITS:
        v, src, gap = trait_weekly(days, _flagged(area, year, z[t]), weeks)
        u = week_uncertainty(src, gap, _measured_sd(days, z[f"{t}_sd"], weeks, src), unc[t])
        fv, fu = field_mean(v, share, inv, F), field_mean(u, share, inv, F)
        if t == "ccc":                                       # field source: majority by share
            sw = np.stack([np.stack([np.bincount(inv, weights=share * (src[w] == k), minlength=F) for k in range(5)], 1)
                           for w in range(len(weeks))])
            fsrc = sw.argmax(2)
        for k, f in enumerate(fids):
            e = fields[str(int(f))]
            e[t] = [None if not np.isfinite(x) else round(float(x), ROUND[t]) for x in fv[:, k]]
            e[f"{t}_unc"] = [None if not np.isfinite(x) else round(float(x), ROUND[t] + 1) for x in fu[:, k]]
    for k, f in enumerate(fids):
        fields[str(int(f))]["src"] = fsrc[:, k].tolist()
    out = os.path.join(DOCS, f"{area}_{year}", "traits.json")
    json.dump({"weeks": [str(w) for w in wk["weeks"]], "fields": fields}, open(out + ".part", "w"), separators=(",", ":"))
    os.replace(out + ".part", out)
    return f"{area} {year}: {F} fields exported"


def _measured_sd(days, sd, weeks, src):
    """Network sd of the clear value used on measured weeks (nearest clear date within 3 days, as trait_weekly)."""
    out = np.full(src.shape, np.nan, np.float32)
    fin = np.isfinite(sd)
    for w, wd in enumerate(weeks):
        done = np.zeros(sd.shape[1], bool)
        for j in np.argsort(np.abs(days - wd), kind="stable"):
            if abs(days[j] - wd) > MEASURED_DAYS:
                break
            take = ~done & fin[j]
            out[w, take], done[take] = sd[j, take], True
    return out


def _pool(fn, nproc):
    jobs = [(a, y) for a in AREAS for y in YEARS]
    log = open(os.path.join(FV2, "traits_log.txt"), "a")
    with Pool(nproc) as pool:
        for msg in pool.imap_unordered(fn, jobs):
            line = f"{pd.Timestamp.now():%Y-%m-%d %H:%M} {fn.__name__}: {msg}"
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()


if __name__ == "__main__":
    cmd, nproc = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2
    if cmd == "retrieve":
        _pool(retrieve_one, nproc)
    elif cmd == "errors":
        errors()
    elif cmd == "export":
        _pool(export_one, nproc)
        from fv2_platform import write_index
        write_index()                                       # new data version: the app reloads the field files
