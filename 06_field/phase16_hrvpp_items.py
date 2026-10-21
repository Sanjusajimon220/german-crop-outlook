"""HR-VPP benchmark items 1-3 (pre-registered 2026-10-17; details fixed 2026-10-21 before any comparison; one look).

1 clear-day consistency: Spearman of our field LAI vs HR-VPP PPI on our measured weeks, per crop and season (no winner).
2 cloudy weeks: per Landsat date (>= 30 fields without a measured week), Spearman(ours, Landsat) vs Spearman(PPI, Landsat);
  ours better if it wins on >= 60 % of all qualifying dates 2021-2024 (ties not won). Primary PPI = QFLAG >= 1, secondary >= 3.
3 season timing: our start / peak / end (half-amplitude rule) vs HR-VPP SOSD / MAXD / EOSD season 1, median |days| (no winner).
Run from D:\\vista: python phase16_hrvpp_items.py -> data/processed/field_v2/benchmark/hrvpp_*.csv and the printed summary.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HRVPP = r"D:\vista\data\raw\hrvpp_portal\hrvpp"
OUT = r"D:\vista\data\processed\field_v2\benchmark"
DOCS = r"D:\german-crop-outlook-field\docs\field"
CROPS = {"wheat", "barley", "rapeseed", "maize", "potato", "sugar beet"}
SEASONS = range(2021, 2025)                                    # HR-VPP ends with 2024 (amendment 2026-10-21)
PPI_DAYS, MIN_FIELDS, WIN_SHARE, COVER = 5, 30, 0.60, 0.5


def decode_ppi(dn):
    v = np.asarray(dn).astype(float)
    return np.where(v == -32768, np.nan, v * 0.0001)


def decode_yydoy(v):
    v = np.asarray(v).astype(int)
    out = np.full(v.shape, np.datetime64("NaT"), dtype="datetime64[D]")
    ok = v > 0
    yy, doy = 2000 + v[ok] // 1000, v[ok] % 1000
    out[ok] = (yy - 1970).astype("datetime64[Y]").astype("datetime64[D]") + (doy - 1)
    return out


def grid_index(x, y, t, shape, src_epsg, dst_epsg):
    """Row / column (nearest cell) in a raster (affine t, CRS dst_epsg) of points given in src_epsg; -1 outside."""
    if src_epsg != dst_epsg:
        from pyproj import Transformer
        x, y = Transformer.from_crs(src_epsg, dst_epsg, always_xy=True).transform(x, y)
    c = np.floor((np.asarray(x) - t.c) / t.a).astype(int)
    r = np.floor((np.asarray(y) - t.f) / t.e).astype(int)
    bad = (r < 0) | (c < 0) | (r >= shape[0]) | (c >= shape[1])
    return np.where(bad, -1, r), np.where(bad, -1, c)


def nearest_date(dates, day, tol):
    if not len(dates):
        return -1
    d = np.abs((dates - day).astype(int))
    k = int(np.argmin(d))
    return k if d[k] <= tol else -1


def season_timing(lai, mask):
    """Week indices of start (first above min + 50 % amplitude), peak (max) and end (last above) inside the mask."""
    idx = np.flatnonzero(mask & np.isfinite(lai))
    if len(idx) < 3:
        return None
    v = lai[idx]
    lo, hi = v.min(), v.max()
    if hi - lo <= 0:
        return None
    above = idx[v > lo + 0.5 * (hi - lo)]
    return int(above[0]), int(idx[np.argmax(v)]), int(above[-1])


def _spearman(a, b):
    return pd.Series(a).rank().corr(pd.Series(b).rank())


def item2_score(rows, min_fields=MIN_FIELDS):
    """rows: (date, landsat, ours, ppi) arrays over fields. Counts dates with >= min_fields and strict wins of ours."""
    out = []
    for d, ls, ours, ppi in rows:
        if len(ls) < min_fields:
            continue
        out.append(dict(date=d, n=len(ls), rho_ours=_spearman(ours, ls), rho_ppi=_spearman(ppi, ls)))
    df = pd.DataFrame(out, columns=["date", "n", "rho_ours", "rho_ppi"])
    wins = int((df.rho_ours > df.rho_ppi).sum())
    return {"dates": len(df), "ours_wins": wins, "table": df}


# ---- real-data run ----
def _field_table(area, year):
    """Our weekly field means (LAI), measured share, field ids / groups; kept fields: crop, >= 0.5 ha, not partial."""
    from areas import out_dir
    from fv2_platform import field_mean, season_mask
    d = out_dir(area, year)
    z = dict(np.load(os.path.join(d, "weekly_pixels_V2.npz")))
    px = pd.read_csv(os.path.join(d, "pixels_nrw.csv.gz"))
    mainf = px.sort_values("share").drop_duplicates(["row", "col"], keep="last").set_index(["row", "col"])
    key = pd.MultiIndex.from_arrays([z["rows"], z["cols"]])
    fid = mainf["field_id"].reindex(key).values
    share = np.nan_to_num(mainf["share"].reindex(key).values)
    geo = json.load(open(os.path.join(DOCS, f"{area}_{year}", "fields.geojson"), encoding="utf-8"))
    props = {int(f["properties"]["id"]): f["properties"] for f in geo["features"]}
    fl = pd.read_csv(os.path.join(d, "fields_nrw.csv")).set_index("field_id")
    keep_ids = [f for f, p in props.items() if f in fl.index and fl.loc[f, "crop_group"] in CROPS
                and fl.loc[f, "area_ha"] >= 0.5 and not p.get("partial", False)]
    ok = np.isin(np.nan_to_num(fid, nan=-1).astype(np.int64), keep_ids)
    fids, inv = np.unique(fid[ok].astype(np.int64), return_inverse=True)
    lai = field_mean(z["lai"][:, ok], share[ok], inv, len(fids))
    meas = field_mean((z["lai_src"][:, ok] == 1).astype(float), np.ones(int(ok.sum())), inv, len(fids))
    groups = fl.loc[fids, "crop_group"].values
    mask = np.array([season_mask(lai[:, k], groups[k]) for k in range(len(fids))], bool).T
    grid = json.load(open(os.path.join(d, "grid_info.json")))
    x = grid["origin_x"] + (z["cols"][ok] + 0.5) * grid["res"]
    y = grid["origin_y"] - (z["rows"][ok] + 0.5) * grid["res"]
    return dict(weeks=z["weeks"].astype("datetime64[D]"), fids=fids, inv=inv, groups=groups, lai=lai, meas=meas, mask=mask,
                x=x, y=y, ok=ok, out_dir=d)


def _field_mean_px(vals, inv, n, need=COVER):
    ok = np.isfinite(vals)
    s = np.bincount(inv[ok], vals[ok], n)
    c = np.bincount(inv[ok], minlength=n)
    tot = np.bincount(inv, minlength=n)
    return np.where(c >= need * np.maximum(tot, 1), s / np.maximum(c, 1), np.nan)


def _field_median_px(vals, inv, n, need=COVER):
    df = pd.DataFrame({"f": inv, "v": vals}).dropna()
    med, cnt, tot = df.groupby("f")["v"].median(), df.groupby("f").size(), np.bincount(inv, minlength=n)
    out = np.full(n, np.nan)
    good = cnt.index[cnt.values >= need * tot[cnt.index]]
    out[good] = med.loc[good].values
    return out


def _st_dates(area):
    st = {}
    for f in glob.glob(os.path.join(HRVPP, area, "ST_*_PPI.tif")):
        b = os.path.basename(f)
        st[np.datetime64(f"{b[3:7]}-{b[7:9]}-{b[9:11]}")] = f
    return st


def run():
    import rasterio
    os.makedirs(OUT, exist_ok=True)
    r1, r2, r2b, r3 = [], [], [], []
    for area in ["pilot", "selhausen", "north", "south"]:
        st = _st_dates(area)
        sdates = np.array(sorted(st), dtype="datetime64[D]")
        with rasterio.open(st[sdates[0]]) as s0:
            T, shape, epsg = s0.transform, s0.shape, s0.crs.to_epsg()
        for year in SEASONS:
            F = _field_table(area, year)
            n = len(F["fids"])
            r, c = grid_index(F["x"], F["y"], T, shape, 25832, epsg)
            inside = r >= 0
            cache = {}

            def ppi_at(day, min_q):
                k = nearest_date(sdates, day, PPI_DAYS)
                if k < 0:
                    return np.full(n, np.nan)
                if (k, min_q) not in cache:
                    fp = st[sdates[k]]
                    with rasterio.open(fp) as a, rasterio.open(fp.replace("_PPI.tif", "_QFLAG.tif")) as q:
                        assert a.transform == T and q.transform == T, "HR-VPP cuts not on one grid"
                        p, qf = decode_ppi(a.read(1)), q.read(1)
                    v = np.full(len(r), np.nan)
                    v[inside] = np.where(qf[r[inside], c[inside]] >= min_q, p[r[inside], c[inside]], np.nan)
                    cache[(k, min_q)] = _field_mean_px(v, F["inv"], n)
                return cache[(k, min_q)]

            # item 1: our measured weeks
            for wi, wd in enumerate(F["weeks"]):
                m = (F["meas"][wi] >= 0.5) & F["mask"][wi] & np.isfinite(F["lai"][wi])
                if not m.any():
                    continue
                pp = ppi_at(wd, 1)
                for k in np.flatnonzero(m & np.isfinite(pp)):
                    r1.append(dict(area=area, season=year, crop=F["groups"][k], field=F["fids"][k], week=str(wd),
                                   ours=F["lai"][wi, k], ppi=pp[k]))
            # item 2: Landsat as truth on cloudy weeks
            l = np.load(os.path.join(F["out_dir"], "landsat_pixels.npz"))
            for j, ld in enumerate(l["dates"].astype("datetime64[D]")):
                wi = nearest_date(F["weeks"], ld, 3)
                if wi < 0:
                    continue
                lsf = _field_mean_px(l["lai"][j][F["ok"]].astype(float), F["inv"], n)
                sel = np.isfinite(lsf) & (F["meas"][wi] < 0.5) & F["mask"][wi] & np.isfinite(F["lai"][wi])
                for q, store in ((1, r2), (3, r2b)):
                    pp = ppi_at(ld, q)
                    s2 = sel & np.isfinite(pp)
                    store.append((f"{area}|{year}|{ld}", lsf[s2], F["lai"][wi, s2], pp[s2]))
            # item 3: season timing vs VPP season 1
            vpp = {}
            for p in ("SOSD", "MAXD", "EOSD", "QFLAG"):
                fs = glob.glob(os.path.join(HRVPP, area, f"VPP_{year}_*_s1_{p}.tif"))
                if fs:
                    with rasterio.open(fs[0]) as a:
                        assert a.transform == T, "VPP cut not on the ST grid"
                        arr = a.read(1)
                    v = np.full(len(r), -1, np.int64)
                    v[inside] = arr[r[inside], c[inside]]
                    vpp[p] = v
            if len(vpp) == 4:
                good = vpp["QFLAG"] >= 2
                med = {}
                for p in ("SOSD", "MAXD", "EOSD"):
                    days = decode_yydoy(np.maximum(vpp[p], 0)).astype("datetime64[D]").astype("int64").astype(float)
                    med[p] = _field_median_px(np.where(good & (vpp[p] > 0), days, np.nan), F["inv"], n)
                for k in range(n):
                    t = season_timing(F["lai"][:, k], F["mask"][:, k])
                    if t is None:
                        continue
                    for name, i, h in zip(("start", "peak", "end"), t, (med["SOSD"][k], med["MAXD"][k], med["EOSD"][k])):
                        if np.isfinite(h):
                            o = float(F["weeks"][i].astype("int64"))
                            r3.append(dict(area=area, season=year, crop=F["groups"][k], field=F["fids"][k], event=name,
                                           diff_days=o - h))
            print(area, year, "fields", n, flush=True)
    t1 = pd.DataFrame(r1)
    t1.to_csv(os.path.join(OUT, "hrvpp_item1_pairs.csv.gz"), index=False)
    s1 = pd.DataFrame([dict(crop=k[0], season=k[1], n=len(g), spearman=_spearman(g.ours, g.ppi)) for k, g in t1.groupby(["crop", "season"])])
    s1.to_csv(os.path.join(OUT, "hrvpp_item1_summary.csv"), index=False)
    print("ITEM 1 (no winner): Spearman ours vs PPI on measured weeks\n", s1.pivot(index="crop", columns="season", values="spearman").round(3).to_string())
    for tag, rows in (("primary", r2), ("secondary", r2b)):
        res = item2_score(rows)
        tb = res["table"]
        tb["season"] = tb.date.str.split("|").str[1]
        tb.to_csv(os.path.join(OUT, f"hrvpp_item2_{tag}.csv"), index=False)
        share = res["ours_wins"] / max(res["dates"], 1)
        per = tb.assign(win=tb.rho_ours > tb.rho_ppi).groupby("season").agg(dates=("win", "size"), ours_wins=("win", "sum"))
        print(f"ITEM 2 ({tag}, PPI QFLAG >= {1 if tag == 'primary' else 3}): qualifying dates {res['dates']}, ours better on "
              f"{res['ours_wins']} ({share:.0%}); median rho ours {tb.rho_ours.median():.3f} vs PPI {tb.rho_ppi.median():.3f}\n{per.to_string()}")
        if tag == "primary":
            print(f"RESULT item 2: {'OURS BETTER' if share >= WIN_SHARE else 'ours NOT better'} (rule: >= {WIN_SHARE:.0%} of dates)")
    t3 = pd.DataFrame(r3)
    t3.to_csv(os.path.join(OUT, "hrvpp_item3_pairs.csv.gz"), index=False)
    s3 = t3.assign(ad=t3.diff_days.abs()).groupby(["event", "crop", "season"]).agg(
        n=("ad", "size"), median_abs_days=("ad", "median"), median_diff=("diff_days", "median")).reset_index()
    s3.to_csv(os.path.join(OUT, "hrvpp_item3_summary.csv"), index=False)
    print("ITEM 3 (no winner): median |ours - HR-VPP| days\n",
          s3.pivot_table(index=["event", "crop"], columns="season", values="median_abs_days").round(0).to_string())
    print("HRVPP ITEMS DONE")


if __name__ == "__main__":
    run()
