"""Flux-tower TEST-site field chain for water v4 (pre-registration in docs/project_log.md 2026-11-06).

Outside NRW there is no field register here, so the tower field is outlined from Sentinel-2 itself:
  outline     10 m pixels within 600 m of the tower whose NDVI series (SCL 4/5 clear, 1 Mar - 30 Sep, >= 8 common dates)
              correlates r >= 0.9 with the tower 3 x 3 median series and has mean NDVI within 0.10; the 4-connected
              region holding the tower pixel.
  LAI         fv2 S2 retrieval (prosail_net.pt, as NRW) on the outline minus a 2-pixel edge; field median on dates with
              >= 50 % of field pixels clear; 3-date rolling median; daily by linear interpolation, 0 outside.
  season      start = first day LAI >= 0.5 (or 1 Mar), peak = max, end = first day after the peak with LAI < 0.5 (or 30 Sep).
  weather     HYRAS cell nearest the tower, SARAH-3 nearest cell, DWD wind station, FAO-56 Penman-Monteith ET0 (v2 code).
Test sites only: never on the platform (areas.TEST_AREAS).
Run from D:\\vista: python phase33_testsite_field.py [AREA YEAR ...]   (default: all test seasons)
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import ndimage

TOWERS = {   # lat, lon, elevation m, district (soil), DWD wind station (rule: nearest with wind on >= 95 % of season days)
    "dekli": dict(lat=50.893042, lon=13.5223, elev=478.0, district="14628", wind_station="06314"),
    "degeb": dict(lat=51.09973, lon=10.91463, elev=163.0, district="16068", wind_station="01270"),
}
CROP = {("dekli", 2018): "maize", ("dekli", 2021): "wheat", ("dekli", 2022): "barley", ("dekli", 2023): "barley",
        ("dekli", 2025): "wheat", ("degeb", 2020): "potato", ("degeb", 2021): "wheat", ("degeb", 2022): "wheat",
        ("degeb", 2024): "wheat"}
RADIUS_PX, HALF = 60, 75          # 600 m search radius; window half-size read around the tower (750 m)


# ---------------------------------------------------------------------------------------------------- pure functions
def outline(nd, tower_rc, r_min=0.9, d_max=0.10, min_dates=8, radius_px=RADIUS_PX):
    """Field mask from an NDVI stack (dates x rows x cols, NaN = not clear)."""
    r0, c0 = tower_rc
    ref = np.nanmedian(nd[:, max(r0 - 1, 0):r0 + 2, max(c0 - 1, 0):c0 + 2].reshape(len(nd), -1), axis=1)
    ok = np.isfinite(nd) & np.isfinite(ref)[:, None, None]
    n = ok.sum(0)
    x = np.where(ok, nd, 0.0)
    y = np.where(ok, ref[:, None, None], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx, my = x.sum(0) / n, y.sum(0) / n
        dx, dy = np.where(ok, x - mx, 0.0), np.where(ok, y - my, 0.0)
        r = (dx * dy).sum(0) / np.sqrt((dx ** 2).sum(0) * (dy ** 2).sum(0))
    rr, cc = np.indices(nd.shape[1:])
    near = (rr - r0) ** 2 + (cc - c0) ** 2 <= radius_px ** 2
    cand = near & (n >= min_dates) & (r >= r_min) & (np.abs(mx - my) <= d_max)
    lab, _ = ndimage.label(cand)                       # default structure = 4-connectivity
    return (lab == lab[r0, c0]) & cand if cand[r0, c0] else np.zeros_like(cand)


def interior(mask, px):
    """Pixels more than px pixels away from the nearest pixel outside the mask."""
    return ndimage.distance_transform_edt(mask) > px


def season_bounds(dates, lai):
    """(start, peak, end) from a daily LAI series (already smoothed)."""
    dates = pd.DatetimeIndex(dates)
    lai = np.asarray(lai, float)
    up = np.flatnonzero(lai >= 0.5)
    start = dates[up[0]] if len(up) else dates[0]
    ip = int(np.nanargmax(lai))
    after = np.flatnonzero(lai[ip:] < 0.5)
    end = dates[ip + after[0]] if len(after) else dates[-1]
    return start, dates[ip], end


# ---------------------------------------------------------------------------------------------------- data chain
def tower_xy(area):
    from weather_districts import utm32
    t = TOWERS[area]
    x, y = utm32(np.array([t["lat"]]), np.array([t["lon"]]))
    return float(np.ravel(x)[0]), float(np.ravel(y)[0])


def _s2_files(area, year):
    from areas import raw_dir
    return sorted(glob.glob(os.path.join(raw_dir(area, year), "s2", "*.tif")))


def ndvi_window(area, year):
    """NDVI stack around the tower (1 Mar - 30 Sep, SCL 4/5, 2-pixel buffer around cloud classes, as fv2)."""
    import rasterio
    from rasterio.windows import Window
    from fv2 import ndvi
    files = _s2_files(area, year)
    with rasterio.open(files[0]) as r:
        tr, H, W = r.transform, r.height, r.width
    x, y = tower_xy(area)
    row, col = rasterio.transform.rowcol(tr, x, y)
    r0, c0 = max(row - HALF, 0), max(col - HALF, 0)
    win = Window(c0, r0, min(2 * HALF + 1, W - c0), min(2 * HALF + 1, H - r0))
    dates, stack = [], []
    for f in files:
        d = pd.Timestamp(os.path.basename(f)[7:17])
        if not (3 <= d.month <= 9) or d.year != year:
            continue
        with rasterio.open(f) as r:
            a = r.read(window=win)
        bad = ndimage.binary_dilation(np.isin(a[10], (3, 8, 9, 10, 11)), iterations=2)
        clear = np.isin(a[10], (4, 5)) & ~bad
        nd = np.where(clear, ndvi(a[2] / 1e4, a[6] / 1e4), np.nan)
        dates.append(d)
        stack.append(nd.astype(np.float32))
    return dict(dates=pd.DatetimeIndex(dates), nd=np.stack(stack), r0=r0, c0=c0, tower_rc=(row - r0, col - c0),
                transform=tr)


def field_lai(area, year, rows, cols):
    """fv2 retrieval on the field pixels -> field median per S2 date (>= 50 % clear), smoothed, daily 1 Mar - 30 Sep."""
    import fv2
    from areas import out_dir
    os.makedirs(out_dir(area, year), exist_ok=True)
    n = len(rows)
    px = dict(rows=rows, cols=cols, inv=np.zeros(n, int), share=np.ones(n))
    fv2.retrieval_s2(area, year, px)
    z = np.load(os.path.join(out_dir(area, year), "s2_pixels_fv2.npz"))
    lai = z["lai"]
    share = np.isfinite(lai).mean(1)
    keep = share >= 0.5
    obs = pd.Series(np.nanmedian(np.where(keep[:, None], lai, np.nan)[keep], axis=1),
                    index=pd.DatetimeIndex(z["dates"])[keep]).sort_index()
    obs = obs.rolling(3, center=True, min_periods=1).median()
    days = pd.date_range(f"{year}-03-01", f"{year}-09-30")
    daily = obs.reindex(obs.index.union(days)).interpolate(method="time", limit_area="inside").reindex(days).fillna(0)
    return obs, daily


def site_weather(area, dates):
    """Daily tasmin, tasmax, hurs, pr (HYRAS nearest cell), rs MJ (SARAH-3 nearest), u10 / u2 (station), ET0."""
    import io
    import zipfile
    from phase29_water_v2 import et0_pm, u2_from_u10
    t = TOWERS[area]
    x, y = tower_xy(area)
    parts = []
    for yr in sorted({d.year for d in dates}):
        h = np.load(os.path.join("data", "processed", "field", f"hyras_testsites_{yr}.npz"))
        j = int(np.argmin((h["x"] - x) ** 2 + (h["y"] - y) ** 2))
        parts.append(pd.DataFrame({k: h[k][:, j] for k in ("tasmin", "tasmax", "hurs", "pr")},
                                  index=pd.DatetimeIndex(h["dates"])))
    w = pd.concat(parts).reindex(dates)
    s = np.load(os.path.join("data", "processed", "weather", "sarah3_germany.npz"))
    i, k = np.argmin(np.abs(s["lat"] - t["lat"])), np.argmin(np.abs(s["lon"] - t["lon"]))
    w["rs"] = pd.Series(s["sis"][:, i, k] * 0.0864, index=pd.DatetimeIndex(s["dates"])).reindex(dates).values
    wp = []
    for f in glob.glob(os.path.join("data", "raw", "dwd_stations*", f"tageswerte_KL_{t['wind_station']}_*.zip")):
        z = zipfile.ZipFile(f)
        d = pd.read_csv(io.BytesIO(z.read([n for n in z.namelist() if n.startswith("produkt")][0])), sep=";",
                        skipinitialspace=True, na_values=[-999])
        wp.append(pd.Series(d["FM"].values, index=pd.to_datetime(d["MESS_DATUM"].astype(str))))
    wind = pd.concat(wp)
    wind = wind[~wind.index.duplicated(keep="last")].sort_index().reindex(dates)
    w["u10"] = wind.fillna(wind.median()).values
    w["u2"] = u2_from_u10(w["u10"].values)
    w["et0"] = et0_pm(w["tasmin"], w["tasmax"], w["hurs"], w["rs"], w["u2"], t["elev"], t["lat"], dates.dayofyear)
    return w


def soil(area):
    """PAW per 10 cm layer (mm) and TEW (mm) from the district soil table (as v2 _soil, ratio 1.0)."""
    s = pd.read_csv(os.path.join("data", "processed", "soil_districts.csv"), dtype={"district": str}).set_index(
        "district").loc[TOWERS[area]["district"]]
    paw = np.array([(s[f"fc_{k}"] - s[f"wp_{k}"]) * 100 for k in range(1, 21)])
    tew = 1000 * (s["fc_1"] - 0.5 * s["wp_1"]) * 0.10
    return paw, tew


def build(area, year):
    """Outline + LAI + season for one test season -> data/processed/field_v2/<area>_<year>/testsite_field.json/.npz + map."""
    from areas import out_dir
    o = out_dir(area, year)
    os.makedirs(o, exist_ok=True)
    W = ndvi_window(area, year)
    mask = outline(W["nd"], W["tower_rc"])
    rr, cc = np.nonzero(interior(mask, 2))
    rows, cols = rr + W["r0"], cc + W["c0"]
    if len(rows) == 0:                                  # pre-registered: too small a field -> season dropped
        info = dict(area=area, year=year, crop=CROP[(area, year)], field_px=int(mask.sum()), lai_px=0, dropped=True,
                    reason="outline has no pixels after the 2-pixel edge")
        json.dump(info, open(os.path.join(o, "testsite_field.json"), "w"), indent=1)
        _map(W, mask, dict(info, field_ha=mask.sum() / 100), os.path.join(o, "testsite_outline.png"))
        print(info, flush=True)
        return info
    obs, daily = field_lai(area, year, rows, cols)
    start, peak, end = season_bounds(daily.index, daily.values)
    tr = W["transform"]
    info = dict(area=area, year=year, crop=CROP[(area, year)], field_px=int(mask.sum()), field_ha=float(mask.sum() / 100),
                lai_px=int(len(rows)), s2_dates=int(len(W["dates"])), lai_dates=int(len(obs)),
                start=str(start.date()), peak=str(peak.date()), end=str(end.date()), peak_lai=float(daily.max()))
    np.savez_compressed(os.path.join(o, "testsite_field.npz"), mask=mask, r0=W["r0"], c0=W["c0"],
                        transform=np.array(tr)[:6], rows=rows, cols=cols)
    daily.rename("lai").to_csv(os.path.join(o, "testsite_lai_daily.csv"), index_label="date")
    json.dump(info, open(os.path.join(o, "testsite_field.json"), "w"), indent=1)
    _map(W, mask, info, os.path.join(o, "testsite_outline.png"))
    print(info, flush=True)
    return info


def _map(W, mask, info, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    peak = np.nanmax(W["nd"], axis=0)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(peak, cmap="Greens", vmin=0, vmax=1)
    ax.contour(mask, levels=[0.5], colors="red", linewidths=1)
    ax.plot(W["tower_rc"][1], W["tower_rc"][0], "k^", ms=7)
    ax.set_title(f"{info['area']} {info['year']} {info['crop']}: {info['field_ha']:.1f} ha", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    from areas import TEST_SEASONS
    a = sys.argv[1:]
    pairs = [(a[i], int(a[i + 1])) for i in range(0, len(a), 2)] or [(s, y) for s, ys in TEST_SEASONS.items() for y in ys]
    for s, y in pairs:
        build(s, y)
