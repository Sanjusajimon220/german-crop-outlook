"""FAO-56 bias study at the three ICOS towers (DE-RuS, DE-Kli, DE-Geb), pre-registered docs/project_log.md 2026-11-06.

Why do the v2 / v3 / v4 water models run HIGH against tower ET? Splits the gap into
  weather inputs  ET0 from our gridded inputs vs ET0 from the tower's own weather (same FAO-56 equations),
                  with a one-input-at-a-time swap (radiation, temperature, humidity, wind)
  crop coefficient Kc_act = ET / ET0 on each side, by growth stage and crop
  tower closure   LE_CORR and the energy balance ratio (H + LE) / (Rn - G)
A STUDY on seasons already used (not a test); any correction it suggests needs a held-out test.
Run from D:\\vista: python phase35_water_bias.py -> data/processed/field_v2/water_bias/
"""
import glob
import os

import numpy as np
import pandas as pd

OUT = os.path.join("data", "processed", "field_v2", "water_bias")
SEASONS = [("selhausen", 2021), ("selhausen", 2022), ("selhausen", 2023), ("selhausen", 2024), ("selhausen", 2025),
           ("dekli", 2018), ("dekli", 2021), ("dekli", 2023), ("dekli", 2025),
           ("degeb", 2021), ("degeb", 2022), ("degeb", 2024)]
SONIC_M = {"selhausen": 2.67, "dekli": 3.5, "degeb": 3.05}          # ICOS INST files (sonic anemometer height)
TOWER_DIR = {"selhausen": "fx_inner", "dekli": "fx_Kli_inner", "degeb": "fx_Geb_inner"}
_es = lambda x: 0.6108 * np.exp(17.27 * x / (x + 237.3))


# ------------------------------------------------------------------------------------------------- pure functions
def et0_ea(tmin, tmax, ea, rs, u2, p, elev, lat, doy):
    """FAO-56 Penman-Monteith (as phase29 et0_pm) with actual vapour pressure ea (kPa) and pressure p (kPa) given."""
    tmin, tmax, ea, rs, u2, p, doy = (np.asarray(x, float) for x in (tmin, tmax, ea, rs, u2, p, doy))
    t = (tmin + tmax) / 2
    gamma = 0.000665 * p
    es = (_es(tmax) + _es(tmin)) / 2
    delta = 4098 * _es(t) / (t + 237.3) ** 2
    phi = np.radians(lat)
    dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * doy)
    dec = 0.409 * np.sin(2 * np.pi / 365 * doy - 1.39)
    ws = np.arccos(-np.tan(phi) * np.tan(dec))
    ra = 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))
    rso = (0.75 + 2e-5 * elev) * ra
    rnl = 4.903e-9 * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2 * (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0))) \
        * (1.35 * np.clip(rs / rso, 0, 1) - 0.35)
    rn = 0.77 * rs - rnl
    return np.maximum((0.408 * delta * rn + gamma * 900 / (t + 273) * u2 * (es - ea)) /
                      (delta + gamma * (1 + 0.34 * u2)), 0.0)


def u2_from_height(u, z):
    """Wind at 2 m from a measurement at z m (FAO-56 eq. 47)."""
    return np.asarray(u, float) * 4.87 / np.log(67.8 * z - 5.42)


def tower_daily(hh):
    """Daily tower weather from half-hours: tmax / tmin (C), ea (kPa, mean of es(TA) - VPD), rs (MJ), ws (m/s), pa (kPa)."""
    t = pd.to_datetime(hh["TIMESTAMP_END"].astype(str), format="%Y%m%d%H%M")
    day = (t - pd.Timedelta(minutes=1)).dt.normalize()
    ea = _es(hh["TA_F"]) - hh["VPD_F"] / 10.0                           # VPD in hPa
    g = pd.DataFrame({"day": day, "ta": hh["TA_F"], "ea": ea, "sw": hh["SW_IN_F"], "ws": hh["WS_F"], "pa": hh["PA_F"]}) \
        .groupby("day")
    out = pd.DataFrame({"tmax": g["ta"].max(), "tmin": g["ta"].min(), "ea": g["ea"].mean(), "rs": g["sw"].mean() * 0.0864,
                        "ws": g["ws"].mean(), "pa": g["pa"].mean(), "n": g["ta"].count()})
    out.index.name = None
    return out


def stages(dates, lai, peak):
    """Growth stage per day: after_peak, else by LAI (< 1, 1-3, >= 3)."""
    dates, lai = pd.DatetimeIndex(dates), np.asarray(lai, float)
    return np.where(dates > peak, "after_peak", np.where(lai < 1, "lai<1", np.where(lai < 3, "lai1-3", "lai>=3")))


def attribution(et_m, et_t, e0_m, e0_t):
    """Mean weekly log ratio of model to tower ET, split into the ET0 part and the crop-coefficient part."""
    et0 = np.log(np.asarray(e0_m) / np.asarray(e0_t))
    kc = np.log((np.asarray(et_m) / np.asarray(e0_m)) / (np.asarray(et_t) / np.asarray(e0_t)))
    return dict(total=float(np.mean(et0 + kc)), et0=float(np.mean(et0)), kc=float(np.mean(kc)), weeks=len(et0))


# ------------------------------------------------------------------------------------------------- data
ONE = {k: np.ones(1) for k in ("taw_f", "root_f", "kcb_f", "tew_f")} | {"dr0": np.zeros(1)}


def model_season(site, year):
    """v2 single run (all factors 1, buckets full) + its weather inputs, 1 March - 30 September."""
    if site == "selhausen":
        from phase29_water_v2 import SITE, site_weather
        from phase30_water_v3 import season_ensemble
        E = season_ensemble(year, ONE)
        S = E["S"]
        crop, start, peak, end = S["crop"], S["start"], S["peak"], S["end"]
        w = site_weather(E["dates"])
        lat, elev = SITE["lat"], SITE["elev"]
    else:
        from phase33_testsite_field import TOWERS, site_weather
        from phase34_water_v4 import season_ensemble
        E = season_ensemble(site, year, ONE)
        info = E["info"]
        crop, start, peak, end = info["crop"], *(pd.Timestamp(info[k]) for k in ("start", "peak", "end"))
        w = site_weather(site, E["dates"])
        lat, elev = TOWERS[site]["lat"], TOWERS[site]["elev"]
    d = pd.DataFrame(dict(lai=E["lai"], et_m=E["ens"]["et"][0], ks_m=E["ens"]["ks"][0], et0_m=w["et0"].values,
                          tmin_m=w["tasmin"].values, tmax_m=w["tasmax"].values, hurs_m=w["hurs"].values,
                          rs_m=w["rs"].values, u2_m=w["u2"].values, rain_m=w["pr"].values), index=E["dates"])
    return d, dict(site=site, year=year, crop=crop, start=start, peak=peak, end=end, lat=lat, elev=elev)


def tower_path(site):
    return glob.glob(os.path.join("data", "raw", "icos", "extract", TOWER_DIR[site], "*FLUXMET_HH*.csv"))[0]


def tower_season(site, dates, lat, elev):
    from phase34_water_v4 import ec_daily
    p = tower_path(site)
    hh = pd.read_csv(p, usecols=["TIMESTAMP_END", "TA_F", "VPD_F", "SW_IN_F", "WS_F", "PA_F"], na_values=[-9999])
    t = tower_daily(hh).reindex(dates)
    t["u2"] = u2_from_height(t["ws"], SONIC_M[site])
    t["et0_t"] = et0_ea(t["tmin"], t["tmax"], t["ea"], t["rs"], t["u2"], t["pa"], elev, lat, dates.dayofyear)
    t["et_t"] = ec_daily(p).reindex(dates)
    t["et_corr"] = ec_daily(p, "LE_CORR").reindex(dates)
    return t


def ebr(site):
    """Energy balance ratio sum(H + LE) / sum(Rn - G), 1 Mar - 30 Sep, half-hours with all four present, per year."""
    d = pd.read_csv(tower_path(site), usecols=["TIMESTAMP_END", "H_F_MDS", "LE_F_MDS", "NETRAD", "G_F_MDS"],
                    na_values=[-9999]).dropna()
    t = pd.to_datetime(d["TIMESTAMP_END"].astype(str), format="%Y%m%d%H%M")
    m = t.dt.month.between(3, 9)
    d, yr = d[m], t[m].dt.year
    g = d.assign(year=yr.values).groupby("year")
    return (g["H_F_MDS"].sum() + g["LE_F_MDS"].sum()) / (g["NETRAD"].sum() - g["G_F_MDS"].sum())


def run():
    os.makedirs(OUT, exist_ok=True)
    days, weeks, swaps, ebrs = [], [], [], {}
    for site, year in SEASONS:
        m, I = model_season(site, year)
        t = tower_season(site, m.index, I["lat"], I["elev"])
        d = m.join(t)
        d["stage"] = stages(d.index, d["lai"], I["peak"])
        ins = (d.index >= I["start"]) & (d.index <= I["end"])
        ok = ins & d["et_t"].notna() & d["et0_t"].notna() & (d["n"] == 48)
        d = d[ok].assign(site=site, year=year, crop=I["crop"])
        # one-input swaps on the model side (ea from the model's own humidity kept unless humidity is swapped)
        es_m = (_es(d["tmax_m"]) + _es(d["tmin_m"])) / 2
        ea_m = d["hurs_m"] / 100 * es_m
        p_m = 101.3 * ((293 - 0.0065 * I["elev"]) / 293) ** 5.26
        doy = d.index.dayofyear
        base = et0_ea(d["tmin_m"], d["tmax_m"], ea_m, d["rs_m"], d["u2_m"], p_m, I["elev"], I["lat"], doy)
        sw = {"radiation": et0_ea(d["tmin_m"], d["tmax_m"], ea_m, d["rs"], d["u2_m"], p_m, I["elev"], I["lat"], doy),
              "temperature": et0_ea(d["tmin"], d["tmax"], ea_m, d["rs_m"], d["u2_m"], p_m, I["elev"], I["lat"], doy),
              "humidity": et0_ea(d["tmin_m"], d["tmax_m"], d["ea"], d["rs_m"], d["u2_m"], p_m, I["elev"], I["lat"], doy),
              "wind": et0_ea(d["tmin_m"], d["tmax_m"], ea_m, d["rs_m"], d["u2"], p_m, I["elev"], I["lat"], doy)}
        swaps.append(dict(site=site, year=year, crop=I["crop"], days=len(d), et0_model=float(d["et0_m"].mean()),
                          et0_tower=float(d["et0_t"].mean()), check_base=float(np.mean(base - d["et0_m"])),
                          **{f"swap_{k}": float(np.mean(v - base)) for k, v in sw.items()},
                          rs_model=float(d["rs_m"].mean()), rs_tower=float(d["rs"].mean()),
                          u2_model=float(d["u2_m"].mean()), u2_tower=float(d["u2"].mean()),
                          ea_model=float(ea_m.mean()), ea_tower=float(d["ea"].mean()),
                          tmean_model=float(((d["tmin_m"] + d["tmax_m"]) / 2).mean()),
                          tmean_tower=float(((d["tmin"] + d["tmax"]) / 2).mean())))
        wk = ((d.index - I["start"]).days // 7).values
        g = d.assign(wk=wk).groupby("wk")
        W = pd.DataFrame(dict(n=g.size(), et_m=g["et_m"].sum(), et_t=g["et_t"].sum(), et_corr=g["et_corr"].sum(),
                              et0_m=g["et0_m"].sum(), et0_t=g["et0_t"].sum(), corr_n=g["et_corr"].count()))
        W = W[W.n >= 5].assign(site=site, year=year, crop=I["crop"])
        weeks.append(W)
        days.append(d)
        if site not in ebrs:
            ebrs[site] = ebr(site)
        print(f"{site} {year} {I['crop']}: {len(d)} days, ET model {d['et_m'].sum():.0f} / tower {d['et_t'].sum():.0f} mm, "
              f"ET0 model {d['et0_m'].sum():.0f} / tower {d['et0_t'].sum():.0f} mm", flush=True)
    D, Wk, SW = pd.concat(days), pd.concat(weeks), pd.DataFrame(swaps)
    D.to_csv(os.path.join(OUT, "days.csv"))
    Wk.to_csv(os.path.join(OUT, "weeks.csv"), index=False)
    SW.to_csv(os.path.join(OUT, "et0_swaps.csv"), index=False)
    report(D, Wk, SW, ebrs)


def report(D, Wk, SW, ebrs):
    L = []
    L.append("== 1. Season totals (valid days in the satellite season), mm")
    tot = D.groupby(["site", "year", "crop"]).agg(days=("et_t", "size"), et_model=("et_m", "sum"), et_tower=("et_t", "sum"),
                                                   et0_model=("et0_m", "sum"), et0_tower=("et0_t", "sum"))
    tot["et_ratio"], tot["et0_ratio"] = tot.et_model / tot.et_tower, tot.et0_model / tot.et0_tower
    tot["kc_model"], tot["kc_tower"] = tot.et_model / tot.et0_model, tot.et_tower / tot.et0_tower
    L.append(tot.round(2).to_string())
    L.append("\n== 2. ET0 inputs: mean per day (model vs tower) and one-input swap effect on model ET0 (mm/day)")
    L.append(SW.drop(columns=["check_base"]).round(2).to_string(index=False))
    L.append(f"(check: rebuilt model ET0 equals the model's ET0 within {SW.check_base.abs().max():.1e} mm/day)")
    L.append("\n== 3. Kc_act by stage (sum ET / sum ET0 on each side)")
    st = D.groupby(["site", "crop", "stage"]).agg(days=("et_t", "size"), et_m=("et_m", "sum"), et_t=("et_t", "sum"),
                                                  e0_m=("et0_m", "sum"), e0_t=("et0_t", "sum"))
    st["kc_model"], st["kc_tower"] = st.et_m / st.e0_m, st.et_t / st.e0_t
    st["kc_ratio"] = st.kc_model / st.kc_tower
    L.append(st[["days", "kc_model", "kc_tower", "kc_ratio"]].round(2).to_string())
    L.append("\n== 4. Attribution, weekly log ratio model / tower (mean; exp() = factor)")
    rows = []
    for key, g in [("ALL", Wk)] + list(Wk.groupby("site")):
        a = attribution(g.et_m, g.et_t, g.et0_m, g.et0_t)
        gc = g[g.corr_n == g.n]
        c = attribution(gc.et_m, gc.et_corr, gc.et0_m, gc.et0_t) if len(gc) else dict(total=np.nan, et0=np.nan, kc=np.nan, weeks=0)
        rows.append(dict(group=key, weeks=a["weeks"], factor_total=np.exp(a["total"]), factor_et0=np.exp(a["et0"]),
                         factor_kc=np.exp(a["kc"]), share_et0=a["et0"] / a["total"] if a["total"] else np.nan,
                         corr_weeks=c["weeks"], corr_factor_total=np.exp(c["total"]), corr_factor_kc=np.exp(c["kc"])))
    A = pd.DataFrame(rows)
    A.to_csv(os.path.join(OUT, "attribution.csv"), index=False)
    L.append(A.round(3).to_string(index=False))
    L.append("\n== 5. Tower energy balance ratio (H + LE) / (Rn - G), Mar-Sep")
    for s, e in ebrs.items():
        L.append(f"{s}: " + ", ".join(f"{y} {v:.2f}" for y, v in e.items() if 2018 <= y <= 2025))
    txt = "\n".join(L)
    open(os.path.join(OUT, "bias_report.txt"), "w").write(txt + "\n")
    print(txt)
    _figure(D, Wk)


def _figure(D, Wk):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    col = {"selhausen": "#1b6ca8", "dekli": "#c0392b", "degeb": "#27ae60"}
    for s, g in Wk.groupby("site"):
        ax[0].scatter(g.et0_t / g.n, g.et0_m / g.n, s=10, c=col[s], label=s, alpha=0.7)
        ax[1].scatter(g.et_t / g.n, g.et_m / g.n, s=10, c=col[s], label=s, alpha=0.7)
    for a, t in zip(ax[:2], ("ET0 per day: tower weather (x) vs our inputs (y)", "ET per day: tower (x) vs model v2 (y)")):
        lim = [0, 7]
        a.plot(lim, lim, "k--", lw=0.8)
        a.set_xlim(lim); a.set_ylim(lim); a.set_title(t, fontsize=9); a.legend(fontsize=7)
    st = D.groupby(["stage"]).apply(lambda g: pd.Series(dict(model=g.et_m.sum() / g.et0_m.sum(), tower=g.et_t.sum() / g.et0_t.sum())))
    st = st.reindex(["lai<1", "lai1-3", "lai>=3", "after_peak"])
    x = np.arange(len(st))
    ax[2].bar(x - 0.2, st.model, 0.4, label="model", color="#888")
    ax[2].bar(x + 0.2, st.tower, 0.4, label="tower", color="#1b6ca8")
    ax[2].set_xticks(x, st.index); ax[2].set_title("Kc_act = ET / ET0 by stage (all sites)", fontsize=9); ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "bias_figure.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    run()
