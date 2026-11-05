"""Job 3: field water use (evapotranspiration) vs flux towers (pre-registered docs/project_log.md 2026-11-05).

Truth: eddy-covariance daily ET (field_check_tr32_water.ec_daily rules: good-quality LE, >= 44 of 48 half-hours).
M1 ours: the growth model's daily actual ET (transpiration + soil evaporation) x k (TR32: leave-one-site-year-out k;
ICOS DE-RuS: k = 1.301). M2 FAO-56 single crop coefficient: ET0 x Kc (FAO shape on our calendar's stage dates) x Ks
(root-zone bucket from the same soil, p = 0.55). Matched days 1 April - harvest. Rule: M1 passes if, pooled, its daily
RMSE <= M2's AND |season ratio - 1| <= M2's AND daily r >= 0.6.
Run from D:\\vista: python phase28_water_job3.py tr32 | icos   -> data/processed/field_v2/water_job3/.
"""
import os

import numpy as np
import pandas as pd

KC_INI, KC_MID = 0.4, 1.15
KC_END = {"winter_wheat": 0.3, "winter_barley": 0.25}
P_DEPLETION = 0.55
ROOT0, ROOT1 = 0.3, 1.3                                   # m: emergence -> heading
OUT = os.path.join("data", "processed", "field_v2", "water_job3")


def kc_curve(dates, st, kc_end):
    """FAO-56 single crop coefficient for a winter cereal: Kc_ini until 1 March, linear to Kc_mid at heading, Kc_mid until
    half of grain filling, linear to Kc_end at maturity, Kc_end after."""
    t = pd.DatetimeIndex(dates)
    reg = pd.Timestamp(f"{st['heading'].year}-03-01")
    h, mt = st["heading"], st["maturity"]
    half = h + (mt - h) / 2
    d = lambda a: (t - a).days.values.astype(float)
    kc = np.full(len(t), KC_INI)
    dev = (t >= reg) & (t < h)
    kc[dev] = KC_INI + (KC_MID - KC_INI) * d(reg)[dev] / max((h - reg).days, 1)
    kc[(t >= h) & (t <= half)] = KC_MID
    late = (t > half) & (t < mt)
    kc[late] = KC_MID + (kc_end - KC_MID) * (d(half)[late] / max((mt - half).days, 1))
    kc[t >= mt] = kc_end
    return kc


def fao56_et(et0, rain, kc, taw, p=P_DEPLETION):
    """Daily actual ET with the FAO-56 water-stress factor Ks from a root-zone depletion bucket (mm), starting full."""
    dr, out = 0.0, np.zeros(len(et0))
    for i in range(len(et0)):
        dr = max(dr - rain[i], 0.0)                        # rain refills (excess drains)
        raw = p * taw[i]
        ks = 1.0 if dr <= raw else max((taw[i] - dr) / max(taw[i] - raw, 1e-9), 0.0)
        et = min(ks * kc[i] * et0[i], max(taw[i] - dr, 0.0))
        out[i] = et
        dr = min(dr + et, taw[i])
    return out


def scores(model, ec):
    model, ec = np.asarray(model, float), np.asarray(ec, float)
    return {"days": int(len(ec)), "rmse": float(np.sqrt(np.mean((model - ec) ** 2))), "r": float(np.corrcoef(model, ec)[0, 1]),
            "ratio": float(model.sum() / ec.sum()), "model_sum": float(model.sum()), "ec_sum": float(ec.sum())}


def _stage_dates(sub, i, start):
    """Calendar dates of emergence, heading and maturity for row i (development curve vs its thresholds)."""
    dev = sub["dev"][i].numpy()
    out = {}
    for name, key in (("emergence", "th_e"), ("heading", "th_h"), ("maturity", "th_m")):
        hit = np.flatnonzero(dev >= float(sub[key][i]))
        out[name] = start + pd.Timedelta(days=int(hit[0]) if len(hit) else len(dev) - 1)
    return out


def _taw(cap_layers, dates, st, layer_cm=20):
    """Plant-available water (mm) within the root depth on each date: root 0.3 m at emergence -> 1.3 m at heading."""
    t = pd.DatetimeIndex(dates)
    frac = np.clip((t - st["emergence"]).days.values / max((st["heading"] - st["emergence"]).days, 1), 0, 1)
    depth_cm = 100 * (ROOT0 + (ROOT1 - ROOT0) * frac)
    top = np.arange(len(cap_layers)) * layer_cm
    share = np.clip((depth_cm[:, None] - top[None, :]) / layer_cm, 0, 1)
    return share @ np.asarray(cap_layers, float)


def tr32():
    """TR32 site-years: M1 (district weather, prior median) with leave-one-site-year-out k vs M2 FAO-56 vs EC."""
    import json
    import torch
    import field_check_tr32_water as C
    pg = C.pg
    os.makedirs(OUT, exist_ok=True)
    sites = [s for s in C.SITES if not s[0].startswith("Selhausen SEF08")]   # SEF08 file defect (check 2, logged)
    crop = "winter_wheat"
    m, d = pg.load_growth(crop)
    model = C.daily_model()(json.load(open(os.path.join(pg.P, "growth_model_winter_wheat_v7_v8.json"))))
    years = sorted({int(s[0][-4:]) for s in sites})
    have = set(m.loc[m["district"] == "05358", "year"])
    pick = [(("05358" if y in have else "05370"), y) for y in years]
    rows = np.array([np.flatnonzero((m["district"] == dd).values & (m["year"] == y).values)[0] for dd, y in pick])
    sub = {k: v[rows] for k, v in d.items() if v.dim() and len(v) == len(m)}
    with torch.no_grad():
        _, out = model.forward(sub, pg.GROWTH[crop]["dry_matter"], record=True)
    daily = out["lai_daily"].numpy().transpose(2, 1, 0)
    per = []
    for name, path, fmt, harvest, _ in sites:
        y = int(name[-4:]); i = years.index(y)
        start = pd.Timestamp(f"{y}-01-01") + pd.Timedelta(days=round(m["doy_sowing"].values[rows][i]) - 1)
        dates = start + pd.to_timedelta(np.arange(daily.shape[1]), unit="D")
        st = _stage_dates(sub, i, start)
        kc = kc_curve(dates, st, KC_END[crop])
        taw = _taw(sub["cap"][i].numpy(), dates, st)
        m2 = np.zeros(len(dates))
        go = dates >= pd.Timestamp(f"{y}-03-01")                       # bucket starts full on 1 March
        m2[go] = fao56_et(sub["et0"][i].numpy()[go], sub["pr"][i].numpy()[go], kc[go], taw[go])
        frame = pd.DataFrame({"m1_raw": daily[i, :, 1], "m2": m2}, index=dates)
        ec = C.ec_daily(path, fmt)
        ec = ec[(ec.index >= f"{y}-04-01") & (ec.index <= harvest)]
        j = frame.reindex(ec.index).dropna()
        per.append(dict(site_year=name, ec=ec.loc[j.index], m1_raw=j["m1_raw"], m2=j["m2"]))
    for k, p in enumerate(per):                                         # leave-one-site-year-out k for M1
        others = [q for n, q in enumerate(per) if n != k]
        p["k"] = sum(q["ec"].sum() for q in others) / sum(q["m1_raw"].sum() for q in others)
        p["m1"] = p["m1_raw"] * p["k"]
    return per


def judge(per):
    """Scores per site-year and pooled; the pre-registered rule on the pooled scores."""
    rows = [dict(site_year=p["site_year"], method=m, **scores(p[m], p["ec"])) for p in per for m in ("m1", "m2")]
    pooled = {m: scores(pd.concat([p[m] for p in per]), pd.concat([p["ec"] for p in per])) for m in ("m1", "m2")}
    rows += [dict(site_year="POOLED", method=m, **v) for m, v in pooled.items()]
    a, b = pooled["m1"], pooled["m2"]
    ok = a["rmse"] <= b["rmse"] and abs(a["ratio"] - 1) <= abs(b["ratio"] - 1) and a["r"] >= 0.6
    return pd.DataFrame(rows), ok


def run():
    os.makedirs(OUT, exist_ok=True)
    per = tr32() + icos()
    r, ok = judge(per)
    r.to_csv(os.path.join(OUT, "job3_scores.csv"), index=False)
    pd.concat([pd.DataFrame({"date": p["ec"].index, "site_year": p["site_year"], "ec": p["ec"].values, "m1": p["m1"].values,
                             "m2": p["m2"].values}) for p in per]).to_csv(os.path.join(OUT, "job3_daily.csv"), index=False)
    for p in per:
        print(p["site_year"], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.get("info", {"k": p.get("k")}).items()})
    with pd.option_context("display.width", 200):
        print(r.round(3).to_string(index=False))
    print(f"RESULT Job 3: {'PASS' if ok else 'FAIL'} (rule: pooled M1 RMSE <= M2, |ratio - 1| <= M2's, daily r >= 0.6)")
    return ok



ICOS_HH = os.path.join("data", "raw", "icos", "extract", "fx_inner", "ICOSETC_DE-RuS_FLUXNET_FLUXMET_HH_2019-2026_v1.3_r1.csv")
ICOS_FIELDS = {2023: 7182626, 2025: 8398633}            # DE-RuS winter wheat seasons (NRW register, Selhausen area)
K_SHOWN = 1.301


def ec_daily_gapfilled(path=ICOS_HH, min_good=0.8):
    """Daily ET (mm) from gap-filled LE_F_MDS on days with >= 80 % measured or good-quality filled half-hours (QC <= 1)."""
    d = pd.read_csv(path, usecols=["TIMESTAMP_END", "LE_F_MDS", "LE_F_MDS_QC"], na_values=[-9999])
    t = pd.to_datetime(d["TIMESTAMP_END"].astype(str), format="%Y%m%d%H%M")
    day = (t - pd.Timedelta(minutes=1)).dt.normalize()
    good = (d["LE_F_MDS_QC"] <= 1) & d["LE_F_MDS"].notna()
    g = pd.DataFrame({"day": day, "le": d["LE_F_MDS"], "good": good}).groupby("day")
    share, n = g["good"].mean(), g["le"].count()
    et = g["le"].sum() * 1800 / 2.45e6
    return et[(share >= min_good) & (n == 48)]


def _field_ensemble_et(crop, year, district, cell):
    """phase23 field ensemble with daily [LAI, actual ET] recorded: returns daily lai, eta (members x days), sowing, cap_f, dev."""
    import inspect, textwrap, torch
    import phase2_growth as pg
    import phase23_ours_field as OF
    from phase2_growth import TAU
    from phase4_field_assim import FieldGrowth, draw
    src = textwrap.dedent(inspect.getsource(pg.Growth.forward))
    src = src.replace("daily.append(lai)", "daily.append(torch.stack([lai, (ta + es) * growing]))")
    ns = {}
    exec(src, {**pg.__dict__, "torch": torch}, ns)

    class FG(FieldGrowth):
        forward = ns["forward"]
    L = OF._load(crop)
    m, d = L["m"], L["d"]
    i = L["row"][(district, year)]
    dates, col, w = OF._weather(year)
    sow = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=round(m.at[i, "doy_sowing"]) - 1)
    first = int((np.datetime64(sow.date()) - dates[0]).astype(int))
    n_days = d["tmin"].shape[1]
    sub = {k: v[i:i + 1].clone() for k, v in d.items() if v.dim() and len(v) == len(m)}
    for key, var in (("tmin", "tasmin"), ("tmax", "tasmax"), ("pr", "pr")):
        sub[key] = torch.tensor(w[var][first:first + n_days, col[cell]], dtype=torch.float32)[None, :]
    with torch.no_grad():
        dev, th = L["cal"]({"tmin": sub["tmin"], "tmax": sub["tmax"], "daylen": sub["daylen"],
                            "rain": torch.cumsum(sub["pr"], 1), "decade": sub["decade"]}, return_dev=True)
    pert = draw(3000, seed=int(district) + year)
    pert["dev_f"] = 1 + (pert["dev_f"] - 1) * OF.DEV_STRETCH
    n = len(pert)
    pt = {k: torch.tensor(v.values, dtype=torch.float32) for k, v in pert.items()}
    base = {k: v.clone() for k, v in sub.items()}
    sub = {k: v.repeat(n, *([1] * (v.dim() - 1))) for k, v in sub.items()}
    devp = dev.repeat(n, 1) * pt["dev_f"][:, None]
    e, h, mt = (th[0, j] for j in L["idx"])
    sub["em"] = torch.sigmoid((devp - e) / TAU); sub["ph"] = torch.clamp((devp - e) / (h - e), 0, 1)
    sub["post"] = torch.sigmoid((devp - h) / TAU); sub["gf"] = torch.clamp((devp - h) / (mt - h), 0, 1)
    sub["mat"] = torch.sigmoid((devp - mt) / TAU)
    sub["cap"] = sub["cap"] * pt["cap_f"][:, None]
    sub["n_supply"] = pt["n_supply"]
    model = FG(L["settings"], pt)
    with torch.no_grad():
        _, out = model(sub, L["dry"], record=True)
    daily = out["lai_daily"].numpy()                        # 2 ([lai, eta]) x days x members
    days = sow + pd.to_timedelta(np.arange(daily.shape[1]), unit="D")
    stages = {nm: sow + pd.Timedelta(days=int(np.flatnonzero(dev[0].numpy() >= float(v))[0]) if (dev[0].numpy() >= float(v)).any() else n_days - 1)
              for nm, v in zip(("emergence", "heading", "maturity"), (e, h, mt))}
    return dict(dates=days, lai=daily[0].T, eta=daily[1].T, cap_f=pert["cap_f"].values, stages=stages,
                cell_pr=base["pr"][0].numpy(), et0=base["et0"][0].numpy(), cap=base["cap"][0].numpy())


def icos():
    """ICOS DE-RuS wheat seasons: M1 (field ensemble, satellite-weighted) and M2 (FAO-56) vs gap-filled EC."""
    import phase23_ours_field as OF
    from phase16_hrvpp_items import _field_table
    from phase4_field_assim_v3 import weights
    from phase24_job2 import soil_logprior
    ec_all = ec_daily_gapfilled()
    FD = os.path.join("data", "processed", "field")
    cells = pd.read_csv(os.path.join(FD, "fields_hyras_cell.csv"))
    dist = pd.read_csv(os.path.join(FD, "fields_district.csv"), dtype={"district": str})
    ratio = OF.soil_ratios()["wheat"]
    per = []
    for year, fid in ICOS_FIELDS.items():
        cell = int(cells[(cells.ID == fid) & (cells.VALIDFROM == year)]["cell"].iloc[0])
        district = dist[(dist.ID == fid) & (dist.VALIDFROM == year)]["district"].iloc[0]
        E = _field_ensemble_et("wheat", year, district, cell)
        F = _field_table("selhausen", year)
        k = list(F["fids"]).index(fid)
        weeks = pd.DatetimeIndex(F["weeks"].astype("datetime64[D]"))
        y, meas, mask = F["lai"][:, k], np.nan_to_num(F["meas"][:, k]) >= 0.5, np.asarray(F["mask"][:, k], bool)
        used = np.flatnonzero(meas & mask & np.isfinite(y) & ~weeks.month.isin([12, 1]))
        day_idx = np.array([(wk - E["dates"][0]).days for wk in weeks[used]])
        ok = (day_idx >= 0) & (day_idx < E["lai"].shape[1])
        r = ratio.get((fid, year))
        wgt = weights(E["lai"].astype(float), day_idx[ok], y[used][ok], np.zeros(ok.sum()), soil_logprior(E["cap_f"], r))
        m1 = pd.Series(wgt @ E["eta"] * K_SHOWN, index=E["dates"])
        st = E["stages"]
        kc = kc_curve(E["dates"], st, KC_END["winter_wheat"])
        taw = _taw(E["cap"] * (np.clip(r, 0.5, 1.5) if r is not None and np.isfinite(r) else 1.0), E["dates"], st)
        go = E["dates"] >= pd.Timestamp(f"{year}-03-01")
        m2 = np.zeros(len(E["dates"]))
        m2[go] = fao56_et(E["et0"][go], E["cell_pr"][go], kc[go], taw[go])
        season_end = weeks[np.flatnonzero(mask & (np.nan_to_num(y) > 0.5))].max()
        ec = ec_all[(ec_all.index >= f"{year}-04-01") & (ec_all.index <= season_end)]
        fr = pd.DataFrame({"m1": m1, "m2": pd.Series(m2, index=E["dates"])}).reindex(ec.index).dropna()
        per.append(dict(site_year=f"ICOS DE-RuS {year}", ec=ec.loc[fr.index], m1=fr["m1"], m2=fr["m2"],
                        info=dict(field=fid, cell=cell, district=district, soil_ratio=r, ess=float(1 / (wgt ** 2).sum()),
                                  used_weeks=int(ok.sum()), season_end=str(season_end.date()))))
    return per


if __name__ == "__main__":
    run()
