"""Water v5 HELD-OUT test (pre-registered docs/project_log.md 2026-11-06): v5 (f_e 0.60, k 0.45) vs v2 at towers never
used before - ICOS BE-Lon, FR-Gri and DE-Kli 2017. Both models run on the tower's own weather (ET0 as study A, rain P_F),
satellite LAI from our Sentinel-2 chain (phase33 outline), soil: silt loam ruling (BE-Lon / FR-Gri) or district table
(DE-Kli). Truth: LE_CORR (raw LE_F_MDS reported); towers with Mar-Sep energy balance ratio > 1.05 do not decide.
PASS if pooled v5 weekly RMSE <= 0.90 x v2 AND |mean weekly log ratio v5 / tower| <= ln 1.10 AND v5 better in >= 60 %
of deciding seasons.
Run from D:\\vista: python phase38_water_heldout.py -> data/processed/field_v2/water_heldout/
"""
import os

import numpy as np
import pandas as pd

OUT = os.path.join("data", "processed", "field_v2", "water_heldout")
SEASONS = [("belon", y) for y in (2017, 2018, 2019, 2020, 2022, 2023, 2024)] + \
          [("frgri", y) for y in (2018, 2019, 2020, 2022, 2023, 2024, 2025)] + [("dekli", 2017)]
V2, V5 = (1.0, 0.70), (0.60, 0.45)
SILT_LOAM = (0.29, 0.15)          # FAO-56 Table 19 silt loam midpoints: theta_FC 0.22-0.36, theta_WP 0.09-0.21


# ------------------------------------------------------------------------------------------------- pure functions
def silt_loam_soil():
    fc, wp = SILT_LOAM
    return np.full(20, (fc - wp) * 100.0), 1000 * (fc - 0.5 * wp) * 0.10


def daily_rain(hh):
    t = pd.to_datetime(hh["TIMESTAMP_END"].astype(str), format="%Y%m%d%H%M")
    day = (t - pd.Timedelta(minutes=1)).dt.normalize()
    r = hh["P_F"].groupby(day).sum(min_count=1)
    r.index.name = None
    return r


def passes(rmse_v5, rmse_v2, log_v5, better, seasons):
    return rmse_v5 <= 0.90 * rmse_v2 and abs(log_v5) <= np.log(1.10) and better >= 0.6 * seasons


# ------------------------------------------------------------------------------------------------- data
def season(site, year):
    """Inputs (tower weather, satellite LAI, soil) + truth for one held-out season."""
    import json
    from areas import out_dir
    from phase29_water_v2 import KCB_MID, P_TAB, ROOT_MAX, _taw
    from phase33_testsite_field import TOWERS, soil
    from phase34_water_v4 import ec_daily
    from phase35_water_bias import ebr, tower_path, tower_season
    info = json.load(open(os.path.join(out_dir(site, year), "testsite_field.json")))
    if info.get("dropped"):
        return None
    crop = info["crop"]
    days = pd.date_range(f"{year}-03-01", f"{year}-09-30")
    lai = pd.read_csv(os.path.join(out_dir(site, year), "testsite_lai_daily.csv"), index_col=0, parse_dates=True)["lai"]
    lai = lai.reindex(days).fillna(0).values
    start, peak, end = (pd.Timestamp(info[k]) for k in ("start", "peak", "end"))
    t = TOWERS[site]
    tw = tower_season(site, days, t["lat"], t["elev"])
    hh = pd.read_csv(tower_path(site), usecols=["TIMESTAMP_END", "P_F"], na_values=[-9999])
    rain = daily_rain(hh).reindex(days)
    et0 = tw["et0_t"].interpolate(limit_area="inside").bfill().ffill()
    paw, tew = soil(site) if site == "dekli" else silt_loam_soil()
    frac = np.clip(np.asarray((days - start).days) / max((peak - start).days, 1), 0, 1)
    taw = _taw(paw, 0.1 + (ROOT_MAX[crop] - 0.1) * frac)
    corr = ec_daily(tower_path(site), "LE_CORR").reindex(days)
    return dict(site=site, year=year, crop=crop, dates=days, start=start, peak=peak, end=end, lai=lai,
                et0=et0.values, rain=rain.fillna(0).values, rain_gaps=int(rain.isna().sum()), et0_gaps=int(tw["et0_t"].isna().sum()),
                taw=taw, tew=tew, p=P_TAB[crop], kcb_mid=KCB_MID[crop], corr=corr, raw=tw["et_t"],
                inside=(days >= start) & (days <= end), ebr=float(ebr(site).get(year, np.nan)), field_ha=info["field_ha"])


def run_model(S, f_e, k):
    from phase29_water_v2 import dual_kc_balance, kcb_from_lai
    return dual_kc_balance(S["et0"], S["rain"], kcb_from_lai(S["lai"], S["kcb_mid"], k), S["lai"], S["taw"], S["p"],
                           S["tew"], ke_scale=f_e)


def look():
    from phase37_water_v5 import deciding, weekly_sums
    os.makedirs(OUT, exist_ok=True)
    rows, pooled = [], {"v2": ([], []), "v5": ([], [])}
    for site, year in SEASONS:
        S = season(site, year)
        if S is None:
            rows.append(dict(site=site, year=year, dropped=True))
            continue
        dec = deciding(S["ebr"])
        row = dict(site=site, year=year, crop=S["crop"], field_ha=S["field_ha"], ebr=S["ebr"], deciding=dec,
                   rain_gap_days=S["rain_gaps"], et0_gap_days=S["et0_gaps"])
        for name, setting in (("v2", V2), ("v5", V5)):
            out = run_model(S, *setting)
            et = pd.Series(out["et"], index=S["dates"])
            for truth in ("corr", "raw"):
                valid = pd.Series(S["inside"], index=S["dates"]) & S[truth].notna()
                m, t = weekly_sums(et, S[truth].fillna(0), valid, S["start"])
                row[f"{name}_rmse_{truth}"] = float(np.sqrt(np.mean((m - t) ** 2))) if len(m) else np.nan
                row[f"{name}_factor_{truth}"] = float(np.exp(np.mean(np.log(m / t)))) if len(m) else np.nan
                row[f"weeks_{truth}"] = len(m)
                if truth == "corr" and dec and len(m):
                    pooled[name][0].append(m); pooled[name][1].append(t)
            row[f"{name}_stress_days"] = int((out["ks"][S["inside"]] < 0.9).sum())
        rows.append(row)
        print(f"{site} {year} {S['crop']} EBR {S['ebr']:.2f} {'DECIDES' if dec else 'not deciding'}: weeks {row['weeks_corr']}, "
              f"RMSE v2 {row['v2_rmse_corr']:.2f} v5 {row['v5_rmse_corr']:.2f}, factor v2 {row['v2_factor_corr']:.2f} "
              f"v5 {row['v5_factor_corr']:.2f}", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "seasons.csv"), index=False)
    D = R[R.get("deciding", False) == True].dropna(subset=["v5_rmse_corr"])      # noqa: E712
    res = {}
    for name in ("v2", "v5"):
        m, t = np.concatenate(pooled[name][0]), np.concatenate(pooled[name][1])
        res[name] = dict(rmse=float(np.sqrt(np.mean((m - t) ** 2))), log=float(np.mean(np.log(m / t))), weeks=len(m),
                         tower_mean=float(t.mean()))
    better = int((D.v5_rmse_corr < D.v2_rmse_corr).sum())
    ok = passes(res["v5"]["rmse"], res["v2"]["rmse"], res["v5"]["log"], better, len(D))
    L = [f"Deciding seasons {len(D)} (of {len(R)}); pooled weeks {res['v5']['weeks']}, tower mean {res['v5']['tower_mean']:.1f} mm/week",
         f"Pooled weekly RMSE (LE_CORR): v2 {res['v2']['rmse']:.2f}, v5 {res['v5']['rmse']:.2f} "
         f"(ratio {res['v5']['rmse'] / res['v2']['rmse']:.2f}, needed <= 0.90)",
         f"Pooled factor model / tower: v2 {np.exp(res['v2']['log']):.2f}, v5 {np.exp(res['v5']['log']):.2f} (needed 0.91-1.10)",
         f"v5 better than v2 in {better} of {len(D)} deciding seasons (needed >= 60 %)",
         f"RESULT v5 held-out: {'PASS' if ok else 'FAIL'}"]
    nd = R[(R.get("deciding", True) == False)]                                    # noqa: E712
    if len(nd):
        L.append(f"Not deciding (EBR > 1.05): {list(zip(nd.site, nd.year))}")
    raw_ok = R.dropna(subset=["v5_rmse_raw"])
    L.append(f"Secondary, raw LE_F_MDS (all seasons): mean RMSE v2 {raw_ok.v2_rmse_raw.mean():.2f} v5 {raw_ok.v5_rmse_raw.mean():.2f}; "
             f"factor v2 {np.exp(np.log(raw_ok.v2_factor_raw).mean()):.2f} v5 {np.exp(np.log(raw_ok.v5_factor_raw).mean()):.2f}")
    txt = "\n".join(L)
    open(os.path.join(OUT, "look_log.txt"), "w").write(txt + "\n")
    print(txt)
    return ok


if __name__ == "__main__":
    look()
