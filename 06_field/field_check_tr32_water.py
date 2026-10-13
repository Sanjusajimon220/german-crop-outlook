"""Field Outlook v1, CHECK 2 (pre-registered, docs/project_log.md): the model's water balance vs measured
evapotranspiration (eddy covariance) on TR32 winter wheat fields in the Rur region (Reichenau et al. 2020, ESSD 12,
2333; data DOI 10.5880/TR32DB.39, CC BY 4.0).

EC daily ET = sum of half-hourly latent heat (LvE, flag <= 1; Merken: SumLvE, no flag) x 1800 s / 2.45 MJ/kg on days
with >= 44 of 48 valid half-hours (mean x 48). Model = the district growth model's daily water balance (same physics
as the field model v4) for district 05358 Dueren, district weather of that year, frozen v8/v9 physics settings;
daily actual ET = transpiration + soil evaporation. Matched days 1 April - harvest.
"""
import inspect
import json
import os
import textwrap

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("VISTA_WEATHER", "v8")
import phase2_growth as pg   # noqa: E402

T = os.path.join("data", "raw", "tr32")
SITES = [  # site-year, EC file, format, harvest date, measured yield (t/ha as reported)
    ("Selhausen SEF08 2008", "selhausen/EC/fluxes_SEF08_EC_000_2008.csv", "A", "2008-08-06", "8.3 dry"),
    ("Selhausen SEF08 2009", "selhausen/EC/fluxes_SEF08_EC_000_2009.csv", "A", "2009-07-28", "8.6"),
    ("Merken MKF02 2009", "merken/EC/fluxes_MKF02_EC_J1l_2009.csv", "B", "2009-08-04", "11.2"),
    ("Merzenhausen MEF01 2011", "merzenhausen/EC/fluxes_MEF01_EC_001_2011.csv", "A", "2011-08-05", "9.3 dry"),
    ("Merzenhausen MEF01 2012", "merzenhausen/EC/fluxes_MEF01_EC_001_2012.csv", "A", "2012-08-03", "8.9 dry"),
    ("Selhausen SEF01 2015", "selhausen/EC/fluxes_SEF01_EC_001_2015.csv", "A", "2015-08-02", "10.5"),
    ("Merzenhausen MEF01 2017", "merzenhausen/EC/fluxes_MEF01_EC_001_2017.csv", "A", "2017-07-22", "-"),
]


def ec_daily(path, fmt):
    d = pd.read_csv(os.path.join(T, path), sep=";", skiprows=[1], na_values="NA", low_memory=False)
    if fmt == "A":
        t, le = pd.to_datetime(d["T_end"]), d["LvE"].where(d["Flag(LvE)"] <= 1)
    else:
        t, le = pd.to_datetime(d["Datetime(end)"]), d["SumLvE"]
    day = (t - pd.Timedelta(minutes=1)).dt.normalize()
    g = le.groupby(day)
    ok = g.count() >= 44
    return (g.mean() * 48 * 1800 / 2.45e6)[ok]          # mm/day


def daily_model():
    """Growth.forward with daily actual ET recorded (patched copy; the original stays unchanged)."""
    src = textwrap.dedent(inspect.getsource(pg.Growth.forward))
    src = src.replace("daily.append(lai)", "daily.append(torch.stack([lai, (ta + es) * growing, (tpot + epot) * growing]))")
    ns = {}
    exec(src, {**pg.__dict__, "torch": torch}, ns)

    class G(pg.Growth):
        forward = ns["forward"]
    return G


def main():
    crop = "winter_wheat"
    m, d = pg.load_growth(crop)
    model = daily_model()(json.load(open(os.path.join(pg.P, "growth_model_winter_wheat_v7_v8.json"))))
    years = sorted({int(s[0][-4:]) for s in SITES})
    # Dueren (05358); a year without a Dueren row (2011: no official yield) uses neighbouring Heinsberg (05370)
    have = set(m.loc[m["district"] == "05358", "year"])
    pick = [(("05358" if y in have else "05370"), y) for y in years]
    rows = np.array([np.flatnonzero((m["district"] == dd).values & (m["year"] == y).values)[0] for dd, y in pick])
    sub = {k: v[rows] for k, v in d.items() if v.dim() and len(v) == len(m)}
    with torch.no_grad():
        _, out = model.forward(sub, pg.GROWTH[crop]["dry_matter"], record=True)
    daily = out["lai_daily"].numpy().transpose(2, 1, 0)   # [lai, eta, etp] x days x rows -> rows x days x 3
    out_rows = []
    for name, path, fmt, harvest, ytxt in SITES:
        y = int(name[-4:])
        i = years.index(y)
        start = pd.Timestamp(f"{y}-01-01") + pd.Timedelta(days=round(m["doy_sowing"].values[rows][i]) - 1)
        mod = pd.DataFrame({"date": start + pd.to_timedelta(np.arange(daily.shape[1]), unit="D"),
                            "lai": daily[i, :, 0], "eta": daily[i, :, 1], "etp": daily[i, :, 2]}).set_index("date")
        ec = ec_daily(path, fmt)
        win = (ec.index >= f"{y}-04-01") & (ec.index <= harvest)
        ec = ec[win]
        j = mod.reindex(ec.index).dropna()
        e = ec.loc[j.index]
        if len(j) < 10:
            out_rows.append(dict(site_year=name, days=len(j)))
            continue
        out_rows.append(dict(site_year=name, days=len(j), ec_mm_day=e.mean(), model_mm_day=j["eta"].mean(),
                             ratio=j["eta"].mean() / e.mean(), r_daily=np.corrcoef(e, j["eta"])[0, 1],
                             ec_sum=e.sum(), model_sum=j["eta"].sum(), model_demand_sum=j["etp"].sum(),
                             district=pick[i][0], measured_yield=ytxt, model_district_yield=float(out["phys_yield"][i])))
    o = pd.DataFrame(out_rows)
    os.makedirs(os.path.join(pg.P, "field_outlook"), exist_ok=True)
    o.to_csv(os.path.join(pg.P, "field_outlook", "check_tr32_water.csv"), index=False)
    with pd.option_context("display.width", 220):
        print(o.round(2).to_string(index=False))
    ok = o.dropna(subset=["ratio"])
    print(f"POOLED: model/EC ratio {ok['model_sum'].sum() / ok['ec_sum'].sum():.2f} (agreement band 1.00-1.25), "
          f"median daily r {ok['r_daily'].median():.2f} (usable if >= 0.6), site-years {len(ok)}")


if __name__ == "__main__":
    main()
