"""Fusion test across years / areas (pre-registered docs/project_log.md 2026-10-16).

Satellite-only fusion, per-pixel neighbour training, LAI and NDVI. For each test season: hidden 30 % of clear S2 values.
  E1 within-season (5 folds by field)   E2 trained on the other season(s)   E3 = E2 + Landsat LAI as measured dates
Run `python phase11_fusion_test.py AREA YEAR AREA YEAR [...]` (every listed season is tested against the others).
Outputs data/processed/field_v2/fusion_cross_check.csv.gz, fusion_cross_summary.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import fv2
from areas import out_dir
from phase7_step3_s1proxy import neighbours, pixel_groups, prev_next

OUT = os.path.join("data", "processed", "field_v2")
CAP = 3_000_000
N_SEASONS = 2                                                     # set in main()
NAMES = [f"{o}_{f}" for o in fv2.ORBITS for f in fv2.RADAR]


def nearest(dates, d, tol=3):
    k = int(np.argmin(np.abs((dates - d).days)))
    return k if abs((dates[k] - d).days) <= tol else -1


class Season:
    """One area-season: measured values on a date axis (S2, optionally + Landsat LAI), radar, context, hiding."""

    def __init__(self, area, year, target, landsat=False, date_hide=False, hide_share=0.3):
        self.area, self.year, self.target = area, year, target
        px = fv2.pixels(area, year)
        self.px = px
        s = np.load(os.path.join(out_dir(area, year), "s2_pixels_fv2.npz"))
        sd = pd.DatetimeIndex(s["dates"])
        y2 = s[target]
        dates, Y, is_s2 = sd, y2.copy(), np.ones(len(sd), bool)
        if landsat and target == "lai":
            l = np.load(os.path.join(out_dir(area, year), "landsat_pixels.npz"))
            ld = pd.DatetimeIndex(l["dates"])
            dates = sd.union(ld)
            Y = np.full((len(dates), y2.shape[1]), np.nan, np.float32)
            is_s2 = np.isin(dates, sd)
            Y[[dates.get_loc(d) for d in sd]] = y2
            for j, d in enumerate(ld):                        # Landsat fills only where S2 has no value that day
                k = dates.get_loc(d)
                Y[k] = np.where(np.isfinite(Y[k]), Y[k], l["lai"][j])
        self.dates, self.Y, self.is_s2 = dates, Y, is_s2
        # hidden: 30 % of clear S2 values (on the S2-only axis, seed 70), mapped onto this axis
        c2 = np.isfinite(y2)
        if date_hide:                                         # E1b: whole S2 dates hidden (real clouds hide all fields)
            T2 = c2.shape[0]
            pick = np.random.default_rng(70).choice(np.arange(1, T2 - 1), int(round(hide_share * T2)), replace=False)
            h2 = np.zeros_like(c2)
            h2[pick] = c2[pick]
        else:
            h2 = c2 & (np.random.default_rng(70).random(c2.shape) < 0.3)
            h2[np.argmax(c2, axis=0), np.arange(c2.shape[1])] = False
            h2[c2.shape[0] - 1 - np.argmax(c2[::-1], axis=0), np.arange(c2.shape[1])] = False
        self.hide = np.zeros(Y.shape, bool)
        self.hide[[dates.get_loc(d) for d in sd]] = h2
        self.kept = np.isfinite(Y) & ~self.hide
        # gap class by S2-only anchors (identical across E2 / E3)
        pv2, nx2 = prev_next(c2 & ~h2)
        d2 = (sd - sd[0]).days.values
        self.gap_s2 = np.where((pv2 >= 0) & (nx2 >= 0), d2[np.maximum(nx2, 0)] - d2[np.maximum(pv2, 0)], 999).astype(np.int16)
        self.s2_index = np.array([dates.get_loc(d) for d in sd])
        rng = np.random.default_rng(7)
        folds = {f: i % 5 for i, f in enumerate(rng.permutation(px["fids"]))}
        self.fold = np.array([folds[f] for f in px["fid"]])
        self.orb = fv2.radar(area, year, px)
        self.k_s1 = {o: np.array([nearest(self.orb[o][0], d) for d in dates]) for o in fv2.ORBITS}
        dist = fv2.district_of(area, year, px)
        self.ctx, self.rain = fv2.context(dates, year, dist)
        self.days = (dates - dates[0]).days.values

    def radar_at(self, i, idx):
        c = []
        for o in fv2.ORBITS:
            k = self.k_s1[o][i]
            for f in fv2.RADAR:
                c.append(np.full(len(idx), np.nan) if k < 0 else self.orb[o][1][f][k, self.px["inv"][idx]])
        return np.stack(c, 1)

    def base(self, i, idx):
        c = self.ctx.iloc[i]
        return dict(crop=self.px["crop"][idx], dsin=np.sin(2 * np.pi * c["doy"] / 365), dcos=np.cos(2 * np.pi * c["doy"] / 365),
                    since_heading=c["since_heading"])

    def live_rows(self, i, a, idx):
        r = self.radar_at(i, idx) - self.radar_at(a, idx)
        d = dict(self.base(i, idx), days=self.days[i] - self.days[a], y_a=self.Y[a, idx],
                 rain=self.rain[self.dates[i]] - self.rain[self.dates[a]], **{f"d{n}": r[:, k] for k, n in enumerate(NAMES)})
        return pd.DataFrame(d)

    def re_rows(self, i, a, b, idx):
        ra, rb = self.radar_at(i, idx) - self.radar_at(a, idx), self.radar_at(i, idx) - self.radar_at(b, idx)
        pos = (self.days[i] - self.days[a]) / (self.days[b] - self.days[a])
        d = dict(self.base(i, idx), pos=pos, gap=self.days[b] - self.days[a], y_a=self.Y[a, idx], y_b=self.Y[b, idx],
                 rain_a=self.rain[self.dates[i]] - self.rain[self.dates[a]], rain_b=self.rain[self.dates[b]] - self.rain[self.dates[i]],
                 **{f"da{n}": ra[:, k] for k, n in enumerate(NAMES)}, **{f"db{n}": rb[:, k] for k, n in enumerate(NAMES)})
        return pd.DataFrame(d)

    def lin(self, i, a, b, idx):
        return self.Y[a, idx] + (self.days[i] - self.days[a]) / (self.days[b] - self.days[a]) * (self.Y[b, idx] - self.Y[a, idx])

    def training(self, rng, fold_out=None):
        """Live pairs and re-analysis triples from kept values (each pixel's own neighbours); optional fold left out."""
        sel = self.kept if fold_out is None else self.kept & (self.fold != fold_out)[None, :]
        pv, nx = neighbours(sel, 3)
        L, Ly, R, Ry, Lf, Rf = [], [], [], [], [], []
        for i in range(len(self.dates)):
            for s in range(3):
                for (a,), idx in pixel_groups(i, sel[i], (pv[s],), 40000, rng):
                    L.append(self.live_rows(i, a, idx)); Ly.append(self.Y[i, idx] - self.Y[a, idx]); Lf.append(self.fold[idx])
            for s in range(2):
                for u in range(2):
                    for (a, b), idx in pixel_groups(i, sel[i], (pv[s], nx[u]), 30000, rng):
                        R.append(self.re_rows(i, a, b, idx)); Ry.append(self.Y[i, idx] - self.lin(i, a, b, idx)); Rf.append(self.fold[idx])
        def cut(X, y, f, n=None):                              # memory: 3 M rows per model over all training seasons
            n = n or CAP // max(1, N_SEASONS - 1)
            X, y, f = pd.concat(X, ignore_index=True).astype(np.float32), np.concatenate(y), np.concatenate(f)
            if len(X) > n:
                k = np.sort(rng.choice(len(X), n, replace=False))
                X, y, f = X.iloc[k].reset_index(drop=True), y[k], f[k]
            X["crop"] = X["crop"].astype(int)
            return X, y, f
        return (*cut(L, Ly, Lf), *cut(R, Ry, Rf))


VARIANTS = {"V0": [], "V1": ["dsin", "dcos"], "V2": ["dsin", "dcos", "since_heading", "rain", "rain_a", "rain_b"]}


def fit(X, y, rng, v="V0"):
    X = X.drop(columns=[c for c in VARIANTS[v] if c in X])
    if len(X) > CAP:
        k = rng.choice(len(X), CAP, replace=False)
        X, y = X.iloc[k], y[k]
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=0,
                                      categorical_features=[X.columns.get_loc("crop")]).fit(X, y)
    m.drop_ = VARIANTS[v]
    return m


def pred(m, X):
    return m.predict(X.drop(columns=[c for c in m.drop_ if c in X]))


def evaluate(S, mL, mR, mask_fold=None):
    """Predictions for the hidden S2 values of season S (optionally one field fold)."""
    prv, nxt = prev_next(S.kept)
    out = []
    for n, i in enumerate(S.s2_index):
        sel = S.hide[i] & (prv[i] >= 0) & (nxt[i] >= 0)
        if mask_fold is not None:
            sel &= S.fold == mask_fold
        for (a, b), g in pd.DataFrame({"a": prv[i, sel], "b": nxt[i, sel], "p": np.flatnonzero(sel)}).groupby(["a", "b"]):
            idx = g["p"].values
            lin = S.lin(i, a, b, idx)
            out.append(pd.DataFrame({"pixel": idx, "date": S.dates[i], "obs": S.Y[i, idx], "line": lin,
                                     "live": S.Y[a, idx] + pred(mL, S.live_rows(i, a, idx)),
                                     "reanalysis": lin + pred(mR, S.re_rows(i, a, b, idx)),
                                     "gap_s2": S.gap_s2[n, idx], "crop": S.px["grp"][idx]}))
    return pd.concat(out, ignore_index=True)


def part(target, tag, ex):
    """Each finished experiment is saved at once (a stopped run resumes without losing work)."""
    d = os.path.join(OUT, "cross_parts", RUN)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{target}_{tag}_{ex}.csv.gz")


RUN = "run"
VARIANT_ARG = None


def run_experiments(pairs, target, landsat, exps, date_hide=False):
    """exps: list of (name, variant); 'E1' = within-season 5 folds; others = trained on the other seasons."""
    tags = [f"{a}_{y}" for a, y in pairs]
    if all(os.path.exists(part(target, t, ex)) for t in tags for ex, _ in exps):
        return
    seasons = [Season(a, y, target, landsat, date_hide) for a, y in pairs]
    print(f"{time.strftime('%H:%M')} {target} landsat={landsat}: seasons loaded", flush=True)
    rng = np.random.default_rng(7)
    train = [S.training(rng) for S in seasons]
    for k, S in enumerate(seasons):
        tag = tags[k]
        for ex, v in exps:
            if os.path.exists(part(target, tag, ex)):
                continue
            if ex.startswith("E1"):
                XL, yL, fL, XR, yR, fR = train[k]
                e = pd.concat([evaluate(S, fit(XL[fL != f], yL[fL != f], rng, v), fit(XR[fR != f], yR[fR != f], rng, v), mask_fold=f)
                               for f in range(5)], ignore_index=True)
            else:
                others = [train[j] for j in range(len(seasons)) if j != k]
                XL = pd.concat([o[0] for o in others], ignore_index=True); yL = np.concatenate([o[1] for o in others])
                XR = pd.concat([o[3] for o in others], ignore_index=True); yR = np.concatenate([o[4] for o in others])
                e = evaluate(S, fit(XL, yL, rng, v), fit(XR, yR, rng, v))
            tmp = part(target, tag, ex) + ".part"
            e.assign(test=tag, exp=ex, target=target).to_csv(tmp, index=False, compression="gzip")
            os.replace(tmp, part(target, tag, ex))                # only complete files count as done
            print(f"{time.strftime('%H:%M')} {target} {tag} {ex} done", flush=True)


def summarise(pairs, names):
    rm = lambda v: float(np.sqrt(np.mean(np.square(v))))
    rows = []
    for t in ("lai", "ndvi"):
        for a, y in pairs:
            for ex in names:
                f = part(t, f"{a}_{y}", ex)
                if not os.path.exists(f):
                    continue
                g0 = pd.read_csv(f)
                g0["gap"] = pd.cut(g0["gap_s2"], [0, 15, 30, 10000], labels=["<=15 d", "16-30 d", ">30 d"])
                for gc, g in g0.groupby("gap", observed=True):
                    rows.append(dict(target=t, test=f"{a}_{y}", exp=ex, gap=gc, n=len(g), line=rm(g.line - g.obs),
                                     live=rm(g.live - g.obs), reanalysis=rm(g.reanalysis - g.obs)))
    s = pd.DataFrame(rows).round(3)
    s.to_csv(os.path.join(OUT, f"fusion_cross_summary_{RUN}.csv"), index=False)
    pd.set_option("display.width", 200)
    print(s.to_string(index=False))
    return s


def main(pairs, mode="loso"):
    global N_SEASONS, RUN
    N_SEASONS = len(pairs)
    RUN = "_".join(f"{a}{y}" for a, y in pairs)
    if mode == "loso":                                       # E1 (V0) + E2 for V0 / V1 / V2, LAI and NDVI, S2 only
        exps = [("E1", "V0"), ("E2_V0", "V0"), ("E2_V1", "V1"), ("E2_V2", "V2")]
        for t in ("lai", "ndvi"):
            run_experiments(pairs, t, False, exps)
        s = summarise(pairs, [e for e, _ in exps])
        q = s[(s.target == "lai") & (s.gap == ">30 d") & s.exp.str.startswith("E2")].groupby("exp")["reanalysis"].mean()
        best = q.idxmin()
        print(f"mean E2 LAI re-analysis > 30 d by variant: {q.round(3).to_dict()} -> chosen {best}")
        for t, tol in (("lai", 0.05), ("ndvi", 0.01)):
            ok_all = True
            for te in s.test.unique():
                z = s[(s.target == t) & (s.test == te)].set_index(["exp", "gap"])
                ok = all(z.at[(best, g), "reanalysis"] <= z.at[("E1", g), "reanalysis"] + tol for g in ("16-30 d", ">30 d")) and \
                     all(z.at[(best, g), "reanalysis"] < z.at[(best, g), "line"] for g in ("<=15 d", "16-30 d", ">30 d"))
                ok_all &= ok
                print(f"RESULT {te} {t} {best}: {'year-robust' if ok else 'NOT year-robust'}")
            print(f"RESULT {t} {best}: {'YEAR-ROBUST in all test seasons' if ok_all else 'not year-robust in all seasons'}")
        open(os.path.join(OUT, f"chosen_variant_{RUN}.txt"), "w").write(best[3:])
    elif mode == "e1b":                                      # honest within-season reference, whole dates hidden
        v = open(os.path.join(OUT, f"chosen_variant_{RUN}.txt")).read().strip()
        for t in ("lai", "ndvi"):
            run_experiments(pairs, t, False, [(f"E1b_{v}", v), (f"E2b_{v}", v)], date_hide=True)
        summarise(pairs, ["E1", f"E1b_{v}", f"E2_{v}", f"E2b_{v}"])
    elif mode == "choose":                                   # owner rule 2026-10-16: V1 vs V2 on LAI (E3) + NDVI (E2)
        score = {}
        for v in ("V1", "V2"):
            parts = []
            for t, ex in (("lai", f"E3_{v}"), ("ndvi", f"E2_{v}")):
                s = summarise(pairs, [ex])
                s = s[s.target == t]
                parts.append(float((s.reanalysis / s.line).mean()))
            score[v] = (parts[0] + parts[1]) / 2
            print(f"{v}: LAI (E3) {parts[0]:.4f}, NDVI (E2) {parts[1]:.4f} -> score {score[v]:.4f}")
        best = min(score, key=score.get)
        print(f"RESULT chosen variant {best} (scores {score})")
        open(os.path.join(OUT, f"chosen_variant_{RUN}.txt"), "w").write(best)
    else:                                                    # mode 'landsat': E3 with the chosen (or given) variant (LAI)
        v = VARIANT_ARG or open(os.path.join(OUT, f"chosen_variant_{RUN}.txt")).read().strip()
        run_experiments(pairs, "lai", True, [(f"E3_{v}", v)])
        s = summarise(pairs, [f"E2_{v}", f"E3_{v}"])
        rm_all = {}
        for te in s.test.unique():
            for ex in (f"E2_{v}", f"E3_{v}"):
                g = pd.read_csv(part("lai", te, ex))
                rm_all[(te, ex)] = float(np.sqrt(((g.reanalysis - g.obs) ** 2).mean()))
        ok = all(rm_all[(te, f"E3_{v}")] < rm_all[(te, f"E2_{v}")] for te in s.test.unique())
        print({k: round(x, 3) for k, x in rm_all.items()})
        print(f"RESULT Landsat: {'ADOPTED' if ok else 'not adopted'} (E3 < E2 in every test season: {ok})")


if __name__ == "__main__":
    a = sys.argv[1:]
    mode = a.pop(0) if a and a[0] in ("loso", "landsat", "e1b", "choose") else "loso"
    VARIANT_ARG = a.pop(0) if a and a[0] in VARIANTS else None
    main([(a[i], int(a[i + 1])) for i in range(0, len(a), 2)], mode)
