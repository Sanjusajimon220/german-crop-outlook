"""Field Outlook v1, CHECK 3 (pre-registered, docs/project_log.md): does the satellite field pattern beat a flat
pattern on measured field yields (YieldSAT wheat, research data)? Level = measured group mean (oracle), so only the
PATTERN is tested. B (slope) estimated leaving the tested farm out."""
import numpy as np
import pandas as pd

CLIP = (0.7, 1.3)
f = pd.read_csv("data/processed/yieldsat/v6_features.csv")
f["idx"] = f["glad_after"]
f["src"] = np.where(f["glad_after"].notna(), "glad_after", "lai_peak_obs")
f.loc[f["idx"].isna(), "idx"] = f.loc[f["idx"].isna(), "lai_peak_obs"]
f = f.dropna(subset=["idx", "truth"])
f = f[f.groupby("dy")["field"].transform("count") >= 2].copy()
f["level"] = f.groupby("dy")["truth"].transform("mean")
f["rg"] = f["idx"] / f.groupby(["dy", "src"])["idx"].transform("mean")
f["ry"] = f["truth"] / f["level"]
out = []
for farm in sorted(f["farm"].unique()):
    tr, te = f[f["farm"] != farm], f[f["farm"] == farm].copy()
    b = ((tr["rg"] - 1) * (tr["ry"] - 1)).sum() / ((tr["rg"] - 1) ** 2).sum()
    rel = np.clip(1 + b * (te["rg"] - 1), *CLIP)
    te["rel"] = rel / rel.groupby(te["dy"]).transform("mean")
    te["pred"], te["b"] = te["level"] * te["rel"], b
    out.append(te)
o = pd.concat(out)
rm = lambda e: float(np.sqrt(np.mean(e ** 2)))
print(f"fields {len(o)}, district-years {o['dy'].nunique()}, farms {o['farm'].nunique()}")
print(o.groupby("farm").agg(n=("field", "size"), b=("b", "first"),
      rmse_flat=("truth", lambda s: rm(s - o.loc[s.index, "level"])),
      rmse_pattern=("truth", lambda s: rm(s - o.loc[s.index, "pred"]))).round(3).to_string())
wr = o.groupby("dy").apply(lambda g: g["pred"].corr(g["truth"]) if len(g) >= 3 else np.nan, include_groups=False).dropna()
flat, pat = rm(o["truth"] - o["level"]), rm(o["truth"] - o["pred"])
print(f"POOLED leave-one-farm-out: RMSE flat {flat:.3f} t/ha, pattern {pat:.3f} t/ha ({100 * (pat / flat - 1):+.1f} %); "
      f"within-group r median {wr.median():.2f} (mean {wr.mean():.2f}, {len(wr)} groups >= 3 fields)")
print("RESULT:", "pattern helps" if pat < flat and wr.median() > 0 else "pattern does NOT help (rule not met)")
o.to_csv("data/processed/field_outlook/check_yieldsat.csv", index=False)
