"""v12 FINAL (2026-10-11): v11.1 release (v10 medians + state-year ranges, 2018-2026) + the range scale factors
adopted by the pre-registered training-year test (ranges_scale.py -> ranges_scale_k.csv, adopt = True):
winter barley district and national. The range half-widths are multiplied by k around the median.
S2 parts of v12 (A) were stopped by the owner (option A, 2026-10-11 01:00) after the 2024 test and the
classification check failed; v12 has no S2 correction.
Output data/processed/ranges_v12_final/, check2026/v12_2026_frozen.csv + md5.
"""
import glob
import hashlib
import os

import pandas as pd

P = os.path.join("data", "processed")
SRC, OUT = os.path.join(P, "ranges_v11i_stateyr_final"), os.path.join(P, "ranges_v12_final")


def main():
    os.makedirs(OUT, exist_ok=True)
    k = pd.read_csv(os.path.join(P, "ranges_scale_k.csv"))
    k = k[k["adopt"]].set_index(["crop", "level"])["k_all"]
    print("adopted scale factors:", k.to_dict())
    for f in sorted(glob.glob(os.path.join(SRC, "ranges_*.csv"))):
        crop = os.path.basename(f)[len("ranges_"):-len("_leadNN.csv")]
        r = pd.read_csv(f, dtype={"region": str})
        r["k"] = [k.get((crop, lvl), 1.0) for lvl in r["level"]]
        r["low80"] = r["median"] - r["k"] * (r["median"] - r["low80"])
        r["high80"] = r["median"] + r["k"] * (r["high80"] - r["median"])
        r.drop(columns="k").to_csv(os.path.join(OUT, os.path.basename(f)), index=False)
    z = pd.concat([pd.read_csv(f, dtype={"region": str}).assign(file=os.path.basename(f))
                   for f in sorted(glob.glob(os.path.join(OUT, "ranges_*.csv")))])
    z26 = z[z["year"] == 2026]
    p = os.path.join(P, "check2026", "v12_2026_frozen.csv")
    z26.to_csv(p, index=False)
    md5 = hashlib.md5(open(p, "rb").read()).hexdigest()
    open(os.path.join(P, "check2026", "frozen_2026_v12_md5.txt"), "w").write(md5 + "  v12_2026_frozen.csv\n")
    print("2026 frozen:", len(z26), "rows, md5", md5)


if __name__ == "__main__":
    main()
