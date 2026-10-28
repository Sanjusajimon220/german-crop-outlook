import numpy as np
import pandas as pd

from phase23_ours_field import nfk_ratio, weekly_from_daily


def test_weekly_from_daily_counts_days_from_sowing_and_zero_outside():
    lai = np.arange(20, dtype=float)[None, :].repeat(2, 0)          # day k after sowing -> LAI k
    weeks = np.array(["2023-10-02", "2023-10-09", "2023-10-16", "2023-10-23"], dtype="datetime64[D]")
    out = weekly_from_daily(lai, pd.Timestamp("2023-10-05"), weeks)
    assert out.shape == (2, 4)
    assert out[0].tolist() == [0.0, 4.0, 11.0, 18.0]                 # before sowing 0; day 4, 11, 18
    short = weekly_from_daily(lai[:, :10], pd.Timestamp("2023-10-05"), weeks)
    assert short[0].tolist() == [0.0, 4.0, 0.0, 0.0]                 # beyond the model horizon 0


def test_nfk_ratio_relative_to_district_median_of_the_same_crop():
    s = pd.DataFrame({"ID": [1, 2, 3, 4, 5], "VALIDFROM": 2024, "CODE": [115, 115, 115, 131, 131],
                      "district": ["a", "a", "a", "a", "a"], "nfk_mm": [100.0, 200.0, 300.0, 50.0, 150.0],
                      "share_mapped": [1.0, 1.0, 1.0, 1.0, 0.2]})
    r = nfk_ratio(s, 115)
    assert r[(1, 2024)] == 0.5 and r[(2, 2024)] == 1.0 and r[(3, 2024)] == 1.5
    b = nfk_ratio(s, 131)
    assert list(b.index) == [(4, 2024)] and b[(4, 2024)] == 1.0      # poorly mapped field 5 left out


def _toy_drivers(n=2, days=10):
    import torch
    from phase2_growth import LAYERS
    o, z = torch.ones(n, days), torch.zeros(n, days)
    return {"tmin": o * 8, "tmax": o * 18, "em": o, "ph": o * 0.5, "post": z, "gf": z, "mat": z, "et0": o * 2,
            "pr": o * 2, "par": o * 8, "cap": torch.full((n, LAYERS), 30.0), "year": torch.full((n,), 2024.0),
            "n_supply": torch.full((n,), 150.0)}


def test_growth_lai_hook_off_by_default_and_resets_lai_on_its_day():
    import torch
    from phase2_growth import Growth
    d = _toy_drivers()
    g = Growth()
    _, base = g(d, 0.86, record=True)
    g.lai_hook = lambda day, lai: torch.full_like(lai, 5.0) if day == 4 else lai
    _, hooked = g(d, 0.86, record=True)
    a, b = base["lai_daily"], hooked["lai_daily"]
    assert torch.equal(a[:, :4], b[:, :4])                     # unchanged before the update day
    assert torch.allclose(b[:, 4], torch.tensor(5.0))          # set on the update day
    assert (b[:, 5:] > a[:, 5:]).all()                         # and grows on from there
    g.lai_hook = None
    _, again = g(d, 0.86, record=True)
    assert torch.equal(again["lai_daily"], a)


def test_enkf_hook_updates_only_its_cut_block_on_its_days():
    import torch
    from phase23_ours_field import enkf_hook
    rng = np.random.default_rng(0)
    hook = enkf_hook([{3: (5.0, 0.25)}, {}], n=50, rng=rng)    # block 0 (first cut) observed on day 3, block 1 not
    lai = torch.tensor(np.r_[np.linspace(1, 3, 50), np.linspace(1, 3, 50)], dtype=torch.float32)
    assert torch.equal(hook(2, lai), lai)                      # no observation that day
    out = hook(3, lai)
    assert torch.equal(out[50:], lai[50:])                     # other cut untouched
    assert out[:50].mean() > 3.5                               # pulled from 2 towards 5
