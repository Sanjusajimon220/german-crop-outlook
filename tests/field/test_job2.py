import datetime as dt

import numpy as np
import pandas as pd

from phase22_wofost_field import enkf_run
from phase24_job2 import split_weeks, summarise, verdict


class FakeModel:
    """LAI grows by 1 per day from day 0; the Kalman update may reset it."""
    def __init__(self, start, lai0):
        self.day, self.lai, self.out = start, lai0, []

    def run_till(self, day):
        while self.day < day:
            self.day += dt.timedelta(days=1)
            self.lai += 1.0
            self.out.append({"day": self.day, "LAI": self.lai})

    def get_variable(self, name):
        return self.lai

    def set_variable(self, name, value):
        self.lai = value

    def run_till_terminate(self):
        self.run_till(self.day + dt.timedelta(days=21))

    def get_output(self):
        return self.out

    def get_summary_output(self):
        return [{}]


def test_enkf_run_pulls_members_to_the_observation_then_runs_freely():
    start = dt.date(2024, 4, 1)
    models = [FakeModel(start, v) for v in np.linspace(0.0, 4.0, 50)]       # LAI 0-4 on 1 April
    weeks = np.array(["2024-04-15", "2024-04-22"], dtype="datetime64[D]")
    rng = np.random.default_rng(0)
    out = enkf_run(models, [dt.date(2024, 4, 8)], np.array([20.0]), np.array([0.0]), weeks, rng)
    assert out.shape == (50, 2)
    # prior mean 9 on 8 April (spread ~1.2), observation 20 (error 0.5) -> about 85 % of the way, + 7 days of growth
    assert 24.0 < out[:, 0].mean() < 26.5
    assert out[:, 0].std() < 1.0                    # spread shrinks
    assert np.allclose(out[:, 1] - out[:, 0], 7.0)  # then free growth of 1 per day


def test_split_weeks_uses_measured_weeks_before_the_cut_and_targets_after():
    weeks = np.array(pd.date_range("2024-01-01", periods=12, freq="W-MON").values, dtype="datetime64[D]")
    share = np.array([1, 1, 1, 0.2, 1, 1, 0.6, 1, 0.4, 1, 1, 1], float)
    mask = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], bool)
    used, target = split_weeks(weeks, share, mask, np.datetime64("2024-02-19"))
    # January weeks (1, 8, 15, 22, 29 Jan) never used; 5 and 12 Feb measured; cut 19 Feb
    assert np.flatnonzero(used).tolist() == [5, 6]
    assert np.flatnonzero(target).tolist() == [7, 9, 10]   # 0.4 share not measured; outside the season not a target


def test_summarise_and_verdict_apply_the_rule_per_crop():
    rows = []
    for cut, d_err in (("04-01", 0.1), ("05-01", 0.4), ("06-01", 0.2)):
        for obs in (2.0, 4.0):
            rows.append(dict(crop="wheat", cut=cut, obs=obs, A=obs + 0.5, B=obs + 0.6, C=obs + 0.3, D=obs + d_err))
    s = summarise(pd.DataFrame(rows), ["A", "B", "C", "D"])
    assert set(s["cut"]) == {"04-01", "05-01", "06-01"}
    assert abs(s.set_index("cut").at["05-01", "rmse_D"] - 0.4) < 1e-9
    v = verdict(s)
    assert v["wheat"]["wins"] == 2 and v["wheat"]["bias_0601"] == 0.2 and v["wheat"]["pass"]
    bad = summarise(pd.DataFrame(rows).assign(C=lambda x: x["obs"] + 0.55, D=lambda x: x["obs"] + 0.35), ["A", "B", "C", "D"])
    vb = verdict(bad)["wheat"]
    assert vb["wins"] == 3 and not vb["pass"]      # D best at every cut but |bias| at 1 June 0.35 > 0.3


def test_scenarios_report_every_level_of_every_dimension():
    from phase24_job2 import EXTRA, RULE, SCENARIOS, scenarios
    r = pd.DataFrame(dict(crop="wheat", cut=["04-01", "05-01"] * 2, obs=1.0, **{m: 1.5 for m in RULE + EXTRA},
                          season=["2024", "2024", "2025", "2025"], area="north", soil=["low", "high"] * 2,
                          spring_rain="drier", size="< 5 ha", distance="1-4 weeks"))
    s = scenarios(r)
    assert set(s.scenario) == set(SCENARIOS)
    assert set(s[s.scenario == "season"].level) == {"2024", "2025"}
    assert (s.rmse_D == 0.5).all()


def test_claim_against_wofost_needs_c_e1_and_e2_beaten():
    rows = []
    for cut in ("04-01", "05-01", "06-01"):
        for obs in (2.0, 4.0):
            rows.append(dict(crop="wheat", cut=cut, obs=obs, A=obs + 0.5, B=obs + 0.6, C=obs + 0.4, D=obs + 0.2,
                             E1=obs + 0.7, E2=obs + (0.1 if cut != "04-01" else 0.5)))
    v = verdict(summarise(pd.DataFrame(rows), ["A", "B", "C", "D", "E1", "E2"]))["wheat"]
    assert v["pass"] and v["wins_wofost"] == 1 and not v["claim_wofost"]   # E2 better at 2 cuts
    rows2 = [dict(r, E2=r["obs"] + 0.5) for r in rows]
    v2 = verdict(summarise(pd.DataFrame(rows2), ["A", "B", "C", "D", "E1", "E2"]))["wheat"]
    assert v2["wins_wofost"] == 3 and v2["claim_wofost"]
