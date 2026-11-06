"""Water v6 stress side (pre-registration 2026-11-06): root-depth scale, depletion-fraction shift, choice, test rule."""
import numpy as np
import pytest

import phase39_water_v6 as V


def test_root_depth_scaled_and_capped():
    frac = np.array([0.0, 0.5, 1.0])
    assert np.allclose(V.root_depth(0.5, 1.0, frac), [0.1, 0.3, 0.5])
    assert np.allclose(V.root_depth(0.5, 2.0, frac), [0.1, 0.55, 1.0])
    assert V.root_depth(1.5, 2.0, frac)[-1] == pytest.approx(2.0)        # capped at the 2 m profile


def test_shifted_p_capped():
    assert V.shifted_p(0.35, 0.0) == pytest.approx(0.35)
    assert V.shifted_p(0.35, 0.2) == pytest.approx(0.55)
    assert V.shifted_p(0.55, 0.30) == pytest.approx(0.80)
    assert V.shifted_p(0.65, 0.30) == pytest.approx(0.80)


def test_choose_ties_within_one_percent_go_closest_to_v5():
    scores = {(1.0, 0.0): 6.0, (1.5, 0.10): 5.00, (1.2, 0.05): 5.03, (2.0, 0.30): 4.99}
    assert V.choose(scores) == (1.2, 0.05)
    assert V.choose({(1.0, 0.0): 5.0, (1.5, 0.1): 4.0}) == (1.5, 0.1)


def test_stressed_log_ratio_uses_only_weeks_below_threshold():
    m, t, ks = np.array([5.0, 10.0, 20.0]), np.array([10.0, 10.0, 10.0]), np.array([0.5, 0.79, 0.9])
    lg, n = V.stressed_log(m, t, ks)
    assert n == 2 and lg == pytest.approx(np.mean(np.log([0.5, 1.0])))


def test_passes_rule():
    ok = dict(rmse6=5.0, rmse5=5.5, log6=0.05, s6=-0.1, s5=-0.3, n_stress=25)
    assert V.verdict(**ok) == "PASS"
    assert V.verdict(**ok | dict(rmse6=5.3)) == "FAIL"                     # 5.3 > 0.95 x 5.5
    assert V.verdict(**ok | dict(log6=np.log(1.12))) == "FAIL"
    assert V.verdict(**ok | dict(s6=0.35)) == "FAIL"                       # |0.35| not < |-0.3|
    assert V.verdict(**ok | dict(n_stress=19)) == "NOT TESTABLE"
