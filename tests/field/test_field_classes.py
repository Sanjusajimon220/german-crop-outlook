import numpy as np

from phase20_field_classes import area_relative, loso_rank_check, pixel_class, pixel_variability, tercile_class, variability_class


def test_area_relative_divides_by_same_crop_median():
    s = np.array([2.0, 4.0, 6.0, 10.0, 20.0, np.nan])
    g = np.array(["wheat", "wheat", "wheat", "maize", "maize", "wheat"], dtype=object)
    r = area_relative(s, g, {"wheat", "maize"})
    assert np.allclose(r[:5], [0.5, 1.0, 1.5, 10 / 15, 20 / 15]) and np.isnan(r[5])


def test_area_relative_ignores_non_arable_groups():
    r = area_relative(np.array([1.0, 2.0]), np.array(["grass / fodder", "wheat"], dtype=object), {"wheat"})
    assert np.isnan(r[0]) and r[1] == 1.0


def test_tercile_class_low_average_high():
    v = {1: 0.8, 2: 0.9, 3: 1.0, 4: 1.1, 5: 1.2, 6: 1.3}
    c = tercile_class(v)
    assert [c[k] for k in range(1, 7)] == ["low", "low", "average", "average", "high", "high"]


def test_variability_class_rules():
    assert variability_class({"spread": 0.03, "held": 5, "tests": 5}) == "uniform"
    assert variability_class({"spread": 0.08, "held": 4, "tests": 5}) == "variable, reliable zones"
    assert variability_class({"spread": 0.08, "held": 3, "tests": 5}) == "variable, less stable zones"
    assert variability_class(None) == "not assessed"
    assert variability_class({"spread": None, "held": 0, "tests": 0}) == "not assessed"


def test_loso_rank_check_perfectly_consistent_fields():
    m = np.array([[1.0, 1.1, 1.2], [0.8, 0.9, 1.0], [1.3, 1.2, 1.4], [0.7, 0.8, 0.75]])   # fields x seasons
    out = loso_rank_check(m)
    assert len(out) == 3 and all(r > 0.7 for r in out.values())


def test_pixel_variability_spread_and_stability():
    base = np.array([0.9, 0.95, 1.0, 1.05, 1.1])
    m = np.vstack([base, base * 1.0 + 0.0, base])                      # seasons x pixels: same pattern every season
    spread, stab = pixel_variability(m)
    assert abs(spread - np.std(base)) < 1e-9 and stab > 0.99
    noisy = np.array([[0.9, 1.1, 1.0, 0.95, 1.05], [1.1, 0.9, 1.05, 1.0, 0.95], [1.0, 1.0, 0.9, 1.1, 1.0]])
    assert pixel_variability(noisy)[1] < 0.5                            # the pattern does not repeat


def test_pixel_variability_needs_three_seasons():
    assert pixel_variability(np.array([[1.0, 1.1], [1.0, 1.1]])) == (None, None)


def test_pixel_class_rules():
    assert pixel_class(0.02, 0.9) == "uniform"
    assert pixel_class(0.04, 0.6) == "variable, stable pattern"
    assert pixel_class(0.04, 0.3) == "variable, less stable pattern"
    assert pixel_class(None, None) == "not assessed"
