import numpy as np
import pytest

from fv2_weekly import FIX2_TARGETS, drop_flagged


def test_flagged_values_become_cloudy():
    Y = np.array([[1.0, 2.0], [0.2, 3.0]], np.float32)
    out = drop_flagged(Y, np.array([[False, False], [True, False]]))
    assert np.isnan(out[1, 0]) and out[1, 1] == 3.0 and Y[1, 0] == np.float32(0.2)   # input left unchanged


def test_flags_must_match_the_season_grid():
    with pytest.raises(ValueError):
        drop_flagged(np.zeros((2, 2), np.float32), np.zeros((3, 2), bool))


def test_fix2_only_for_lai():
    assert FIX2_TARGETS == {"lai"}
