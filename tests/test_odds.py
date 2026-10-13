import numpy as np
import pytest

from src.engine.odds import (american_to_implied, devig_moneylines, devig_power,
                             map_played_prob, map_prob_from_series, p_map_from_odds,
                             series_win_prob)


def test_american_to_implied():
    assert american_to_implied(-150) == pytest.approx(0.6)
    assert american_to_implied(+150) == pytest.approx(0.4)
    assert american_to_implied(100) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        american_to_implied(50)


def test_power_devig_sums_to_one_and_matches_power_form():
    imp = np.array([american_to_implied(-200), american_to_implied(+170)])
    fair = devig_power(imp)
    assert fair.sum() == pytest.approx(1.0, abs=1e-9)
    # power method: fair_i = imp_i ** k for one common k
    k = np.log(fair) / np.log(imp)
    assert k[0] == pytest.approx(k[1], rel=1e-6)
    assert k[0] > 1  # removing overround -> exponent > 1
    # favourite keeps (relatively) more than multiplicative de-vig would give it
    assert fair[0] > imp[0] / imp.sum()


def test_devig_symmetric_and_no_vig_passthrough():
    a, b = devig_moneylines(-110, -110)
    assert a == pytest.approx(0.5) and b == pytest.approx(0.5)
    fair = devig_power([0.3, 0.7])
    assert fair == pytest.approx([0.3, 0.7])


@pytest.mark.parametrize("bo", [1, 3, 5])
@pytest.mark.parametrize("p", [0.2, 0.5, 0.63, 0.9])
def test_series_inversion_roundtrip(bo, p):
    assert map_prob_from_series(series_win_prob(p, bo), bo) == pytest.approx(p, abs=1e-9)


def test_series_formulas():
    p = 0.6
    assert series_win_prob(p, 3) == pytest.approx(p**2 + 2 * p**2 * (1 - p))
    # BO5: win 3-0, 3-1, 3-2
    q = 1 - p
    assert series_win_prob(p, 5) == pytest.approx(p**3 * (1 + 3 * q + 6 * q**2))
    assert series_win_prob(0.5, 5) == pytest.approx(0.5)


def test_map_ml_overrides_series():
    p_series_only = p_map_from_odds(-300, +240, 3)
    p_with_map = p_map_from_odds(-300, +240, 3, map1_ml=(-140, +120))
    assert p_with_map == pytest.approx(devig_moneylines(-140, 120)[0])
    assert p_series_only != pytest.approx(p_with_map)
    assert 0.5 < p_series_only < devig_moneylines(-300, 240)[0]


def test_map3_occurrence_probability():
    for p in (0.5, 0.7, 0.9):
        assert map_played_prob(p, 3, 3) == pytest.approx(2 * p * (1 - p))
        assert map_played_prob(p, 3, 2) == 1.0
    p = 0.6
    q = 1 - p
    assert map_played_prob(p, 5, 4) == pytest.approx(1 - p**3 - q**3)
    assert map_played_prob(p, 5, 5) == pytest.approx(6 * p**2 * q**2)
