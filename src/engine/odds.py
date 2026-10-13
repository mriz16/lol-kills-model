"""Moneyline handling: American odds, power-method de-vig, series -> map inversion."""
from __future__ import annotations

import numpy as np


def american_to_implied(ml: float) -> float:
    """Implied probability (with vig) from American odds."""
    ml = float(ml)
    if ml == 0 or -100 < ml < 100:
        raise ValueError(f"invalid American odds: {ml}")
    return 100.0 / (ml + 100.0) if ml > 0 else -ml / (-ml + 100.0)


def devig_power(implied: list[float] | np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Power method: find k with sum(p_i ** k) == 1; fair p_i = p_i ** k."""
    p = np.asarray(implied, dtype=float)
    if np.any(p <= 0) or np.any(p >= 1):
        raise ValueError("implied probabilities must be in (0, 1)")
    if abs(p.sum() - 1.0) < tol:
        return p.copy()
    lo, hi = 0.01, 100.0  # sum(p**k) is decreasing in k
    for _ in range(200):
        k = 0.5 * (lo + hi)
        s = np.sum(p ** k)
        if s > 1.0:
            lo = k
        else:
            hi = k
        if hi - lo < tol:
            break
    fair = p ** (0.5 * (lo + hi))
    return fair / fair.sum()


def devig_moneylines(ml_a: float, ml_b: float) -> tuple[float, float]:
    fair = devig_power([american_to_implied(ml_a), american_to_implied(ml_b)])
    return float(fair[0]), float(fair[1])


def series_win_prob(p: float | np.ndarray, best_of: int):
    """P(win series) given independent per-map win prob p."""
    p = np.asarray(p, dtype=float)
    if best_of == 1:
        out = p
    elif best_of == 3:
        out = p ** 2 * (3 - 2 * p)
    elif best_of == 5:
        out = p ** 3 * (10 - 15 * p + 6 * p ** 2)
    else:
        raise ValueError(f"unsupported best_of {best_of}")
    return out if out.ndim else float(out)


def map_prob_from_series(p_series: float, best_of: int, tol: float = 1e-12) -> float:
    """Invert series_win_prob (monotone in p) by bisection."""
    if not 0.0 <= p_series <= 1.0:
        raise ValueError("p_series must be in [0, 1]")
    if best_of == 1:
        return float(p_series)
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if series_win_prob(mid, best_of) < p_series:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def p_map_from_odds(series_ml_a: float, series_ml_b: float, best_of: int,
                    map1_ml: tuple[float, float] | None = None) -> float:
    """Per-map win probability for team A.

    If a Map 1 moneyline pair (A, B) is given it is used directly (de-vigged); otherwise
    the de-vigged series price is inverted.
    """
    if map1_ml is not None:
        return devig_moneylines(*map1_ml)[0]
    p_series, _ = devig_moneylines(series_ml_a, series_ml_b)
    return map_prob_from_series(p_series, best_of)


def map_played_prob(p: float, best_of: int, map_no: int) -> float:
    """Probability that map `map_no` is played in a best_of series with per-map p."""
    need = best_of // 2 + 1
    if map_no <= need:
        return 1.0
    if map_no > best_of:
        return 0.0
    # Map m is played iff after m-1 maps neither side has `need` wins.
    from math import comb
    n = map_no - 1
    q = 1 - p
    return float(sum(comb(n, k) * p ** k * q ** (n - k)
                     for k in range(n + 1) if k < need and n - k < need))
