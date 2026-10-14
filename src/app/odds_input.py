"""Parsing and summarizing the moneyline inputs of a slate block."""
from __future__ import annotations

import re

from src.engine.odds import devig_moneylines, map_prob_from_series, series_win_prob


def parse_american(text) -> float | None:
    """'-150' / '+130' / '130' / 'EV' / 'even' -> float; '' / None -> None.

    Raises ValueError for anything that is not valid American odds (|ml| >= 100).
    """
    if text is None:
        return None
    if isinstance(text, (int, float)):
        val = float(text)
    else:
        s = str(text).strip().lower().replace("−", "-").replace(" ", "")
        if s == "":
            return None
        if s in ("ev", "even", "evens", "pk", "pick"):
            return 100.0
        if not re.fullmatch(r"[+-]?\d+(\.\d+)?", s):
            raise ValueError(f"not American odds: {text!r}")
        val = float(s)
    if -100 < val < 100:
        raise ValueError(f"American odds must be <= -100 or >= +100, got {text!r}")
    return val


def odds_summary(ml_a, ml_b, best_of: int, map1_a=None, map1_b=None) -> dict:
    """De-vigged series and per-map win % for team A (and B) from text or numbers.

    The Map 1 moneyline is used for p_map only when both sides are given. Returns
    {p_series_a, p_series_b, p_map_a, p_map_b, source, error}.
    """
    out = {"p_series_a": None, "p_series_b": None, "p_map_a": None, "p_map_b": None,
           "source": None, "error": None}
    try:
        a, b = parse_american(ml_a), parse_american(ml_b)
        m1a, m1b = parse_american(map1_a), parse_american(map1_b)
    except ValueError as exc:
        out["error"] = str(exc)
        return out
    if (m1a is None) != (m1b is None):
        out["error"] = "enter the Map 1 moneyline for both teams (or neither)"
        return out
    if a is None or b is None:
        if m1a is None:
            out["error"] = "enter the series moneyline for both teams"
            return out
    if a is not None and b is not None:
        ps, _ = devig_moneylines(a, b)
        out["p_series_a"], out["p_series_b"] = ps, 1 - ps
        out["p_map_a"], out["source"] = map_prob_from_series(ps, best_of), "series"
    if m1a is not None:
        pm, _ = devig_moneylines(m1a, m1b)
        out["p_map_a"], out["source"] = pm, "map1"
        if out["p_series_a"] is None:  # series price implied by the map price
            ps = float(series_win_prob(pm, best_of))
            out["p_series_a"], out["p_series_b"] = ps, 1 - ps
    out["p_map_b"] = 1 - out["p_map_a"]
    return out
