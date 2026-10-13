"""Prop outputs, payout breakevens/edges and slip joint probabilities."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import PRIZEPICKS_POWER, UNDERDOG
from src.engine.sim import SeriesSim


def breakeven(payout: float, n_legs: int) -> float:
    """Per-leg win probability at which an all-or-nothing n-leg slip is break-even."""
    return (1.0 / payout) ** (1.0 / n_legs)


BREAKEVENS = {
    "prizepicks_power": {n: breakeven(m, n) for n, m in PRIZEPICKS_POWER.items()},
    "underdog": {n: breakeven(m, n) for n, m in UNDERDOG.items()},
}


def fair_line(kills: np.ndarray) -> float:
    """Half-point line where P(over) is closest to 50%."""
    med = float(np.median(kills))
    cands = np.arange(max(0.5, np.floor(med) - 2.5), med + 3.0, 1.0)
    p_over = np.array([(kills > c).mean() for c in cands])
    return float(cands[np.argmin(np.abs(p_over - 0.5))])


def prop_summary(kills: np.ndarray, line: float) -> dict:
    """P(over), P(under), P(push) counted separately, plus distribution stats and edges."""
    n = len(kills)
    over = int(np.count_nonzero(kills > line))
    under = int(np.count_nonzero(kills < line))
    push = n - over - under
    out = {
        "line": line,
        "p_over": over / n, "p_under": under / n, "p_push": push / n,
        "mean": float(kills.mean()), "median": float(np.median(kills)),
        "fair_line": fair_line(kills),
    }
    # Edge on each side vs per-leg breakeven. A push is treated as a loss here (it never
    # wins an all-or-nothing leg); books that reduce the slip on push make this conservative.
    for book, table in BREAKEVENS.items():
        for legs, be in table.items():
            out[f"edge_over_{book}_{legs}L"] = out["p_over"] - be
            out[f"edge_under_{book}_{legs}L"] = out["p_under"] - be
    return out


def project_player(sim: SeriesSim, player: str, market: str, line: float) -> dict:
    return {"player": player, "market": market, **prop_summary(sim.range_kills(player, market), line)}


@dataclass
class Leg:
    matchup: str          # key into the sims dict
    player: str
    market: str
    line: float
    side: str             # "over" | "under"


def leg_hits(sim: SeriesSim, leg: Leg) -> np.ndarray:
    k = sim.range_kills(leg.player, leg.market)
    if leg.side == "over":
        return k > leg.line
    if leg.side == "under":
        return k < leg.line
    raise ValueError(leg.side)


def slip_probability(sims: dict[str, SeriesSim], legs: list[Leg]) -> dict:
    """Joint P(all legs hit).

    Legs in the same matchup are evaluated on the shared simulation arrays (so team-level
    correlation is preserved); independent matchups are multiplied.
    """
    by_matchup: dict[str, list[Leg]] = {}
    for leg in legs:
        by_matchup.setdefault(leg.matchup, []).append(leg)
    p_all, naive = 1.0, 1.0
    for key, group in by_matchup.items():
        sim = sims[key]
        joint = np.ones(sim.n_sims, dtype=bool)
        for leg in group:
            h = leg_hits(sim, leg)
            joint &= h
            naive *= h.mean()
        p_all *= joint.mean()
    n = len(legs)
    out = {"n_legs": n, "p_all_hit": p_all, "p_independent": naive}
    for book, pay in (("prizepicks_power", PRIZEPICKS_POWER), ("underdog", UNDERDOG)):
        if n in pay:
            out[f"ev_{book}"] = p_all * pay[n] - 1.0
    return out
