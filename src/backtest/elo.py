"""Simple map-level Elo, trained walk-forward, used as the backtest's p_map source."""
from __future__ import annotations

from src import config as C


class Elo:
    def __init__(self, k: float = C.ELO_K, league_init: dict | None = None,
                 default: float = C.ELO_DEFAULT_INIT):
        self.k = k
        self.league_init = league_init or C.ELO_LEAGUE_INIT
        self.default = default
        self.r: dict[str, float] = {}

    def rating(self, team: str, league: str | None = None) -> float:
        if team not in self.r:
            self.r[team] = float(self.league_init.get(league, self.default))
        return self.r[team]

    def p_map(self, a: str, b: str, league: str | None = None) -> float:
        ra, rb = self.rating(a, league), self.rating(b, league)
        return 1.0 / (1.0 + 10 ** (-(ra - rb) / 400.0))

    def update_map(self, a: str, b: str, a_won: bool, league: str | None = None) -> None:
        p = self.p_map(a, b, league)
        d = self.k * ((1.0 if a_won else 0.0) - p)
        self.r[a] += d
        self.r[b] -= d
