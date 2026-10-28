"""Vectorized Monte Carlo series simulator."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from src.config import N_SIMS, POSITIONS
from src.engine.calibration import inflate_samples
from src.engine.features import MatchupParams


@dataclass
class SeriesSim:
    params: MatchupParams
    a_wins: np.ndarray        # (S, M) bool, team A won map m (only meaningful if played)
    played: np.ndarray        # (S, M) bool
    team_kills: np.ndarray    # (S, M, 2) int, zero on unplayed maps
    player_kills: np.ndarray  # (S, M, 2, 5) int, zero on unplayed maps
    # S (locked 2026-10-14): {(position, market): k, "_pool": k}; None = no inflation.
    # range_kills returns mean + k (x - mean) per sim (calibration.inflate_samples) with a
    # fixed seed per (player slot, market), so slips see the same inflated arrays.
    spread_k: dict | None = None

    @property
    def n_sims(self) -> int:
        return self.played.shape[0]

    def player_index(self, player: str) -> tuple[int, int]:
        for s in (0, 1):
            if player in self.params.players[s]:
                return s, self.params.players[s].index(player)
        raise KeyError(player)

    def range_kills(self, player: str, market: str) -> np.ndarray:
        """Player kills over a market range; only maps actually played count."""
        s, j = self.player_index(player)
        lo, hi = parse_market(market, self.played.shape[1])
        k = self.player_kills[:, lo - 1:hi, s, j].sum(axis=1)
        if self.spread_k:
            kk = self.spread_k.get((POSITIONS[j], market), self.spread_k.get("_pool", 1.0))
            k = inflate_samples(k, float(kk), rng=1000 * (5 * s + j) + 10 * lo + hi)
        return k

    def team_range_kills(self, side: int, market: str) -> np.ndarray:
        lo, hi = parse_market(market, self.played.shape[1])
        return self.team_kills[:, lo - 1:hi, side].sum(axis=1)


def parse_market(market: str, max_maps: int) -> tuple[int, int]:
    """'Maps 1-2' -> (1, 2); 'Map 3' -> (3, 3). Clipped to the series length."""
    m = re.fullmatch(r"\s*maps?\s*(\d+)\s*(?:-\s*(\d+))?\s*", market, flags=re.I)
    if not m:
        raise ValueError(f"unknown market {market!r}")
    lo = int(m.group(1))
    hi = int(m.group(2) or lo)
    if lo < 1 or hi < lo:
        raise ValueError(f"bad market range {market!r}")
    if lo > max_maps:
        raise ValueError(f"{market!r} impossible in a BO{max_maps}")
    return lo, min(hi, max_maps)


def _draw(rng, pool, size):
    values, probs = pool
    if len(values) == 1:
        return np.full(size, values[0])
    return values[rng.choice(len(values), size=size, p=probs)]


def simulate_series(params: MatchupParams, n_sims: int = N_SIMS,
                    rng: np.random.Generator | int | None = None) -> SeriesSim:
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    S, M = n_sims, params.best_of
    need = M // 2 + 1

    # (a) map winners and which maps are played (stop at clinch)
    a_wins = rng.random((S, M)) < params.p_map
    wa = np.cumsum(a_wins, axis=1)
    wb = np.cumsum(~a_wins, axis=1)
    prev_a = np.concatenate([np.zeros((S, 1), int), wa[:, :-1]], axis=1)
    prev_b = np.concatenate([np.zeros((S, 1), int), wb[:, :-1]], axis=1)
    played = (prev_a < need) & (prev_b < need)

    # (b)-(e) team kills: bootstrap from the W or L pool by drawn outcome
    team_kills = np.zeros((S, M, 2), dtype=np.int64)
    for s in (0, 1):
        won = a_wins if s == 0 else ~a_wins
        w_draw = _draw(rng, params.pools[s][1], (S, M))
        l_draw = _draw(rng, params.pools[s][0], (S, M))
        x = np.where(won, w_draw, l_draw) * params.script_mult
        x = np.maximum(x, 0.0)
        if params.nb_r is None:
            base = np.floor(x)
            k = base + (rng.random((S, M)) < (x - base))  # stochastic rounding keeps mean
        else:  # toggle C: smoothed bootstrap, NB kernel centred on the drawn value
            r = params.nb_r
            k = rng.negative_binomial(r, r / (r + x))
        team_kills[:, :, s] = np.where(played, k, 0).astype(np.int64)

    # (f) Dirichlet-multinomial allocation to the five players
    player_kills = np.zeros((S, M, 2, 5), dtype=np.int64)
    for s in (0, 1):
        g = rng.standard_gamma(params.alpha[s], size=(S, M, 5))
        if params.alpha_wl is not None:     # toggle Q: shares by map outcome
            won = a_wins if s == 0 else ~a_wins
            g_w = rng.standard_gamma(params.alpha_wl[s, 1], size=(S, M, 5))
            g_l = rng.standard_gamma(params.alpha_wl[s, 0], size=(S, M, 5))
            g = np.where(won[:, :, None], g_w, g_l)
        if params.scale is not None:
            g = g * params.scale[s]
        shares = g / g.sum(axis=-1, keepdims=True)
        player_kills[:, :, s, :] = rng.multinomial(team_kills[:, :, s], shares)

    return SeriesSim(params, a_wins, played, team_kills, player_kills)
