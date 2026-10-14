"""Turn match history into simulator inputs (pools, adjustments, player shares).

Everything here is computed "as of" a cutoff timestamp so the same code serves live
projection (cutoff = now) and the walk-forward backtest (cutoff = series start).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import config as C


@dataclass
class EngineConfig:
    patch_decay: float = C.PATCH_DECAY
    patch_lookback: int = C.PATCH_LOOKBACK
    opp_full_games: int = C.OPP_ADJ_FULL_GAMES
    opp_clamp: tuple = C.OPP_ADJ_CLAMP
    shrink_games: int = C.SHRINK_GAMES
    min_outcome_pool: int = C.MIN_OUTCOME_POOL
    game_script: bool = False
    game_script_clamp: tuple = C.GAME_SCRIPT_CLAMP
    kappa: float = C.DEFAULT_KAPPA
    # --- upper-tail toggles (all off by default; see CLAUDE.md) ---
    # A) per-position concentration (normalized gammas with per-position scale)
    position_kappa: bool = False
    kappa_pos: dict | None = None      # fitted; None -> fit on the fly when enabled
    # B) Kish effective-sample floor: blend toward league W/L pool until n_eff >= floor
    neff_floor: float = 0.0            # 0 = off
    # C) smoothed bootstrap: NB(mean = drawn value, r = league dispersion)
    smooth_bootstrap: bool = False
    nb_r: dict | None = None           # league -> r; None -> fit on the fly when enabled
    position_priors: dict = field(default_factory=lambda: dict(C.POSITION_PRIORS))
    # --- mean-bias toggles (off by default; see CLAUDE.md / reports/backtest_bias.md) ---
    # M) meta drift: scale pool values by (league kills/game over the latest N patches) /
    #    (decay-weighted league kills/game) of the league each pool game was played in
    meta_drift_patches: int = 0        # 0 = off; N = 1, 2, 3 tested
    # P) position priors refit walk-forward per league (replaces POSITION_PRIORS)
    fit_priors: bool = False
    # O) outcome-conditional opponent adjustment: when side s draws from its W (L) pool,
    #    use the opponent's kills allowed in its LOSSES (WINS) relative to the average
    #    normalized kills of winners (losers). Fixes the plain adjRatio double-counting
    #    team strength that the W/L pools already carry. Not part of the locked spec (c).
    opp_by_outcome: bool = False
    # G) reference only: global multiplier on team kills (fit walk-forward by the runner)
    kill_mult: float = 1.0


def locked_config(**overrides) -> EngineConfig:
    """The locked engine config (CLAUDE.md, signed off 2026-10-14): shrink30 + A + B50,
    game-script off, smooth bootstrap off. Use this for live projections."""
    cfg = dict(shrink_games=30, position_kappa=True, neff_floor=50.0, game_script=False,
               smooth_bootstrap=False)
    cfg.update(overrides)
    return EngineConfig(**cfg)


@dataclass
class MatchupParams:
    """Everything the simulator needs for one series. Index 0 = team A, 1 = team B."""
    teams: tuple[str, str]
    best_of: int
    p_map: float
    # pools[side][outcome] = (values, probs); outcome 1 = win, 0 = loss
    pools: list
    alpha: np.ndarray            # (2, 5) Dirichlet parameters
    players: list                # [[5 names], [5 names]] in POSITIONS order
    script_mult: float = 1.0
    info: dict = field(default_factory=dict)
    # gamma scale per player for the share draw; all-ones = plain Dirichlet(alpha)
    scale: np.ndarray | None = None
    # NB dispersion for the smoothed bootstrap; None = stochastic rounding
    nb_r: float | None = None


class History:
    """Columnar, time-sorted view of processed OE player/team rows."""

    def __init__(self, players: pd.DataFrame, teams: pd.DataFrame):
        t = teams.sort_values(["date", "gameid", "side"]).reset_index(drop=True)
        opp = t.groupby("gameid")["teamname"].transform(lambda s: s.iloc[::-1].values)
        self.t_time = t["date"].values.astype("datetime64[ns]").astype(np.int64)
        self.t_team = t["teamname"].astype(str).values
        self.t_opp = opp.astype(str).values
        self.t_league = t["league"].astype(str).values
        self.t_patch = t["patch_ord"].values.astype(np.int32)
        self.t_result = t["result"].values.astype(np.int8)
        self.t_kills = t["teamkills"].values.astype(float)
        self.t_deaths = t["teamdeaths"].values.astype(float)
        self.t_len = t["gamelength"].values.astype(float) / 60.0
        self.team_idx = _index(self.t_team)
        self.league_idx = _index(self.t_league)

        p = players.sort_values(["date", "gameid"]).reset_index(drop=True)
        self.p_time = p["date"].values.astype("datetime64[ns]").astype(np.int64)
        self.p_player = p["playername"].astype(str).values
        self.p_patch = p["patch_ord"].values.astype(np.int32)
        self.p_kills = p["kills"].values.astype(float)
        self.p_teamkills = p["teamkills"].values.astype(float)
        self.p_pos = p["position"].astype(str).values
        self.p_league = p["league"].astype(str).values
        self.p_league_idx = _index(self.p_league)
        self.player_idx = _index(self.p_player)
        self.leagues = sorted(self.league_idx)

    # ------------------------------------------------------------------ helpers ----
    @staticmethod
    def _prior(idx: np.ndarray, times: np.ndarray, cutoff: int) -> np.ndarray:
        return idx[: np.searchsorted(times[idx], cutoff, side="left")]

    def _weights(self, patches: np.ndarray, patch: int, cfg: EngineConfig):
        back = np.clip(patch - patches, 0, None)
        keep = back <= cfg.patch_lookback
        return keep, cfg.patch_decay ** back

    def team_rows(self, team, cutoff, patch, cfg):
        idx = self.team_idx.get(team)
        if idx is None:
            return np.empty(0, dtype=int), np.empty(0)
        idx = self._prior(idx, self.t_time, cutoff)
        keep, w = self._weights(self.t_patch[idx], patch, cfg)
        return idx[keep], w[keep]

    def recent_rows(self, cutoff, patch, cfg, n=4000):
        end = np.searchsorted(self.t_time, cutoff, side="left")
        idx = np.arange(max(0, end - n), end)
        keep, w = self._weights(self.t_patch[idx], patch, cfg)
        return idx[keep], w[keep]

    def home_league(self, team, cutoff) -> str | None:
        idx = self.team_idx.get(team)
        if idx is None:
            return None
        idx = self._prior(idx, self.t_time, cutoff)[-60:]
        lg = [x for x in self.t_league[idx] if x in C.DOMESTIC_LEAGUES and x != "EM"]
        if not lg:
            lg = [x for x in self.t_league[idx] if x in C.DOMESTIC_LEAGUES]
        if not lg:
            return None
        vals, counts = np.unique(lg, return_counts=True)
        return str(vals[np.argmax(counts)])

    # ------------------------------------------------------------------ environments --
    def measured_env(self, league, cutoff, patch, cfg):
        """(avg team kills per map, n games) for a league as of cutoff."""
        idx = self.league_idx.get(league)
        if idx is None:
            return np.nan, 0
        idx = self._prior(idx, self.t_time, cutoff)
        if league in C.INTERNATIONAL_LEAGUES:
            # events are sparse across patches: plain average of the most recent games
            idx = idx[-240:]
            if len(idx) == 0:
                return np.nan, 0
            return float(self.t_kills[idx].mean()), len(idx) // 2
        keep, w = self._weights(self.t_patch[idx], patch, cfg)
        idx, w = idx[keep], w[keep]
        if len(idx) == 0:
            return np.nan, 0
        return float(np.average(self.t_kills[idx], weights=w)), len(idx) // 2

    def env_table(self, cutoff, patch, cfg) -> dict:
        """Kill environment for every league (used to normalize pool entries)."""
        rows, w = self.recent_rows(cutoff, patch, cfg)
        glob = float(np.average(self.t_kills[rows], weights=w)) if len(rows) else 14.0
        out = {"_global": glob}
        for lg in self.leagues:
            m, n = self.measured_env(lg, cutoff, patch, cfg)
            if lg in C.INTERNATIONAL_LEAGUES:
                wt = min(1.0, n / C.EVENT_ENV_FULL_WEIGHT_GAMES)
                prior = glob * C.EVENT_ENV_PRIOR.get(lg, 1.0)
                out[lg] = prior if n == 0 else wt * m + (1 - wt) * prior
            else:
                out[lg] = m if n >= 10 else glob
        return out

    def target_env(self, league, team_a, team_b, cutoff, patch, cfg, envs) -> float:
        if league not in C.INTERNATIONAL_LEAGUES:
            return envs.get(league, envs["_global"])
        homes = [self.home_league(t, cutoff) for t in (team_a, team_b)]
        home_env = np.mean([envs.get(h, envs["_global"]) if h else envs["_global"]
                            for h in homes])
        prior = home_env * C.EVENT_ENV_PRIOR.get(league, 1.0)
        m, n = self.measured_env(league, cutoff, patch, cfg)
        if n == 0:
            return float(prior)
        wt = min(1.0, n / C.EVENT_ENV_FULL_WEIGHT_GAMES)
        return float(wt * m + (1 - wt) * prior)

    def recent_patch_env(self, idx, n_patches, min_games=C.DRIFT_MIN_GAMES):
        """Plain team kills/game over the latest n patches present in rows idx (extended
        one patch at a time until min_games games). nan if idx is empty."""
        if len(idx) == 0:
            return np.nan
        pts = self.t_patch[idx]
        uniq = np.unique(pts)[::-1]
        k = min(n_patches, len(uniq))
        while True:
            sel = pts >= uniq[k - 1]
            if sel.sum() >= 2 * min_games or k == len(uniq):
                return float(self.t_kills[idx[sel]].mean())
            k += 1

    def drift_table(self, cutoff, patch, cfg, envs) -> dict:
        """Toggle M: league -> recent-N-patch kills/game / decay-weighted kills/game.

        Leagues whose env fell back to the global env, and international events, use the
        global ratio. Clamped to DRIFT_CLAMP."""
        n = cfg.meta_drift_patches
        g_idx, _ = self.recent_rows(cutoff, patch, cfg)
        g_rec = self.recent_patch_env(g_idx, n)
        glob = g_rec / envs["_global"] if np.isfinite(g_rec) else 1.0
        out = {"_global": float(np.clip(glob, *C.DRIFT_CLAMP))}
        for lg in self.leagues:
            idx = self.league_idx[lg]
            idx = self._prior(idx, self.t_time, cutoff)
            idx = idx[self.t_patch[idx] <= patch]
            if lg in C.INTERNATIONAL_LEAGUES or len(idx) < 2 * C.DRIFT_MIN_GAMES:
                out[lg] = out["_global"]
                continue
            m, n_games = self.measured_env(lg, cutoff, patch, cfg)
            if n_games < 10 or not np.isfinite(m):
                out[lg] = out["_global"]
                continue
            out[lg] = float(np.clip(self.recent_patch_env(idx, n) / m, *C.DRIFT_CLAMP))
        return out

    def fitted_priors(self, league, cutoff, patch, cfg) -> dict:
        """Toggle P: position kill-share priors from prior data in `league` (decay-
        weighted, patch lookback), blended with the all-league fit by min(1, n/PRIOR_FULL).
        International events use the all-league fit."""
        end = np.searchsorted(self.p_time, cutoff, side="left")
        all_idx = np.arange(max(0, end - 60000), end)

        def fit(idx):
            keep, w = self._weights(self.p_patch[idx], patch, cfg)
            idx, w = idx[keep], w[keep]
            out, n = {}, len(idx) // 10
            for pos in C.POSITIONS:
                m = self.p_pos[idx] == pos
                den = np.sum(w[m] * self.p_teamkills[idx[m]])
                out[pos] = float(np.sum(w[m] * self.p_kills[idx[m]]) / den) if den > 0 \
                    else C.POSITION_PRIORS[pos]
            tot = sum(out.values())
            return {k: v / tot for k, v in out.items()}, n

        glob, n_glob = fit(all_idx)
        if n_glob < 50:
            glob = dict(C.POSITION_PRIORS)
        lg_idx = self.p_league_idx.get(league)
        if lg_idx is None or league in C.INTERNATIONAL_LEAGUES:
            return glob
        loc, n = fit(self._prior(lg_idx, self.p_time, cutoff))
        wt = min(1.0, n / C.PRIOR_FULL_GAMES)
        return {k: wt * loc[k] + (1 - wt) * glob[k] for k in C.POSITIONS}

    # ------------------------------------------------------------------ players ------
    def player_share(self, player, cutoff, patch, cfg):
        """(patch-weighted kill share over all games, n games)."""
        idx = self.player_idx.get(player)
        if idx is None:
            return np.nan, 0
        idx = self._prior(idx, self.p_time, cutoff)
        keep, w = self._weights(self.p_patch[idx], patch, cfg)
        idx, w = idx[keep], w[keep]
        denom = np.sum(w * self.p_teamkills[idx])
        if len(idx) == 0 or denom <= 0:
            return np.nan, len(idx)
        return float(np.sum(w * self.p_kills[idx]) / denom), len(idx)

    def player_mean_kills(self, player, cutoff, patch, cfg):
        idx = self.player_idx.get(player)
        if idx is None:
            return np.nan, 0
        idx = self._prior(idx, self.p_time, cutoff)
        keep, w = self._weights(self.p_patch[idx], patch, cfg)
        idx, w = idx[keep], w[keep]
        if len(idx) == 0:
            return np.nan, 0
        return float(np.average(self.p_kills[idx], weights=w)), len(idx)


def _index(values: np.ndarray) -> dict:
    order = np.argsort(values, kind="stable")
    sv = values[order]
    uniq, starts = np.unique(sv, return_index=True)
    ends = list(starts[1:]) + [len(sv)]
    return {u: np.sort(order[s:e]) for u, s, e in zip(uniq, starts, ends)}


def fit_kappa(hist: History, cutoff: int, n_rows: int = 30000) -> float:
    """Method-of-moments Dirichlet concentration from player kill shares before cutoff."""
    end = np.searchsorted(hist.p_time, cutoff, side="left")
    sl = slice(max(0, end - n_rows), end)
    k, K, pl = hist.p_kills[sl], hist.p_teamkills[sl], hist.p_player[sl]
    if len(k) < 500:
        return C.DEFAULT_KAPPA
    df = pd.DataFrame({"k": k, "K": K, "pl": pl})
    sums = df.groupby("pl")[["k", "K"]].transform("sum")
    p = (sums["k"] / sums["K"].where(sums["K"] > 0)).fillna(0.2).values
    return _mom_kappa(k, K, p)


def fit_kappa_by_position(hist: History, cutoff: int, n_rows: int = 30000) -> dict:
    """Toggle A: method-of-moments concentration fitted separately per position."""
    end = np.searchsorted(hist.p_time, cutoff, side="left")
    sl = slice(max(0, end - n_rows), end)
    k, K = hist.p_kills[sl], hist.p_teamkills[sl]
    pl, pos = hist.p_player[sl], hist.p_pos[sl]
    out = {}
    if len(k) < 500:
        return {p: C.DEFAULT_KAPPA for p in C.POSITIONS}
    df = pd.DataFrame({"k": k, "K": K, "pl": pl})
    sums = df.groupby("pl")[["k", "K"]].transform("sum")
    p = (sums["k"] / sums["K"].where(sums["K"] > 0)).fillna(0.2).values
    for position in C.POSITIONS:
        m = pos == position
        out[position] = _mom_kappa(k[m], K[m], p[m]) if m.sum() >= 100 else C.DEFAULT_KAPPA
    return out


def _mom_kappa(k, K, p) -> float:
    resid = np.sum((k - K * p) ** 2)
    base = K * p * (1 - p)
    if resid <= base.sum():
        return 1e4

    def f(kappa):
        return np.sum(base * (K + kappa) / (1 + kappa)) - resid

    lo, hi = 0.1, 1e4
    if f(lo) < 0:
        return lo
    for _ in range(100):
        mid = np.sqrt(lo * hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def fit_nb_dispersion(hist: History, cutoff: int, patch: int,
                      cfg: EngineConfig | None = None) -> dict:
    """Toggle C: per-league NB dispersion r of team kills around their (team, outcome) mean.

    Var = mu + mu^2 / r, fitted by method of moments on the within-(team, result) variance
    of prior games in the patch lookback window. Returns {league: r, "_global": r}.
    """
    cfg = cfg or EngineConfig()
    end = np.searchsorted(hist.t_time, cutoff, side="left")
    idx = np.arange(max(0, end - 8000), end)
    keep, _ = hist._weights(hist.t_patch[idx], patch, cfg)
    idx = idx[keep]
    df = pd.DataFrame({"lg": hist.t_league[idx], "tm": hist.t_team[idx],
                       "res": hist.t_result[idx], "x": hist.t_kills[idx]})

    def r_of(d: pd.DataFrame) -> float:
        g = d.groupby(["tm", "res"])["x"]
        n = g.transform("size")
        d = d[n >= 2]
        if len(d) < 30:
            return np.nan
        dev = d["x"] - d.groupby(["tm", "res"])["x"].transform("mean")
        n_groups = d.groupby(["tm", "res"]).ngroups
        var = float(np.sum(dev ** 2) / max(1, len(d) - n_groups))
        mu = float(d["x"].mean())
        if var <= mu:
            return 1000.0
        return float(np.clip(mu ** 2 / (var - mu), 1.0, 1000.0))

    glob = r_of(df)
    out = {"_global": 30.0 if np.isnan(glob) else glob}
    for lg, d in df.groupby("lg"):
        r = r_of(d)
        out[lg] = out["_global"] if np.isnan(r) else r
    return out


def kish_neff(w: np.ndarray) -> float:
    s = w.sum()
    return float(s * s / np.sum(w * w)) if s > 0 else 0.0


def _blend_to_floor(vals, wts, lvals, lwts, floor):
    """Toggle B: mix team pool (weight lam) with league pool (1-lam), max lam with
    Kish n_eff >= floor. Returns (values, probs, lam)."""
    ne_t = kish_neff(wts)
    if ne_t >= floor or len(lvals) == 0:
        return vals, wts / wts.sum(), 1.0
    ne_l = kish_neff(lwts)

    def ne(lam):
        return 1.0 / (lam ** 2 / ne_t + (1 - lam) ** 2 / ne_l)

    if ne_l < floor:  # league pool can't reach the floor: use the lam maximizing n_eff
        lam = ne_t / (ne_t + ne_l)
    else:
        lo, hi = 0.0, 1.0  # ne(0) = ne_l >= floor > ne(1) = ne_t
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if ne(mid) >= floor:
                lo = mid
            else:
                hi = mid
        lam = lo
    p = np.concatenate([wts / wts.sum() * lam, lwts / lwts.sum() * (1 - lam)])
    return np.concatenate([vals, lvals]), p, lam


def build_matchup(hist: History, team_a: str, team_b: str, league: str, best_of: int,
                  p_map: float, lineups: tuple[list, list], cutoff, patch: int,
                  cfg: EngineConfig | None = None) -> MatchupParams:
    """Assemble MatchupParams.

    lineups: ([5 player names A], [5 player names B]) in POSITIONS order.
    cutoff: pandas Timestamp / ns int; only history strictly before it is used.
    """
    cfg = cfg or EngineConfig()
    cutoff = int(pd.Timestamp(cutoff).value) if not isinstance(cutoff, (int, np.integer)) \
        else int(cutoff)
    envs = hist.env_table(cutoff, patch, cfg)
    target = hist.target_env(league, team_a, team_b, cutoff, patch, cfg, envs)
    if cfg.meta_drift_patches > 0:
        drift = hist.drift_table(cutoff, patch, cfg, envs)
        # dividing by env/drift == normalize by origin env, then apply origin drift
        env_of = np.vectorize(lambda lg: envs.get(lg, envs["_global"])
                              / drift.get(lg, drift["_global"]), otypes=[float])
        env_raw = np.vectorize(lambda lg: envs.get(lg, envs["_global"]), otypes=[float])
    else:
        drift = None
        env_of = env_raw = np.vectorize(lambda lg: envs.get(lg, envs["_global"]),
                                        otypes=[float])

    teams = (team_a, team_b)
    rows = [hist.team_rows(t, cutoff, patch, cfg) for t in teams]
    g_idx, g_w = hist.recent_rows(cutoff, patch, cfg)
    g_norm = hist.t_kills[g_idx] / env_of(hist.t_league[g_idx]) if len(g_idx) else g_idx
    g_raw = hist.t_kills[g_idx] / env_raw(hist.t_league[g_idx]) if len(g_idx) else g_idx

    info = {"target_env": target, "envs": envs, "drift": drift}
    if cfg.neff_floor > 0:
        lg_rows = hist.league_idx.get(league)
        if lg_rows is None or league in C.INTERNATIONAL_LEAGUES:
            l_idx, l_w = g_idx, g_w
        else:
            l_idx = hist._prior(lg_rows, hist.t_time, cutoff)
            keep, l_w = hist._weights(hist.t_patch[l_idx], patch, cfg)
            l_idx, l_w = l_idx[keep], l_w[keep]
        l_norm = hist.t_kills[l_idx] / env_of(hist.t_league[l_idx]) if len(l_idx) else l_idx
    pools = []
    for s in (0, 1):
        o = 1 - s
        # (c) opponent adjustment: opponent kills-allowed relative to its leagues' averages
        o_idx, o_w = rows[o]
        if len(o_idx):
            allowed = hist.t_deaths[o_idx] / env_raw(hist.t_league[o_idx])
            raw = float(np.average(allowed, weights=o_w))
        else:
            raw = 1.0
        fade = min(1.0, len(o_idx) / cfg.opp_full_games)
        adj = float(np.clip(1.0 + (raw - 1.0) * fade, *cfg.opp_clamp))
        info[f"opp_adj_{s}"] = adj
        adj_by = {0: adj, 1: adj}
        if cfg.opp_by_outcome:
            adj_by = _opp_adj_by_outcome(hist, o_idx, o_w, g_idx, g_w, g_raw, env_raw, cfg)
            info[f"opp_adj_{s}_by_outcome"] = adj_by

        # (b)+(d) W/L pools, normalized by origin-league env, rescaled to target env
        idx, w = rows[s]
        norm = hist.t_kills[idx] / env_of(hist.t_league[idx]) if len(idx) else np.empty(0)
        side_pools = [None, None]
        for outcome in (0, 1):
            adj = adj_by[outcome]
            sel = hist.t_result[idx] == outcome
            vals, wts = norm[sel], w[sel]
            n = int(sel.sum())
            if cfg.neff_floor > 0 and n > 0:
                lsel = hist.t_result[l_idx] == outcome
                vals, probs, lam = _blend_to_floor(vals, wts, l_norm[lsel], l_w[lsel],
                                                   cfg.neff_floor)
                info[f"neff_lam_{s}_{outcome}"] = lam
                side_pools[outcome] = (vals * target * adj * cfg.kill_mult, probs)
                info[f"n_pool_{s}_{outcome}"] = n
                continue
            if n < cfg.min_outcome_pool:
                gsel = hist.t_result[g_idx] == outcome
                gv, gw = g_norm[gsel], g_w[gsel]
                if len(gv):
                    share_team = n / cfg.min_outcome_pool
                    wts = wts / wts.sum() * share_team if n else wts
                    gw = gw / gw.sum() * (1 - share_team)
                    vals, wts = np.concatenate([vals, gv]), np.concatenate([wts, gw])
            if len(vals) == 0:  # no history at all: neutral environment
                vals, wts = np.array([1.3 if outcome else 0.7]), np.array([1.0])
            probs = wts / wts.sum()
            side_pools[outcome] = (vals * target * adj * cfg.kill_mult, probs)
            info[f"n_pool_{s}_{outcome}"] = n
        pools.append(side_pools)

    # (e) game-script multiplier
    script = 1.0
    if cfg.game_script:
        script = _game_script(hist, rows, league, envs, env_raw, cutoff, patch, cfg)
    info["script_mult_raw"] = script

    # (f) player shares from overall pools, shrunk toward position priors
    alpha = np.zeros((2, 5))
    scale = np.ones((2, 5))
    kpos = None
    if cfg.position_kappa:
        kpos = cfg.kappa_pos or fit_kappa_by_position(hist, cutoff)
    priors = hist.fitted_priors(league, cutoff, patch, cfg) if cfg.fit_priors \
        else cfg.position_priors
    info["position_priors"] = priors
    for s in (0, 1):
        shares = np.zeros(5)
        for j, (pos, pl) in enumerate(zip(C.POSITIONS, lineups[s])):
            prior = priors[pos]
            obs, n = hist.player_share(pl, cutoff, patch, cfg)
            wt = min(1.0, n / cfg.shrink_games) if cfg.shrink_games > 0 else 1.0
            shares[j] = prior if np.isnan(obs) else wt * obs + (1 - wt) * prior
            info[f"share_{s}_{j}"] = (shares[j], n)
        shares = np.clip(shares, 1e-3, None)
        shares = shares / shares.sum()
        if kpos is None:
            alpha[s] = cfg.kappa * shares
        else:
            # independent Gamma(k_j * s_j, 1/k_j): mean s_j, var s_j / k_j, then normalize
            kv = np.array([kpos[p] for p in C.POSITIONS])
            alpha[s] = kv * shares
            scale[s] = 1.0 / kv

    nb_r = None
    if cfg.smooth_bootstrap:
        table = cfg.nb_r or fit_nb_dispersion(hist, cutoff, patch, cfg)
        nb_r = float(table.get(league, table["_global"]))
        info["nb_r"] = nb_r

    return MatchupParams(teams=teams, best_of=best_of, p_map=float(p_map), pools=pools,
                         alpha=alpha, players=[list(lineups[0]), list(lineups[1])],
                         script_mult=script, info=info, scale=scale, nb_r=nb_r)


def _opp_adj_by_outcome(hist, o_idx, o_w, g_idx, g_w, g_raw, env_raw, cfg) -> dict:
    """Toggle O: {outcome of side s: adj}. Side s winning = opponent losing, so the
    opponent's kills allowed in its losses are compared with the average normalized kills
    of winning teams (and vice versa). Faded by min(1, n/opp_full_games), clamped."""
    out = {}
    for outcome in (0, 1):
        ref_sel = hist.t_result[g_idx] == outcome
        ref = float(np.average(g_raw[ref_sel], weights=g_w[ref_sel])) \
            if ref_sel.any() else 1.0
        sel = hist.t_result[o_idx] == 1 - outcome
        if sel.any() and ref > 0:
            allowed = hist.t_deaths[o_idx[sel]] / env_raw(hist.t_league[o_idx[sel]])
            raw = float(np.average(allowed, weights=o_w[sel])) / ref
        else:
            raw = 1.0
        fade = min(1.0, int(sel.sum()) / cfg.opp_full_games)
        out[outcome] = float(np.clip(1.0 + (raw - 1.0) * fade, *cfg.opp_clamp))
    return out


def _game_script(hist, rows, league, envs, env_of, cutoff, patch, cfg) -> float:
    """Projected combined KPM x projected length vs the league baseline."""
    rates, lens = [], []
    for idx, w in rows:
        if len(idx) == 0:
            continue
        comb = (hist.t_kills[idx] + hist.t_deaths[idx]) / (2 * env_of(hist.t_league[idx]))
        rates.append(np.average(comb / hist.t_len[idx], weights=w))
        lens.append(np.average(hist.t_len[idx], weights=w))
    if len(rates) < 2:
        return 1.0
    lg_idx = hist.league_idx.get(league)
    if lg_idx is None or league in C.INTERNATIONAL_LEAGUES:
        b_idx, b_w = hist.recent_rows(cutoff, patch, cfg)
    else:
        b_idx = hist._prior(lg_idx, hist.t_time, cutoff)
        keep, b_w = hist._weights(hist.t_patch[b_idx], patch, cfg)
        b_idx, b_w = b_idx[keep], b_w[keep]
    if len(b_idx) < 20:
        return 1.0
    b_comb = (hist.t_kills[b_idx] + hist.t_deaths[b_idx]) / (2 * env_of(hist.t_league[b_idx]))
    base_rate = np.average(b_comb / hist.t_len[b_idx], weights=b_w)
    base_len = np.average(hist.t_len[b_idx], weights=b_w)
    mult = (np.mean(rates) * np.mean(lens)) / (base_rate * base_len)
    return float(np.clip(mult, *cfg.game_script_clamp))
