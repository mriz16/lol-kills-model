"""Slate -> projections: rosters, lineups, simulations, the prop table and slips.

No Streamlit here; `streamlit_app.py` wraps these with caching and widgets.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from src import config as C
from src.app.names import LineMatch, match_lines
from src.app.odds_input import odds_summary
import json

from src.engine.calibration import (Calibrator, calibrated_prop, fit_calibrator_asof,
                                    fit_spread_asof, fit_w_asof, lambda_for, w_adjust)
from src.engine.features import (EngineConfig, History, MatchupParams, build_matchup,
                                 fit_kappa_by_position, locked_config)
from src.engine.props import Leg, breakeven, prop_summary, slip_probability
from src.engine.sim import SeriesSim, simulate_series

PLATFORMS = {"PrizePicks": C.PRIZEPICKS_POWER, "Underdog": C.UNDERDOG}
MARKET_BY_FORMAT = {1: "Map 1", 3: "Maps 1-2", 5: "Maps 1-3"}
ROSTER_DAYS = 60            # players seen for the team in the last N days of its games
THIN_EDGE = 0.03            # uncalibrated fallback only: flag edges under 3 points
LOW_SAMPLE_GAMES = 20
CAUTION_P_MORE = 0.25       # "less" at a whole line with P(more) < 25%: overs hit ~9 pts more
# Post-sim layers (reports/backtest_spread.md, reports/calibration_fit.md), all fitted
# as of the store's cutoff from files written by scripts/refit_calibration.py:
#  S (locked 2026-10-14): spread k per position x market <- data/spread_rows.parquet
#  I (display): beta calibration of P(over) on the S pmfs <- data/calib_pairs.parquet
#  W (display): p_line + lambda (p_cal - p_line) per market <- data/w_rows.parquet, or
#    data/w_lambda_real.json (refit on real PrizePicks lines, scripts/refit_w_real.py)
#    for cutoffs after the rows it was fitted on.
CALIB_PATH = C.DATA_DIR / "calib_pairs.parquet"
SPREAD_PATH = C.DATA_DIR / "spread_rows.parquet"
W_PATH = C.DATA_DIR / "w_rows.parquet"
W_REAL_PATH = C.DATA_DIR / "w_lambda_real.json"
CALIB_MODEL = "L+O+S"
CALIB_METHOD = "beta"
# Minimum W-adjusted edge for a PLAY (reports/calibration_fit.md: after W the top-1/3/5
# picks per slate are within ~1 pt of their shown P; 3 pts clears that by 2 se). Applies
# only to calibrated probabilities.
MIN_EDGE = 0.03
SIDE_BALANCE = 0.70         # warn when > 70% of a slate's plays are one side
SIDE_BALANCE_MIN_PLAYS = 3
# Floors and goblins (reports/audit_unders.md). floorNN = the most kills the player reaches
# in NN% of sims (with S). In the walk-forward backtest P(X >= f) of the S pmfs is
# calibrated to within ~1 pt from 80% to 98%, except supports (overstated 3-5 pts).
FLOOR_QS = (0.80, 0.90)
# SAFE FLOOR: shown P(more) >= 85%, not a support. At lines >= 2 kills under a PP-like
# line the raw P(more) in 85-95% hit within 1-3 pts of its prediction; the shown (W) P is
# lower than raw, so the flag is conservative there. Lines only 1 kill under the market
# line ran ~7 pts hot (80% hit at a raw 87%): trust the flag on goblins 2+ kills under.
SAFE_FLOOR_P = 0.85
# SUP MORE: support overs at PP-like lines hit 55.8% at a shown 67.3% (n=267), below the
# 2-pick breakeven; support unders were fine (68.2% at 70.5%).


def per_leg_breakeven(platform: str, legs: int) -> float:
    """(1 / multiplier) ** (1 / legs) from the engine payout tables."""
    return breakeven(PLATFORMS[platform][legs], legs)


# --------------------------------------------------------------------- data store ----
class DataStore:
    """Processed OE frames + History + rosters, built once per data version."""

    def __init__(self, players: pd.DataFrame, teams: pd.DataFrame, asof=None,
                 calib_pairs: pd.DataFrame | None = None,
                 spread_rows: pd.DataFrame | None = None,
                 w_rows: pd.DataFrame | None = None, w_real: dict | None = None):
        """asof (a date): backfill mode. Only games before that date (00:00 UTC) are kept,
        so pools, rosters, lineups, kappas and the S / I / W fits use prior data only.
        calib_pairs / spread_rows / w_rows / w_real default to the files in data/ (pass
        an empty frame / {} to switch a layer off)."""
        self.asof = pd.Timestamp(asof).normalize() if asof is not None else None
        if self.asof is not None:
            players = players[pd.to_datetime(players["date"]) < self.asof]
            teams = teams[pd.to_datetime(teams["date"]) < self.asof]
            if teams.empty:
                raise ValueError(f"no games before {self.asof.date()}")
        self.players, self.teams = players, teams
        self.hist = History(players, teams)
        self.last_date = pd.Timestamp(teams["date"].max())
        self.cutoff = int(self.asof.value) if self.asof is not None else \
            int((self.last_date + pd.Timedelta(days=1)).value)
        self.patch = int(teams["patch_ord"].max())
        self.patch_name = str(teams.loc[teams["patch_ord"] == self.patch, "patch"].iloc[-1])
        self.kappa_pos = fit_kappa_by_position(self.hist, self.cutoff)
        self._rosters = _build_rosters(players)
        last = teams.sort_values("date").groupby("teamname", observed=True)["date"].max()
        # most recently active first: the order of the team pickers
        self.team_names = [str(t) for t in last.sort_values(ascending=False).index]
        self.calibrator = load_calibrator(self.cutoff, calib_pairs)
        self.spread_k = load_spread(self.cutoff, spread_rows)
        self.w_lambda, self.w_source = load_w(self.cutoff, w_rows, w_real)

    @property
    def backfill(self) -> bool:
        return self.asof is not None

    @property
    def calibrated(self) -> bool:
        return self.calibrator is not None and self.calibrator.pool is not None

    def roster(self, team: str) -> list[dict]:
        return self._rosters.get(team, [])

    def default_lineup(self, team: str) -> list[str | None]:
        """Starters of the team's most recent game, in POSITIONS order."""
        r = self.roster(team)
        out = []
        for pos in C.POSITIONS:
            cands = [x for x in r if x["role"] == pos]
            out.append(cands[0]["player"] if cands else None)
        return out

    def home_league(self, team: str) -> str | None:
        return self.hist.home_league(team, self.cutoff)

    def infer_league(self, team_a: str, team_b: str) -> str:
        """Same home league -> that league; otherwise the latest international event
        either team played in, else Worlds."""
        ha, hb = self.home_league(team_a), self.home_league(team_b)
        if ha and ha == hb:
            return ha
        t = self.teams
        intl = t[t["teamname"].astype(str).isin([team_a, team_b])
                 & t["league"].astype(str).isin(C.INTERNATIONAL_LEAGUES)]
        if len(intl) and (self.last_date - intl["date"].max()).days <= 30:
            return str(intl.sort_values("date")["league"].iloc[-1])
        return "WLDs"

    def config(self, **overrides) -> EngineConfig:
        return locked_config(kappa_pos=self.kappa_pos, **overrides)


def load_calibrator(cutoff, pairs: pd.DataFrame | None = None) -> Calibrator | None:
    """Calibrator fitted on backtest pairs of series started >= 1 day before cutoff.
    None when there is no pairs file (the app then shows raw probabilities)."""
    if pairs is None:
        if not CALIB_PATH.exists():
            return None
        pairs = pd.read_parquet(CALIB_PATH)
    if pairs.empty:
        return None
    if "model" in pairs:
        pairs = pairs[pairs["model"] == CALIB_MODEL]
    cal = fit_calibrator_asof(pairs, pd.Timestamp(cutoff), CALIB_METHOD)
    return cal if cal.pool is not None else None


def _read(df, path):
    if df is not None:
        return df
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def load_spread(cutoff, rows: pd.DataFrame | None = None) -> dict | None:
    """S: {(position, market): k, "_pool": k} from backtest rows before cutoff."""
    rows = _read(rows, SPREAD_PATH)
    if rows.empty:
        return None
    k = fit_spread_asof(rows, pd.Timestamp(cutoff))
    return k if len(k) > 1 or k["_pool"] != 1.0 else None


def load_w(cutoff, rows: pd.DataFrame | None = None, real: dict | None = None
           ) -> tuple[dict | None, str]:
    """W: ({market: lambda, "_pool": lambda}, source). Real-line lambda when its fit ended
    before the cutoff, else the proxy-line fit as of cutoff."""
    if real is None and W_REAL_PATH.exists():
        real = json.loads(W_REAL_PATH.read_text())
    if real and real.get("lambda") and \
            pd.Timestamp(cutoff) > pd.Timestamp(real["fitted_through"]) + pd.Timedelta(days=1):
        return dict(real["lambda"]), f"real PrizePicks lines (n={real.get('n_rows', '?')})"
    rows = _read(rows, W_PATH)
    if rows.empty:
        return None, "none"
    lam = fit_w_asof(rows, pd.Timestamp(cutoff), per_leg_breakeven("PrizePicks", 2))
    if lam.get("_n", 0) == 0 and lam["_pool"] == 1.0:
        return None, "none"
    return lam, "proxy lines"


def _build_rosters(players: pd.DataFrame) -> dict[str, list[dict]]:
    """team -> [{player, team, role, games, last}] for players seen in the team's last
    ROSTER_DAYS days, most recent starters first. Role = most frequent position."""
    p = players[["date", "gameid", "teamname", "playername", "position"]].copy()
    for c in ("teamname", "playername", "position"):
        p[c] = p[c].astype(str)
    last_team = p.groupby("teamname")["date"].transform("max")
    p = p[p["date"] >= last_team - pd.Timedelta(days=ROSTER_DAYS)]
    g = p.groupby(["teamname", "playername"])
    agg = g.agg(games=("gameid", "size"), last=("date", "max"))
    role = g["position"].agg(lambda s: s.value_counts().index[0])
    agg["role"] = role
    agg = agg.reset_index().sort_values(["teamname", "last", "games"],
                                        ascending=[True, False, False])
    out: dict[str, list[dict]] = {}
    for team, d in agg.groupby("teamname", sort=False):
        out[team] = [dict(player=r.playername, team=team, role=r.role, games=int(r.games),
                          last=r.last) for r in d.itertuples(index=False)]
    return out


# --------------------------------------------------------------------- matchups ----
@dataclass
class MatchupInput:
    team_a: str
    team_b: str
    best_of: int = 3
    platform: str = "PrizePicks"
    ml_a: str = ""
    ml_b: str = ""
    map1_a: str = ""
    map1_b: str = ""
    lines: str = ""
    league: str = ""            # "" = infer

    @property
    def key(self) -> str:
        return f"{self.team_a} vs {self.team_b}"

    @property
    def market(self) -> str:
        return MARKET_BY_FORMAT[int(self.best_of)]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MatchupResult:
    inp: MatchupInput
    league: str
    odds: dict
    sim: SeriesSim
    params: MatchupParams
    lines: list[LineMatch]
    rows: list[dict] = field(default_factory=list)


def check_matchup(store: DataStore, m: MatchupInput) -> tuple[list[LineMatch], dict, list[str]]:
    """Validate a block before running: (line matches, odds summary, problems)."""
    problems = []
    if m.team_a not in store.team_names or m.team_b not in store.team_names:
        problems.append("pick both teams")
    elif m.team_a == m.team_b:
        problems.append("team A and team B are the same")
    odds = odds_summary(m.ml_a, m.ml_b, int(m.best_of), m.map1_a, m.map1_b)
    if odds["error"]:
        problems.append(odds["error"])
    roster = store.roster(m.team_a) + store.roster(m.team_b)
    lines = match_lines(m.lines, roster)
    if not lines:
        problems.append("no lines entered")
    bad = [lm for lm in lines if not lm.ok]
    if bad:
        problems.append(f"{len(bad)} line(s) to fix: " + ", ".join(lm.raw.strip() for lm in bad))
    return lines, odds, problems


def lineups_for(store: DataStore, m: MatchupInput, lines: list[LineMatch]) -> tuple[list, list]:
    """Latest starters, with any matched line player swapped into their role."""
    lus = []
    for team in (m.team_a, m.team_b):
        lu = store.default_lineup(team)
        for lm in lines:
            if lm.ok and lm.team == team:
                lu[C.POSITIONS.index(lm.role)] = lm.player
        # an empty slot (new roster) gets a placeholder that falls back to the prior
        lu = [pl if pl else f"_{team}_{pos}" for pl, pos in zip(lu, C.POSITIONS)]
        lus.append(lu)
    return lus[0], lus[1]


def simulate_matchup(store: DataStore, m: MatchupInput, lines, odds, league: str,
                     n_sims: int = C.N_SIMS, seed: int = 7) -> tuple[SeriesSim, MatchupParams]:
    lu = lineups_for(store, m, lines)
    params = build_matchup(store.hist, m.team_a, m.team_b, league, int(m.best_of),
                           odds["p_map_a"], lu, store.cutoff, store.patch, store.config())
    return simulate_series(params, n_sims=n_sims, rng=seed), params


def run_matchup(store: DataStore, m: MatchupInput, legs: int, n_sims: int = C.N_SIMS,
                seed: int = 7, sim_fn=None) -> MatchupResult:
    lines, odds, problems = check_matchup(store, m)
    if problems:
        raise ValueError(f"{m.key}: " + "; ".join(problems))
    league = m.league or store.infer_league(m.team_a, m.team_b)
    sim, params = (sim_fn or simulate_matchup)(store, m, lines, odds, league, n_sims, seed)
    res = MatchupResult(m, league, odds, sim, params, lines)
    # S is part of the locked model: every range_kills of this sim is inflated
    sim.spread_k = getattr(store, "spread_k", None)
    res.sim = sim
    cal = getattr(store, "calibrator", None)
    res.rows = [prop_row(res, lm, legs, cal, getattr(store, "w_lambda", None))
                for lm in lines]
    return res


# --------------------------------------------------------------------- prop rows ----
def flags_for(side: str, edge: float, p_more: float, line: float, n_games: int,
              calibrated: bool = False, role: str = "", kind: str = "standard") -> list[str]:
    """Calibrated: BELOW MIN under the minimum edge. Uncalibrated: THIN EDGE under 3 pts
    (and never a PLAY: the minimum edge applies only after calibration). Goblin / demon
    lines get their kind as a flag (MORE only, payout not in the edge)."""
    out = []
    if kind != "standard":
        out.append(kind.upper())
    if calibrated and edge < MIN_EDGE:
        out.append("BELOW MIN")
    elif not calibrated and edge < THIN_EDGE:
        out.append("THIN EDGE")
    if n_games < LOW_SAMPLE_GAMES:
        out.append("LOW SAMPLE")
    if side == "less" and p_more < CAUTION_P_MORE and float(line).is_integer():
        out.append("CAUTION")
    if p_more >= SAFE_FLOOR_P and role != "sup":
        out.append("SAFE FLOOR")
    if side == "more" and role == "sup":
        out.append("SUP MORE")
    return out


def _side(p_more: float, p_less: float, kind: str = "standard") -> tuple[str, float]:
    if kind != "standard":          # goblins / demons can only be played MORE
        return "more", p_more
    return ("more", p_more) if p_more >= p_less else ("less", p_less)


def floors(kills: np.ndarray, qs=FLOOR_QS) -> dict:
    """{"floor80": f, ...}: the largest f with P(kills >= f) >= q."""
    n = len(kills)
    counts = np.bincount(np.asarray(kills, dtype=np.int64))
    at_least = 1.0 - np.concatenate([[0.0], np.cumsum(counts)[:-1]]) / n   # P(X >= f)
    return {f"floor{int(round(100 * q))}": int(np.nonzero(at_least >= q - 1e-12)[0].max())
            for q in qs}


def prop_row(res: MatchupResult, lm: LineMatch, legs: int,
             calibrator: Calibrator | None = None, w_lambda: dict | None = None) -> dict:
    """One prop. Shown P / side / edge = W(I(model)) when a calibrator is given; cal_* =
    I only; raw_* = the model (with S when the sim carries spread_k)."""
    m, sim = res.inp, res.sim
    kills = sim.range_kills(lm.player, m.market)
    s = prop_summary(kills, lm.line)
    cal = calibrated_prop(kills, lm.line, calibrator, lm.role, m.market)
    is_cal = calibrator is not None and calibrator.pool is not None
    lam = lambda_for(w_lambda, m.market) if is_cal else 1.0
    w_more, w_less = w_adjust(cal["p_over"], cal["p_under"], cal["p_push"], lam)
    be = per_leg_breakeven(m.platform, legs)
    kind = getattr(lm, "kind", "standard")
    raw_side, raw_p = _side(s["p_over"], s["p_under"], kind)
    cal_side, cal_p = _side(cal["p_over"], cal["p_under"], kind)
    side, p_side = _side(float(w_more), float(w_less), kind)
    side_idx, j = sim.player_index(lm.player)
    share, n_games = res.params.info.get(f"share_{side_idx}_{j}", (np.nan, 0))
    opp = m.team_b if lm.team == m.team_a else m.team_a
    p_map = res.params.p_map if side_idx == 0 else 1 - res.params.p_map
    row = dict(
        matchup=m.key, platform=m.platform, league=res.league, best_of=int(m.best_of),
        market=m.market, player=lm.player, team=lm.team, opponent=opp, role=lm.role,
        line=float(lm.line), p_more=float(w_more), p_less=float(w_less),
        p_push=cal["p_push"], mean=s["mean"], median=s["median"],
        fair_line=s["fair_line"], side=side, p_side=p_side, breakeven=be,
        edge=p_side - be, calibrated=is_cal, w_lambda=lam,
        k_spread=_spread_of(sim, lm.role, m.market), cal_side=cal_side,
        cal_p_more=cal["p_over"], cal_p_less=cal["p_under"], cal_p_push=cal["p_push"],
        cal_p_side=cal_p, cal_edge=cal_p - be, raw_side=raw_side, raw_p_more=s["p_over"],
        raw_p_less=s["p_under"], raw_p_push=s["p_push"], raw_p_side=raw_p,
        raw_edge=raw_p - be, n_games=int(n_games), share=float(share), p_map=p_map,
        kind=kind, **floors(kills),
    )
    # goblin / demon payouts are not the standard multiplier: never a PLAY on this edge
    row["play"] = bool(is_cal and row["edge"] >= MIN_EDGE and kind == "standard")
    row["flags"] = " ".join(flags_for(side, row["edge"], row["p_more"], row["line"], n_games,
                                      is_cal, lm.role, kind))
    row["why"] = explain(res, lm, row)
    return row


def _spread_of(sim: SeriesSim, role: str, market: str) -> float:
    k = getattr(sim, "spread_k", None)
    return float(k.get((role, market), k.get("_pool", 1.0))) if k else 1.0


def pool_mean(pool) -> float:
    vals, probs = pool
    return float(np.sum(np.asarray(vals) * np.asarray(probs)))


def explain(res: MatchupResult, lm: LineMatch, row: dict) -> str:
    """One plain-English line: team kills in wins/losses, kill share, opp adj, win %."""
    prm, info = res.params, res.params.info
    s = 0 if lm.team == res.inp.team_a else 1
    win_k, loss_k = pool_mean(prm.pools[s][1]), pool_mean(prm.pools[s][0])
    p_map = prm.p_map if s == 0 else 1 - prm.p_map
    p_ser = res.odds["p_series_a"] if s == 0 else res.odds["p_series_b"]
    by = info.get(f"opp_adj_{s}_by_outcome")
    if by:
        adj = f"opponent adj x{by[1]:.2f} in wins, x{by[0]:.2f} in losses"
    else:
        adj = f"opponent adj x{info.get(f'opp_adj_{s}', 1.0):.2f}"
    n = row["n_games"]
    share = f"{row['share']:.0%}" if np.isfinite(row["share"]) else "prior"
    raw = (f" P(more) / P(less): shown {row['p_more']:.1%} / {row['p_less']:.1%} "
           f"(W, lambda {row['w_lambda']:.2f}); calibrated {row['cal_p_more']:.1%} / "
           f"{row['cal_p_less']:.1%}; raw model {row['raw_p_more']:.1%} / "
           f"{row['raw_p_less']:.1%}."
           if row.get("calibrated") else " Uncalibrated (no calibration data).")
    return (f"{lm.team} projects {win_k:.1f} team kills/map in wins and {loss_k:.1f} in "
            f"losses ({res.league} env {info.get('target_env', float('nan')):.1f}); "
            f"{lm.player} ({lm.role}) takes {share} of team kills ({n} games); {adj}; "
            f"{lm.team} wins {p_map:.0%} of maps ({p_ser:.0%} series). "
            f"Model mean {row['mean']:.1f} over {res.inp.market} vs line {row['line']:g}."
            + raw)


def results_table(results: list[MatchupResult]) -> pd.DataFrame:
    rows = [r for res in results for r in res.rows]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("edge", ascending=False).reset_index(drop=True)
    df["leg_id"] = [f"{r.matchup}|{r.player}|{r.line:g}" for r in df.itertuples()]
    return df


def safest_overs(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Rows by shown P(more), highest first (goblins included): the slate's safest overs.
    Supports are kept but carry SUP MORE (their overs ran hot in the backtest)."""
    if df is None or df.empty:
        return pd.DataFrame()
    return df.sort_values("p_more", ascending=False).head(n).reset_index(drop=True)


def rebreakeven(df: pd.DataFrame, legs: int) -> pd.DataFrame:
    """Recompute breakeven / edge / flags for a different slip size (no re-sim)."""
    if df.empty:
        return df
    df = df.copy()
    df["breakeven"] = [per_leg_breakeven(p, legs) for p in df["platform"]]
    df["edge"] = df["p_side"] - df["breakeven"]
    df["raw_edge"] = df["raw_p_side"] - df["breakeven"]
    df["cal_edge"] = df["cal_p_side"] - df["breakeven"]
    kind = df["kind"] if "kind" in df else pd.Series("standard", index=df.index)
    df["play"] = df["calibrated"].astype(bool) & (df["edge"] >= MIN_EDGE) & \
        (kind == "standard")
    df["flags"] = [" ".join(flags_for(r.side, r.edge, r.p_more, r.line, r.n_games,
                                      bool(r.calibrated), r.role, k))
                   for r, k in zip(df.itertuples(), kind)]
    return df.sort_values("edge", ascending=False).reset_index(drop=True)


def side_balance(df: pd.DataFrame) -> dict | None:
    """Warning when more than SIDE_BALANCE of the slate's plays are one side (needs at
    least SIDE_BALANCE_MIN_PLAYS plays). None if balanced."""
    if df is None or df.empty or "play" not in df:
        return None
    plays = df[df["play"].astype(bool)]
    n = len(plays)
    if n < SIDE_BALANCE_MIN_PLAYS:
        return None
    share = (plays["side"] == "less").mean()
    side, frac = ("less", share) if share >= 0.5 else ("more", 1 - share)
    if frac <= SIDE_BALANCE:
        return None
    return {"side": side, "share": float(frac), "n": n}


# --------------------------------------------------------------------- slips ----
def slip_summary(sims: dict[str, SeriesSim], picks: list[dict], platform: str,
                 payout: float | None = None) -> dict:
    """picks: [{matchup, player, market, line, side ('more'|'less')}].

    Joint probability from shared sims within a matchup, multiplied across matchups.
    payout overrides the platform's multiplier (goblin / demon legs change it).
    """
    legs = [Leg(p["matchup"], p["player"], p["market"], float(p["line"]),
                "over" if p["side"] == "more" else "under") for p in picks]
    sp = slip_probability(sims, legs)
    n = len(legs)
    table = PLATFORMS[platform]
    mult = payout if payout else table.get(n)
    # Shown (W) and calibrated joints: the shared-sim joint scaled by each leg's shown (or
    # calibrated) / raw P, which keeps the within-matchup correlation of the sims; capped
    # at the weakest leg.
    p_raw = sp["p_all_hit"]
    joint = {"": p_raw, "cal_": p_raw}
    if picks and all(p.get("calibrated") and p.get("raw_p_more") is not None for p in picks):
        for pre in joint:
            ratio = np.prod([_leg_p(p, pre) / max(_leg_p(p, "raw_"), 1e-9) for p in picks])
            joint[pre] = float(min(p_raw * ratio, min(_leg_p(p, pre) for p in picks)))
    out = {"n_legs": n, "p_joint": joint[""], "p_joint_cal": joint["cal_"],
           "p_joint_raw": p_raw, "p_independent": sp["p_independent"], "payout": mult,
           "ev": None, "ev_cal": None, "ev_raw": None}
    if mult is not None:
        out["ev"] = joint[""] * mult - 1.0
        out["ev_cal"] = joint["cal_"] * mult - 1.0
        out["ev_raw"] = p_raw * mult - 1.0
    return out


def _leg_p(p: dict, pre: str) -> float:
    """P of the leg's chosen side: pre "" shown (W), "cal_" calibrated, "raw_" raw. The
    leg's side may differ from the row's model side."""
    return float(p[pre + "p_" + ("more" if p["side"] == "more" else "less")])
