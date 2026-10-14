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
from src.engine.features import (EngineConfig, History, MatchupParams, build_matchup,
                                 fit_kappa_by_position, locked_config)
from src.engine.props import Leg, breakeven, prop_summary, slip_probability
from src.engine.sim import SeriesSim, simulate_series

PLATFORMS = {"PrizePicks": C.PRIZEPICKS_POWER, "Underdog": C.UNDERDOG}
MARKET_BY_FORMAT = {1: "Map 1", 3: "Maps 1-2", 5: "Maps 1-3"}
ROSTER_DAYS = 60            # players seen for the team in the last N days of its games
THIN_EDGE = 0.03            # flag edges under 3 points
LOW_SAMPLE_GAMES = 20
CAUTION_P_MORE = 0.25       # "less" at a whole line with P(more) < 25%: overs hit ~9 pts more


def per_leg_breakeven(platform: str, legs: int) -> float:
    """(1 / multiplier) ** (1 / legs) from the engine payout tables."""
    return breakeven(PLATFORMS[platform][legs], legs)


# --------------------------------------------------------------------- data store ----
class DataStore:
    """Processed OE frames + History + rosters, built once per data version."""

    def __init__(self, players: pd.DataFrame, teams: pd.DataFrame):
        self.players, self.teams = players, teams
        self.hist = History(players, teams)
        self.last_date = pd.Timestamp(teams["date"].max())
        self.cutoff = int((self.last_date + pd.Timedelta(days=1)).value)
        self.patch = int(teams["patch_ord"].max())
        self.patch_name = str(teams.loc[teams["patch_ord"] == self.patch, "patch"].iloc[-1])
        self.kappa_pos = fit_kappa_by_position(self.hist, self.cutoff)
        self._rosters = _build_rosters(players)
        last = teams.sort_values("date").groupby("teamname", observed=True)["date"].max()
        # most recently active first: the order of the team pickers
        self.team_names = [str(t) for t in last.sort_values(ascending=False).index]

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
    res.rows = [prop_row(res, lm, legs) for lm in lines]
    return res


# --------------------------------------------------------------------- prop rows ----
def flags_for(side: str, edge: float, p_more: float, line: float, n_games: int) -> list[str]:
    out = []
    if edge < THIN_EDGE:
        out.append("THIN EDGE")
    if n_games < LOW_SAMPLE_GAMES:
        out.append("LOW SAMPLE")
    if side == "less" and p_more < CAUTION_P_MORE and float(line).is_integer():
        out.append("CAUTION")
    return out


def prop_row(res: MatchupResult, lm: LineMatch, legs: int) -> dict:
    m, sim = res.inp, res.sim
    kills = sim.range_kills(lm.player, m.market)
    s = prop_summary(kills, lm.line)
    side = "more" if s["p_over"] >= s["p_under"] else "less"
    p_side = s["p_over"] if side == "more" else s["p_under"]
    be = per_leg_breakeven(m.platform, legs)
    side_idx, j = sim.player_index(lm.player)
    share, n_games = res.params.info.get(f"share_{side_idx}_{j}", (np.nan, 0))
    opp = m.team_b if lm.team == m.team_a else m.team_a
    row = dict(
        matchup=m.key, platform=m.platform, league=res.league, best_of=int(m.best_of),
        market=m.market, player=lm.player, team=lm.team, opponent=opp, role=lm.role,
        line=float(lm.line), p_more=s["p_over"], p_less=s["p_under"], p_push=s["p_push"],
        mean=s["mean"], median=s["median"], fair_line=s["fair_line"], side=side,
        p_side=p_side, breakeven=be, edge=p_side - be, n_games=int(n_games),
        share=float(share),
    )
    row["flags"] = " ".join(flags_for(side, row["edge"], row["p_more"], row["line"], n_games))
    row["why"] = explain(res, lm, row)
    return row


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
    return (f"{lm.team} projects {win_k:.1f} team kills/map in wins and {loss_k:.1f} in "
            f"losses ({res.league} env {info.get('target_env', float('nan')):.1f}); "
            f"{lm.player} ({lm.role}) takes {share} of team kills ({n} games); {adj}; "
            f"{lm.team} wins {p_map:.0%} of maps ({p_ser:.0%} series). "
            f"Model mean {row['mean']:.1f} over {res.inp.market} vs line {row['line']:g}.")


def results_table(results: list[MatchupResult]) -> pd.DataFrame:
    rows = [r for res in results for r in res.rows]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("edge", ascending=False).reset_index(drop=True)
    df["leg_id"] = [f"{r.matchup}|{r.player}|{r.line:g}" for r in df.itertuples()]
    return df


def rebreakeven(df: pd.DataFrame, legs: int) -> pd.DataFrame:
    """Recompute breakeven / edge / flags for a different slip size (no re-sim)."""
    if df.empty:
        return df
    df = df.copy()
    df["breakeven"] = [per_leg_breakeven(p, legs) for p in df["platform"]]
    df["edge"] = df["p_side"] - df["breakeven"]
    df["flags"] = [" ".join(flags_for(r.side, r.edge, r.p_more, r.line, r.n_games))
                   for r in df.itertuples()]
    return df.sort_values("edge", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------- slips ----
def slip_summary(sims: dict[str, SeriesSim], picks: list[dict], platform: str) -> dict:
    """picks: [{matchup, player, market, line, side ('more'|'less')}].

    Joint probability from shared sims within a matchup, multiplied across matchups.
    """
    legs = [Leg(p["matchup"], p["player"], p["market"], float(p["line"]),
                "over" if p["side"] == "more" else "under") for p in picks]
    sp = slip_probability(sims, legs)
    n = len(legs)
    table = PLATFORMS[platform]
    mult = table.get(n)
    out = {"n_legs": n, "p_joint": sp["p_all_hit"], "p_independent": sp["p_independent"],
           "payout": mult, "ev": None}
    if mult is not None:
        out["ev"] = sp["p_all_hit"] * mult - 1.0
    return out
