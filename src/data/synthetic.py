"""Synthetic match data in Oracle's Elixir schema.

Used for tests (small fixture) and as a sandbox dataset for the backtest when the real OE
CSV cannot be downloaded. The generative process is deliberately *not* the engine's model
(kills are generated as a game total that is then split winner/loser, so team kills are
correlated within a game; shares drift per player; patches shift the environment), so a
backtest on it is a meaningful smoke test of mechanics and calibration - but it is NOT
evidence about real-world edge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from zlib import crc32

import numpy as np
import pandas as pd

from src.config import POSITION_PRIORS, POSITIONS

def _stable_hash(s: str) -> int:
    return crc32(s.encode())


ANIMALS = [
    "Tigers", "Wolves", "Falcons", "Sharks", "Dragons", "Ravens", "Bears", "Vipers",
    "Lions", "Owls", "Foxes", "Hawks", "Cobras", "Bulls", "Orcas", "Lynx",
]

# league: n_teams, strength mean, combined kills/game env, regular BO, playoff BO
LEAGUE_SPEC = {
    "LCK": (10, 0.40, 26.5, 3, 5),
    "LPL": (10, 0.40, 31.0, 3, 5),
    "LEC": (10, 0.00, 28.5, 1, 5),
    "LCS": (8, -0.20, 26.0, 1, 5),
    "LCKC": (8, -0.45, 28.0, 3, 3),
    "LFL": (8, -0.55, 30.0, 1, 5),
}
EM_EXTRA_TEAMS = 6
EVENT_TRUE_FACTOR = {"FST": 0.86, "MSI": 1.03, "WLDs": 0.97}
BASE_LEN = 31.5


@dataclass
class Team:
    name: str
    league: str
    strength: float
    aggression: float
    roster: dict = field(default_factory=dict)  # pos -> (playername, base_share)


class Generator:
    def __init__(self, seed: int = 7, leagues: dict | None = None):
        self.rng = np.random.default_rng(seed)
        self.spec = leagues or LEAGUE_SPEC
        self.teams: dict[str, Team] = {}
        self.rows: list[dict] = []
        self.game_counter = 0
        self.player_counter = 0
        self.partial_rate = 0.02
        for lg, (n, mu, *_r) in self.spec.items():
            for i in range(n):
                self._new_team(f"{lg} {ANIMALS[i]}", lg, mu)
        if "LFL" in self.spec:
            for i in range(EM_EXTRA_TEAMS):
                self._new_team(f"ERL {ANIMALS[i]}", "ERL", -0.6)

    # ---------------------------------------------------------------- entities ----
    def _new_player(self, pos: str):
        self.player_counter += 1
        share = POSITION_PRIORS[pos] * float(np.exp(self.rng.normal(0, 0.18)))
        return (f"P{self.player_counter:04d}", share)

    def _new_team(self, name, league, mu):
        t = Team(name, league, float(self.rng.normal(mu, 0.35)),
                 float(np.exp(self.rng.normal(0, 0.08))))
        t.roster = {p: self._new_player(p) for p in POSITIONS}
        self.teams[name] = t

    def offseason(self):
        for t in self.teams.values():
            t.strength = 0.6 * t.strength + 0.4 * self.spec.get(t.league, (0, -0.6))[1] \
                + self.rng.normal(0, 0.25)
            for p in POSITIONS:
                if self.rng.random() < 0.3:
                    t.roster[p] = self._new_player(p)

    def drift(self):
        for t in self.teams.values():
            t.strength += self.rng.normal(0, 0.12)

    def league_teams(self, league):
        return [t for t in self.teams.values() if t.league == league]

    # ---------------------------------------------------------------- environment --
    def patch_for(self, date: pd.Timestamp) -> str:
        n = (date.dayofyear - 1) // 14 + 1
        return f"{date.year % 100}.{n:02d}"

    def patch_mult(self, patch: str) -> float:
        h = _stable_hash(f"{patch}|{self.seed_salt}") % 10_000 / 10_000
        return float(np.exp((h - 0.5) * 0.16))

    seed_salt = 0

    def env_for(self, league, a: Team, b: Team) -> float:
        if league in self.spec:
            base = self.spec[league][2]
        elif league == "EM":
            base = 29.0
        else:
            home = [self.spec.get(t.league, (0, 0, 29.0))[2] for t in (a, b)]
            base = float(np.mean(home)) * EVENT_TRUE_FACTOR.get(league, 1.0)
        return base

    # ---------------------------------------------------------------- games --------
    def play_series(self, league, a: Team, b: Team, best_of: int, date, split, playoffs):
        need = best_of // 2 + 1
        wins = {a.name: 0, b.name: 0}
        g = 0
        subs = {}
        for t in (a, b):  # occasional one-series substitute
            if self.rng.random() < 0.03:
                pos = POSITIONS[self.rng.integers(5)]
                subs[(t.name, pos)] = self._new_player(pos)
        while max(wins.values()) < need:
            g += 1
            blue, red = (a, b) if (g % 2 == 1) else (b, a)
            w = self.play_game(league, blue, red, date + pd.Timedelta(minutes=50 * g),
                               g, split, playoffs, subs)
            wins[w.name] += 1
        return a if wins[a.name] >= need else b

    def play_game(self, league, blue: Team, red: Team, date, game_no, split, playoffs, subs):
        rng = self.rng
        self.game_counter += 1
        gid = f"SYN{self.game_counter:06d}"
        p_blue = 1 / (1 + np.exp(-(1.25 * (blue.strength - red.strength) + 0.06)))
        blue_wins = rng.random() < p_blue
        win, lose = (blue, red) if blue_wins else (red, blue)
        mismatch = abs(2 * p_blue - 1)
        length = max(20.0, BASE_LEN * np.exp(rng.normal(0, 0.14)) * (1 - 0.10 * mismatch))
        patch = self.patch_for(date)
        env = self.env_for(league, blue, red) * self.patch_mult(patch)
        env *= blue.aggression * red.aggression
        mean_total = env * (length / BASE_LEN) ** 0.7
        r = 14.0
        total = rng.negative_binomial(r, r / (r + mean_total))
        share = rng.beta(7.0 + 4 * mismatch, 3.3)
        wk = int(round(total * share))
        kills = {win.name: wk, lose.name: total - wk}
        complete = "complete" if rng.random() > self.partial_rate else "partial"

        team_players = {}
        for t in (blue, red):
            roster = {p: subs.get((t.name, p), t.roster[p]) for p in POSITIONS}
            base = np.array([roster[p][1] for p in POSITIONS])
            base = base / base.sum()
            shares = rng.dirichlet(30 * base)
            pk = rng.multinomial(kills[t.name], shares)
            team_players[t.name] = (roster, pk)

        year = date.year
        for side, t, opp in (("Blue", blue, red), ("Red", red, blue)):
            roster, pk = team_players[t.name]
            deaths = rng.multinomial(kills[opp.name], [0.21, 0.19, 0.19, 0.19, 0.22])
            assists = rng.multinomial(int(kills[t.name] * 1.9), [0.17, 0.23, 0.2, 0.15, 0.25])
            res = int(t is win)
            common = dict(
                gameid=gid, datacompleteness=complete, url="", league=league, year=year,
                split=split, playoffs=int(playoffs), date=date.strftime("%Y-%m-%d %H:%M:%S"),
                game=game_no, patch=patch, side=side, teamname=t.name,
                teamid=f"oe:team:{_stable_hash(t.name) % 10**8:08d}", gamelength=round(length * 60),
                result=res, teamkills=kills[t.name], teamdeaths=kills[opp.name],
            )
            base_pid = 1 if side == "Blue" else 6
            for i, p in enumerate(POSITIONS):
                name = roster[p][0]
                self.rows.append(dict(
                    common, participantid=base_pid + i, position=p, playername=name,
                    playerid=f"oe:player:{name}", champion="Champ",
                    kills=int(pk[i]), deaths=int(deaths[i]), assists=int(assists[i]),
                ))
            self.rows.append(dict(
                common, participantid=100 if side == "Blue" else 200, position="team",
                playername=None, playerid=None, champion=None,
                kills=kills[t.name], deaths=kills[opp.name], assists=int(assists.sum()),
            ))
        return win

    # ---------------------------------------------------------------- schedule -----
    def round_robin(self, league, teams, start, end, best_of, split, double=True):
        pairs = list(combinations(teams, 2))
        if double:
            pairs = pairs + [(b, a) for a, b in pairs]
        order = self.rng.permutation(len(pairs))
        days = (end - start).days
        for k, idx in enumerate(order):
            a, b = pairs[idx]
            day = start + pd.Timedelta(days=int(k * days / len(pairs)))
            self.play_series(league, a, b, best_of, day + pd.Timedelta(hours=8 + (k % 3) * 3),
                             split, False)

    def knockout(self, league, teams, start, best_of, split):
        teams = sorted(teams, key=lambda t: -t.strength + self.rng.normal(0, 0.3))
        day = start
        while len(teams) > 1:
            nxt = []
            half = len(teams) // 2
            for i in range(half):
                a, b = teams[i], teams[-(i + 1)]
                nxt.append(self.play_series(league, a, b, best_of,
                                            day + pd.Timedelta(hours=9 + 4 * i), split, True))
            teams = nxt
            day += pd.Timedelta(days=3)

    def standings_top(self, league, n):
        ts = self.league_teams(league)
        return sorted(ts, key=lambda t: -t.strength)[:n]

    def season(self, year: int, end: pd.Timestamp | None = None):
        Y = lambda m, d: pd.Timestamp(year, m, d)  # noqa: E731
        plan = [
            ("split1", Y(1, 12), Y(3, 10)),
            ("FST", Y(3, 22), Y(3, 30)),
            ("split2", Y(4, 5), Y(6, 15)),
            ("MSI", Y(6, 25), Y(7, 10)),
            ("split3", Y(7, 18), Y(9, 5)),
            ("EM", Y(9, 6), Y(9, 20)),
            ("WLDs", Y(9, 28), Y(11, 2)),
        ]
        for name, start, stop in plan:
            if end is not None and start > end:
                return
            self.drift()
            if name.startswith("split"):
                for lg, (_n, _mu, _env, reg_bo, po_bo) in self.spec.items():
                    ts = self.league_teams(lg)
                    self.round_robin(lg, ts, start, stop - pd.Timedelta(days=8), reg_bo,
                                     name, double=(name != "split1"))
                    self.knockout(lg, self.standings_top(lg, 4),
                                  stop - pd.Timedelta(days=6), po_bo, name)
            elif name == "EM" and "LFL" in self.spec:
                ts = self.standings_top("LFL", 4) + self.league_teams("ERL")
                self.round_robin("EM", ts, start, stop - pd.Timedelta(days=4), 1, "EM",
                                 double=False)
                self.knockout("EM", sorted(ts, key=lambda t: -t.strength)[:4],
                              stop - pd.Timedelta(days=3), 5, "EM")
            elif name in EVENT_TRUE_FACTOR:
                majors = [lg for lg in ("LCK", "LPL", "LEC", "LCS") if lg in self.spec]
                k = {"FST": 2, "MSI": 2, "WLDs": 4}[name]
                ts = [t for lg in majors for t in self.standings_top(lg, k)]
                bo = 1 if name == "WLDs" else 3
                self.round_robin(name, ts, start, stop - pd.Timedelta(days=9), bo, name,
                                 double=False)
                self.knockout(name, sorted(ts, key=lambda t: -t.strength)[:4],
                              stop - pd.Timedelta(days=7), 5, name)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def generate(years=(2025, 2026), end="2026-10-12", seed=7, leagues=None,
             extra_ldl=True) -> pd.DataFrame:
    gen = Generator(seed=seed, leagues=leagues)
    gen.seed_salt = seed
    end_ts = pd.Timestamp(end)
    for i, y in enumerate(years):
        if i:
            gen.offseason()
        gen.season(y, end=end_ts)
    df = gen.frame()
    if extra_ldl:  # out-of-scope league rows to exercise the league filter
        ldl = Generator(seed=seed + 1, leagues={"LDL": (4, -0.6, 30.0, 3, 3)})
        ldl.round_robin("LDL", ldl.league_teams("LDL"), pd.Timestamp(years[0], 2, 1),
                        pd.Timestamp(years[0], 2, 20), 3, "split1", double=False)
        ld = ldl.frame()
        ld["gameid"] = "LDL" + ld["gameid"]
        df = pd.concat([df, ld], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= end_ts + pd.Timedelta(days=1)]
    df["date"] = df["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df.reset_index(drop=True)


def generate_fixture(seed=11) -> pd.DataFrame:
    """Small fixture: two leagues, one split + FST + partial split 2 (~1.5k rows)."""
    spec = {"LCK": (6, 0.4, 26.5, 3, 5), "LEC": (6, 0.0, 28.5, 1, 5)}
    return generate(years=(2025,), end="2025-04-20", seed=seed, leagues=spec)
