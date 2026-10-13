"""Oracle's Elixir CSV processing: filter, split player/team rows, infer series, compact."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.config import LEAGUES, POSITIONS

REQUIRED_COLUMNS = [
    "gameid", "datacompleteness", "league", "year", "split", "playoffs", "date", "game",
    "patch", "participantid", "side", "position", "playername", "playerid", "teamname",
    "teamid", "champion", "gamelength", "result", "kills", "deaths", "assists",
    "teamkills", "teamdeaths",
]

PLAYER_COLUMNS = [
    "gameid", "league", "year", "split", "playoffs", "date", "game", "patch", "side",
    "position", "playername", "playerid", "teamname", "teamid", "champion", "gamelength",
    "result", "kills", "deaths", "assists", "teamkills", "teamdeaths",
]
TEAM_COLUMNS = [
    "gameid", "league", "year", "split", "playoffs", "date", "game", "patch", "side",
    "teamname", "teamid", "gamelength", "result", "kills", "deaths", "assists",
    "teamkills", "teamdeaths",
]


# Columns kept in the committed parquet (everything the engine/backtest reads).
SERIES_COLUMNS = ["series_id", "map_no", "best_of", "series_complete", "patch_ord"]
PLAYER_KEEP = ["gameid", "league", "date", "patch", "side", "position", "playername",
               "teamname", "result", "kills", "teamkills"] + SERIES_COLUMNS
TEAM_KEEP = ["gameid", "league", "date", "patch", "side", "teamname", "gamelength",
             "result", "teamkills", "teamdeaths"] + SERIES_COLUMNS


def patch_key(patch) -> tuple:
    """Sortable key for OE patch values ("14.01", 14.1, "25.S1.1" ...)."""
    if patch is None or (isinstance(patch, float) and np.isnan(patch)):
        return (-1,)
    s = str(patch)
    if re.fullmatch(r"\d+\.\d+", s):
        major, minor = s.split(".")
        # OE floats lose zero padding (14.1 == 14.10); treat minor as 2-digit decimal.
        return (int(major), int(round(float("0." + minor) * 100)))
    return tuple(int(x) for x in re.findall(r"\d+", s)) or (-1,)


def add_patch_ordinal(*frames: pd.DataFrame) -> None:
    """Add a global `patch_ord` (0..n) consistently across frames (in place)."""
    keys = sorted({patch_key(p) for f in frames for p in f["patch"].unique()})
    ordmap = {k: i for i, k in enumerate(keys)}
    for f in frames:
        f["patch_ord"] = f["patch"].map(lambda p: ordmap[patch_key(p)]).astype("int16")


def filter_raw(raw: pd.DataFrame, leagues=LEAGUES) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"OE data missing columns: {missing}")
    df = raw[(raw["datacompleteness"] == "complete") & raw["league"].isin(leagues)].copy()
    df["date"] = pd.to_datetime(df["date"])
    # Drop games that are not exactly 2 team rows + 10 player rows.
    counts = df.groupby("gameid")["position"].agg(
        n_team=lambda s: (s == "team").sum(), n_player=lambda s: s.isin(POSITIONS).sum()
    )
    good = counts.index[(counts.n_team == 2) & (counts.n_player == 10)]
    return df[df["gameid"].isin(good)]


def infer_series(teams: pd.DataFrame) -> pd.DataFrame:
    """Return per-game frame with series_id, map_no and best_of.

    OE has no series id. Games of the same team pair in the same league are grouped in
    date order; a new series starts when the game number does not increase or when more
    than 1 day passed. best_of = 2*max(map wins)-1 (BO1/BO3/BO5).
    """
    t = teams.sort_values(["date", "gameid", "side"])
    g = t.groupby("gameid", sort=False).agg(
        date=("date", "first"), league=("league", "first"), game=("game", "first"),
        teams=("teamname", lambda s: tuple(sorted(s))),
    ).reset_index().sort_values(["date", "gameid"])
    winners = t[t["result"] == 1].set_index("gameid")["teamname"]
    g["winner"] = g["gameid"].map(winners)

    series_ids = np.empty(len(g), dtype=object)
    last: dict = {}
    counter = 0
    for i, (league, pair, game, date) in enumerate(
        zip(g["league"], g["teams"], g["game"], g["date"])
    ):
        key = (league, pair)
        prev = last.get(key)
        game = int(game) if pd.notna(game) else 1
        if prev is None or game <= prev[1] or (date - prev[2]) > pd.Timedelta(days=1):
            counter += 1
            sid = f"S{counter:06d}"
        else:
            sid = prev[0]
        last[key] = (sid, game, date)
        series_ids[i] = sid
    g["series_id"] = series_ids
    # Use OE's game number as the map number; it stays right when an incomplete game was
    # filtered out. Fall back to order within the series when it is missing.
    order = g.groupby("series_id").cumcount() + 1
    game = pd.to_numeric(g["game"], errors="coerce")
    g["map_no"] = game.where(game.notna() & (game >= 1), order).astype(int)
    wins = g.groupby(["series_id", "winner"]).size().groupby("series_id").max()
    max_map = g.groupby("series_id")["map_no"].max()
    n_maps = g.groupby("series_id").size()
    bo = np.maximum(2 * wins - 1, max_map)
    bo = bo.where(bo <= 1, bo + (bo % 2 == 0))  # round even up to the next odd format
    g["best_of"] = g["series_id"].map(bo).astype("int8")
    # A series is complete when maps 1..n are all present (no filtered-out game).
    g["series_complete"] = g["series_id"].map(max_map == n_maps)
    return g[["gameid", "series_id", "map_no", "best_of", "series_complete"]]


def process(raw: pd.DataFrame, leagues=LEAGUES) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw OE frame -> (players, teams) compact frames with series info."""
    df = filter_raw(raw, leagues)
    is_team = df["position"] == "team"
    teams = df.loc[is_team, TEAM_COLUMNS].copy()
    players = df.loc[~is_team & df["position"].isin(POSITIONS), PLAYER_COLUMNS].copy()
    series = infer_series(teams)
    teams = teams.merge(series, on="gameid")
    players = players.merge(series, on="gameid")
    add_patch_ordinal(teams, players)
    return compact(players[PLAYER_KEEP]), compact(teams[TEAM_KEEP])


def compact(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ("league", "split", "side", "position", "teamname", "teamid", "playername",
              "playerid", "champion", "patch", "series_id"):
        if c in df.columns:
            df[c] = df[c].astype(str).astype("category")
    for c in ("kills", "deaths", "assists", "teamkills", "teamdeaths"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int16")
    if "series_complete" in df.columns:
        df["series_complete"] = df["series_complete"].astype(bool)
    for c in ("result", "playoffs", "game", "map_no", "best_of"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int8")
    if "year" in df.columns:
        df["year"] = df["year"].astype("int16")
    if "gamelength" in df.columns:
        df["gamelength"] = df["gamelength"].astype("float32")
    return df.sort_values(["date", "gameid"]).reset_index(drop=True)
