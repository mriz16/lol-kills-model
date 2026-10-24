"""Oracle's Elixir CSV processing: filter, split player/team rows, infer series, compact."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.config import LEAGUE_CODE_ALIASES, LEAGUES, PARTIAL_OK_LEAGUES, POSITIONS

# Only the OE columns the pipeline reads (filter, partial check, series, engine, audit).
# The app's uploader parses just these with explicit dtypes (src.data.upload).
REQUIRED_COLUMNS = [
    "gameid", "datacompleteness", "league", "year", "date", "game", "patch", "side",
    "position", "playername", "teamname", "gamelength", "result", "kills", "deaths",
    "assists", "teamkills", "teamdeaths",
]

PLAYER_COLUMNS = [
    "gameid", "league", "year", "date", "game", "patch", "side", "position", "playername",
    "teamname", "gamelength", "result", "kills", "deaths", "assists", "teamkills",
    "teamdeaths", "partial",
]
TEAM_COLUMNS = [
    "gameid", "league", "year", "date", "game", "patch", "side", "teamname", "gamelength",
    "result", "kills", "deaths", "assists", "teamkills", "teamdeaths", "partial",
]


# Columns kept in the committed parquet (everything the engine/backtest reads).
SERIES_COLUMNS = ["series_id", "map_no", "best_of", "series_complete", "patch_ord"]
# partial: game is datacompleteness == "partial" and passed partial_check (LPL 2025).
PLAYER_KEEP = ["gameid", "league", "date", "patch", "side", "position", "playername",
               "teamname", "result", "kills", "teamkills", "partial"] + SERIES_COLUMNS
TEAM_KEEP = ["gameid", "league", "date", "patch", "side", "teamname", "gamelength",
             "result", "teamkills", "teamdeaths", "partial"] + SERIES_COLUMNS


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


def normalize_patch(patch: pd.Series) -> pd.Series:
    """OE patch values as the strings the parquet carries: numeric patches go through float
    (a CSV parsed as text gives "15.10", parsed as numbers 15.1 -> "15.1"; both -> "15.1"),
    anything else ("25.S1.1") is kept as is."""
    num = pd.to_numeric(patch, errors="coerce")
    return patch.astype(object).where(num.isna(), num.astype(str))


def add_patch_ordinal(*frames: pd.DataFrame) -> None:
    """Add a global `patch_ord` (0..n) consistently across frames (in place)."""
    keys = sorted({patch_key(p) for f in frames for p in f["patch"].unique()})
    ordmap = {k: i for i, k in enumerate(keys)}
    for f in frames:
        f["patch_ord"] = f["patch"].map(lambda p: ordmap[patch_key(p)]).astype("int16")


PARTIAL_PLAYER_FIELDS = ["kills", "deaths", "assists", "result", "gamelength"]


def partial_check(raw: pd.DataFrame, leagues=PARTIAL_OK_LEAGUES) -> pd.DataFrame:
    """Per-game verdict for "partial" games in `leagues` (league codes already aliased).

    A game passes only if it has 2 team + 10 player rows, kills/deaths/assists/result/
    gamelength are non-null for all 10 players, every player has a name, and the sum of
    player kills equals team kills on both sides (team row `teamkills`, else `kills`).
    Returns one row per game: gameid, league, year, passed, reason ("" when passed).
    """
    cols = ["gameid", "league", "year", "passed", "reason"]
    part = raw[(raw["datacompleteness"] == "partial") & raw["league"].isin(leagues)]
    if part.empty:
        return pd.DataFrame(columns=cols)
    is_team = part["position"] == "team"
    pl = part[part["position"].isin(POSITIONS)]
    tm = part[is_team]
    g = part.groupby("gameid").agg(league=("league", "first"), year=("year", "first"))
    g["n_team"] = tm.groupby("gameid").size()
    g["n_player"] = pl.groupby("gameid").size()
    g = g.fillna({"n_team": 0, "n_player": 0})
    null_stats = pl[PARTIAL_PLAYER_FIELDS].isna().any(axis=1).groupby(pl["gameid"]).any()
    null_name = pl["playername"].isna().groupby(pl["gameid"]).any()
    pk = pd.to_numeric(pl["kills"], errors="coerce").groupby(
        [pl["gameid"], pl["side"]]).sum(min_count=1)
    team_k = pd.to_numeric(tm["teamkills"], errors="coerce").fillna(
        pd.to_numeric(tm["kills"], errors="coerce"))
    tk = team_k.groupby([tm["gameid"], tm["side"]]).first()
    both = pd.concat({"pk": pk, "tk": tk}, axis=1)
    mismatch = (both["pk"] != both["tk"]).groupby(level=0).any()  # NaN != x -> mismatch

    reason = pd.Series("", index=g.index, dtype=object)
    checks = [  # first failing check wins
        ((g["n_team"] != 2) | (g["n_player"] != 10), "row count != 2 team + 10 player"),
        (null_stats.reindex(g.index, fill_value=True), "null player kills/deaths/assists/"
                                                       "result/gamelength"),
        (null_name.reindex(g.index, fill_value=True), "null playername"),
        (mismatch.reindex(g.index, fill_value=True), "player kills != team kills"),
    ]
    for bad, why in reversed(checks):
        reason[bad.astype(bool)] = why
    g["reason"] = reason
    g["passed"] = reason == ""
    return g.reset_index()[cols]


def _fill_partial(df: pd.DataFrame) -> pd.DataFrame:
    """Fill team-level fields of partial games that passed partial_check from the player
    rows: teamkills/teamdeaths (player and team rows), team-row result and gamelength."""
    part = df["partial"]
    if not part.any():
        return df
    df = df.copy()
    pl = df[part & df["position"].isin(POSITIONS)]
    k = pd.to_numeric(pl["kills"], errors="coerce").groupby([pl["gameid"], pl["side"]]).sum()
    glen = pl.groupby("gameid")["gamelength"].first()
    key = pd.MultiIndex.from_arrays([df["gameid"], df["side"]])
    other = df["side"].map({"Blue": "Red", "Red": "Blue"})
    okey = pd.MultiIndex.from_arrays([df["gameid"], other])
    own = pd.Series(k.reindex(key).values, index=df.index)
    opp = pd.Series(k.reindex(okey).values, index=df.index)
    for col, fill in (("teamkills", own), ("teamdeaths", opp)):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[part, col] = df.loc[part, col].fillna(fill[part])
    res = pl.groupby(["gameid", "side"])["result"].first()
    df["result"] = pd.to_numeric(df["result"], errors="coerce")
    df.loc[part, "result"] = df.loc[part, "result"].fillna(
        pd.Series(res.reindex(key).values, index=df.index)[part])
    df["gamelength"] = pd.to_numeric(df["gamelength"], errors="coerce")
    df.loc[part, "gamelength"] = df.loc[part, "gamelength"].fillna(
        df.loc[part, "gameid"].map(glen))
    return df


def filter_raw(raw: pd.DataFrame, leagues=LEAGUES) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"OE data missing columns: {missing}")
    raw = raw.assign(league=raw["league"].replace(LEAGUE_CODE_ALIASES)) \
        if LEAGUE_CODE_ALIASES else raw
    chk = partial_check(raw[raw["league"].isin(leagues)])
    ok_partial = set(chk.loc[chk["passed"], "gameid"])
    keep = (raw["datacompleteness"] == "complete") | raw["gameid"].isin(ok_partial)
    df = raw[keep & raw["league"].isin(leagues)].copy()
    df["partial"] = df["datacompleteness"] != "complete"
    df = _fill_partial(df)
    df["patch"] = normalize_patch(df["patch"])
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
    for c in ("series_complete", "partial"):
        if c in df.columns:
            df[c] = df[c].astype(bool)
    for c in ("result", "playoffs", "game", "map_no", "best_of"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int8")
    if "year" in df.columns:
        df["year"] = df["year"].astype("int16")
    if "gamelength" in df.columns:
        df["gamelength"] = df["gamelength"].astype("float32")
    return df.sort_values(["date", "gameid"]).reset_index(drop=True)
