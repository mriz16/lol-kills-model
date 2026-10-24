"""Oracle's Elixir file uploaded in the app -> processed frames merged into the data.

The OE yearly CSV (or a .zip of it) is parsed with only the columns the pipeline reads
(`src.data.oe.REQUIRED_COLUMNS`) and explicit dtypes, run through the same rules as
before (complete games + partial games passing the kill checks, league aliases, series
inference, audit), then merged into the existing frames by `gameid`: an uploaded game
replaces the stored one, every other stored game (all of 2025 included) is kept.
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field

import pandas as pd

from src.config import LEAGUES
from src.data.audit import audit
from src.data.oe import (PLAYER_KEEP, REQUIRED_COLUMNS, SERIES_COLUMNS, TEAM_KEEP,
                         add_patch_ordinal, compact, infer_series, process)

# Low-cardinality text -> category; names / ids stay plain strings (they are grouped on,
# and a categorical groupby would expand to every category).
DTYPES = {
    "gameid": str, "datacompleteness": "category", "league": "category", "year": "Int16",
    "date": str, "game": "float32", "patch": str, "side": "category",
    "position": "category", "playername": str, "teamname": str, "gamelength": "float32",
    "result": "float32", "kills": "float32", "deaths": "float32", "assists": "float32",
    "teamkills": "float32", "teamdeaths": "float32",
}
assert set(DTYPES) == set(REQUIRED_COLUMNS)


class UploadError(ValueError):
    """The uploaded file is not a usable OE match file (message is shown in the app)."""


def _open_csv(src, name: str):
    """(binary stream of the CSV, its file name). A .zip must hold a .csv (largest wins)."""
    src.seek(0)
    magic = src.read(4)
    src.seek(0)
    if name.lower().endswith(".zip") or magic == b"PK\x03\x04":
        try:
            zf = zipfile.ZipFile(src)
        except zipfile.BadZipFile as exc:
            raise UploadError(f"{name}: not a valid .zip ({exc})") from exc
        csvs = [i for i in zf.infolist()
                if i.filename.lower().endswith(".csv") and not i.is_dir()
                and not i.filename.startswith("__MACOSX/")]
        if not csvs:
            raise UploadError(f"{name}: the .zip has no .csv inside")
        best = max(csvs, key=lambda i: i.file_size)
        return zf.open(best), best.filename
    return src, name


def read_oe_csv(src, name: str = "upload.csv") -> pd.DataFrame:
    """Raw OE rows (REQUIRED_COLUMNS only) from CSV or .zip; src is bytes or a seekable
    binary file object (e.g. Streamlit's UploadedFile, read in place without a copy)."""
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    src.seek(0)
    head = src.read(2048).lstrip().lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
        raise UploadError(f"{name}: this is a web page, not the OE CSV")
    stream, inner = _open_csv(src, name)
    header = pd.read_csv(stream, nrows=0).columns
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing:
        raise UploadError(f"{inner}: not an Oracle's Elixir match file, missing columns "
                          f"{missing}")
    stream, inner = _open_csv(src, name)
    try:
        raw = pd.read_csv(stream, usecols=REQUIRED_COLUMNS, dtype=DTYPES)
    except (ValueError, pd.errors.ParserError) as exc:
        raise UploadError(f"{inner}: could not parse ({exc})") from exc
    if raw.empty:
        raise UploadError(f"{inner}: 0 rows")
    return raw


def _plain(df: pd.DataFrame) -> pd.DataFrame:
    """Categoricals -> object so frames with different categories concat cleanly."""
    df = df.copy()
    for c in df.columns:
        if isinstance(df[c].dtype, pd.CategoricalDtype):
            df[c] = df[c].astype(str).astype(object)
    return df


def merge_frames(old_players: pd.DataFrame, old_teams: pd.DataFrame,
                 new_players: pd.DataFrame, new_teams: pd.DataFrame
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Union by gameid (new rows replace old ones of the same game), then series and patch
    ordinals are rebuilt over the union so a series split across uploads joins up."""
    new_ids = set(new_teams["gameid"].astype(str))
    teams = pd.concat([_plain(old_teams[~old_teams["gameid"].astype(str).isin(new_ids)]),
                       _plain(new_teams)], ignore_index=True)
    players = pd.concat([_plain(old_players[~old_players["gameid"].astype(str)
                                            .isin(new_ids)]),
                         _plain(new_players)], ignore_index=True)
    # the parquet has no OE `game` column: map_no is that game number
    teams["game"] = teams["map_no"]
    teams = teams.drop(columns=[c for c in SERIES_COLUMNS if c in teams.columns])
    players = players.drop(columns=[c for c in SERIES_COLUMNS if c in players.columns])
    teams["gameid"] = teams["gameid"].astype(str)
    players["gameid"] = players["gameid"].astype(str)
    teams["date"] = pd.to_datetime(teams["date"])
    players["date"] = pd.to_datetime(players["date"])
    series = infer_series(teams)
    teams = teams.drop(columns="game").merge(series, on="gameid")
    players = players.merge(series, on="gameid")
    add_patch_ordinal(teams, players)
    teams, players = compact(teams[TEAM_KEEP]), compact(players[PLAYER_KEEP])
    lost = set(old_teams["gameid"].astype(str)) - set(teams["gameid"])
    if lost:  # by construction impossible; never write a frame that lost games
        raise AssertionError(f"merge dropped {len(lost)} stored games")
    return players, teams


@dataclass
class UploadResult:
    players: pd.DataFrame
    teams: pd.DataFrame
    audit_md: str
    source: str
    games_added: int
    games_updated: int
    player_rows_added: int
    ranges: pd.DataFrame          # per league: games in upload, previous / new date range
    warnings: list[str] = field(default_factory=list)


def date_ranges(teams: pd.DataFrame) -> pd.DataFrame:
    t = teams[["league", "date", "gameid"]].astype({"league": str})
    return t.groupby("league").agg(first=("date", "min"), last=("date", "max"),
                                   games=("gameid", "nunique"))


def process_upload(data, name: str, old_players: pd.DataFrame | None,
                   old_teams: pd.DataFrame | None, leagues=LEAGUES) -> UploadResult:
    raw = read_oe_csv(data, name)
    new_players, new_teams = process(raw, leagues)
    if new_teams.empty:
        raise UploadError(f"{name}: 0 complete in-scope games after filtering "
                          "(wrong file, or league codes changed?)")
    audit_md, missing = audit(raw, new_players, new_teams, [name])
    del raw
    if old_teams is None or old_teams.empty:
        players, teams = new_players, new_teams
        old_ids: set = set()
        old_rng = pd.DataFrame(columns=["first", "last", "games"])
    else:
        players, teams = merge_frames(old_players, old_teams, new_players, new_teams)
        old_ids = set(old_teams["gameid"].astype(str))
        old_rng = date_ranges(old_teams)
    up_ids = set(new_teams["gameid"].astype(str))
    added = up_ids - old_ids
    new_rng = date_ranges(teams)
    up = new_teams.astype({"league": str}).groupby("league")["gameid"].nunique()
    added_by = new_teams[new_teams["gameid"].astype(str).isin(added)].astype(
        {"league": str}).groupby("league")["gameid"].nunique()
    ranges = pd.DataFrame({
        "in_upload": up, "added": added_by,
        "before_last": old_rng["last"] if len(old_rng) else pd.NaT,
        "first": new_rng["first"], "last": new_rng["last"], "games": new_rng["games"]})
    ranges = ranges.fillna({"in_upload": 0, "added": 0}).astype(
        {"in_upload": int, "added": int}).reset_index(names="league")
    ranges = ranges.sort_values("last", ascending=False).reset_index(drop=True)

    warnings = [f"League codes not in this file: {', '.join(missing)}."] if missing else []
    warnings += _audit_warnings(audit_md)
    if not added:
        warnings.append("No new games: every game in this file was already stored.")
    if len(old_rng) and new_teams["date"].max() < old_teams["date"].max():
        warnings.append(f"This file ends {new_teams['date'].max():%Y-%m-%d}, before the "
                        f"stored data ({old_teams['date'].max():%Y-%m-%d}): an old file?")
    return UploadResult(players, teams, audit_md, name, len(added), len(up_ids & old_ids),
                        int(new_players["gameid"].astype(str).isin(added).sum()), ranges,
                        warnings)


def _audit_warnings(md: str) -> list[str]:
    """Bullets under '## Warnings', international teams with no domestic games, and
    partial-game failures from the audit markdown."""
    out, in_warn = [], False
    for line in md.splitlines():
        if line.startswith("## "):
            in_warn = line.strip() == "## Warnings"
            continue
        if in_warn and line.startswith("- "):
            out.append(line[2:])
        if line.startswith("**Teams at international events with no domestic games") \
                and not line.rstrip().endswith("none."):
            out.append(line.replace("**", ""))
        if line.startswith("Failure reasons:"):
            out.append("Some partial games failed the kill checks and were dropped "
                       "(see the audit).")
    return out
