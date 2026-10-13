"""Data audit printed by the refresh job and saved to reports/data_audit.md."""
from __future__ import annotations

import pandas as pd

from src.config import POSITIONS

EXPECTED_LEAGUES = ["LCK", "LPL", "LEC", "LCS", "LCKC", "EM", "LFL", "FST", "WLDs", "MSI"]


def _md(df: pd.DataFrame) -> str:
    df = df.astype(str)
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    return "\n".join([head, sep] + ["| " + " | ".join(r) + " |" for r in df.values.tolist()])


def audit(raw: pd.DataFrame, players: pd.DataFrame, teams: pd.DataFrame,
          sources: list[str] | None = None) -> tuple[str, list[str]]:
    """Return (markdown report, list of missing expected league codes)."""
    raw_games = raw.drop_duplicates("gameid")[["gameid", "league", "datacompleteness"]]
    found = sorted(raw["league"].dropna().astype(str).unique())
    missing = [lg for lg in EXPECTED_LEAGUES if lg not in found]

    per_league = raw_games.groupby("league").agg(
        raw_games=("gameid", "size"),
        incomplete_games=("datacompleteness", lambda s: int((s != "complete").sum())),
    )
    per_league["raw_rows"] = raw.groupby("league").size()
    kept = teams.groupby("league", observed=True).agg(
        kept_games=("gameid", "nunique"), kept_series=("series_id", "nunique"))
    per_league = per_league.join(kept, how="left").fillna(0).astype(int)
    per_league["in_scope"] = [lg in EXPECTED_LEAGUES for lg in per_league.index]
    per_league = per_league.reset_index().sort_values(
        ["in_scope", "raw_games"], ascending=[False, False])

    dates = pd.to_datetime(raw["date"], errors="coerce")
    patches = raw["patch"].dropna().astype(str)
    from src.data.oe import patch_key
    pk = sorted(patches.unique(), key=patch_key)
    in_scope = raw_games[raw_games["league"].isin(EXPECTED_LEAGUES)]
    series = teams.drop_duplicates("series_id")
    lines = [
        "# Data audit", "",
        f"Sources: {', '.join(sources) if sources else 'n/a'}  ",
        f"Raw rows: {len(raw):,}; raw games: {len(raw_games):,}  ",
        f"Date range: {dates.min()} -> {dates.max()}  ",
        f"Patch range: {pk[0] if pk else 'n/a'} -> {pk[-1] if pk else 'n/a'} "
        f"({len(pk)} patches)  ",
        f"Kept (complete, in-scope): {teams['gameid'].nunique():,} games, "
        f"{series.shape[0]:,} series, {len(players):,} player rows  ",
        f"Incomplete in-scope games dropped: "
        f"{int((in_scope['datacompleteness'] != 'complete').sum()):,}  ",
        f"Series with a missing map (excluded from backtest): "
        f"{int((~series['series_complete'].astype(bool)).sum()) if len(series) else 0}  ",
        f"Best-of inferred: {series['best_of'].value_counts().sort_index().to_dict()}", "",
        "## Expected league codes", "",
        "MISSING: " + ", ".join(missing) if missing else "All expected league codes found.",
        "", "## League codes found in the file", "", ", ".join(found), "",
        "## Per league", "", _md(per_league), "",
    ]
    pos = players["position"].astype(str).value_counts()
    odd = sorted(set(pos.index) - set(POSITIONS))
    if odd:
        lines += [f"Unexpected positions in player rows: {odd}", ""]
    return "\n".join(lines), missing
