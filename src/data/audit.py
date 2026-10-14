"""Data audit printed by the refresh job and saved to reports/data_audit.md."""
from __future__ import annotations

import pandas as pd

from src.config import LEAGUE_CODE_ALIASES, POSITIONS

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
    found_canonical = set(found) | {LEAGUE_CODE_ALIASES.get(lg, lg) for lg in found}
    missing = [lg for lg in EXPECTED_LEAGUES if lg not in found_canonical]

    per_league = raw_games.groupby("league").agg(
        raw_games=("gameid", "size"),
        incomplete_games=("datacompleteness", lambda s: int((s != "complete").sum())),
    )
    per_league["raw_rows"] = raw.groupby("league").size()
    # Count kept games under the raw OE code so aliased codes get their own row.
    raw_code = raw_games.set_index("gameid")["league"]
    kept = teams.assign(league=teams["gameid"].map(raw_code)).groupby("league").agg(
        kept_games=("gameid", "nunique"), kept_series=("series_id", "nunique"))
    per_league = per_league.join(kept, how="left").fillna(0).astype(int)
    # Aliased codes (e.g. 2025 "LTA N" -> "LCS") are counted under their canonical code.
    per_league["maps_to"] = [LEAGUE_CODE_ALIASES.get(lg, "") for lg in per_league.index]
    per_league["in_scope"] = [lg in EXPECTED_LEAGUES or LEAGUE_CODE_ALIASES.get(lg)
                              in EXPECTED_LEAGUES for lg in per_league.index]
    per_league = per_league.reset_index().sort_values(
        ["in_scope", "raw_games"], ascending=[False, False])

    dates = pd.to_datetime(raw["date"], errors="coerce")
    patches = raw["patch"].dropna().astype(str)
    from src.data.oe import patch_key
    pk = sorted(patches.unique(), key=patch_key)
    in_scope = raw_games[raw_games["league"].replace(LEAGUE_CODE_ALIASES)
                         .isin(EXPECTED_LEAGUES)]
    series = teams.drop_duplicates("series_id")
    lines = [
        "# Data audit", "",
        f"Sources: {', '.join(sources) if sources else 'n/a'}  ",
        f"Raw rows: {len(raw):,}; raw games: {len(raw_games):,}  ",
        f"Date range: {dates.min()} -> {dates.max()}  ",
        f"Patch range: {pk[0] if pk else 'n/a'} -> {pk[-1] if pk else 'n/a'} "
        f"({len(pk)} patches)  ",
        f"Kept (complete + passing partial, in-scope): {teams['gameid'].nunique():,} games, "
        f"{series.shape[0]:,} series, {len(players):,} player rows  ",
        f"Incomplete in-scope games: "
        f"{int((in_scope['datacompleteness'] != 'complete').sum()):,} "
        f"(kept as passing partial: "
        f"{int(teams.drop_duplicates('gameid')['partial'].sum()) if 'partial' in teams else 0:,})  ",
        f"Series with a missing map (excluded from backtest): "
        f"{int((~series['series_complete'].astype(bool)).sum()) if len(series) else 0}  ",
        f"Best-of inferred: {series['best_of'].value_counts().sort_index().to_dict()}", "",
        "## Expected league codes", "",
        "MISSING: " + ", ".join(missing) if missing else "All expected league codes found.",
        "", "## League codes found in the file", "", ", ".join(found), "",
        "## League code aliases", "",
        (", ".join(f"{k} -> {v}" for k, v in LEAGUE_CODE_ALIASES.items())
         if LEAGUE_CODE_ALIASES else "none"), "",
        "## Per league", "", _md(per_league), "",
    ]
    inc = per_league[per_league["in_scope"]]
    inc = inc[inc["incomplete_games"] > 0.25 * inc["raw_games"]]
    if len(inc):
        lines += ["## Warnings", ""] + [
            f"- {r.league}: {r.incomplete_games}/{r.raw_games} games not 'complete' "
            f"(see partial games below)" for r in inc.itertuples()] + [""]
    lines += partial_section(raw)
    pos = players["position"].astype(str).value_counts()
    odd = sorted(set(pos.index) - set(POSITIONS))
    if odd:
        lines += [f"Unexpected positions in player rows: {odd}", ""]
    return "\n".join(lines), missing


def partial_section(raw: pd.DataFrame) -> list[str]:
    """Pass/fail counts of "partial" games in PARTIAL_OK_LEAGUES (src.data.oe.partial_check)."""
    from src.config import PARTIAL_OK_LEAGUES
    from src.data.oe import partial_check
    aliased = raw.assign(league=raw["league"].replace(LEAGUE_CODE_ALIASES))
    chk = partial_check(aliased)
    lines = [f"## Partial games ({', '.join(PARTIAL_OK_LEAGUES)})", "",
             "Kept only if kills/deaths/assists/result/gamelength and playername are "
             "non-null for all 10 players and player kills sum to team kills on both "
             "sides.", ""]
    if chk.empty:
        return lines + ["No partial games in these leagues.", ""]
    tab = chk.groupby(["league", "year"]).agg(
        partial_games=("gameid", "size"), passed=("passed", "sum"))
    tab["failed"] = tab["partial_games"] - tab["passed"]
    lines += [_md(tab.reset_index()), ""]
    fails = chk[~chk["passed"]]
    if len(fails):
        why = fails.groupby(["league", "year", "reason"]).size().rename("games")
        lines += ["Failure reasons:", "", _md(why.reset_index()), ""]
    lpl25 = chk[(chk["league"] == "LPL") & (chk["year"].astype(str) == "2025")]
    lines += [f"**LPL 2025 partial games: {int(lpl25['passed'].sum())} pass, "
              f"{int((~lpl25['passed']).sum())} fail.**", ""]
    return lines
