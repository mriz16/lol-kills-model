import pandas as pd

from src.backtest.elo import Elo
from src.backtest.metrics import crps_discrete, log_score
from src.data.oe import patch_key, process
from src.data.teams import resolve


def test_process_filters_and_splits(raw, processed):
    players, teams = processed
    assert set(players["league"].unique()) <= {"LCK", "LEC", "FST"}  # LDL dropped
    assert "partial" in set(raw["datacompleteness"])
    kept = set(teams["gameid"])
    partial = set(raw.loc[raw["datacompleteness"] != "complete", "gameid"])
    # partial games are kept only when they pass the kill-consistency check (the
    # fixture's FST partial game does) and are flagged
    from src.data.oe import partial_check
    ok = set(partial_check(raw).query("passed")["gameid"])
    assert kept & partial == ok & partial and ok
    assert teams.loc[teams["gameid"].isin(partial), "partial"].all()
    assert (teams.groupby("gameid").size() == 2).all()
    assert (players.groupby("gameid").size() == 10).all()
    # team kills = sum of player kills
    s = players.groupby(["gameid", "teamname"], observed=True)["kills"].sum()
    tk = teams.set_index(["gameid", "teamname"])["teamkills"]
    assert (s.sort_index().values == tk.sort_index().values).all()
    assert players["kills"].dtype.name == "int16"


def test_series_inference(processed):
    _, teams = processed
    g = teams.drop_duplicates("gameid")
    ser = g.groupby("series_id").agg(bo=("best_of", "first"), n=("gameid", "size"),
                                     league=("league", "first"))
    assert set(ser["bo"]) <= {1, 3, 5}
    assert (ser["n"] <= ser["bo"]).all()
    # LEC regular season in the fixture is BO1, LCK is BO3
    assert (ser.loc[ser.league == "LEC", "bo"] == 1).mean() > 0.6
    assert (ser.loc[ser.league == "LCK", "bo"] >= 3).all()


def test_patch_key():
    assert patch_key("14.01") < patch_key("14.10") < patch_key("15.01")
    assert patch_key(14.1) == patch_key("14.10")
    assert patch_key("25.S1.1") < patch_key("25.S1.2")


def test_team_resolve():
    known = ["Gen.G", "T1", "Hanwha Life Esports", "G2 Esports", "Team Liquid"]
    assert resolve("GEN", known) == "Gen.G"
    assert resolve("gen.g esports", known) == "Gen.G"
    assert resolve("HLE", known) == "Hanwha Life Esports"
    assert resolve("t1", known) == "T1"
    assert resolve("G2", known) == "G2 Esports"
    assert resolve("Team Liqiud", known) == "Team Liquid"  # typo -> fuzzy
    assert resolve("Totally Unknown Squad", known) is None
    # OE 2026 names
    assert resolve("DK", ["Dplus Kia", "T1"]) == "Dplus Kia"
    assert resolve("BRO", ["HANJIN BRION", "T1"]) == "HANJIN BRION"
    assert resolve("DNF", ["DN SOOPers", "T1"]) == "DN SOOPers"


def test_metrics():
    import numpy as np
    s = np.array([3] * 100)
    assert crps_discrete(s, 3) == 0.0
    assert crps_discrete(s, 5) == 2.0
    assert log_score(np.array([1, 2, 2, 3]), 2) < log_score(np.array([1, 2, 2, 3]), 3)


def test_elo():
    e = Elo()
    assert e.p_map("A", "B", "LCK") == 0.5
    assert e.p_map("C", "D", "LCS") == 0.5
    assert e.p_map("A", "C") > 0.5  # LCK prior > LCS prior
    e.update_map("A", "B", True)
    assert e.p_map("A", "B") > 0.5


def test_refresh_script_on_local_csv(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, str(root / "scripts" / "refresh_data.py"), "--csv",
                        str(root / "tests" / "fixtures" / "oe_fixture.csv"), "--out",
                        str(tmp_path), "--audit", str(tmp_path / "audit.md")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "audit.md").exists()
    assert len(pd.read_parquet(tmp_path / "players.parquet")) > 0


def test_league_code_aliases(raw, monkeypatch):
    import src.data.oe as oe
    monkeypatch.setattr(oe, "LEAGUE_CODE_ALIASES", {"LDL": "LPL"})
    players, teams = oe.process(raw)
    assert "LPL" in set(teams["league"].astype(str))


def test_lta_north_maps_to_lcs(raw):
    import src.data.oe as oe
    from src.config import LEAGUE_CODE_ALIASES
    assert LEAGUE_CODE_ALIASES["LTA N"] == "LCS"
    lcs = raw["league"] == "LEC"  # fixture has no LCS: relabel one league as LTA N
    players, teams = oe.process(raw.assign(league=raw["league"].mask(lcs, "LTA N")))
    assert "LCS" in set(teams["league"].astype(str))
    assert "LTA N" not in set(teams["league"].astype(str))


def _lpl_partial(raw, n=4):
    """Relabel the first n LCK games as LPL 'partial' (team rows stripped of teamkills)."""
    lck = raw[(raw["league"] == "LCK") & (raw["datacompleteness"] == "complete")]
    ids = list(dict.fromkeys(lck["gameid"]))[:n]
    r = raw.copy()
    m = r["gameid"].isin(ids)
    r.loc[m, "league"] = "LPL"
    r.loc[m, "datacompleteness"] = "partial"
    r.loc[m, ["teamkills", "teamdeaths"]] = None  # OE partial rows may lack these
    r.loc[m & (r["position"] == "team"), "kills"] = \
        raw.loc[m & (raw["position"] == "team"), "teamkills"]
    return r, ids


def test_partial_check_and_keep(raw):
    from src.data.oe import partial_check, process
    r, ids = _lpl_partial(raw)
    # game 1: a player's deaths missing; game 2: player kills don't sum to team kills
    p1 = r.index[(r["gameid"] == ids[1]) & (r["position"] == "mid")][0]
    r.loc[p1, "deaths"] = None
    p2 = r.index[(r["gameid"] == ids[2]) & (r["position"] == "top")][0]
    r.loc[p2, "kills"] = r.loc[p2, "kills"] + 1
    chk = partial_check(r).set_index("gameid")
    assert chk.loc[ids[0], "passed"] and chk.loc[ids[3], "passed"]
    assert chk.loc[ids[1], "reason"].startswith("null player")
    assert chk.loc[ids[2], "reason"] == "player kills != team kills"
    assert "FST" in set(chk["league"])    # every in-scope league is checked
    assert "LDL" not in set(chk["league"])  # out-of-scope leagues are not

    players, teams = process(r)
    kept = set(teams["gameid"])
    assert {ids[0], ids[3]} <= kept and not ({ids[1], ids[2]} & kept)
    assert teams.loc[teams["gameid"].isin([ids[0], ids[3]]), "partial"].all()
    other = set(raw.loc[raw["datacompleteness"] != "complete", "gameid"])
    assert not teams.loc[~teams["gameid"].isin(set(ids) | other), "partial"].any()
    # team kills / deaths filled from the player rows
    g = teams[teams["gameid"] == ids[0]].sort_values("side")
    pk = players[players["gameid"] == ids[0]].groupby("side", observed=True)["kills"].sum()
    assert list(g["teamkills"]) == list(pk.sort_index())
    assert list(g["teamdeaths"]) == list(pk.sort_index())[::-1]
    pl = players[players["gameid"] == ids[0]]
    assert (pl["teamkills"] == pl["side"].astype(str).map(pk)).all()
    assert set(g["result"]) == {0, 1}


def test_audit_reports_partial(raw):
    from src.data.audit import audit
    from src.data.oe import process
    r, ids = _lpl_partial(raw)
    p2 = r.index[(r["gameid"] == ids[2]) & (r["position"] == "top")][0]
    r.loc[p2, "kills"] = r.loc[p2, "kills"] + 1
    players, teams = process(r)
    text, _ = audit(r, players, teams)
    assert "LPL 2025 partial games: 3 pass, 1 fail" in text
    assert "player kills != team kills" in text


def test_worlds_feeder_leagues_in_scope():
    from src.config import (DOMESTIC_LEAGUES, ELO_LEAGUE_INIT, INTERNATIONAL_LEAGUES,
                            LEAGUE_CODE_ALIASES, LEAGUES, PARTIAL_OK_LEAGUES)
    # every region that sends teams to Worlds / MSI, plus the international events
    assert {"LCK", "LPL", "LEC", "LCS", "CBLOL", "LCP"} <= set(DOMESTIC_LEAGUES)
    assert {"WLDs", "MSI", "FST", "EWC"} <= set(INTERNATIONAL_LEAGUES)
    assert LEAGUE_CODE_ALIASES["LTA S"] == "CBLOL"
    assert set(PARTIAL_OK_LEAGUES) == set(LEAGUES)
    assert set(DOMESTIC_LEAGUES) <= set(ELO_LEAGUE_INIT)


def test_lcp_and_partial_worlds_games_kept(raw):
    """An LCP team (e.g. MVK) and its partial Worlds games make it into the data."""
    import src.data.oe as oe
    from src.data.audit import audit
    lec = raw["league"] == "LEC"
    fst = raw["league"] == "FST"
    r = raw.assign(league=raw["league"].mask(lec, "LCP").mask(fst, "WLDs"))
    players, teams = oe.process(r)
    assert {"LCP", "WLDs"} <= set(teams["league"].astype(str))
    assert teams.loc[teams["league"] == "WLDs", "partial"].any()   # partial Worlds game kept
    text, _ = audit(r, players, teams)
    assert "## International events" in text
    assert "no domestic games in the data:** " in text
