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
    assert not (kept & partial)
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
