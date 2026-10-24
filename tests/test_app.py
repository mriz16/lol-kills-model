"""Phase 2 app logic: name matching, odds input, breakevens, OCR parser, pick log."""
import base64
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.app import grading, picklog
from src.app.names import match_lines, match_player, parse_kind, parse_line, search_teams
from src.app.ocr import parse_board_text, props_to_lines
from src.app.odds_input import odds_summary, parse_american
from src.app.slate import (DataStore, MatchupInput, check_matchup, flags_for, floors,
                           per_leg_breakeven, rebreakeven, results_table, run_matchup,
                           safest_overs, slip_summary)
from src.engine.odds import series_win_prob

ROSTER = [
    dict(player="Chovy", team="Gen.G", role="mid"),
    dict(player="Canyon", team="Gen.G", role="jng"),
    dict(player="Ruler", team="Gen.G", role="bot"),
    dict(player="Faker", team="T1", role="mid"),
    dict(player="Keria", team="T1", role="sup"),
    dict(player="Doran", team="T1", role="top"),
    dict(player="Oner", team="T1", role="jng"),
]


# ------------------------------------------------------------------ name matching ----
def test_parse_line_formats():
    assert parse_line("Loki 10.5") == ("Loki", 10.5)
    assert parse_line("Hans Sama: 5.5") == ("Hans Sama", 5.5)
    assert parse_line("Faker,\t7") == ("Faker", 7.0)
    assert parse_line("Faker") is None


def test_parse_goblin_demon_tags():
    assert parse_line("Chovy 7.5 g") == ("Chovy", 7.5)
    assert [parse_kind(t) for t in ("Chovy 7.5", "Chovy 7.5 g", "Chovy 7.5goblin",
                                    "Chovy 12.5 D", "Hans Sama: 5.5 demon")] == \
        ["standard", "goblin", "goblin", "demon", "demon"]
    out = match_lines("Chovy 10.5\nChovy 7.5 g\nChovy 8.5 g", ROSTER)
    assert [(m.status, m.kind) for m in out] == [("ok", "standard"), ("ok", "goblin"),
                                                 ("ambiguous", "goblin")]


def test_match_player_exact_prefix_fuzzy():
    assert match_player("chovy", ROSTER)[0] == "ok"
    st, c = match_player("CHOVY", ROSTER)
    assert c[0][:3] == ("Chovy", "Gen.G", "mid")
    st, c = match_player("Ker", ROSTER)          # unique prefix -> Keria
    assert st == "ok" and c[0][0] == "Keria"
    st, c = match_player("Fakr", ROSTER)         # typo
    assert st == "ok" and c[0][0] == "Faker"
    assert match_player("Zeus", ROSTER)[0] == "unmatched"


def test_match_player_ambiguous():
    roster = ROSTER + [dict(player="Chovy2", team="T1", role="mid")]
    st, c = match_player("Cho", roster)          # prefix of two players
    assert st == "ambiguous" and {x[0] for x in c} == {"Chovy", "Chovy2"}


def test_match_lines_autofills_team_and_role_and_flags_problems():
    out = match_lines("Chovy 12.5\nkeria 1.5\nZeus 7.5\nnonsense\n\nchovy 11.5", ROSTER)
    assert [m.status for m in out] == ["ok", "ok", "unmatched", "bad line", "ambiguous"]
    assert (out[0].team, out[0].role, out[0].line) == ("Gen.G", "mid", 12.5)
    assert (out[1].player, out[1].team, out[1].role) == ("Keria", "T1", "sup")


def test_search_teams_uses_alias_table():
    known = ["Gen.G", "Gen.G Global Academy", "Gentle Mates", "Hanwha Life Esports",
             "T1", "Bilibili Gaming", "Top Esports"]
    assert search_teams("gen", known)[0] == "Gen.G"
    assert search_teams("HLE", known)[0] == "Hanwha Life Esports"
    assert search_teams("blg", known)[0] == "Bilibili Gaming"
    assert search_teams("tes", known)[0] == "Top Esports"
    assert search_teams("", known) == []


# ------------------------------------------------------------------ odds input ----
def test_parse_american():
    assert parse_american("-150") == -150
    assert parse_american("+130") == 130
    assert parse_american(" 200 ") == 200
    assert parse_american("EV") == 100
    assert parse_american("−120") == -120  # unicode minus from copy/paste
    assert parse_american("") is None and parse_american(None) is None
    for bad in ("50", "-99", "abc", "1.5x"):
        with pytest.raises(ValueError):
            parse_american(bad)


def test_odds_summary_series_and_map1():
    s = odds_summary("-200", "+160", 3)
    assert s["error"] is None and s["source"] == "series"
    assert s["p_series_a"] + s["p_series_b"] == pytest.approx(1.0)
    assert 0.5 < s["p_map_a"] < s["p_series_a"]
    assert series_win_prob(s["p_map_a"], 3) == pytest.approx(s["p_series_a"], abs=1e-9)
    m = odds_summary("-200", "+160", 3, "-140", "+110")
    assert m["source"] == "map1" and m["p_map_a"] != pytest.approx(s["p_map_a"])
    assert m["p_map_a"] + m["p_map_b"] == pytest.approx(1.0)
    # map 1 alone is enough; series is implied
    only = odds_summary("", "", 3, "-140", "+110")
    assert only["error"] is None and only["p_series_a"] > only["p_map_a"]


def test_odds_summary_errors():
    assert "both teams" in odds_summary("-150", "", 3)["error"]
    assert "Map 1" in odds_summary("-150", "+120", 3, "-130", "")["error"]
    assert odds_summary("-50", "+120", 3)["error"]


# ------------------------------------------------------------------ breakeven ----
def test_per_leg_breakeven_from_payout_tables():
    assert per_leg_breakeven("PrizePicks", 2) == pytest.approx((1 / 3) ** 0.5)
    assert per_leg_breakeven("PrizePicks", 3) == pytest.approx((1 / 5) ** (1 / 3))
    assert per_leg_breakeven("PrizePicks", 6) == pytest.approx((1 / 40) ** (1 / 6))
    assert per_leg_breakeven("Underdog", 3) == pytest.approx((1 / 6) ** (1 / 3))
    for plat, table in (("PrizePicks", C.PRIZEPICKS_POWER), ("Underdog", C.UNDERDOG)):
        for n, mult in table.items():
            assert per_leg_breakeven(plat, n) ** n * mult == pytest.approx(1.0)


def test_flags():
    assert flags_for("more", 0.01, 0.6, 7.5, 50) == ["THIN EDGE"]
    assert flags_for("more", 0.10, 0.6, 7.5, 12) == ["LOW SAMPLE"]
    assert flags_for("less", 0.10, 0.20, 7.0, 50) == ["CAUTION"]
    assert flags_for("less", 0.10, 0.20, 7.5, 50) == []      # half-point line
    assert flags_for("less", 0.10, 0.30, 7.0, 50) == []      # P(more) >= 25%


def test_flags_goblin_safe_floor_sup_more():
    assert flags_for("more", 0.30, 0.90, 4.5, 50, True, "mid", "goblin") == \
        ["GOBLIN", "SAFE FLOOR"]
    assert flags_for("more", 0.30, 0.90, 1.5, 50, True, "sup", "goblin") == \
        ["GOBLIN", "SUP MORE"]
    assert flags_for("more", 0.10, 0.62, 2.5, 50, True, "sup") == ["SUP MORE"]
    assert flags_for("less", 0.10, 0.38, 2.5, 50, True, "sup") == []
    assert flags_for("less", 0.05, 0.38, 2.5, 50, True, "sup") == ["BELOW MIN"]


def test_floors():
    k = np.repeat(np.arange(10), 10)            # uniform 0..9
    assert floors(k) == {"floor80": 2, "floor90": 1}
    assert floors(np.full(50, 4)) == {"floor80": 4, "floor90": 4}


# ------------------------------------------------------------------ OCR parser ----
CARD = """GEN • MID
Chovy
vs T1 Sun 4:00am
10.5 MAPS 1-2 Kills
"""


def test_ocr_parser_sample_card():
    props = parse_board_text(CARD)
    assert len(props) == 1
    p = props[0]
    assert (p.name, p.line, p.market, p.team, p.role, p.opponent) == \
        ("Chovy", 10.5, "Maps 1-2", "GEN", "mid", "T1")
    assert props_to_lines(props) == "Chovy 10.5"


def test_ocr_parser_board_with_noise_and_split_lines():
    text = CARD + """More Less
T1 * TOP
Doran
vs GEN Sun 4:00am
6.5
MAPS 1-2 Kills
Trending 4.1K
HLE e ADC
Viper
vs KT Sat 1:00am
14 MAPS 1~3 Kills
"""
    props = parse_board_text(text)
    assert [(p.name, p.line, p.market, p.role) for p in props] == [
        ("Chovy", 10.5, "Maps 1-2", "mid"), ("Doran", 6.5, "Maps 1-2", "top"),
        ("Viper", 14.0, "Maps 1-3", "bot")]
    assert props_to_lines(props).splitlines() == ["Chovy 10.5", "Doran 6.5", "Viper 14"]


def test_ocr_parser_real_tesseract_quirks():
    # From tesseract on a rendered board: a garbled header and a stray digit before MAPS
    text = "Teh Ute)?\n\nDoran\nvs GEN Sun 4:00am\n\n6.5 MAPS 1-2 Kills\nLess More\n" \
           "HLE * ADC\n\nViper\n\nvs KT Sat 1:00am\n\n14 3 MAPS 1-3 Kills\n"
    props = parse_board_text(text)
    assert [(p.name, p.line, p.market) for p in props] == [
        ("Doran", 6.5, "Maps 1-2"), ("Viper", 14.0, "Maps 1-3")]


# ------------------------------------------------------------------ pick log ----
LEG = dict(league="LCK", best_of=3, matchup="Gen.G vs T1", team="Gen.G", opponent="T1",
           player="Chovy", role="mid", market="Maps 1-2", line=10.5, side="more",
           p_more=0.58, p_less=0.42, p_push=0.0, mean=11.2, fair_line=10.5)


def _rows(slip_id="S1", n=2):
    legs = [dict(LEG, player=f"P{i}") for i in range(n)]
    return picklog.make_pick_rows(legs, {"p_joint": 0.35, "payout": 3.0, "ev": 0.05},
                                  "PrizePicks", 10, slip_id=slip_id,
                                  now=datetime(2026, 10, 14, 12, tzinfo=timezone.utc))


def test_log_write_read_local(tmp_path):
    store = picklog.LocalStore(tmp_path)
    assert picklog.read_picks(store).empty
    picklog.append_picks(store, _rows("S1"))
    picklog.append_picks(store, _rows("S2", 3))
    picklog.append_picks(store, _rows("S2", 3))       # idempotent on pick_id
    df = picklog.read_picks(store)
    assert len(df) == 5 and list(df.columns[:len(picklog.PICK_COLUMNS)]) == picklog.PICK_COLUMNS
    r = df.iloc[0]
    assert (r.pick_id, r.slip_id, r.date, r.platform, r.side) == \
        ("S1-1", "S1", "2026-10-14", "PrizePicks", "more")
    assert r.p_more == pytest.approx(0.58) and r.line == 10.5 and r.n_legs == 2
    assert (tmp_path / picklog.PICKS_PATH).exists()


def test_log_conflict_is_retried(tmp_path):
    store = picklog.LocalStore(tmp_path)
    picklog.append_picks(store, _rows("S1"))
    real_read, calls = store.read, {"n": 0}

    def stale_read(path):  # first read returns a stale sha -> ConflictError -> retry
        text, sha = real_read(path)
        calls["n"] += 1
        return (text, "stale") if calls["n"] == 1 else (text, sha)

    store.read = stale_read
    picklog.append_picks(store, _rows("S2"))
    assert calls["n"] >= 3  # stale read -> conflict -> re-read -> write
    assert len(picklog.read_picks(picklog.LocalStore(tmp_path))) == 4


class _Resp:
    def __init__(self, status, body=None):
        self.status_code, self._body = status, body or {}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            err = RuntimeError(self.status_code)
            err.response = self  # like requests.HTTPError
            raise err


class FakeGitHub:
    """Minimal in-memory GitHub contents API."""

    def __init__(self):
        self.headers, self.files, self.branches = {}, {}, {"main"}

    def get(self, url, params=None, timeout=None, headers=None):
        if "/repos/o/r" not in url:
            return _Resp(404)
        if url.endswith("/branches/picks-log"):
            return _Resp(200 if "picks-log" in self.branches else 404)
        if url.endswith("/git/ref/heads/main"):
            return _Resp(200, {"object": {"sha": "abc"}})
        if url.endswith("/repos/o/r"):
            return _Resp(200, {"default_branch": "main"})
        path = url.split("/contents/")[1]
        if path not in self.files:
            return _Resp(404)
        text, sha = self.files[path]
        return _Resp(200, {"encoding": "base64", "sha": sha,
                           "content": base64.b64encode(text.encode()).decode()})

    def post(self, url, json=None, timeout=None):
        self.branches.add(json["ref"].split("/")[-1])
        return _Resp(201)

    def put(self, url, json=None, timeout=None):
        path = url.split("/contents/")[1]
        cur = self.files.get(path)
        if (cur and json.get("sha") != cur[1]) or (cur is None and json.get("sha")):
            return _Resp(409, {"message": "sha does not match"})
        assert json["branch"] == "picks-log"
        n = len(self.files) + 1 + (0 if cur is None else int(cur[1][1:]))
        self.files[path] = (base64.b64decode(json["content"]).decode(), f"s{n}")
        return _Resp(200 if cur else 201)


def test_log_write_read_github_api():
    fake = FakeGitHub()
    store = picklog.GitHubStore("tok", "o/r", session=fake)
    assert fake.headers["Authorization"] == "Bearer tok"
    picklog.append_picks(store, _rows("S1"))
    assert "picks-log" in fake.branches       # created from main on the first write
    picklog.append_picks(store, _rows("S2"))
    df = picklog.read_picks(store)
    assert list(df["slip_id"]) == ["S1", "S1", "S2", "S2"]


def test_log_check_creates_branch_and_reports_errors():
    fake = FakeGitHub()
    ok, msg = picklog.check_store(picklog.GitHubStore("tok", "o/r", session=fake))
    assert ok and "created now" in msg and "picks-log" in fake.branches
    ok, msg = picklog.check_store(picklog.GitHubStore("tok", "o/r", session=fake))
    assert ok and "created" not in msg
    ok, msg = picklog.check_store(picklog.GitHubStore("tok", "o/other", session=fake))
    assert not ok and "cannot see o/other" in msg

    class ReadOnly(FakeGitHub):
        def post(self, url, json=None, timeout=None):
            return _Resp(403)
    ok, msg = picklog.check_store(picklog.GitHubStore("tok", "o/r", session=ReadOnly()))
    assert not ok and "Contents: Read and write" in msg


# ------------------------------------------------------------------ app on fixture ----
@pytest.fixture(scope="module")
def store(processed):
    # no calibration pairs: raw probabilities, independent of data/calib_pairs.parquet
    return DataStore(*processed, calib_pairs=pd.DataFrame(), spread_rows=pd.DataFrame(),
                     w_rows=pd.DataFrame(), w_real={})


def _fixture_matchup(store, lines=None):
    a, b = "LCK Tigers", "LCK Ravens"
    lu = store.default_lineup(a)[:2] + store.default_lineup(b)[:1]
    text = lines or "\n".join(f"{pl.lower()} 4.5" for pl in lu)
    return MatchupInput(a, b, 3, "PrizePicks", "-150", "+125", lines=text)


def test_run_matchup_on_fixture(store):
    m = _fixture_matchup(store)
    lines, odds, problems = check_matchup(store, m)
    assert problems == [] and all(lm.ok for lm in lines)
    res = run_matchup(store, m, legs=2, n_sims=4000)
    df = results_table([res])
    assert len(df) == 3 and df["edge"].is_monotonic_decreasing
    row = df.iloc[0]
    assert row.p_more + row.p_less + row.p_push == pytest.approx(1.0)
    assert row.market == "Maps 1-2" and row.breakeven == pytest.approx((1 / 3) ** 0.5)
    assert "team kills/map in wins" in row.why and "of maps" in row.why
    df5 = rebreakeven(df, 5)
    assert df5["breakeven"].iloc[0] == pytest.approx((1 / 20) ** 0.2)
    legs = [dict(r) for _, r in df.head(2).iterrows()]
    s = slip_summary({res.inp.key: res.sim}, legs, "PrizePicks")
    assert s["payout"] == 3.0 and s["ev"] == pytest.approx(s["p_joint"] * 3 - 1)


def test_goblin_rows_more_only_never_play(store):
    a, b = "LCK Tigers", "LCK Ravens"
    pl = store.default_lineup(a)[2]
    m = MatchupInput(a, b, 3, "PrizePicks", "-150", "+125",
                     lines=f"{pl} 30.5\n{pl} 0.5 g")
    df = results_table([run_matchup(store, m, legs=2, n_sims=4000)])
    std, gob = df[df["kind"] == "standard"].iloc[0], df[df["kind"] == "goblin"].iloc[0]
    assert std["side"] == "less" and gob["side"] == "more" and not gob["play"]
    assert "GOBLIN" in gob["flags"] and gob["floor90"] <= gob["floor80"] <= gob["median"]
    assert not rebreakeven(df, 3)["play"][df["kind"] == "goblin"].any()
    assert safest_overs(df).iloc[0]["kind"] == "goblin"
    legs = [dict(gob, side="more")]
    sims = {m.key: run_matchup(store, m, legs=2, n_sims=4000).sim}
    assert slip_summary(sims, legs * 2, "PrizePicks", payout=2.0)["payout"] == 2.0


def test_check_matchup_reports_bad_names(store):
    m = _fixture_matchup(store, lines="Nobody 3.5")
    _, _, problems = check_matchup(store, m)
    assert any("to fix" in p for p in problems)
    with pytest.raises(ValueError):
        run_matchup(store, m, legs=2, n_sims=1000)


def test_grade_picks_on_fixture(processed):
    players, teams = processed
    t = teams[teams["best_of"] == 3].sort_values("date")
    sid = t["series_id"].iloc[0]
    g = t[t["series_id"] == sid]
    team = str(g["teamname"].iloc[0])
    opp = str(g[g["teamname"].astype(str) != team]["teamname"].iloc[0])
    pl = players[(players["series_id"] == sid) & (players["teamname"].astype(str) == team)]
    name = str(pl["playername"].iloc[0])
    mine = pl[pl["playername"].astype(str) == name]
    actual = int(mine[mine["map_no"].isin([1, 2])]["kills"].sum())
    logged = (pd.Timestamp(g["date"].min()) - pd.Timedelta(hours=3)).tz_localize("UTC")
    base = dict(LEG, team=team, opponent=opp, player=name)
    legs = [dict(base, line=actual - 0.5, side="more"), dict(base, line=actual + 0.5),
            dict(base, line=float(actual))]
    rows = picklog.make_pick_rows(legs, {"p_joint": 0.3, "payout": 5.0, "ev": 0.1},
                                  "PrizePicks", 10, slip_id="G1", now=logged.to_pydatetime())
    picks = pd.DataFrame(rows)
    graded = grading.grade_picks(picks, players, teams, now=datetime(2026, 1, 1))
    assert list(graded["status"]) == ["win", "loss", "push"]
    assert (graded["actual"] == actual).all() and (graded["series_id"] == str(sid)).all()
    s = grading.record_summary(picks, graded)
    assert (s["legs_won"], s["legs_lost"], s["legs_push"]) == (1, 1, 1)
    assert s["slips_lost"] == 1 and s["profit"] == -10 and s["roi"] == -1.0
    # a pick with no series in the window stays pending, then voids
    late = picks.assign(logged_at="2025-12-20T00:00:00+00:00")
    assert set(grading.grade_picks(late, players, teams, now=datetime(2025, 12, 21))
               ["status"]) == {"pending"}
    assert set(grading.grade_picks(late, players, teams, now=datetime(2026, 2, 1))
               ["status"]) == {"void"}


def test_settle_slip_push_reduces_legs():
    picks = pd.DataFrame(_rows("S1", 3))
    graded = pd.DataFrame({"pick_id": picks["pick_id"], "status": ["win", "win", "push"],
                           "actual": 1, "series_id": "x", "graded_at": "t"})
    s = grading.settle_slips(picks, graded).iloc[0]
    assert s.status == "win" and s.profit == pytest.approx(10 * (C.PRIZEPICKS_POWER[2] - 1))
    graded["status"] = ["win", "push", "void"]
    assert grading.settle_slips(picks, graded).iloc[0].status == "refund"
    final = graded.assign(status=["win", "win", "win"])
    merged = grading.merge_graded(final, graded.assign(status="pending"))
    assert list(merged["status"]) == ["win", "win", "win"]
    assert np.isclose(grading.record_summary(picks, final)["roi"], C.PRIZEPICKS_POWER[3] - 1)
