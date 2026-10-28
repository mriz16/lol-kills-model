"""Quick slate parser, goblin calibration (research), Flex payouts, played-line log."""
import numpy as np
import pandas as pd

from src import config as C
from src.app import grading, picklog, slatelog
from src.app.quick import insert_lines, parse_quick_slate, to_text
from src.app.slate import flex_summary, poisson_binomial
from src.engine.calibration import GoblinCal, goblin_depth, goblin_pairs

TEAMS = ["Hanwha Life Esports", "Gen.G", "T1", "Dplus Kia", "Bilibili Gaming",
         "Team Liquid", "G2 Esports"]


def test_quick_slate_headers_and_lines():
    text = """HLE vs GEN -150 +120
Zeka 3.5
Chovy 4.5 g

T1 -120 vs BLG +100 m1 -110 -110 bo5
Faker 3.5
G2 vs T1"""
    ms, loose = parse_quick_slate(text, TEAMS)
    assert not loose and len(ms) == 3
    a, b, c = ms
    assert (a.team_a, a.team_b, a.ml_a, a.ml_b, a.best_of) == \
        ("Hanwha Life Esports", "Gen.G", "-150", "+120", None)
    assert a.lines == ["Zeka 3.5", "Chovy 4.5 g"] and not a.problems
    assert (b.team_a, b.team_b, b.ml_a, b.ml_b, b.m1_a, b.m1_b, b.best_of) == \
        ("T1", "Bilibili Gaming", "-120", "+100", "-110", "-110", 5)
    assert c.problems and "moneylines" in c.problems[0]


def test_quick_slate_loose_lines_and_insert():
    ms, loose = parse_quick_slate("Faker 3.5\nHLE vs GEN -150 +120\nZeka 3.5", TEAMS)
    assert loose and len(ms) == 1
    text = "HLE vs GEN -150 +120\nZeka 3.5\n\nT1 vs BLG ev ev\nFaker 3.5"
    out = insert_lines(text, 0, "Chovy 4.5")
    ms, _ = parse_quick_slate(out, TEAMS)
    assert ms[0].lines == ["Zeka 3.5", "Chovy 4.5"] and ms[1].lines == ["Faker 3.5"]
    blocks = [dict(team_a="T1", team_b="Gen.G", ml_a="-150", ml_b="+120", bo=1,
                   lines="Faker 3.5")]
    ms, _ = parse_quick_slate(to_text(blocks), ["T1", "Gen.G"])
    assert ms[0].best_of == 1 and ms[0].lines == ["Faker 3.5"]


def test_goblin_depth():
    assert goblin_depth(4.0, 3.5) == 1 and goblin_depth(4.5, 1.5) == 3


def test_goblin_cal_only_maps_supports():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.5, 0.95, 6000)
    pairs = pd.DataFrame({"position": ["sup"] * 3000 + ["mid"] * 3000, "p": p,
                          "y": (rng.uniform(size=6000) < np.where(
                              np.arange(6000) < 3000, p - 0.08, p)).astype(int)})
    cal = GoblinCal.fit(pairs)
    assert set(cal.maps) == {"sup"}
    assert np.isclose(cal(0.8, "mid"), 0.8) and cal(0.8, "sup") < 0.77


def test_goblin_pairs_shape():
    base = pd.DataFrame({"date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                         "position": ["mid", "sup"], "market": ["Map 1"] * 2,
                         "proxy_mean": [3.5, 1.5], "y": [3, 0]})
    P = np.zeros((2, 10))
    P[0, 3], P[1, 0] = 1.0, 1.0
    g = goblin_pairs(base, P)
    assert sorted(g[g["position"] == "mid"]["depth"]) == [1, 2, 3]   # 2.5, 1.5, 0.5
    assert (g[g["position"] == "mid"]["y"] == 1).all() and (g[g["position"] == "mid"]["p"] == 1).all()
    assert list(g[g["position"] == "sup"]["depth"]) == [1]


def _row(**kw):
    r = dict(matchup="A vs B", platform="PrizePicks", league="LCK", best_of=1,
             market="Map 1", player="Zeka", team="A", opponent="B", role="mid", line=3.5,
             p_more=0.52, p_less=0.48, p_push=0.0, p_side=0.52, side="more", mean=4.4,
             median=4.0, fair_line=4.5, breakeven=0.577, edge=-0.05, calibrated=True,
             w_lambda=0.9, cal_p_more=0.53, cal_p_less=0.47, cal_p_push=0.0, cal_edge=0.0,
             raw_side="more", raw_p_more=0.55, raw_p_less=0.45, raw_p_push=0.0,
             raw_edge=0.0, k_spread=1.0, n_games=50, share=0.24, p_map=0.5, kind="standard",
             floor80=2, floor90=1, flags="")
    r.update(kw)
    return r


def test_poisson_binomial_and_flex_table():
    d = poisson_binomial([0.5, 0.5])
    assert np.allclose(d, [0.25, 0.5, 0.25])
    picks = [dict(_row(player=f"P{i}"), p_more=0.6, cal_p_more=0.6) for i in range(5)]

    class FakeSim:
        n_sims = 4

        def range_kills(self, player, market):
            return np.array([5, 5, 1, 1])

    f = flex_summary({"A vs B": FakeSim()}, picks)
    pb = poisson_binomial([0.6] * 5)
    assert np.isclose(f["ev"], 10 * pb[5] + 2 * pb[4] + 0.4 * pb[3] - 1)
    assert np.isclose(f["ev_raw"], 0.5 * 10 - 1)       # all 5 hit together in half the sims


def test_settle_flex_and_goblin_power():
    legs = [dict(_row(player=f"P{i}"), side="less") for i in range(5)]
    rows = picklog.make_pick_rows(legs, {"p_joint": 0.1, "payout": 10.0, "ev": 0.2},
                                  "PrizePicks", 25, slip_id="F1", mode="flex")
    picks = pd.DataFrame(rows)
    graded = pd.DataFrame({"pick_id": picks["pick_id"],
                           "status": ["loss", "loss", "win", "win", "win"]})
    s = grading.settle_slips(picks, graded).iloc[0]
    assert s.status == "partial" and np.isclose(s.profit, 25 * (0.4 - 1))
    gl = [dict(_row(player=f"G{i}"), kind="goblin") for i in range(6)]
    rows = picklog.make_pick_rows(gl, {"p_joint": 0.1, "payout": 8.25, "ev": -0.1},
                                  "PrizePicks", 25, slip_id="G1")
    picks = pd.DataFrame(rows)
    graded = pd.DataFrame({"pick_id": picks["pick_id"], "status": ["win"] * 6})
    s = grading.settle_slips(picks, graded).iloc[0]
    assert s.status == "win" and np.isclose(s.profit, 25 * 7.25)


def test_played_rows_and_missing(tmp_path):
    store = picklog.LocalStore(tmp_path)
    df = pd.DataFrame([_row()])
    rows = slatelog.make_slate_rows(df, "2026-10-24", False, pd.Timestamp("2026-10-24"),
                                    slate_id="s1")
    slatelog.append_slate(store, rows)
    assert slatelog.missing_rows(store, [rows[0]["row_id"]]) == []
    assert slatelog.missing_rows(store, [rows[0]["row_id"]], played=True)
    g = dict(_row(), line=1.5, kind="goblin", side="more", p_side=0.83)
    grow = slatelog.make_slate_rows(pd.DataFrame([g]), "2026-10-24", False,
                                    pd.Timestamp("2026-10-24"), slate_id="s1", played=True)
    slatelog.append_slate(store, grow)
    slates = slatelog.read_slates(store)
    assert len(slates) == 2 and slates["kind"].tolist() == ["standard", "goblin"]
    graded = pd.DataFrame({"row_id": slates["row_id"], "status": ["win", "win"],
                           "actual": [5, 5]})
    pr = slatelog.played_rows(slates, graded)
    assert list(pr["kind"]) == ["goblin"] and list(pr["result"]) == ["hit"]
    e = slatelog.evaluate(slates, graded)
    gob = e["goblin"].set_index("group")
    assert gob.loc["goblins played", "n"] == 1
    assert e["table"]["n"].max() == 1          # the goblin row is not in the standard table
