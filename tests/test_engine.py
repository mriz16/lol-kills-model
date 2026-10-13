import numpy as np
import pandas as pd
import pytest

from src.engine.features import EngineConfig, History, MatchupParams, build_matchup, fit_kappa
from src.engine.odds import map_played_prob
from src.engine.props import (BREAKEVENS, Leg, fair_line, leg_hits, prop_summary,
                              slip_probability)
from src.engine.sim import parse_market, simulate_series


def toy_params(p_map=0.6, best_of=3, script=1.0):
    win = (np.array([14.0, 16.0, 18.0, 20.0]), np.full(4, 0.25))
    loss = (np.array([6.0, 8.0, 10.0]), np.full(3, 1 / 3))
    alpha = np.array([[4.2, 5.75, 6.1, 7.8, 1.2]] * 2)
    return MatchupParams(
        teams=("A", "B"), best_of=best_of, p_map=p_map, pools=[[loss, win], [loss, win]],
        alpha=alpha, players=[[f"a{i}" for i in range(5)], [f"b{i}" for i in range(5)]],
        script_mult=script,
    )


def test_map3_occurrence_in_sim_matches_closed_form():
    sim = simulate_series(toy_params(p_map=0.65), n_sims=40_000, rng=1)
    assert sim.played[:, 2].mean() == pytest.approx(map_played_prob(0.65, 3, 3), abs=0.01)
    assert sim.played[:, :2].all()
    # no kills on unplayed maps
    assert (sim.team_kills[~sim.played] == 0).all()
    assert (sim.player_kills.sum(axis=-1) == sim.team_kills).all()


def test_bo5_clinch():
    sim = simulate_series(toy_params(p_map=0.55, best_of=5), n_sims=20_000, rng=3)
    wins_a = (sim.a_wins & sim.played).sum(1)
    wins_b = (~sim.a_wins & sim.played).sum(1)
    assert ((wins_a == 3) ^ (wins_b == 3)).all()
    assert sim.played[:, 3].mean() == pytest.approx(map_played_prob(0.55, 5, 4), abs=0.012)


def test_seeded_end_to_end_reproducible():
    a = simulate_series(toy_params(), n_sims=5000, rng=42)
    b = simulate_series(toy_params(), n_sims=5000, rng=42)
    assert np.array_equal(a.player_kills, b.player_kills)
    k = a.range_kills("a3", "Maps 1-2")
    assert k.shape == (5000,)
    # expected bot kills over 2 maps: share * E[team kills per map] * 2
    share = 7.8 / 25.05
    e_team = 0.6 * 17 + 0.4 * 8
    assert k.mean() == pytest.approx(2 * share * e_team, rel=0.03)


def test_market_ranges_count_only_played_maps():
    sim = simulate_series(toy_params(p_map=0.5), n_sims=10_000, rng=5)
    m3 = sim.range_kills("b1", "Map 3")
    assert (m3[~sim.played[:, 2]] == 0).all()
    total = sim.range_kills("b1", "Maps 1-3")
    assert np.array_equal(total, sim.range_kills("b1", "Maps 1-2") + m3)
    assert parse_market("Maps 1-5", 3) == (1, 3)
    with pytest.raises(ValueError):
        parse_market("Map 4", 3)


def test_push_handling_whole_number_line():
    kills = np.array([3, 4, 4, 5, 6, 4, 2, 7, 4, 5])
    s = prop_summary(kills, 4.0)
    assert s["p_push"] == pytest.approx(0.4)
    assert s["p_over"] == pytest.approx(0.4)
    assert s["p_under"] == pytest.approx(0.2)
    assert s["p_over"] + s["p_under"] + s["p_push"] == pytest.approx(1.0)
    h = prop_summary(kills, 4.5)
    assert h["p_push"] == 0.0
    assert h["p_over"] + h["p_under"] == pytest.approx(1.0)
    # push is not a win for either side of a slip leg
    sim = simulate_series(toy_params(), n_sims=2000, rng=9)
    k = sim.range_kills("a0", "Map 1")
    line = float(np.median(k))
    over = leg_hits(sim, Leg("m", "a0", "Map 1", line, "over"))
    under = leg_hits(sim, Leg("m", "a0", "Map 1", line, "under"))
    assert not (over & under).any()
    assert (~over & ~under).mean() == pytest.approx((k == line).mean())


def test_breakevens_and_edges():
    assert BREAKEVENS["prizepicks_power"][2] == pytest.approx(1 / np.sqrt(3))
    assert BREAKEVENS["underdog"][3] == pytest.approx(6 ** (-1 / 3))
    s = prop_summary(np.array([0, 10] * 50), 4.5)
    assert s["edge_over_prizepicks_power_2L"] == pytest.approx(0.5 - 1 / np.sqrt(3))


def test_fair_line():
    k = np.random.default_rng(0).poisson(4.2, 50_000)
    fl = fair_line(k)
    assert fl % 1 == 0.5
    assert abs((k > fl).mean() - 0.5) <= abs((k > fl + 1).mean() - 0.5)


def test_slip_same_matchup_shared_and_cross_multiplied():
    s1 = simulate_series(toy_params(), n_sims=20_000, rng=11)
    s2 = simulate_series(toy_params(p_map=0.4), n_sims=20_000, rng=12)
    sims = {"m1": s1, "m2": s2}
    # Two teammates' overs are positively correlated through team kills
    legs = [Leg("m1", "a2", "Maps 1-2", 4.5, "over"), Leg("m1", "a3", "Maps 1-2", 5.5, "over")]
    res = slip_probability(sims, legs)
    h = leg_hits(s1, legs[0]) & leg_hits(s1, legs[1])
    assert res["p_all_hit"] == pytest.approx(h.mean())
    assert res["p_all_hit"] > res["p_independent"]
    cross = [legs[0], Leg("m2", "b2", "Maps 1-2", 4.5, "over")]
    r2 = slip_probability(sims, cross)
    assert r2["p_all_hit"] == pytest.approx(
        leg_hits(s1, cross[0]).mean() * leg_hits(s2, cross[1]).mean())
    assert "ev_prizepicks_power" in r2


def test_build_matchup_from_fixture(processed):
    players, teams = processed
    hist = History(players, teams)
    last = teams[teams["league"] == "LCK"].sort_values("date").iloc[-1]
    sid = last["series_id"]
    g = teams[teams["series_id"] == sid]
    a, b = g[g["map_no"] == 1].sort_values("side")["teamname"].astype(str)
    m1 = players[(players["series_id"] == sid) & (players["map_no"] == 1)]
    lineups = tuple(
        m1[m1["teamname"].astype(str) == t].set_index("position").loc[
            ["top", "jng", "mid", "bot", "sup"], "playername"].astype(str).tolist()
        for t in (a, b))
    cutoff = g["date"].min()
    for script in (False, True):
        cfg = EngineConfig(game_script=script, kappa=fit_kappa(hist, int(cutoff.value)))
        prm = build_matchup(hist, a, b, "LCK", 3, 0.55, lineups, cutoff,
                            int(g["patch_ord"].iloc[0]), cfg)
        assert 0.8 <= prm.info["opp_adj_0"] <= 1.2
        assert cfg.game_script_clamp[0] <= prm.script_mult <= cfg.game_script_clamp[1]
        for s in (0, 1):
            for o in (0, 1):
                v, p = prm.pools[s][o]
                assert p.sum() == pytest.approx(1.0) and (v >= 0).all()
            # win pool should average more kills than the loss pool
            assert np.dot(*prm.pools[s][1]) > np.dot(*prm.pools[s][0])
        sim = simulate_series(prm, n_sims=4000, rng=0)
        assert 5 < sim.team_kills[:, 0, :].mean() * 2 < 45


def test_no_lookahead(processed):
    """Projection must not change when future games are added to history."""
    players, teams = processed
    cut = teams["date"].quantile(0.6)
    h_full = History(players, teams)
    h_past = History(players[players.date < cut], teams[teams.date < cut])
    t = teams[teams.date < cut]["teamname"].astype(str).value_counts().index[:2].tolist()
    lineup = [["x1", "x2", "x3", "x4", "x5"], ["y1", "y2", "y3", "y4", "y5"]]
    patch = int(teams[teams.date < cut]["patch_ord"].max())
    a = build_matchup(h_full, t[0], t[1], "LCK", 3, 0.5, lineup, cut, patch)
    b = build_matchup(h_past, t[0], t[1], "LCK", 3, 0.5, lineup, cut, patch)
    for s in (0, 1):
        for o in (0, 1):
            assert np.allclose(a.pools[s][o][0], b.pools[s][o][0])


def test_shrinkage_toward_position_prior(processed):
    players, teams = processed
    hist = History(players, teams)
    cutoff = pd.Timestamp("2030-01-01")
    patch = int(teams["patch_ord"].max())
    t = teams["teamname"].astype(str).iloc[0]
    o = teams["teamname"].astype(str).iloc[1]
    unknown = [["n1", "n2", "n3", "n4", "n5"]] * 2
    prm = build_matchup(hist, t, o, "LCK", 1, 0.5, unknown, cutoff, patch,
                        EngineConfig(kappa=10.0))
    shares = prm.alpha[0] / prm.alpha[0].sum()
    pri = np.array([0.168, 0.230, 0.243, 0.312, 0.047])
    assert shares == pytest.approx(pri / pri.sum(), rel=1e-6)
