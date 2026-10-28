"""Win / lose dependence of lines, slip conflicts, player research tables, R/Q/F toggles."""
import numpy as np
import pandas as pd
import pytest

from src.app import research
from src.app.slate import (DataStore, MatchupInput, needs_for, needs_team, results_table,
                           run_matchup, series_a_wins, slip_by_winner, slip_conflicts)
from src.backtest.walkforward import ALL_VARIANTS
from src.engine.features import build_matchup, locked_config
from src.engine.sim import simulate_series

A, B = "LCK Tigers", "LCK Ravens"


@pytest.fixture(scope="module")
def store(processed):
    return DataStore(*processed, calib_pairs=pd.DataFrame(), spread_rows=pd.DataFrame(),
                     w_rows=pd.DataFrame(), w_real={}, form_rows=pd.DataFrame())


@pytest.fixture(scope="module")
def res(store):
    lu = store.default_lineup(A)[:3] + store.default_lineup(B)[:2]
    m = MatchupInput(A, B, 3, "PrizePicks", "-150", "+125",
                     lines="\n".join(f"{pl} 4.5" for pl in lu))
    return run_matchup(store, m, legs=2, n_sims=6000)


def test_series_winner_and_script_split(res):
    sim = res.sim
    aw = series_a_wins(sim)
    assert aw.shape == (sim.n_sims,)
    # BO3: the series winner took 2 maps
    wa = (sim.a_wins & sim.played).sum(axis=1)
    assert (wa[aw] == 2).all() and (wa[~aw] < 2).all()
    df = results_table([res])
    for r in df.to_dict("records"):
        assert 0 <= r["p_if_win"] <= 1 and 0 <= r["p_if_loss"] <= 1
        # the split averages back to the raw P of the shown side
        side = r["side"]
        p = r["p_team_series"] * r[f"{side}_if_win"] + \
            (1 - r["p_team_series"]) * r[f"{side}_if_loss"]
        assert p == pytest.approx(r[f"raw_p_{side}"], abs=1e-9)
        assert r["needs"] == needs_for(r, side)
        if r["needs"]:
            assert f"NEEDS {r['needs'].upper()}" in r["flags"]
    # more kills when the team wins: MORE leans on winning
    more = df.iloc[0]
    assert more["more_if_win"] > more["more_if_loss"]


def test_needs_and_conflicts():
    base = dict(matchup="A vs B", team="A", opponent="B", more_if_win=0.7, more_if_loss=0.3,
                less_if_win=0.3, less_if_loss=0.7)
    over = dict(base, side="more")
    under = dict(base, side="less")
    assert needs_for(over, "more") == "win" and needs_for(under, "less") == "loss"
    assert needs_team(over) == "A" and needs_team(under) == "B"
    flat = dict(base, side="more", more_if_win=0.55, more_if_loss=0.45)
    assert needs_team(flat) is None
    assert slip_conflicts([over, under]) == ["A vs B: legs need A and B to win"]
    opp_over = dict(base, team="B", opponent="A", side="more")
    assert slip_conflicts([over, opp_over])                     # both overs, both teams
    assert slip_conflicts([over, dict(opp_over, side="less")]) == []   # both need A
    other = dict(under, matchup="C vs D", team="C", opponent="D")
    assert slip_conflicts([over, other]) == []


def test_slip_by_winner(res):
    df = results_table([res])
    legs = [dict(r) for r in df.head(2).to_dict("records")]
    out = slip_by_winner({res.inp.key: res.sim}, legs)
    assert len(out) == 1 and out[0]["legs"] == 2
    o = out[0]
    from src.app.slate import slip_summary
    p = slip_summary({res.inp.key: res.sim}, legs, "PrizePicks")["p_joint_raw"]
    assert o["p_a"] * o["hit_if_a"] + (1 - o["p_a"]) * o["hit_if_b"] == pytest.approx(p)


def test_market_totals_and_summaries(processed):
    players, _ = processed
    name = str(players["playername"].iloc[0])
    t1 = research.market_totals(players, name, "Map 1")
    t2 = research.market_totals(players, name, "Maps 1-2")
    assert len(t1) and (t1["maps"] == 1).all() and (t2["maps"] == 2).all()
    assert t1["date"].is_monotonic_decreasing and t1["opponent"].ne("").all()
    # Map 1 kills of a series = the player's map-1 row
    p = players[(players["playername"].astype(str) == name) & (players["map_no"] == 1)]
    sid = t1["series_id"].iloc[0]
    assert t1["kills"].iloc[0] == int(p[p["series_id"].astype(str) == sid]["kills"].sum())
    s = research.dist_summary([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert (s["floor90"], s["floor80"], s["ceil80"], s["ceil90"]) == (1, 2, 8, 9)
    lad = research.line_ladder([1, 3, 5, 7], [True, True, False, False], lines=[2.5, 4.5],
                               sim=np.array([0, 5, 5, 5]))
    assert list(lad["over"]) == [0.75, 0.5] and list(lad["over_win"]) == [0.5, 0.0]
    assert list(lad["over_loss"]) == [1.0, 1.0] and list(lad["model_over"]) == [0.75, 0.75]
    pm = research.pmf_compare([1, 1, 2], np.array([1, 2, 2, 2]))
    assert pm.groupby("source")["share"].sum().round(9).tolist() == [1.0, 1.0]


def test_archetypes(processed):
    players, _ = processed
    st = research.map_stats(players, days=100000, min_maps=5)
    a = research.archetypes(st)
    assert len(a) and set(a["type"]) <= {"BOOM/BUST", "BALANCED", "STEADY", ""}
    assert (a.loc[a["role"].isin(["top", "sup"]), "type"] == "").all()
    assert a.loc[a["role"] == "mid", "type"].ne("").all()


@pytest.mark.parametrize("name", ["L+O+R1", "L+O+Q10", "L+O+F20"])
def test_projection_toggles_run(store, name):
    cfg = ALL_VARIANTS[name]
    lu = (store.default_lineup(A), store.default_lineup(B))
    p = build_matchup(store.hist, A, B, "LCK", 3, 0.6, lu, store.cutoff, store.patch, cfg)
    sim = simulate_series(p, n_sims=2000, rng=1)
    assert sim.player_kills.sum(axis=(2, 3)).tolist() == sim.team_kills.sum(axis=2).tolist()
    if name.endswith("Q10"):
        assert p.alpha_wl is not None and p.alpha_wl.shape == (2, 2, 5)
        # shares by outcome keep the per-position concentration
        assert np.allclose(p.alpha_wl.sum(axis=2).mean(), p.alpha.sum(axis=1).mean(),
                           rtol=0.2)
    base = build_matchup(store.hist, A, B, "LCK", 3, 0.6, lu, store.cutoff, store.patch,
                         locked_config(kappa_pos=store.kappa_pos))
    assert base.alpha_wl is None


def test_form_scale_and_fit():
    from src.engine.calibration import fit_form, form_scale, scale_pmf, scale_samples
    rng = np.random.default_rng(0)
    x = rng.poisson(6, 200000)
    c = form_scale(6.0, 2.0, 0.25)
    assert c == pytest.approx(6.5 / 6.0)
    s = scale_samples(x, c, rng=1)
    assert s.mean() == pytest.approx(6.5, abs=0.03)
    pmf = np.bincount(x, minlength=40)[:40] / len(x)
    assert np.allclose(scale_pmf(pmf, c), np.bincount(s, minlength=40)[:40] / len(s),
                       atol=0.004)
    assert form_scale(6.0, np.nan, 0.3) == 1.0 and form_scale(6.0, 2.0, 0.0) == 1.0
    gap = rng.normal(0, 2, 3000)
    rows = pd.DataFrame({"market": ["Maps 1-2"] * 2000 + ["Map 1"] * 1000, "gap": gap,
                         "err": 0.3 * gap + rng.normal(0, 1, 3000),
                         "date": pd.Timestamp("2026-01-01")})
    b = fit_form(rows)
    assert b["Maps 1-2"] == pytest.approx(0.3, abs=0.05) and b["Map 1"] == pytest.approx(
        0.3, abs=0.08)
    assert fit_form(rows.head(100)) == {"_pool": 0.0}


def test_prior_form_uses_only_earlier_series(processed):
    players, _ = processed
    name = str(players["playername"].iloc[0])
    pf = research.prior_form(players, "Map 1", n_last=3)
    mine = pf[pf["player"] == name]
    t = research.market_totals(players, name, "Map 1").sort_values("date")
    assert len(mine) == len(t)
    m = mine.set_index("series_id").loc[t["series_id"]]
    assert m["n_prev"].iloc[0] == 0 and m["n_prev"].iloc[-1] == min(3, len(t) - 1)
    assert m["avg_prev"].iloc[-1] == pytest.approx(t["kills"].iloc[-4:-1].mean())


def test_app_applies_form(processed):
    players, teams = processed
    st_ = DataStore(players, teams, calib_pairs=pd.DataFrame(), spread_rows=pd.DataFrame(),
                    w_rows=pd.DataFrame(), w_real={},
                    form_rows=pd.DataFrame({"date": pd.Timestamp("2000-01-01"),
                                            "market": "Maps 1-2", "gap": np.arange(600.0),
                                            "err": 0.5 * np.arange(600.0)}))
    assert st_.form_beta["Maps 1-2"] == pytest.approx(0.5)
    lu = st_.default_lineup(A)[:2]
    m = MatchupInput(A, B, 3, "PrizePicks", "-150", "+125",
                     lines="\n".join(f"{pl} 4.5" for pl in lu))
    r = run_matchup(st_, m, legs=2, n_sims=4000)
    assert r.sim.form and all(k[1] == "Maps 1-2" for k in r.sim.form)
    for (pl, mk), c in r.sim.form.items():
        raw = r.sim.player_kills[:, :2, *r.sim.player_index(pl)].sum(axis=1).mean()
        avg, _ = research.recent_avg(st_.players, pl, mk)
        assert c * raw == pytest.approx(raw + 0.5 * (avg - raw))
        assert r.sim.range_kills(pl, mk).mean() == pytest.approx(c * raw, rel=0.02)
