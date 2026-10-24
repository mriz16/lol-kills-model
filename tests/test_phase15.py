import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtest.walkforward import (VARIANTS, _range_actual, pick_best_combo,
                                      primary_market, run_backtest, summarize)
from src.data.audit import audit
from src.data.oe import PLAYER_KEEP, TEAM_KEEP
from src.engine.features import (EngineConfig, History, _blend_to_floor, build_matchup,
                                 fit_kappa_by_position, fit_nb_dispersion, kish_neff)
from src.engine.sim import simulate_series
from tests.test_engine import toy_params

ROOT = Path(__file__).resolve().parents[1]


def _matchup(processed, **cfg):
    players, teams = processed
    hist = History(players, teams)
    t = teams["teamname"].astype(str).value_counts().index[:2].tolist()
    cutoff = teams["date"].max() + pd.Timedelta(days=1)
    patch = int(teams["patch_ord"].max())
    lineup = [["x1", "x2", "x3", "x4", "x5"], ["y1", "y2", "y3", "y4", "y5"]]
    return hist, build_matchup(hist, t[0], t[1], "LCK", 3, 0.5, lineup, cutoff, patch,
                               EngineConfig(**cfg))


# ---------------------------------------------------------------- toggle A ----------
def test_fit_kappa_by_position(processed):
    players, teams = processed
    hist = History(players, teams)
    k = fit_kappa_by_position(hist, int(teams["date"].max().value) + 1)
    assert set(k) == {"top", "jng", "mid", "bot", "sup"}
    assert all(v > 0 for v in k.values())


def test_position_kappa_keeps_mean_shares_and_changes_spread():
    p = toy_params()
    shares = p.alpha[0] / p.alpha[0].sum()
    kv = np.array([10.0, 50.0, 50.0, 50.0, 3.0])
    p.alpha = np.vstack([kv * shares] * 2)
    p.scale = np.vstack([1.0 / kv] * 2)
    sim = simulate_series(p, n_sims=40_000, rng=0)
    tk = sim.team_kills[:, 0, 0]
    obs = sim.player_kills[:, 0, 0, :].sum(0) / tk.sum()
    assert obs == pytest.approx(shares, abs=0.01)
    # low-kappa support is over-dispersed relative to high-kappa mid (per unit share)
    pk = sim.player_kills[:, 0, 0, :] / np.maximum(tk, 1)[:, None]
    cv = pk.std(0) / pk.mean(0)
    assert cv[4] > cv[2]


def test_build_matchup_position_kappa(processed):
    _, prm = _matchup(processed, position_kappa=True)
    assert not np.allclose(prm.scale, 1.0)
    shares = prm.alpha * prm.scale
    assert shares.sum(1) == pytest.approx([1.0, 1.0])


# ---------------------------------------------------------------- toggle B ----------
def test_kish_neff():
    assert kish_neff(np.ones(10)) == pytest.approx(10)
    assert kish_neff(np.array([1.0, 0.0, 0.0])) == pytest.approx(1)
    w = 0.65 ** np.arange(12)
    assert 2 < kish_neff(w) < 6


@pytest.mark.parametrize("floor", [15, 25])
def test_blend_reaches_floor(floor):
    v, w = np.arange(6.0), 0.65 ** np.arange(6)
    lv, lw = np.arange(200.0), np.ones(200)
    vals, probs, lam = _blend_to_floor(v, w, lv, lw, floor)
    assert probs.sum() == pytest.approx(1.0)
    assert kish_neff(probs) == pytest.approx(floor, rel=1e-3)
    assert 0 < lam < 1
    # already above the floor: untouched
    vals2, probs2, lam2 = _blend_to_floor(lv, lw, v, w, floor)
    assert lam2 == 1.0 and len(vals2) == 200


def test_build_matchup_neff_floor(processed):
    _, prm = _matchup(processed, neff_floor=25)
    for s in (0, 1):
        for o in (0, 1):
            assert kish_neff(prm.pools[s][o][1]) >= 25 - 1e-6


# ---------------------------------------------------------------- toggle C ----------
def test_nb_dispersion_fit(processed):
    players, teams = processed
    hist = History(players, teams)
    r = fit_nb_dispersion(hist, int(teams["date"].max().value) + 1,
                          int(teams["patch_ord"].max()))
    assert "_global" in r and "LCK" in r
    assert all(1.0 <= v <= 1000.0 for v in r.values())


def test_smoothed_bootstrap_keeps_mean_adds_variance():
    a = simulate_series(toy_params(), n_sims=60_000, rng=1)
    p = toy_params()
    p.nb_r = 8.0
    b = simulate_series(p, n_sims=60_000, rng=1)
    ka, kb = a.team_kills[:, 0, 0], b.team_kills[:, 0, 0]
    assert kb.mean() == pytest.approx(ka.mean(), rel=0.02)
    assert kb.var() > ka.var() * 1.5
    assert kb.max() > ka.max()  # can exceed every pool value: the point of toggle C


def test_build_matchup_smoothing(processed):
    _, prm = _matchup(processed, smooth_bootstrap=True)
    assert prm.nb_r is not None and prm.nb_r >= 1


# ---------------------------------------------------------------- backtest ----------
def test_primary_market_is_what_is_bet():
    assert primary_market(1) == "Map 1"
    assert primary_market(3) == "Maps 1-2"
    assert primary_market(5) == "Maps 1-3"
    assert _range_actual([2, 3, 4, 1], "Maps 1-3") == 9
    assert _range_actual([2, None, 4], "Maps 1-3") is None
    assert _range_actual([2, 3], "Maps 1-3") is None


def test_variant_list():
    assert set(VARIANTS) >= {"base", "shrink30", "shrink30+A", "shrink30+B25",
                             "shrink30+B35", "shrink30+B50", "shrink30+C"}
    assert VARIANTS["base"].shrink_games == 20
    assert all(v.shrink_games == 30 for k, v in VARIANTS.items() if k != "base")
    t = pd.DataFrame({"variant": ["shrink30+B25", "shrink30+B35", "shrink30+C"],
                      "crps": [1.7, 1.6, 1.5]})
    assert pick_best_combo(t) == "shrink30+A+B35"  # C is not a B candidate


def test_backtest_tail_report_and_quick(processed):
    players, teams = processed
    v = {k: VARIANTS[k] for k in ("base", "shrink30+C")}
    start = teams["date"].max() - pd.Timedelta(days=30)
    res = run_backtest(players, teams, variants=v, warmup_days=20, n_sims=500,
                       eval_start=start, verbose=False)
    assert (res["date"] >= start).all()
    assert set(res["market"]) <= {"Map 1", "Maps 1-2", "Maps 1-3"}
    s = summarize(res)
    tail = s["tail"]
    assert {"top", "jng", "mid", "bot", "sup", "ALL"} <= set(tail["position"])
    assert set(tail["variant"]) == {"base", "shrink30+C"}


# ---------------------------------------------------------------- data workflow -----
def test_slim_parquet_columns(processed):
    players, teams = processed
    assert list(players.columns) == PLAYER_KEEP
    assert list(teams.columns) == TEAM_KEEP


def test_audit_flags_missing_leagues(raw, processed):
    text, missing = audit(raw, *processed, ["fixture.csv"])
    assert "LPL" in missing and "LCK" not in missing and "FST" not in missing
    assert "MISSING:" in text and "Patch range" in text and "Incomplete" in text


def test_refresh_fails_loudly_on_html(tmp_path):
    bad = tmp_path / "2026_LoL_esports_match_data_from_OraclesElixir.csv"
    bad.write_text("<!DOCTYPE html><html><body>Google Drive - quota exceeded</body></html>")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "refresh_data.py"), "--csv",
                        str(bad), "--out", str(tmp_path / "o"), "--audit",
                        str(tmp_path / "a.md")], capture_output=True, text=True)
    assert r.returncode == 1
    assert "web page" in r.stdout


def test_refresh_fails_loudly_on_empty(tmp_path):
    empty = tmp_path / "e.csv"
    from src.data.oe import REQUIRED_COLUMNS
    empty.write_text(",".join(REQUIRED_COLUMNS) + "\n")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "refresh_data.py"), "--csv",
                        str(empty), "--out", str(tmp_path / "o"), "--audit",
                        str(tmp_path / "a.md")], capture_output=True, text=True)
    assert r.returncode == 1
    assert "0 rows" in r.stdout
