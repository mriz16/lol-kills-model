"""Mean-bias toggles: M (meta drift), P (fitted position priors), G (global multiplier)."""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.backtest.walkforward import ALL_VARIANTS, LOCKED, run_backtest
from src.engine.features import EngineConfig, History
from tests.test_phase15 import _matchup


def _boost_last_patch(processed, factor=1.3):
    """Copy of the fixture with LCK team kills on the latest 3 LCK patches (>= 20 games,
    the DRIFT_MIN_GAMES window) scaled up."""
    players, teams = (f.copy() for f in processed)
    lck = teams["league"].astype(str) == "LCK"
    recent = sorted(teams.loc[lck, "patch_ord"].unique())[-3:]
    m = lck & teams["patch_ord"].isin(recent)
    teams.loc[m, "teamkills"] = (teams.loc[m, "teamkills"] * factor).round().astype("int16")
    return players, teams


def test_drift_table_detects_rising_env(processed):
    players, teams = _boost_last_patch(processed)
    hist = History(players, teams)
    cutoff = int((teams["date"].max() + pd.Timedelta(days=1)).value)
    patch = int(teams["patch_ord"].max())
    cfg = EngineConfig(meta_drift_patches=1)
    envs = hist.env_table(cutoff, patch, cfg)
    d = hist.drift_table(cutoff, patch, cfg, envs)
    assert d["LCK"] > 1.0
    assert C.DRIFT_CLAMP[0] <= min(d.values()) and max(d.values()) <= C.DRIFT_CLAMP[1]
    # the unmodified data has no rise of that size
    h0 = History(*processed)
    d0 = h0.drift_table(cutoff, patch, cfg, h0.env_table(cutoff, patch, cfg))
    assert d["LCK"] > d0["LCK"]


def test_meta_drift_scales_pools(processed):
    data = _boost_last_patch(processed)
    _, off = _matchup(data)
    _, on = _matchup(data, meta_drift_patches=1)
    d = on.info["drift"]["LCK"]
    assert d > 1.0
    mean = [sum(np.dot(*pool) for side in m.pools for pool in side) for m in (off, on)]
    assert mean[1] > mean[0]
    assert off.info["drift"] is None


def test_fitted_priors(processed):
    players, teams = processed
    hist = History(players, teams)
    cutoff = int((teams["date"].max() + pd.Timedelta(days=1)).value)
    patch = int(teams["patch_ord"].max())
    pr = hist.fitted_priors("LCK", cutoff, patch, EngineConfig(patch_decay=1.0,
                                                                patch_lookback=999))
    assert set(pr) == set(C.POSITIONS) and sum(pr.values()) == pytest.approx(1.0)
    # matches the empirical kill share (mostly LCK weight once the league is big)
    lck = players[players["league"].astype(str) == "LCK"]
    emp = lck.groupby("position", observed=True)["kills"].sum() / lck["kills"].sum()
    for pos in C.POSITIONS:
        assert pr[pos] == pytest.approx(emp[pos], abs=0.03)
    _, prm = _matchup(processed, fit_priors=True)
    assert prm.info["position_priors"] != C.POSITION_PRIORS


def test_kill_mult_scales_pools(processed):
    _, a = _matchup(processed)
    _, b = _matchup(processed, kill_mult=1.1)
    for s in (0, 1):
        for o in (0, 1):
            assert np.allclose(b.pools[s][o][0], 1.1 * a.pools[s][o][0])


def test_g_variant_walk_forward(processed, monkeypatch):
    import src.backtest.walkforward as wf
    monkeypatch.setattr(wf, "G_MIN_ROWS", 50)
    players, teams = processed
    v = {k: ALL_VARIANTS[k] for k in (LOCKED, "L+G")}
    res = run_backtest(players, teams, variants=v, n_sims=500, warmup_days=20,
                       verbose=False)
    g = res[res.variant == "L+G"].sort_values("date")
    assert g["kill_mult"].iloc[0] == 1.0          # no prior data yet
    assert g["kill_mult"].nunique() > 1           # refit as series accumulate
    with pytest.raises(ValueError):
        run_backtest(players, teams, variants={"L+G": ALL_VARIANTS["L+G"]}, n_sims=100,
                     verbose=False)


def test_bias_report_builds(processed, monkeypatch):
    import src.backtest.walkforward as wf
    from scripts.bias_report import build
    monkeypatch.setattr(wf, "G_MIN_ROWS", 50)
    players, teams = processed
    v = {k: ALL_VARIANTS[k] for k in (LOCKED, "L+G", "L+M1", "L+P")}
    res = run_backtest(players, teams, variants=v, n_sims=300, warmup_days=20,
                       verbose=False)
    nop = res[res.league != "LCK"]  # stand-in for the no-partial run
    text = build(res, nop.assign(league=nop.league.replace("LEC", "LPL")),
                 "# t", "data", None)
    for h in ("## Variants vs locked", "## Bias by position", "## Under-25% buckets",
              "## CRPS by league", "## LPL with and without"):
        assert h in text


def test_locked_config_matches_backtest_variant():
    from src.engine.features import locked_config
    assert locked_config() == ALL_VARIANTS[LOCKED]
    assert not locked_config().game_script and not locked_config().smooth_bootstrap


def test_opp_by_outcome(processed):
    _, off = _matchup(processed)
    _, on = _matchup(processed, opp_by_outcome=True)
    for s in (0, 1):
        by = on.info[f"opp_adj_{s}_by_outcome"]
        assert set(by) == {0, 1}
        assert all(C.OPP_ADJ_CLAMP[0] <= v <= C.OPP_ADJ_CLAMP[1] for v in by.values())
        # pools differ from the plain adjustment only by the per-outcome factor
        a = off.info[f"opp_adj_{s}"]
        for o in (0, 1):
            assert np.allclose(on.pools[s][o][0] * a, off.pools[s][o][0] * by[o])
