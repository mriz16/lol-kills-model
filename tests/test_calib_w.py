"""W (line shrinkage), S in the app, the calibration refit pipeline and the real-line refit."""
import json

import numpy as np
import pandas as pd
import pytest

from scripts import refit_calibration, refit_w_real
from src.app import slatelog
from src.app.slate import (MIN_EDGE, DataStore, MatchupInput, per_leg_breakeven,
                           results_table, run_matchup, slip_summary)
from src.backtest.calib_pipeline import (BE_2L, build, compare_params, derive_min_edge,
                                         load_rows, params_summary, write_artifacts)
from src.backtest.walkforward import ALL_VARIANTS, LIVE_LOCKED, run_backtest
from src.engine.calibration import fit_w, select_top, w_adjust

RNG = np.random.default_rng(5)


# ------------------------------------------------------------------ W ----
def test_w_adjust_keeps_mass_and_order():
    more, less = w_adjust(0.7, 0.2, 0.1, 0.5)
    assert more + less == pytest.approx(0.9)
    assert more == pytest.approx(0.45 + 0.5 * 0.25) and more > less
    assert w_adjust(0.7, 0.3, 0.0, 1.0) == pytest.approx((0.7, 0.3))
    assert w_adjust(0.7, 0.3, 0.0, 0.0) == pytest.approx((0.5, 0.5))


def _w_rows(n_slates=300, per=15, lam_true=0.7, seed=0):
    rng = np.random.default_rng(seed)
    n = n_slates * per
    p_more = rng.uniform(0.15, 0.85, n)
    true = 0.5 + lam_true * (p_more - 0.5)
    over = rng.random(n) < true
    return pd.DataFrame({
        "slate": np.repeat(np.arange(n_slates), per),
        "date": pd.Timestamp("2025-01-01") + pd.to_timedelta(np.repeat(np.arange(n_slates),
                                                                         per), "D"),
        "market": rng.choice(["Map 1", "Maps 1-2"], n), "line": 4.5, "p_more": p_more,
        "p_less": 1 - p_more, "p_push": 0.0, "y": np.where(over, 6, 3)})


def test_fit_w_recovers_shrinkage_and_calibrates_top5():
    rows = _w_rows()
    lam = fit_w(rows, BE_2L)
    assert lam["_pool"] == pytest.approx(0.7, abs=0.06)
    assert lam["Map 1"] == pytest.approx(0.7, abs=0.08)
    before = select_top(rows, None, BE_2L)
    after = select_top(rows, lam, BE_2L)
    assert before.groupby("slate").size().max() == 5
    assert before["p_w"].mean() - before["hit"].mean() > 0.05      # overstated
    assert abs(after["p_w"].mean() - after["hit"].mean()) < 0.02    # calibrated
    assert fit_w(rows.head(30), BE_2L) == {"_pool": 1.0, "_n": pytest.approx(10, abs=10)}


# ------------------------------------------------------------------ pipeline ----
@pytest.fixture(scope="module")
def fixture_rows(processed, tmp_path_factory):
    players, teams = processed
    res = run_backtest(players, teams, {LIVE_LOCKED: ALL_VARIANTS[LIVE_LOCKED]},
                       warmup_days=20, n_sims=500, verbose=False, keep_pmf=True)
    path = tmp_path_factory.mktemp("rows") / "rows.parquet"
    res.to_parquet(path, index=False)
    return path


def test_build_params_write_and_drift(processed, fixture_rows, tmp_path):
    players, _ = processed
    rows = load_rows([fixture_rows])[LIVE_LOCKED]
    art = build(rows, players)
    assert art["PS"].shape == art["P0"].shape and np.allclose(art["PSI"].sum(axis=1), 1)
    assert {"lam", "p_more", "p_less", "p_push", "slate"} <= set(art["wrows"].columns)
    assert set(art["pairs"]["model"]) == {"L+O", "L+O+S"}
    params = params_summary(art)
    assert {"spread_k", "w_lambda", "calib_pool", "min_edge_suggested"} <= set(params)
    files = write_artifacts(art, tmp_path, params)
    assert all(f.exists() for f in files)
    assert compare_params(params, params) == []
    moved = json.loads(json.dumps(params))
    moved["w_lambda"]["_pool"] = params["w_lambda"]["_pool"] + 0.5
    moved["spread_k"]["_pool"] = params["spread_k"]["_pool"] + 0.2
    bad = compare_params(moved, params)
    assert any("W lambda[_pool]" in b for b in bad) and any("S k[_pool]" in b for b in bad)
    assert compare_params(None, params) == []
    me, tab = derive_min_edge(art["wrows"])
    assert 0 <= me <= 1 and len(tab) == 4


def test_refit_script_drift_check_blocks_write(processed, fixture_rows, tmp_path):
    out, rep = tmp_path / "data", tmp_path / "fit.md"
    args = ["--rows", str(fixture_rows), "--out", str(out), "--report", str(rep)]
    assert refit_calibration.main(args) == 0
    params = json.loads((out / "calib_params.json").read_text())
    params["w_lambda"]["_pool"] += 0.9
    (out / "calib_params.json").write_text(json.dumps(params))
    stamp = (out / "w_rows.parquet").stat().st_mtime_ns
    assert refit_calibration.main(args) == 1                        # drift: no write
    assert (out / "w_rows.parquet").stat().st_mtime_ns == stamp
    assert "FAILED" in rep.read_text() and "before" in rep.read_text()
    assert refit_calibration.main(args + ["--force"]) == 0


# ------------------------------------------------------------------ app: S and W ----
def _pairs(n=20_000, distort=1.3):
    p = RNG.uniform(0.05, 0.95, n)
    return pd.DataFrame({"date": pd.Timestamp("2025-01-01"), "position": "mid",
                         "market": "Maps 1-2", "offset": 0.5, "p": p,
                         "y": (RNG.random(n) < p ** distort).astype(int), "model": "L+O+S"})


def _spread_rows(k=1.3, n=2000):
    return pd.DataFrame({"date": pd.Timestamp("2025-01-01"), "position": "mid",
                         "market": "Maps 1-2", "mean": 5.0, "var": 4.0,
                         "y": 5.0 + RNG.normal(0, 2 * k, n)})


def _store(processed, spread=None, w=None, real=None):
    players, teams = processed
    return DataStore(players, teams, calib_pairs=_pairs(),
                     spread_rows=spread if spread is not None else pd.DataFrame(),
                     w_rows=w if w is not None else pd.DataFrame(),
                     w_real=real if real is not None else {})


def _run(store):
    a, b = "LCK Tigers", "LCK Ravens"
    lu = store.default_lineup(a)[:3] + store.default_lineup(b)[:2]
    m = MatchupInput(a, b, 3, "PrizePicks", "-150", "+125",
                     lines="\n".join(f"{pl} {x}" for pl, x in zip(lu, (4.5, 5, 3.5, 6.5, 2.5))))
    return run_matchup(store, m, legs=2, n_sims=4000)


def test_app_applies_s_to_sims(processed):
    plain = _run(_store(processed))
    wide = _run(_store(processed, spread=_spread_rows()))
    assert plain.sim.spread_k is None and wide.sim.spread_k is not None
    mid = [lm.player for lm in wide.lines if lm.role == "mid"][0]
    a = plain.sim.range_kills(mid, "Maps 1-2")
    b = wide.sim.range_kills(mid, "Maps 1-2")
    assert b.var() > a.var() * 1.3 and abs(b.mean() - a.mean()) < 0.3   # clip at 0
    assert np.array_equal(b, wide.sim.range_kills(mid, "Maps 1-2"))   # stable for slips
    k = results_table([wide]).set_index("player")["k_spread"]
    assert k[mid] == pytest.approx(wide.sim.spread_k[("mid", "Maps 1-2")])


def test_app_applies_w_and_real_lambda(processed):
    store = _store(processed, w=_w_rows(lam_true=0.6))
    assert store.w_source == "proxy lines" and store.w_lambda["_pool"] < 0.75
    df = results_table([_run(store)])
    for r in df.itertuples():
        lam = r.w_lambda
        pl = (1 - r.cal_p_push) / 2
        assert r.p_more == pytest.approx(pl + lam * (r.cal_p_more - pl))
        assert r.p_less == pytest.approx(pl + lam * (r.cal_p_less - pl))
        assert r.play == (r.edge >= MIN_EDGE)
        assert abs(r.p_side - pl) <= abs(r.cal_p_side - pl) + 1e-12
    real = {"lambda": {"_pool": 0.5, "Maps 1-2": 0.4}, "n_rows": 320,
            "fitted_through": "2024-01-01"}
    rs = _store(processed, w=_w_rows(), real=real)
    assert rs.w_source.startswith("real") and rs.w_lambda["Maps 1-2"] == 0.4
    late = dict(real, fitted_through="2030-01-01")             # fitted after the cutoff
    assert _store(processed, w=_w_rows(), real=late).w_source == "proxy lines"


def test_slip_uses_shown_probabilities(processed):
    store = _store(processed, spread=_spread_rows(), w=_w_rows(lam_true=0.6))
    res = _run(store)
    legs = [dict(r) for _, r in results_table([res]).head(2).iterrows()]
    s = slip_summary({res.inp.key: res.sim}, legs, "PrizePicks")

    def p(l, pre):
        return l[f"{pre}p_{l['side']}"]
    for pre, key in (("", "p_joint"), ("cal_", "p_joint_cal")):
        ratio = np.prod([p(l, pre) / p(l, "raw_") for l in legs])
        assert s[key] == pytest.approx(min(s["p_joint_raw"] * ratio,
                                           min(p(l, pre) for l in legs)))
    assert s["ev"] == pytest.approx(s["p_joint"] * 3 - 1)
    assert s["ev_cal"] == pytest.approx(s["p_joint_cal"] * 3 - 1)


# ------------------------------------------------------------------ real-line refit ----
def _log(n_rows, lam_true=0.6):
    w = _w_rows(n_slates=n_rows // 15, per=15, lam_true=lam_true, seed=2)
    slates = pd.DataFrame({
        "row_id": [f"r{i}" for i in range(len(w))], "slate_id": w["slate"].astype(str),
        "slate_date": w["date"].dt.date.astype(str), "platform": "PrizePicks",
        "market": w["market"], "line": w["line"], "p_more": 0.5, "p_less": 0.5,
        "p_push": 0.0, "cal_p_more": w["p_more"], "cal_p_less": w["p_less"],
        "cal_p_push": 0.0})
    graded = pd.DataFrame({"row_id": slates["row_id"], "status": "win",
                           "actual": w["y"].astype(float)})
    return slates, graded


def test_real_line_rows_and_refit():
    slates, graded = _log(600)
    rows = slatelog.graded_real_rows(slates, graded)
    assert len(rows) == 600 and rows["p_more"].equals(slates["cal_p_more"].astype(float))
    other = slates.assign(platform="Underdog")
    assert slatelog.graded_real_rows(other, graded).empty
    out = refit_w_real.fit_real(rows)
    assert out["n_rows"] == 600 and out["lambda"]["_pool"] < 0.9
    assert abs(out["top5_after"]["pred"] - out["top5_after"]["hit"]) < \
        abs(out["top5_before"]["pred"] - out["top5_before"]["hit"])
    with pytest.raises(ValueError):
        refit_w_real.fit_real(rows.head(299))


def test_breakeven_constant_matches_app():
    assert BE_2L == pytest.approx(per_leg_breakeven("PrizePicks", 2))
