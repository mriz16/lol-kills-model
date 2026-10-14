"""Spread toggles (S, K, I), PIT, selection check, slate log and backfill."""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.app import grading, slatelog
from src.app.picklog import LocalStore
from src.app.slate import (MIN_EDGE, DataStore, MatchupInput, flags_for, rebreakeven,
                           results_table, run_matchup, side_balance, slip_summary)
from src.backtest.walkforward import ALL_VARIANTS, PMF_MAX, run_backtest
from src.engine.calibration import (OFFSETS, BetaMap, Calibrator, IsoMap, calibrate_pmf,
                                    calibrated_prop, fit_spread, inflate_pmf,
                                    inflate_samples, pairs_from_rows, pav)
from src.engine.features import EngineConfig, History, player_kappa
from scripts.spread_report import pit_hist, pit_stats, pmf_metrics, proxy_lines

RNG = np.random.default_rng(0)


# ------------------------------------------------------------------ S ----
def test_inflate_pmf_matches_samples_and_keeps_mean():
    x = RNG.poisson(6.0, 200_000)
    pmf = np.bincount(x, minlength=40)[:40] / len(x)
    for k in (1.0, 1.1, 1.3):
        exact = inflate_pmf(pmf, k)
        emp = np.bincount(inflate_samples(x, k, rng=1), minlength=40)[:40] / len(x)
        assert exact.sum() == pytest.approx(1.0)
        assert np.abs(exact - emp).max() < 0.004
        ks = np.arange(40)
        assert exact @ ks == pytest.approx(pmf @ ks, abs=0.02)   # clip at 0 is tiny here
        var = exact @ ks ** 2 - (exact @ ks) ** 2
        assert var == pytest.approx(k * k * 6.0 + (0.17 if k != 1 else 0), rel=0.06)


def test_inflate_samples_keeps_sim_order():
    x = RNG.poisson(5.0, 5000)
    y = inflate_samples(x, 1.2, rng=3)
    big, small = x >= 8, x <= 2
    assert y[big].min() >= y[small].max()


def test_fit_spread_ratio_and_shrinkage():
    n = 4000
    rows = pd.DataFrame({"position": np.repeat(["top", "mid"], n // 2),
                         "market": "Maps 1-2", "mean": 5.0, "var": 4.0})
    rows["y"] = rows["mean"] + RNG.normal(0, 1, n) * np.where(rows["position"] == "top",
                                                              2.4, 2.0)
    k = fit_spread(rows)
    assert k[("top", "Maps 1-2")] == pytest.approx(1.2, abs=0.05)
    assert k[("mid", "Maps 1-2")] == pytest.approx(1.0, abs=0.05)
    assert fit_spread(rows.head(100)) == {"_pool": 1.0}       # too few rows: identity


# ------------------------------------------------------------------ I ----
def test_pav_is_monotone_and_weighted():
    out = pav(np.array([0.1, 0.5, 0.3, 0.7]), np.array([1.0, 1.0, 3.0, 1.0]))
    assert np.all(np.diff(out) >= 0)
    assert out[1] == pytest.approx((0.5 + 0.9) / 4) and out[2] == out[1]


@pytest.mark.parametrize("M", [IsoMap, BetaMap])
def test_calibration_maps_recover_distortion(M):
    p = RNG.uniform(0.02, 0.98, 60_000)
    true = p ** 1.3                      # model overstates P
    y = (RNG.random(len(p)) < true).astype(float)
    m = M.fit(p, y)
    grid = np.array([0.2, 0.5, 0.8])
    assert np.abs(m(grid) - grid ** 1.3).max() < 0.03
    assert m(0.0) == 0.0 and m(1.0) == 1.0


def _pairs(n=8000, distort=1.0, date="2025-01-01"):
    p = RNG.uniform(0.05, 0.95, n)
    y = (RNG.random(n) < p ** distort).astype(int)
    return pd.DataFrame({"date": pd.Timestamp(date), "position": "mid", "market": "Maps 1-2",
                         "offset": 0.5, "p": p, "y": y})


def test_calibrator_identity_when_calibrated_and_pool_fallback():
    cal = Calibrator.fit(_pairs(), "beta")
    grid = np.array([0.1, 0.5, 0.9])
    assert np.abs(cal(grid, "mid", "Maps 1-2") - grid).max() < 0.03
    assert np.abs(cal(grid, "top", "Map 1") - grid).max() < 0.03      # pool only
    assert Calibrator.fit(_pairs(100)).pool is None                     # too few pairs


def test_calibrate_pmf_only_inside_window_and_valid():
    x = RNG.poisson(8.0, 20_000)
    pmf = np.bincount(x, minlength=40)[:40] / len(x)
    cal = Calibrator.fit(_pairs(distort=1.5), "beta")
    out = calibrate_pmf(pmf, cal, "mid", "Maps 1-2", center=8.0)
    assert out.sum() == pytest.approx(1.0) and (out >= 0).all()
    s_raw, s_new = 1 - np.cumsum(pmf), 1 - np.cumsum(out)
    assert np.all(s_new[8:11] < s_raw[8:11])          # overstated overs pulled down
    assert np.allclose(s_new[:2], s_raw[:2])          # far below the window: raw
    p = calibrated_prop(x, 8.5, cal, "mid", "Maps 1-2")
    assert p["p_over"] + p["p_under"] == pytest.approx(1.0) and p["p_push"] < 1e-9
    w = calibrated_prop(x, 8.0, cal, "mid", "Maps 1-2")
    assert w["p_over"] + w["p_under"] + w["p_push"] == pytest.approx(1.0)
    raw = calibrated_prop(x, 8.5, None, "mid", "Maps 1-2")
    assert raw["p_over"] == pytest.approx((x > 8.5).mean())


def test_pairs_from_rows_offsets():
    pmf = np.zeros(PMF_MAX)
    pmf[[3, 4, 5]] = [0.25, 0.5, 0.25]
    rows = pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "position": ["top"],
                         "market": ["Map 1"], "line": [4.0], "y": [5], "pmf": [pmf]})
    pr = pairs_from_rows(rows).set_index("offset")
    assert list(pr.index) == list(OFFSETS)
    assert pr.loc[0.5, "p"] == pytest.approx(0.25) and pr.loc[0.5, "y"] == 1
    assert pr.loc[-0.5, "p"] == pytest.approx(0.75) and pr.loc[1.5, "y"] == 0


# ------------------------------------------------------------------ PIT ----
def test_pit_flat_for_correct_forecasts_and_metrics():
    n, mu = 20_000, 6.0
    pmf = np.bincount(RNG.poisson(mu, 400_000), minlength=PMF_MAX)[:PMF_MAX] / 400_000
    y = RNG.poisson(mu, n)
    P = np.tile(pmf, (n, 1))
    m = pmf_metrics(P, y, np.full(n, 6.0))
    h = pit_hist(m["pit_lo"].to_numpy(), m["pit_hi"].to_numpy())
    assert np.abs(h - 0.1).max() < 0.01
    st = pit_stats(m["pit_lo"].to_numpy(), m["pit_hi"].to_numpy())
    assert st["pit_var_ratio"] == pytest.approx(1.0, abs=0.03)
    narrow = np.bincount(RNG.poisson(mu, 400_000) // 2 + 3, minlength=PMF_MAX)[:PMF_MAX]
    mn = pmf_metrics(np.tile(narrow / narrow.sum(), (n, 1)), y, np.full(n, 6.0))
    assert pit_stats(mn["pit_lo"].to_numpy(), mn["pit_hi"].to_numpy())["pit_var_ratio"] > 1.1
    assert mn["crps"].mean() > m["crps"].mean()


def test_proxy_lines_use_prior_series_only(processed):
    players, _ = processed
    pr = proxy_lines(players)
    p = players.copy()
    p["series_id"] = p["series_id"].astype(str)
    p["playername"] = p["playername"].astype(str)
    pl = p["playername"].value_counts().index[0]
    mine = p[p["playername"] == pl]
    order = mine.groupby("series_id")["date"].min().sort_values()
    first = pr[(pr["player"] == pl) & (pr["series_id"] == order.index[0])]
    assert first["proxy_line"].isna().all()           # no history before the first series
    tot = mine[mine["map_no"] == 1].set_index("series_id")["kills"].reindex(order.index)
    k = 7                                             # 8th series: last 7 map-1 totals
    want = np.floor(np.median(tot.iloc[:k].astype(float))) + 0.5
    got = pr[(pr["player"] == pl) & (pr["series_id"] == order.index[k])
             & (pr["market"] == "Map 1")]["proxy_line"].iloc[0]
    if order.iloc[k] > order.iloc[k - 1]:
        assert got == want
    assert ((pr["proxy_line"].dropna() % 1) == 0.5).all()


# ------------------------------------------------------------------ K ----
def test_player_kappa_shrinks_to_position(processed):
    players, teams = processed
    hist = History(players, teams)
    cutoff = int((teams["date"].max() + pd.Timedelta(days=1)).value)
    patch = int(teams["patch_ord"].max())
    pl = players["playername"].astype(str).value_counts().index[0]
    assert player_kappa(hist, "nobody", 20.0, cutoff, patch,
                        EngineConfig(player_kappa_n0=30)) == (20.0, 0)
    k_loose, n = player_kappa(hist, pl, 20.0, cutoff, patch, EngineConfig(player_kappa_n0=1))
    k_tight, _ = player_kappa(hist, pl, 20.0, cutoff, patch, EngineConfig(player_kappa_n0=1e9))
    assert n > 5 and k_tight == pytest.approx(20.0, rel=1e-3) and k_loose != k_tight


def test_backtest_keep_pmf_and_k_variant(processed):
    players, teams = processed
    v = {k: ALL_VARIANTS[k] for k in ("L+O", "L+O+K30")}
    start = teams["date"].max() - pd.Timedelta(days=20)
    res = run_backtest(players, teams, variants=v, warmup_days=20, n_sims=500,
                       eval_start=start, verbose=False, keep_pmf=True)
    model = res[res["variant"] != "naive_poisson"]
    assert len(model) and model["pmf"].map(len).eq(PMF_MAX).all()
    assert model["pmf"].map(np.sum).between(0.999, 1.001).all()
    assert (model["var"] > 0).all()


# ------------------------------------------------------------------ app ----
def test_flags_calibrated_min_edge():
    assert flags_for("more", MIN_EDGE - 0.01, 0.6, 7.5, 50, calibrated=True) == ["BELOW MIN"]
    assert flags_for("more", MIN_EDGE + 0.01, 0.6, 7.5, 50, calibrated=True) == []
    assert flags_for("more", 0.01, 0.6, 7.5, 50) == ["THIN EDGE"]


def test_side_balance():
    df = pd.DataFrame({"play": [True] * 4 + [False], "side": ["less"] * 3 + ["more", "more"]})
    assert side_balance(df)["share"] == 0.75                # 3 of 4 plays: > 70%
    df = pd.DataFrame({"play": [True] * 4, "side": ["less"] * 4})
    w = side_balance(df)
    assert w["side"] == "less" and w["share"] == 1.0 and w["n"] == 4
    assert side_balance(df.head(2)) is None                 # too few plays
    df = pd.DataFrame({"play": [True] * 10, "side": ["less"] * 7 + ["more"] * 3})
    assert side_balance(df) is None                         # exactly 70%: not more than


def _store(processed, asof=None, pairs=None):
    players, teams = processed
    return DataStore(players, teams, asof=asof,
                     calib_pairs=pairs if pairs is not None else _pairs(10, date="2020-01-01"))


def test_datastore_backfill_uses_prior_data_only(processed):
    players, teams = processed
    asof = pd.Timestamp("2025-03-01")
    s = _store(processed, asof=asof)
    assert s.backfill and (pd.to_datetime(s.teams["date"]) < asof).all()
    assert s.cutoff == asof.value and s.last_date < asof
    assert len(s.teams) < len(teams)
    with pytest.raises(ValueError):
        _store(processed, asof="2000-01-01")


def test_calibrator_asof_in_store(processed):
    pairs = pd.concat([_pairs(5000, 1.4, "2025-01-01"), _pairs(5000, 1.4, "2025-06-01")])
    early = _store(processed, asof="2025-03-01", pairs=pairs)
    assert early.calibrated and early.calibrator.n_pairs == 5000    # later pairs excluded
    none = _store(processed, asof="2025-01-15", pairs=pairs.iloc[5000:])
    assert not none.calibrated


def _run(store, legs=2):
    a, b = "LCK Tigers", "LCK Ravens"
    lu = store.default_lineup(a)[:2] + store.default_lineup(b)[:2]
    m = MatchupInput(a, b, 3, "PrizePicks", "-150", "+125",
                     lines="\n".join(f"{pl} {x}" for pl, x in zip(lu, (4.5, 5, 3.5, 6.5))))
    return run_matchup(store, m, legs=legs, n_sims=4000)


def test_calibrated_rows_raw_on_side_and_slip(processed):
    pairs = _pairs(20_000, 1.4, "2025-01-01").assign(position="mid")
    store = _store(processed, pairs=pairs)
    res = _run(store)
    df = results_table([res])
    assert df["calibrated"].all()
    for r in df.itertuples():
        assert r.p_more + r.p_less + r.p_push == pytest.approx(1.0)
        assert r.raw_p_more + r.raw_p_less + r.raw_p_push == pytest.approx(1.0)
        assert r.play == (r.edge >= MIN_EDGE)
        assert "Raw model P(more)" in r.why
    assert (df["p_more"] != df["raw_p_more"]).any()
    d5 = rebreakeven(df, 5)
    assert (d5["raw_edge"] == d5["raw_p_side"] - d5["breakeven"]).all()
    legs = [dict(r) for _, r in df.head(2).iterrows()]
    s = slip_summary({res.inp.key: res.sim}, legs, "PrizePicks")
    ratio = np.prod([(l["p_more"] if l["side"] == "more" else l["p_less"])
                     / (l["raw_p_more"] if l["side"] == "more" else l["raw_p_less"])
                     for l in legs])
    assert s["p_joint"] == pytest.approx(min(s["p_joint_raw"] * ratio,
                                             min(l["p_side"] for l in legs)))
    assert s["ev_raw"] == pytest.approx(s["p_joint_raw"] * 3 - 1)


def test_uncalibrated_store_never_plays(processed):
    df = results_table([_run(_store(processed))])
    assert not df["calibrated"].any() and not df["play"].any()
    assert (df["p_more"] == df["raw_p_more"]).all()


# ------------------------------------------------------------------ slate log ----
def test_slate_log_append_replace_mark_played(tmp_path, processed):
    store = _store(processed)
    df = results_table([_run(store)])
    log = LocalStore(tmp_path)
    rows = slatelog.make_slate_rows(df, "2025-04-01", False, store.cutoff, slate_id="S1")
    assert len({r["row_id"] for r in rows}) == len(df)
    slatelog.append_slate(log, rows)
    slatelog.mark_played(log, [rows[0]["row_id"]])
    # re-running the same slate replaces the rows and keeps the played flag
    rows2 = slatelog.make_slate_rows(df, "2025-04-01", False, store.cutoff, slate_id="S2")
    out = slatelog.append_slate(log, rows2)
    assert len(out) == len(df) and set(out["slate_id"]) == {"S2"}
    got = slatelog.read_slates(log).set_index("row_id")["played"]
    assert got[rows[0]["row_id"]] == 1 and got.sum() == 1
    # another date is another set of rows
    slatelog.append_slate(log, slatelog.make_slate_rows(df, "2025-04-02", True,
                                                        store.cutoff))
    s = slatelog.read_slates(log)
    assert len(s) == 2 * len(df) and s["backfill"].sum() == len(df)


def test_grade_slates_anchor_on_slate_date(processed):
    players, teams = processed
    t = teams[teams["best_of"] == 3].sort_values("date")
    sid = t["series_id"].iloc[-1]
    g = t[t["series_id"] == sid]
    team = str(g["teamname"].iloc[0])
    opp = str(g[g["teamname"].astype(str) != team]["teamname"].iloc[0])
    pl = players[(players["series_id"] == sid) & (players["teamname"].astype(str) == team)]
    name = str(pl["playername"].iloc[0])
    mine = pl[pl["playername"].astype(str) == name]
    actual = int(mine[mine["map_no"].isin([1, 2])]["kills"].sum())
    day = pd.Timestamp(g["date"].min()).normalize().date().isoformat()
    base = dict(row_id="r1", slate_date=day, logged_at="2030-01-01T00:00:00+00:00",
                team=team, opponent=opp, player=name, market="Maps 1-2", line=actual - 0.5,
                side="more", p_more=0.6, p_less=0.4, raw_side="more", raw_p_more=0.65,
                raw_p_less=0.35, mean=actual - 0.2, played=1, backfill=1)
    sl = pd.DataFrame([base, dict(base, row_id="r2", line=actual + 0.5, side="more",
                                  raw_side="less", played=0)])
    gr = grading.grade_slates(sl, players, teams, now=datetime(2030, 1, 2))
    assert list(gr["status"]) == ["win", "loss"] and (gr["actual"] == actual).all()
    e = slatelog.evaluate(sl, gr)
    t = e["table"].set_index(["model", "group"])
    assert t.loc[("calibrated", "all"), "hit"] == 0.5
    assert t.loc[("raw", "all"), "hit"] == 1.0          # raw said less on r2: right
    assert t.loc[("calibrated", "played"), "n"] == 1
    assert e["bias"] == pytest.approx(-0.2)
