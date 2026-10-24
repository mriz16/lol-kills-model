"""Audit: are the model's unders real? Plus floor / goblin calibration.

    python scripts/audit_unders.py                       # backtest since 2026-03-15 (~3 min)
    python scripts/audit_unders.py --rows reports/backtest_rows_audit.parquet   # reuse rows

Writes reports/audit_unders.md. PrizePicks lines are not logged yet, so "PP-like" lines are
proxies built from each player's own history before the series:
  mean10 = trailing mean of the player's last 10 totals in that market (>= 6), to the
           nearest half point (what a book that tracks recent averages would post)
  med10  = floor(trailing median of the last 10) + 0.5 (the calibration pipeline's proxy)
Sections: (1) model-free under rate at those lines, (2) mean bias of the live model,
(3) the app's shown probabilities (S + I + W) at PP-like lines by side / position /
market, (4) lower-tail (floor) calibration at lines relative to the model median and to
the PP-like line (goblin proxies), (5) Worlds 2026 environment.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_backtest import _md  # noqa: E402
from src.app.slate import SAFE_FLOOR_P, per_leg_breakeven  # noqa: E402
from src.backtest.walkforward import ALL_VARIANTS, LIVE_LOCKED, run_backtest  # noqa: E402
from src.config import DATA_DIR, REPORTS_DIR  # noqa: E402
from src.engine.calibration import (calibrate_pmf, fit_calibrator_asof, fit_spread_asof,  # noqa: E402
                                    fit_w_asof, inflate_pmf, lambda_for)

MARKETS = {1: ("Map 1", 1, 1), 3: ("Maps 1-2", 1, 2), 5: ("Maps 1-3", 1, 3)}
BE_2L = per_leg_breakeven("PrizePicks", 2)


def market_totals(players: pd.DataFrame) -> pd.DataFrame:
    """Player totals per series over the bet market, with trailing-10 proxy lines."""
    parts = []
    for bo, (mkt, lo, hi) in MARKETS.items():
        q = players[(players["best_of"] == bo) & players["map_no"].between(lo, hi)]
        g = q.groupby(["series_id", "playername", "position", "league"], observed=True).agg(
            y=("kills", "sum"), date=("date", "min"), maps=("map_no", "nunique")).reset_index()
        parts.append(g[g["maps"] == hi - lo + 1].assign(market=mkt))
    d = pd.concat(parts).sort_values("date")
    roll = d.groupby(["playername", "market"])["y"].transform
    d["mean10"] = roll(lambda s: s.shift(1).rolling(10, min_periods=6).mean())
    d["med10"] = roll(lambda s: s.shift(1).rolling(10, min_periods=6).median())
    d = d.dropna(subset=["mean10"])
    d["L_mean"] = np.round(d["mean10"] - 0.5) + 0.5          # nearest x.5 to the mean
    d["L_med"] = np.floor(d["med10"]) + 0.5
    return d.rename(columns={"playername": "player"})


def model_free(d: pd.DataFrame) -> str:
    out = []
    for L, name in (("L_mean", "trailing mean"), ("L_med", "trailing median")):
        u = d["y"] < d[L]
        t = d.assign(u=u).groupby("market").agg(n=("u", "size"), under=("u", "mean"))
        t["line"] = name
        p = d.assign(u=u).pivot_table(index="market", columns="position", values="u")
        out.append(_md(t.reset_index().round(3)))
        out.append("")
        out.append(f"Under rate by position ({name} line):")
        out.append("")
        out.append(_md(p.reset_index().round(3)))
        out.append("")
    gap = (d["mean10"] - d["med10"]).groupby(d["market"]).mean().round(2)
    out.append("Trailing mean minus trailing median (right skew): "
               + ", ".join(f"{k} {v:+.2f}" for k, v in gap.items()) + " kills.")
    return "\n".join(out)


def s_pmfs(rows: pd.DataFrame) -> np.ndarray:
    k = fit_spread_asof(pd.read_parquet(DATA_DIR / "spread_rows.parquet"),
                        rows["date"].min().normalize())
    P = np.stack(rows["pmf"].to_numpy()).astype(float)
    return np.stack([inflate_pmf(P[i], k.get((pos, mkt), k["_pool"]))
                     for i, (pos, mkt) in enumerate(zip(rows["position"], rows["market"]))])


def bucket(d: pd.DataFrame, by: list[str], edges) -> pd.DataFrame:
    d = d.assign(b=pd.cut(d["p"], edges))
    t = d.groupby(by + ["b"], observed=True).agg(n=("hit", "size"), pred=("p", "mean"),
                                                 hit=("hit", "mean")).reset_index()
    t["diff"] = t["hit"] - t["pred"]
    t["se"] = np.sqrt(t["hit"] * (1 - t["hit"]) / t["n"])
    t["b"] = t["b"].astype(str)
    return t[t["n"] >= 50].round(3)


def shown_at_lines(rows: pd.DataFrame, PS: np.ndarray, line_col: str) -> pd.DataFrame:
    """The app's shown P (S pmf -> I beta -> W) at each row's PP-like line."""
    pairs = pd.read_parquet(DATA_DIR / "calib_pairs.parquet")
    pairs = pairs[pairs["model"] == "L+O+S"]
    t0 = rows["date"].min().normalize()
    cal = fit_calibrator_asof(pairs, t0, "beta")
    lam = fit_w_asof(pd.read_parquet(DATA_DIR / "w_rows.parquet"), t0, BE_2L)
    out = []
    for i, r in enumerate(rows.itertuples()):
        pmf = PS[r.i]
        med = float(np.floor(np.searchsorted(np.cumsum(pmf), 0.5) + 0.5))
        P = calibrate_pmf(pmf, cal, r.position, r.market, med)
        line = getattr(r, line_col)
        po = float(P[np.arange(len(P)) > line].sum())
        L = lambda_for(lam, r.market)
        wo, wu = 0.5 + L * (po - 0.5), 0.5 + L * (0.5 - po)
        side = "more" if wo >= wu else "less"
        hit = (r.y > line) if side == "more" else (r.y < line)
        out.append((side, max(wo, wu), float(hit), r.position, r.market, r.league))
    return pd.DataFrame(out, columns=["side", "p", "hit", "position", "market", "league"])


def floors_section(rows: pd.DataFrame, PS: np.ndarray, x: pd.DataFrame) -> str:
    cdf = np.cumsum(PS, axis=1)
    y = rows["y"].to_numpy()
    med = np.argmax(cdf >= 0.5, axis=1)
    n = np.arange(len(rows))
    parts = []
    for off in range(-7, 5):
        f = med + off
        ok = f >= 1
        p = 1 - cdf[n, np.clip(f - 1, 0, None)]
        parts.append(pd.DataFrame({"p": p[ok], "hit": (y >= f)[ok].astype(float),
                                   "position": rows["position"].to_numpy()[ok]}))
    c = pd.concat(parts)
    edges = [0, .05, .15, .3, .5, .7, .8, .85, .9, .95, .98, 1]
    L = ["### P(kills >= f) at f = model median -7 .. +4 (S pmf)", "",
         _md(bucket(c, [], edges)), "", "P >= 0.80 by position:", "",
         _md(bucket(c[c["p"] >= 0.8], ["position"], [0.8, 1.0])), ""]
    # goblin proxies: the PP-like line minus g kills
    parts = []
    for g in (1, 2, 3, 4):
        line = x["L_mean"].to_numpy() - g
        ok = line > 0
        po = 1 - cdf[x["i"].to_numpy(), np.floor(np.clip(line, 0, None)).astype(int)]
        parts.append(pd.DataFrame({"g": g, "p": po[ok],
                                   "hit": (x["y"].to_numpy() > line)[ok].astype(float),
                                   "position": x["position"].to_numpy()[ok]}))
    c = pd.concat(parts)
    L += ["### Goblin proxies: P(more) at the trailing-mean line minus g kills (raw S)", "",
          _md(bucket(c, ["g"], [.5, .7, .8, .85, .9, .95, 1])), "",
          "Raw P(more) >= 0.80 by position:", "",
          _md(bucket(c[c["p"] >= 0.8], ["position"], [0.8, 1.0])), ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA_DIR)
    ap.add_argument("--rows", type=Path, help="reuse backtest rows (with pmf)")
    ap.add_argument("--since", default="2026-03-15")
    ap.add_argument("--sims", type=int, default=20_000)
    args = ap.parse_args(argv)
    players = pd.read_parquet(args.data / "players.parquet")
    teams = pd.read_parquet(args.data / "teams.parquet")
    if args.rows:
        rows = pd.read_parquet(args.rows)
    else:
        rows = run_backtest(players, teams, {LIVE_LOCKED: ALL_VARIANTS[LIVE_LOCKED]},
                            n_sims=args.sims, eval_start=pd.Timestamp(args.since),
                            keep_pmf=True)
        rows.to_parquet(REPORTS_DIR / "backtest_rows_audit.parquet", index=False)
    rows = rows[rows["variant"] == LIVE_LOCKED].reset_index(drop=True)
    rows["date"] = pd.to_datetime(rows["date"])
    PS = s_pmfs(rows)
    rows["i"] = np.arange(len(rows))
    rows["mean_s"] = (PS * np.arange(PS.shape[1])).sum(axis=1)
    d = market_totals(players)
    x = rows.merge(d[["series_id", "player", "market", "L_mean", "L_med"]],
                   on=["series_id", "player", "market"], how="inner")

    res = rows["y"] - rows["mean_s"]
    bias_m = res.groupby(rows["date"].dt.to_period("M")).agg(["mean", "size"]).round(3)
    bias_p = res.groupby([rows["market"], rows["position"]]).mean().unstack().round(3)
    L = [f"# Audit: are the unders real? (data through {teams['date'].max():%Y-%m-%d})", "",
         f"Walk-forward backtest of the live model (L+O, {args.sims} sims) on "
         f"{len(rows):,} player-series since {args.since}, S applied as of the start. "
         "PrizePicks lines are not logged yet (no picks-log branch), so PP-like lines are "
         "built from each player's own last 10 totals in the market before the series.", "",
         "## 1. Model-free: unders at lines that track recent averages", "",
         model_free(d), "",
         "## 2. Mean bias of the live model (actual - mean, kills per player-series)", "",
         f"Overall {res.mean():+.3f} (se {res.std() / np.sqrt(len(res)):.3f}).", "",
         _md(bias_m.reset_index().astype(str)), "", _md(bias_p.reset_index()), ""]
    for col, name in (("L_mean", "trailing-mean"), ("L_med", "trailing-median")):
        sh = shown_at_lines(x, PS, col)
        plays = sh[sh["p"] >= BE_2L + 0.03]
        t = plays.groupby("side").agg(n=("hit", "size"), pred=("p", "mean"),
                                      hit=("hit", "mean")).reset_index()
        t["2-pick ROI (same-side pairs, indep.)"] = 3 * t["hit"] ** 2 - 1
        L += [f"## 3{'ab'[col == 'L_med']}. Shown P (S + I + W) at {name} lines", "",
              f"Model side: {np.mean(sh['side'] == 'less'):.1%} LESS. PLAY = shown P >= "
              f"{BE_2L + 0.03:.3f} (2-pick breakeven + 3 pts).", "",
              _md(t.round(3)), "", "By shown-P bucket:", "",
              _md(bucket(sh, ["side"], [.5, .55, BE_2L, BE_2L + .03, .65, .7, .75, 1])), "",
              "PLAYs by position and by market:", "",
              _md(plays.groupby(["position", "side"]).agg(
                  n=("hit", "size"), pred=("p", "mean"), hit=("hit", "mean"))
                  .reset_index().round(3)), "",
              _md(plays.groupby(["market", "side"]).agg(
                  n=("hit", "size"), pred=("p", "mean"), hit=("hit", "mean"))
                  .reset_index().round(3)), ""]
    L += ["## 4. Floors and goblins", "", floors_section(rows, PS, x),
          f"SAFE FLOOR in the app = shown P(more) >= {SAFE_FLOOR_P:.0%}, not a support.", ""]
    w = teams[teams["league"] == "WLDs"]
    env = w.groupby(w["date"].dt.to_period("M")).agg(games=("gameid", "nunique"),
                                                     kills=("teamkills", "mean")).round(2)
    wr = rows[rows["league"] == "WLDs"]
    L += ["## 5. Worlds environment", "",
          "WLDs team kills per map by month (2026-09 = LPL regional finals filed as WLDs):",
          "", _md(env.reset_index().astype(str)), "",
          "Live model on WLDs rows: " + ", ".join(
              f"{k}: actual - mean {v:+.2f} (n={n})" for k, v, n in zip(
                  wr["date"].dt.to_period("M").astype(str).unique(),
                  (wr["y"] - wr["mean_s"]).groupby(wr["date"].dt.to_period("M")).mean(),
                  wr.groupby(wr["date"].dt.to_period("M")).size())), ""]
    path = REPORTS_DIR / "audit_unders_tables.md"
    path.write_text("\n".join(L))
    print("\n".join(L))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
