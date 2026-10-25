"""Goblin lines and the under / over balance, walk-forward -> reports/goblin_fit.md.

    python scripts/goblin_report.py --rows reports/backtest_rows_calib.parquet
    python scripts/goblin_report.py --since 2025-06-01        # runs the backtest (~10 min)

Rows: run_backtest(keep_pmf=True) of the live locked variant. S is applied walk-forward
(weekly refit, as the app does). PP-like line = the player's trailing-10 market mean to the
nearest x.5 (calib_pipeline.proxy_lines `proxy_mean`). Goblin lines = that line minus 1..4
kills. G (the app's goblin map, GoblinCal) is refit weekly on pairs >= 1 day before.
Section 3 checks the standard-line sides with the app's shown P (S + I + W).
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
from src.app.slate import MIN_EDGE, goblin_breakeven  # noqa: E402
from src.backtest.calib_pipeline import (BE_2L, LAG, REFIT, apply_w, build,  # noqa: E402
                                         line_rows, load_rows, refit_dates)
from src.backtest.walkforward import ALL_VARIANTS, LIVE_LOCKED, run_backtest  # noqa: E402
from src.config import DATA_DIR, REPORTS_DIR  # noqa: E402
from src.engine.calibration import GoblinCal, goblin_pairs  # noqa: E402

INTL = ("WLDs", "MSI", "FST", "EWC")


def walk_forward_g(G: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=G.index)
    for t in refit_dates(G["date"]):
        sel = (G["date"] >= t) & (G["date"] < t + REFIT)
        if not sel.any():
            continue
        cal = GoblinCal.fit(G[G["date"] <= t - LAG])
        if not cal.maps:
            continue
        for pos, g in G[sel].groupby("position"):
            out[g.index] = cal(g["p"].to_numpy(float), pos)
    return out


def buckets(d: pd.DataFrame, col: str, by: list[str]) -> pd.DataFrame:
    d = d.assign(bucket=pd.cut(d[col], [0, .5, .6, .7, .75, .8, .85, .9, .95, 1]))
    t = d.groupby(by + ["bucket"], observed=True).agg(
        n=("y", "size"), pred=(col, "mean"), hit=("y", "mean")).reset_index()
    t["gap"] = t["hit"] - t["pred"]
    t["se"] = np.sqrt(t["hit"] * (1 - t["hit"]) / t["n"])
    t["bucket"] = t["bucket"].astype(str)
    return t[t["n"] >= 80].round(3)


def picks(E: pd.DataFrame, col: str, thr: float) -> pd.DataFrame:
    """Per player-series: the highest goblin line (smallest depth) with P >= thr."""
    return E[E[col] >= thr].sort_values("depth").groupby("row", sort=False).head(1)


def summary(s: pd.DataFrame, col: str, by: list[str] | None = None) -> pd.DataFrame:
    g = s.groupby(by) if by else s.assign(all="all").groupby("all")
    t = g.agg(n=("y", "size"), pred=(col, "mean"), hit=("y", "mean")).reset_index()
    t["gap"] = t["hit"] - t["pred"]
    t["se"] = np.sqrt(t["hit"] * (1 - t["hit"]) / t["n"])
    return t.round(3)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA_DIR)
    ap.add_argument("--rows", type=Path, nargs="*")
    ap.add_argument("--since", default="2025-06-01")
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "goblin_fit.md")
    args = ap.parse_args(argv)
    players = pd.read_parquet(args.data / "players.parquet")
    if args.rows:
        rows = load_rows(args.rows)[LIVE_LOCKED]
    else:
        teams = pd.read_parquet(args.data / "teams.parquet")
        tmp = REPORTS_DIR / "backtest_rows_goblin.parquet"
        run_backtest(players, teams, {LIVE_LOCKED: ALL_VARIANTS[LIVE_LOCKED]},
                     eval_start=pd.Timestamp(args.since), keep_pmf=True,
                     verbose=False).to_parquet(tmp, index=False)
        rows = load_rows([tmp])[LIVE_LOCKED]
    art = build(rows, players)
    b, PS, PSI = art["base"], art["PS"], art["PSI"]
    G = goblin_pairs(b.assign(row=np.arange(len(b))), PS, extra=("row", "league"))
    G["pg"] = walk_forward_g(G)
    E = G[G["pg"].notna()].copy()
    be = goblin_breakeven()
    thr = be + MIN_EDGE
    L = [f"# Goblin lines and side balance ({len(b):,} player-series since "
         f"{b['date'].min():%Y-%m-%d}, data through {b['date'].max():%Y-%m-%d})", "",
         "Walk-forward, live locked model with S. PP-like line = trailing-10 market mean "
         "to the nearest x.5. Goblin lines = that line minus 1..4 kills. Raw = the S "
         "model's P(more); G = the app's goblin P (raw, supports through their beta map, "
         "refit weekly).", "",
         f"Goblin breakeven = (1 / 8.25) ^ (1/6) = {be:.1%} (6-pick goblin Power at "
         f"8.25x); a goblin PLAY needs P >= {thr:.1%} (breakeven + {100 * MIN_EDGE:.0f} "
         "pts).", "",
         "## 1. Goblin picks (highest line with P >= the play bar)", ""]
    for col, name in (("p", "raw S model"), ("pg", "G (shown in the app)")):
        s = picks(E, col, thr)
        L += [f"### {name}", "", _md(summary(s, col)), "",
              _md(summary(s, col, ["position"])), "", _md(summary(s, col, ["depth"])), "",
              _md(summary(s, col, ["market"])), "",
              "International events: " + _md(summary(s[s["league"].isin(INTL)], col)), ""]
    L += ["## 2. Calibration of P(more) at goblin lines", "",
          "Raw, by depth:", "", _md(buckets(E, "p", ["depth"])), "",
          "G, by position (P >= 0.7):", "", _md(buckets(E[E["pg"] >= .7], "pg", ["position"])),
          ""]
    L += ["## 3. Standard lines: are the unders inflated?", "",
          "The app's shown P (S + I + W) at PP-like lines, PLAY = shown P >= 2-pick "
          f"breakeven + 3 pts ({BE_2L + MIN_EDGE:.1%}).", ""]
    w, _ = apply_w(line_rows(b, PSI, "proxy_mean"))
    p0 = 0.5 * (1 - w["p_push"])
    pm, pl = p0 + w["lam"] * (w["p_more"] - p0), p0 + w["lam"] * (w["p_less"] - p0)
    d = pd.DataFrame({"side": np.where(pm >= pl, "more", "less"), "p": np.maximum(pm, pl),
                      "market": w["market"], "date": w["date"]})
    d["y"] = np.where(d["side"] == "more", w["y"] > w["line"], w["y"] < w["line"]).astype(float)
    pl_ = d[d["p"] >= BE_2L + MIN_EDGE]
    rec = pl_[pl_["date"] >= "2026-03-01"]
    L += [f"Model side LESS on {np.mean(d['side'] == 'less'):.1%} of lines.", "",
          "All rows:", "", _md(summary(pl_, "p", ["side"])), "",
          "Since 2026-03:", "", _md(summary(rec, "p", ["side"])), "",
          "Since 2026-03, Map 1 (BO1):", "",
          _md(summary(rec[rec["market"] == "Map 1"], "p", ["side"])), ""]
    mean = (PS * np.arange(PS.shape[1])).sum(axis=1)
    r = pd.DataFrame({"r": b["y"] - mean, "position": b["position"], "date": b["date"]})
    r = r[r["date"] >= "2026-03-01"]
    t = r.groupby("position")["r"].agg(["mean", "size", "std"]).reset_index()
    t["se"] = t["std"] / np.sqrt(t["size"])
    L += ["Mean error (actual - model mean) since 2026-03, kills per player-series: "
          f"{r['r'].mean():+.3f}.", "", _md(t.drop(columns="std").round(3)), ""]
    args.out.write_text("\n".join(L))
    print("\n".join(L))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
