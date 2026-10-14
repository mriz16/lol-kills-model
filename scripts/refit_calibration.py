"""Refit the app's post-sim calibration (S, I, W) and write data/ + reports/calibration_fit.md.

    python scripts/refit_calibration.py                 # runs the L+O backtest (keep pmf)
    python scripts/refit_calibration.py --rows reports/backtest_rows_spread_L+O.parquet
    python scripts/refit_calibration.py --force         # write even if the drift check fails

Steps: walk-forward backtest of the live locked config with pmfs -> S (spread k) -> I
(beta calibration on the S pmfs) -> W (lambda per market so the top-5 W-adjusted edges per
slate at proxy lines are calibrated) -> data/calib_pairs.parquet, data/spread_rows.parquet,
data/w_rows.parquet, data/calib_params.json. The new parameters are compared with the
committed data/calib_params.json; if any moved more than DRIFT_LIMITS the script writes the
report, does NOT touch data/, and exits 1 (the weekly workflow then fails loudly).
Run by .github/workflows/calibrate.yml (weekly + manual).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_backtest import _md  # noqa: E402
from src.backtest.calib_pipeline import (DRIFT_LIMITS, build, compare_params,  # noqa: E402
                                         derive_min_edge, load_rows, params_summary,
                                         sel_summary, select_rows, write_artifacts)
from src.backtest.walkforward import ALL_VARIANTS, LIVE_LOCKED, run_backtest  # noqa: E402
from src.config import DATA_DIR, REPORTS_DIR  # noqa: E402


def before_after(wrows: pd.DataFrame) -> pd.DataFrame:
    """Top-1/3/5 per slate, before W (lambda = 1, calibrated P) and after W."""
    out = []
    for top in (1, 3, 5):
        for use in (False, True):
            c = select_rows(wrows, use, top)
            r = sel_summary(c, f"top {top}", "p_w")
            r["W"] = "after" if use else "before"
            out.append(r)
    cols = ["selection", "W", "n", "pred", "hit", "diff", "se", "n_more", "pred_more",
            "hit_more", "n_less", "pred_less", "hit_less"]
    return pd.DataFrame(out)[cols]


def report(art: dict, params: dict, old: dict | None, problems: list[str]) -> str:
    w = art["wrows"]
    lam = {k: v for k, v in params["w_lambda"].items() if k != "_n"}
    hist = pd.DataFrame([{"refit": t.date(), **{k: v for k, v in f.items()}}
                         for t, f in list(art["w_fits"].items())[-8:]])
    me, me_tab = derive_min_edge(w)
    k = pd.Series(params["spread_k"]).rename("k").reset_index().rename(
        columns={"index": "position|market"})
    L = [
        "# Calibration fit (S, I, W)", "",
        f"Fitted {params['fitted_at']} on {params['rows']:,} backtest player-series through "
        f"{params['last_series'][:10]} (walk-forward, weekly refits, fits use series >= 1 "
        "day before). Base = live locked config L+O; S and I-beta on top; W on the S+I "
        "probabilities at proxy lines (rolling median of the player's last 10 market "
        "totals, floored, + 0.5).", "",
        "## Drift check", "",
        ("No previous fit." if not old else
         ("**FAILED**: " + "; ".join(problems) if problems else "Passed.")) +
        f" Limits: S k ±{DRIFT_LIMITS['spread_k']}, I map ±{DRIFT_LIMITS['calib_pool']}, "
        f"W lambda ±{DRIFT_LIMITS['w_lambda']}.", "",
        "## W: lambda by market (current fit)", "",
        "Displayed P = p_line + lambda (p_cal - p_line), p_line = (1 - P(push)) / 2.", "",
        _md(pd.DataFrame([lam])), "",
        f"Selected picks in the fit: {params['w_lambda'].get('_n', 0)}.", "",
        "### Weekly fits (latest 8)", "", _md(hist), "",
        "## Top edges per slate, before vs after W (walk-forward)", "",
        "`pred` = mean shown P of the picked side (before: calibrated S+I; after: W), "
        "`hit` = hit rate, `diff` = hit - pred.", "",
        _md(before_after(w)), "",
        "## Minimum edge on W-adjusted edges", "",
        f"**{100 * me:.0f} pts** = worst of (-diff + 2 se) below, rounded up.", "",
        _md(me_tab), "",
        "## S: spread k (position|market)", "", _md(k, 3), "",
        "## I: pooled beta map", "",
        _md(pd.DataFrame([dict(zip([f"p={g}" for g in params["calib_grid"]],
                                   params["calib_pool"] or []))])), "",
    ]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, nargs="*",
                    help="existing backtest rows with pmfs (skip the backtest)")
    ap.add_argument("--data", type=Path, default=DATA_DIR)
    ap.add_argument("--out", type=Path, default=DATA_DIR,
                    help="where to write the calibration artifacts")
    ap.add_argument("--report", type=Path, default=REPORTS_DIR / "calibration_fit.md")
    ap.add_argument("--sims", type=int, default=20_000)
    ap.add_argument("--force", action="store_true", help="write despite a failed drift check")
    args = ap.parse_args(argv)

    players = pd.read_parquet(args.data / "players.parquet")
    if args.rows:
        rows = load_rows(args.rows)[LIVE_LOCKED]
    else:
        teams = pd.read_parquet(args.data / "teams.parquet")
        res = run_backtest(players, teams, {LIVE_LOCKED: ALL_VARIANTS[LIVE_LOCKED]},
                           n_sims=args.sims, keep_pmf=True)
        tmp = REPORTS_DIR / "backtest_rows_calib.parquet"
        res.to_parquet(tmp, index=False)
        rows = load_rows([tmp])[LIVE_LOCKED]
    art = build(rows, players)
    params = params_summary(art)
    prev = args.out / "calib_params.json"
    old = json.loads(prev.read_text()) if prev.exists() else None
    problems = compare_params(old, params)
    text = report(art, params, old, problems)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(text)
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as fh:
            fh.write(text)
    if problems and not args.force:
        print("error: drift check failed; data/ not updated (rerun with --force to accept)",
              file=sys.stderr)
        return 1
    for p in write_artifacts(art, args.out, params):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
