"""Refit W's lambda on real PrizePicks lines from the graded slate log.

    python scripts/refit_w_real.py --log-root ../picks-log           # dry run: print only
    python scripts/refit_w_real.py --log-root ../picks-log --write   # data/w_lambda_real.json

Needs at least 300 graded PrizePicks rows (win / loss / push) in logs/slates.csv +
logs/slates_graded.csv (pending rows are graded live against data/*.parquet first). Fits
lambda per market exactly like the proxy-line fit (top-5 W-adjusted edges per slate
calibrated), with slate = one logged slate and the calibrated (pre-W) probabilities the
app logged. With --write the app uses this lambda instead of the proxy-line one for
cutoffs after the last fitted slate date.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.app import grading, picklog, slatelog  # noqa: E402
from src.backtest.calib_pipeline import BE_2L  # noqa: E402
from src.config import DATA_DIR  # noqa: E402
from src.engine.calibration import W_TOP_N, fit_w, select_top  # noqa: E402

REAL_MIN_PICKS = 60         # selected picks (top-5 per slate) before lambda departs from 1


def fit_real(rows: pd.DataFrame, min_rows: int = slatelog.REAL_W_MIN_ROWS) -> dict:
    if len(rows) < min_rows:
        raise ValueError(f"{len(rows)} graded PrizePicks rows; need {min_rows}")
    lam = fit_w(rows, BE_2L, top_n=W_TOP_N, min_picks=REAL_MIN_PICKS)
    before = select_top(rows, None, BE_2L)
    after = select_top(rows, lam, BE_2L)
    return {
        "lambda": {k: round(float(v), 4) for k, v in lam.items() if k != "_n"},
        "n_rows": int(len(rows)), "n_picks": int(lam.get("_n", 0)),
        "fitted_through": rows["date"].max().date().isoformat(),
        "fitted_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "top5_before": {"pred": round(float(before["p_w"].mean()), 4),
                        "hit": round(float(before["hit"].mean()), 4), "n": int(len(before))},
        "top5_after": {"pred": round(float(after["p_w"].mean()), 4),
                       "hit": round(float(after["hit"].mean()), 4), "n": int(len(after))},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-root", type=Path, required=True,
                    help="checkout of the picks-log branch (has logs/slates.csv)")
    ap.add_argument("--data", type=Path, default=DATA_DIR)
    ap.add_argument("--min-rows", type=int, default=slatelog.REAL_W_MIN_ROWS)
    ap.add_argument("--write", action="store_true", help="write data/w_lambda_real.json")
    args = ap.parse_args(argv)

    store = picklog.LocalStore(args.log_root)
    slates = slatelog.read_slates(store)
    players = pd.read_parquet(args.data / "players.parquet")
    teams = pd.read_parquet(args.data / "teams.parquet")
    graded = grading.merge_graded(slatelog.read_slates_graded(store),
                                  grading.grade_slates(slates, players, teams), id_col="row_id")
    rows = slatelog.graded_real_rows(slates, graded)
    try:
        out = fit_real(rows, args.min_rows)
    except ValueError as exc:
        print(f"not refitting: {exc}")
        return 0
    print(json.dumps(out, indent=1))
    proxy = DATA_DIR / "calib_params.json"
    if proxy.exists():
        print("proxy-line lambda:", json.loads(proxy.read_text()).get("w_lambda"))
    if args.write:
        path = args.data / "w_lambda_real.json"
        path.write_text(json.dumps(out, indent=1) + "\n")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
