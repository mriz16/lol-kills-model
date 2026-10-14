"""Grade the logged picks against the current OE parquet.

    python scripts/grade_picks.py --log-root ../picks-log    # checkout of the picks-log branch

Reads <log-root>/logs/picks.csv (written by the app), grades every pick that is not final
yet, and writes <log-root>/logs/graded.csv. Final grades (win/loss/push/void) are kept.
Prints the record (and appends it to $GITHUB_STEP_SUMMARY when set).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.app import grading, picklog  # noqa: E402
from src.config import DATA_DIR  # noqa: E402


def summary_md(picks: pd.DataFrame, graded: pd.DataFrame) -> str:
    s = grading.record_summary(picks, graded)
    roi = f"{s['roi']:+.1%}" if pd.notna(s["roi"]) else "n/a"
    lines = [
        "## Logged picks", "",
        f"- Legs W-L-P: {s['legs_won']}-{s['legs_lost']}-{s['legs_push']} "
        f"(void {s['legs_void']}, pending {s['legs_pending']})",
        f"- Slips W-L: {s['slips_won']}-{s['slips_lost']} (refund {s['slips_refund']}, "
        f"pending {s['slips_pending']})",
        f"- Staked ${s['staked']:.2f}, profit ${s['profit']:+.2f}, ROI {roi}",
    ]
    if pd.notna(s["leg_hit"]):
        lines.append(f"- Leg hit rate {s['leg_hit']:.1%} vs model {s['exp_leg_hit']:.1%}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-root", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=DATA_DIR)
    args = ap.parse_args(argv)

    store = picklog.LocalStore(args.log_root)
    picks = picklog.read_picks(store)
    if picks.empty:
        print("no logged picks")
        return 0
    players = pd.read_parquet(args.data / "players.parquet")
    teams = pd.read_parquet(args.data / "teams.parquet")
    old = picklog.read_graded(store)
    new = grading.grade_picks(picks, players, teams)
    graded = grading.merge_graded(old, new)
    out = args.log_root / picklog.GRADED_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    graded.to_csv(out, index=False)
    text = summary_md(picks, graded)
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
