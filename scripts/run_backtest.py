"""Walk-forward backtest runner.

    python scripts/run_backtest.py                       # data/players.parquet (real OE)
    python scripts/run_backtest.py --data data/synthetic # synthetic sandbox data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.backtest.walkforward import run_backtest, summarize  # noqa: E402
from src.config import DATA_DIR, REPORTS_DIR  # noqa: E402


def _md(df: pd.DataFrame, floatfmt: int = 4) -> str:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda x: "" if pd.isna(x) else f"{x:.{floatfmt}f}")
        else:
            df[c] = df[c].astype(str)
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = "\n".join("| " + " | ".join(r) + " |" for r in df.values.tolist())
    return "\n".join([head, sep, body])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA_DIR)
    ap.add_argument("--sims", type=int, default=20_000)
    ap.add_argument("--warmup-days", type=int, default=75)
    ap.add_argument("--out", type=Path, default=REPORTS_DIR)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    players = pd.read_parquet(args.data / "players.parquet")
    teams = pd.read_parquet(args.data / "teams.parquet")
    res = run_backtest(players, teams, n_sims=args.sims, warmup_days=args.warmup_days)
    table, calib, multi = summarize(res)

    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    res.to_parquet(args.out / f"backtest_rows{tag}.parquet", index=False)
    table.to_csv(args.out / f"backtest_summary{tag}.csv", index=False)

    lines = [f"# Backtest ({args.data}, {args.sims} sims)", "",
             f"Player-series scored: {int(table.loc[table.variant == 'base', 'n'].iloc[0])}",
             "", "## Variants", "", _md(table), ""]
    for v, t in calib.items():
        lines += [f"## Calibration at line = rounded median ({v})", "", _md(t[
            ["bucket", "n", "pred", "obs", "diff", "se"]]), ""]
    for v, t in multi.items():
        lines += [f"## Calibration, half-point lines median +/- 0.5, 1.5 ({v})", "",
                  _md(t[["bucket", "n", "pred", "obs", "diff", "se"]]), ""]
    by_league = (res[res.variant.isin(["base", "naive_poisson"])]
                 .groupby(["league", "variant"])[["crps", "logscore"]].mean()
                 .unstack("variant").round(4))
    by_league.columns = [f"{a}_{b}" for a, b in by_league.columns]
    lines += ["## By league (base vs naive)", "", _md(by_league.reset_index()), ""]
    by_pos = (res[res.variant == "base"].groupby("position")
              .agg(n=("y", "size"), mean_pred=("mean", "mean"), mean_y=("y", "mean"),
                   crps=("crps", "mean")).reset_index())
    lines += ["## By position (base)", "", _md(by_pos), ""]
    (args.out / f"backtest{tag}.md").write_text("\n".join(lines))
    print("\n".join(lines[:8]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
