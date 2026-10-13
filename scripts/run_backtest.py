"""Walk-forward backtest runner.

    python scripts/run_backtest.py                       # data/*.parquet (real OE)
    python scripts/run_backtest.py --data data/synthetic # synthetic sandbox data
    python scripts/run_backtest.py --quick               # score the latest 6 months only
    python scripts/run_backtest.py --variants base shrink30+C
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.backtest.walkforward import VARIANTS, run_backtest, summarize  # noqa: E402
from src.config import DATA_DIR, REPORTS_DIR  # noqa: E402

COLS = ["bucket", "n", "pred", "obs", "diff", "se"]


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


def report(res: pd.DataFrame, title: str) -> tuple[str, dict]:
    s = summarize(res)
    table, tail = s["table"], s["tail"]
    model_rows = table[table["variant"] != "naive_poisson"]
    winner = model_rows.sort_values("crps")["variant"].iloc[0]
    n_scored = int(table.loc[table.variant == table.variant.iloc[0], "n"].iloc[0])
    lines = [
        f"# {title}", "",
        f"Player-series scored per variant: {n_scored}  ",
        "Markets: Map 1 (BO1), Maps 1-2 (BO3), Maps 1-3 (BO5).  ",
        f"Lowest CRPS: **{winner}**. A+best B/C combo: **{s['combo']}**.", "",
        "## Variants", "",
        "`tail_pred`/`tail_obs`: mean P(over) vs hit rate at lines rounded median + 1.5/2.5/3.5."
        " `crps_vs_base` is paired, with its standard error.", "",
        _md(table), "",
        "## Upper tail by position (lines >= median + 1.5)", "",
        _md(tail[tail.position == "ALL"].drop(columns="position")), "",
    ]
    for pos in ["top", "jng", "mid", "bot", "sup"]:
        lines += [f"### {pos}", "", _md(tail[tail.position == pos].drop(columns="position")),
                  ""]
    focus = list(dict.fromkeys(["base", winner]))
    for v in focus:
        lines += [f"## Calibration at line = rounded median ({v})", "",
                  _md(s["calib"][v][COLS]), "",
                  f"## Calibration, half-point lines median +/- 0.5, 1.5 ({v})", "",
                  _md(s["multi"][v][COLS]), ""]
    by_league = (res[res.variant.isin(["base", winner, "naive_poisson"])]
                 .groupby(["league", "variant"])["crps"].mean().unstack("variant").round(4))
    lines += ["## CRPS by league", "", _md(by_league.reset_index()), ""]
    by_mkt = (res[res.variant.isin(["base", winner])]
              .groupby(["market", "variant"])["crps"].agg(["size", "mean"]).round(4))
    lines += ["## CRPS by market", "", _md(by_mkt.reset_index()), ""]
    return "\n".join(lines), s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA_DIR)
    ap.add_argument("--sims", type=int, default=20_000)
    ap.add_argument("--warmup-days", type=int, default=75)
    ap.add_argument("--quick", action="store_true", help="score the latest 6 months only")
    ap.add_argument("--variants", nargs="*", help=f"subset of: {', '.join(VARIANTS)}")
    ap.add_argument("--out", type=Path, default=REPORTS_DIR)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    players = pd.read_parquet(args.data / "players.parquet")
    teams = pd.read_parquet(args.data / "teams.parquet")
    variants = VARIANTS
    if args.variants:
        unknown = set(args.variants) - set(VARIANTS)
        if unknown:
            ap.error(f"unknown variants: {sorted(unknown)}")
        variants = {k: VARIANTS[k] for k in args.variants}
    eval_start = None
    if args.quick:
        eval_start = pd.to_datetime(teams["date"]).max() - pd.DateOffset(months=6)
    res = run_backtest(players, teams, variants=variants, n_sims=args.sims,
                       warmup_days=args.warmup_days, eval_start=eval_start)
    if res.empty:
        print("error: no series scored", file=sys.stderr)
        return 1

    tag = f"_{args.tag}" if args.tag else ""
    if args.quick:
        tag += "_quick"
    title = f"Backtest ({args.data}, {args.sims} sims{', last 6 months' if args.quick else ''})"
    text, s = report(res, title)
    args.out.mkdir(parents=True, exist_ok=True)
    res.to_parquet(args.out / f"backtest_rows{tag}.parquet", index=False)
    s["table"].to_csv(args.out / f"backtest_summary{tag}.csv", index=False)
    s["tail"].to_csv(args.out / f"backtest_tail{tag}.csv", index=False)
    (args.out / f"backtest{tag}.md").write_text(text)
    print(text.split("## Upper tail")[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
