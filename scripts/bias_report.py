"""Mean-bias report: toggles M / P / G vs the locked config -> reports/backtest_bias.md.

Reads backtest row files written by `run_backtest.py --rows-only`:

    python scripts/run_backtest.py --rows-only --tag g  --variants shrink30+A+B50 L+G
    python scripts/run_backtest.py --rows-only --tag m1 --variants L+M1 L+M1+P   # ... m2, m3
    python scripts/run_backtest.py --rows-only --tag nop --drop-partial 2025 \\
        --variants shrink30+A+B50 <candidate>
    python scripts/bias_report.py --rows reports/backtest_rows_{g,m1,m2,m3}.parquet \\
        --nopartial reports/backtest_rows_nop.parquet --intro notes.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_backtest import COLS, _md  # noqa: E402
from src.backtest.metrics import calibration_table, ece  # noqa: E402
from src.backtest.walkforward import LOCKED, _multi_line, tail_report  # noqa: E402
from src.config import POSITIONS, REPORTS_DIR  # noqa: E402

KEY = ["series_id", "player"]


def load(paths) -> pd.DataFrame:
    parts = [pd.read_parquet(p) for p in paths]
    res = pd.concat(parts, ignore_index=True)
    return res.drop_duplicates(["variant"] + KEY).reset_index(drop=True)


def paired(res: pd.DataFrame, v: str, col: str, ref: str = LOCKED) -> tuple[float, float]:
    a = res[res.variant == ref].set_index(KEY)[col]
    b = res[res.variant == v].set_index(KEY)[col]
    d = (b - a).dropna()
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else np.nan


def se_mean(x: pd.Series) -> float:
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def low_bucket(p: pd.Series, y: pd.Series, cut: float = 0.25) -> tuple[int, float, float]:
    m = p < cut
    return int(m.sum()), float(p[m].mean()), float(y[m].mean())


def variant_table(model: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    tail = tail_report(model)
    rows = []
    for v in order:
        g = model[model.variant == v]
        err = g["mean"] - g["y"]
        mp, my = _multi_line(g)
        t = tail[(tail.variant == v) & (tail.position == "ALL")].iloc[0]
        dc, dcse = paired(model, v, "crps")
        dl, dlse = paired(model, v, "logscore")
        rows.append(dict(
            variant=v, n=len(g), crps=g["crps"].mean(), crps_vs_locked=dc,
            crps_se=dcse, logscore=g["logscore"].mean(), log_vs_locked=dl, log_se=dlse,
            bias=err.mean(), bias_se=se_mean(err),
            ece_median=ece(g["p_over"], g["over"]), ece_half=ece(mp, my),
            tail_pred=t.pred, tail_obs=t.obs, tail_obs_over_pred=t.obs_over_pred,
            kill_mult=g["kill_mult"].mean() if "kill_mult" in g else np.nan))
    return pd.DataFrame(rows)


def bias_by(model: pd.DataFrame, order: list[str], col: str, levels) -> pd.DataFrame:
    e = model.assign(err=model["mean"] - model["y"])
    t = e.groupby(["variant", col], observed=True)["err"].mean().unstack(col)
    t = t.reindex(index=order, columns=levels)
    ref = e[e.variant == LOCKED].groupby(col, observed=True)["err"].agg(se_mean)
    out = t.reset_index()
    se_row = {"variant": "(se, locked)"} | {c: ref.get(c, np.nan) for c in levels}
    return pd.concat([out, pd.DataFrame([se_row])], ignore_index=True)


def low_table(model: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    rows = []
    for v in order:
        g = model[model.variant == v]
        n1, p1, o1 = low_bucket(g["p_over"], g["over"])
        mp, my = _multi_line(g)
        n2, p2, o2 = low_bucket(mp, my)
        rows.append(dict(variant=v, median_n=n1, median_pred=p1, median_obs=o1,
                         median_gap=o1 - p1, half_n=n2, half_pred=p2, half_obs=o2,
                         half_gap=o2 - p2))
    return pd.DataFrame(rows)


def tail_table(model: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    t = tail_report(model)
    t = t.pivot(index="variant", columns="position", values="obs_over_pred")
    return t.reindex(index=order, columns=["ALL"] + POSITIONS).reset_index()


def league_table(model: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    ref = model[model.variant == LOCKED].set_index(KEY)
    out = ref.groupby("league").agg(n=("crps", "size"), crps_locked=("crps", "mean"))
    for v in order:
        if v == LOCKED:
            continue
        o = model[model.variant == v].set_index(KEY)
        d = (o["crps"] - ref["crps"]).dropna()
        lg = ref.loc[d.index, "league"]
        out[v] = d.groupby(lg).mean()
        out[f"{v} se"] = d.groupby(lg).agg(se_mean)
    return out.sort_values("n", ascending=False).reset_index()


def month_table(model: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    e = model.assign(err=model["mean"] - model["y"],
                     month=pd.to_datetime(model["date"]).dt.to_period("Q").astype(str))
    t = e.groupby(["month", "variant"])["err"].mean().unstack("variant")
    t = t.reindex(columns=order)
    t.insert(0, "n", e[e.variant == LOCKED].groupby("month").size())
    g = e[e.variant == "L+G"].groupby("month")["kill_mult"].mean() \
        if "L+G" in set(e.variant) else None
    if g is not None:
        t["G kill_mult"] = g
    return t.reset_index().rename(columns={"month": "quarter"})


def lpl_section(withp: pd.DataFrame, nop: pd.DataFrame, variants: list[str]) -> list[str]:
    """LPL with vs without the 2025 partial games (paired on the same series)."""
    lines = ["## LPL with and without 2025 partial data", "",
             "`without` = same run with the 805 LPL 2025 partial games removed from the "
             "data (they are neither history nor scored). LPL 2026 series are scored in "
             "both runs, so that comparison is paired; LPL 2025 series exist only with "
             "the partial data.", ""]
    rows = []
    for v in variants:
        a = withp[(withp.variant == v)].set_index(KEY)
        b = nop[(nop.variant == v)].set_index(KEY)
        if b.empty:
            continue
        lpl_a = a[a.league == "LPL"]
        yr = pd.to_datetime(lpl_a["date"]).dt.year
        common = lpl_a.index.intersection(b.index)
        d = (lpl_a.loc[common, "crps"] - b.loc[common, "crps"])
        allc = a.index.intersection(b.index)
        d_all = a.loc[allc, "crps"] - b.loc[allc, "crps"]
        other = allc[a.loc[allc, "league"] != "LPL"]
        d_oth = a.loc[other, "crps"] - b.loc[other, "crps"]
        rows += [
            dict(variant=v, slice="LPL 2025 (with only)", n=int((yr == 2025).sum()),
                 crps_with=lpl_a.loc[yr == 2025, "crps"].mean(), crps_without=np.nan,
                 diff=np.nan, se=np.nan,
                 bias_with=(lpl_a.loc[yr == 2025, "mean"] - lpl_a.loc[yr == 2025, "y"]).mean(),
                 bias_without=np.nan),
            dict(variant=v, slice="LPL 2026 (paired)", n=len(common),
                 crps_with=lpl_a.loc[common, "crps"].mean(),
                 crps_without=b.loc[common, "crps"].mean(), diff=d.mean(), se=se_mean(d),
                 bias_with=(lpl_a.loc[common, "mean"] - lpl_a.loc[common, "y"]).mean(),
                 bias_without=(b.loc[common, "mean"] - b.loc[common, "y"]).mean()),
            dict(variant=v, slice="non-LPL (paired)", n=len(other),
                 crps_with=a.loc[other, "crps"].mean(),
                 crps_without=b.loc[other, "crps"].mean(), diff=d_oth.mean(),
                 se=se_mean(d_oth), bias_with=np.nan, bias_without=np.nan),
            dict(variant=v, slice="all common (paired)", n=len(allc),
                 crps_with=a.loc[allc, "crps"].mean(),
                 crps_without=b.loc[allc, "crps"].mean(), diff=d_all.mean(),
                 se=se_mean(d_all), bias_with=np.nan, bias_without=np.nan),
        ]
    return lines + [_md(pd.DataFrame(rows)), ""]


def build(res: pd.DataFrame, nop: pd.DataFrame | None, intro: str, data_line: str,
          focus: str | None) -> str:
    model = res[res.variant != "naive_poisson"]
    order = [LOCKED] + sorted(v for v in model.variant.unique() if v != LOCKED)
    vt = variant_table(model, order)
    best = focus or vt[vt.variant != "L+G"].sort_values("crps")["variant"].iloc[0]
    lines = [intro.rstrip(), "", "## Generated tables", "", data_line + "  ",
             f"Player-series per variant: {int(vt['n'].iloc[0])}. Reference: **{LOCKED}** "
             f"(locked). Paired differences are variant minus locked (negative CRPS / log "
             f"score = better). Bias = mean predicted - actual kills per player-series.", "",
             "## Variants vs locked", "", _md(vt), "",
             "## Bias by position", "", _md(bias_by(model, order, "position", POSITIONS)), "",
             "## Bias by market (BO1 Map 1, BO3 Maps 1-2, BO5 Maps 1-3)", "",
             _md(bias_by(model, order, "market", ["Map 1", "Maps 1-2", "Maps 1-3"])), "",
             "## Bias by quarter (and G multiplier)", "", _md(month_table(model, order)), "",
             "## Under-25% buckets (P(over) < 0.25)", "",
             "`median_*`: at line = rounded median; `half_*`: half-point lines "
             "median +/- 0.5, 1.5. gap = observed - predicted.", "",
             _md(low_table(model, order)), "",
             "## Upper tail: hit rate / predicted at median + 1.5/2.5/3.5", "",
             _md(tail_table(model, order)), ""]
    for v in dict.fromkeys([LOCKED, best]):
        g = model[model.variant == v]
        lines += [f"## Calibration at line = rounded median ({v})", "",
                  _md(calibration_table(g["p_over"], g["over"])[COLS]), "",
                  f"## Calibration, half-point lines ({v})", "",
                  _md(calibration_table(*_multi_line(g))[COLS]), ""]
    lines += ["## CRPS by league (variant - locked, paired)", "",
              _md(league_table(model, order)), ""]
    if nop is not None:
        lines += lpl_section(model, nop[nop.variant != "naive_poisson"],
                             [v for v in dict.fromkeys([LOCKED, best])])
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, nargs="+", required=True)
    ap.add_argument("--nopartial", type=Path, nargs="*")
    ap.add_argument("--intro", type=Path)
    ap.add_argument("--focus", help="variant to show full calibration for (default: "
                                    "lowest CRPS excluding G)")
    ap.add_argument("--data-line", default="")
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "backtest_bias.md")
    args = ap.parse_args(argv)
    res = load(args.rows)
    nop = load(args.nopartial) if args.nopartial else None
    intro = args.intro.read_text() if args.intro else "# Backtest: mean-bias toggles"
    args.out.write_text(build(res, nop, intro, args.data_line, args.focus))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
