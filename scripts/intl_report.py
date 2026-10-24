"""International-event backtest: locked config vs base on Worlds / MSI / FST only, with the
domestic leagues over the same window as the yardstick.

    python scripts/intl_report.py                      # run + report (~1 h, 20k sims)
    python scripts/intl_report.py --rows reports/backtest_rows_intl.parquet   # report only

Writes reports/backtest_intl.md. Series OE files under WLDs between two teams of the same
home league (the LPL regional finals in late September) are reported separately: they
are not international games.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_backtest import _md  # noqa: E402
from src.backtest.metrics import ece  # noqa: E402
from src.backtest.walkforward import ALL_VARIANTS, LIVE_LOCKED, run_backtest  # noqa: E402
from src.config import DATA_DIR, INTERNATIONAL_LEAGUES, REPORTS_DIR  # noqa: E402
from src.engine.features import History  # noqa: E402

EVENTS = {("WLDs", 2025): "Worlds 2025", ("MSI", 2026): "MSI 2026",
          ("FST", 2026): "FST 2026", ("WLDs", 2026): "Worlds 2026 (so far)",
          ("MSI", 2025): "MSI 2025", ("FST", 2025): "FST 2025", ("EWC", 2025): "EWC 2025",
          ("EWC", 2026): "EWC 2026"}
HEADLINE = ["Worlds 2025", "MSI 2026", "FST 2026"]
SINCE = "2025-09-01"


def tag_rows(res: pd.DataFrame, players, teams) -> pd.DataFrame:
    hist = History(players, teams)
    t = teams.astype({"series_id": str, "teamname": str})
    pair = t.groupby("series_id")["teamname"].agg(lambda s: tuple(sorted(s.unique())))
    res = res.copy()
    res["series_id"] = res["series_id"].astype(str)
    res["year"] = pd.to_datetime(res["date"]).dt.year
    # a series under an international code between two teams of one home league, played
    # before the event's first cross-region series = a regional final OE filed there
    ser = t.drop_duplicates("series_id").astype({"league": str})
    ser = ser[ser["league"].isin(INTERNATIONAL_LEAGUES)].copy()
    ser["year"] = ser["date"].dt.year
    ser["same"] = [len(h) == 1 and None not in h for h in (
        {hist.home_league(tm, int(d.value)) for tm in pair[sid]}
        for sid, d in zip(ser["series_id"], ser["date"]))]
    start = ser[~ser["same"]].groupby(["league", "year"])["date"].min()
    ev_start = pd.Series([start.get((lg, y), pd.Timestamp.max) for lg, y in
                          zip(ser["league"], ser["year"])], index=ser.index)
    same_home = dict(zip(ser["series_id"], ser["same"] & (ser["date"] < ev_start)))
    intl = res["league"].isin(INTERNATIONAL_LEAGUES)
    qual = res["series_id"].map(same_home).fillna(False).astype(bool)
    ev = [EVENTS.get((lg, y), f"{lg} {y}") for lg, y in zip(res["league"], res["year"])]
    res["group"] = np.where(~intl, "domestic", np.where(qual, "regional qualifiers filed as intl",
                                                         ev))
    return res


def null_ece(p: np.ndarray, draws: int = 200, seed: int = 0) -> float:
    """Mean ECE of a perfectly calibrated forecaster with these P(over) values and this
    sample size (outcomes drawn from p): the noise floor small samples sit on."""
    rng = np.random.default_rng(seed)
    ps = pd.Series(p)
    return float(np.mean([ece(ps, pd.Series((rng.random(len(p)) < p).astype(float)))
                          for _ in range(draws)]))


def metrics(g: pd.DataFrame) -> dict:
    hp = np.concatenate([g[f"hl{j}_p"].to_numpy(float) for j in range(4)])
    hy = np.concatenate([g[f"hl{j}_y"].to_numpy(float) for j in range(4)])
    ok = ~np.isnan(hp)
    return dict(series=g["series_id"].nunique(), n=len(g), crps=g["crps"].mean(),
                crps_rel=g["crps"].mean() / g["y"].mean(), logscore=g["logscore"].mean(),
                bias=(g["mean"] - g["y"]).mean(), mean_y=g["y"].mean(),
                ece_median=ece(g["p_over"], g["over"]),
                ece_median_null=null_ece(g["p_over"].to_numpy(float)),
                ece_half=ece(pd.Series(hp[ok]), pd.Series(hy[ok])))


def paired(res: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Per group: CRPS(a) - CRPS(b) on the same (series, player) rows, with its se."""
    key = ["series_id", "player"]
    x = res[res["variant"] == a].set_index(key)
    y = res[res["variant"] == b].set_index(key)
    j = x[["crps", "group", "market"]].join(y[["crps"]], rsuffix="_b", how="inner")
    j["d"] = j["crps"] - j["crps_b"]
    out = j.groupby("group")["d"].agg(n="size", diff="mean", se=lambda d: d.std(ddof=1)
                                      / np.sqrt(len(d)))
    return out.reset_index()


def report(res: pd.DataFrame, title: str, variants: list[str]) -> str:
    res = res[res["variant"].isin(variants)]
    rows = []
    for (grp, v), g in res.groupby(["group", "variant"]):
        rows.append(dict(group=grp, variant=v, **metrics(g)))
    tab = pd.DataFrame(rows)
    order = HEADLINE + sorted(set(tab["group"]) - set(HEADLINE) - {"domestic"}) + ["domestic"]
    tab["o"] = tab["group"].map({g: i for i, g in enumerate(order)})
    tab = tab.sort_values(["o", "variant"]).drop(columns="o").round(4)

    intl3 = res[res["group"].isin(HEADLINE)]
    pooled = []
    for v in variants:
        for name, g in (("international (3 events)", intl3[intl3.variant == v]),
                        ("domestic, same window", res[(res.group == "domestic")
                                                      & (res.variant == v)])):
            pooled.append(dict(group=name, variant=v, **metrics(g)))
    # same market (Maps 1-3 is what MSI / FST / Worlds knockouts bet): like with like
    for v in variants:
        for mkt in sorted(intl3["market"].unique()):
            for name, g in (("international", intl3), ("domestic", res[res.group == "domestic"])):
                gg = g[(g.variant == v) & (g.market == mkt)]
                if len(gg):
                    pooled.append(dict(group=f"{name} / {mkt}", variant=v, **metrics(gg)))
    pooled = pd.DataFrame(pooled).round(4)

    lines = [f"# {title}", "",
             "Walk-forward (each series projected from prior data only, Elo p_map), primary "
             "market = what is bet (BO1 Map 1, BO3 Maps 1-2, BO5 Maps 1-3). "
             f"`{variants[0]}` vs `{variants[1]}`. crps_rel = CRPS / mean actual kills "
             "(CRPS grows with the kill total, so compare markets on crps_rel). ECE at the "
             "line = rounded median (`ece_median`) and at half-point lines median "
             "+/- 0.5, 1.5 (`ece_half`), 5% buckets. `ece_median_null` = the ECE a perfectly "
             "calibrated model would show at the same sample size (outcomes drawn from its "
             "own P(over), 200 draws): ECE below it is noise, not miscalibration.", "",
             "## Per event", "", _md(tab), "",
             "## International (Worlds 2025 + MSI 2026 + FST 2026) vs domestic", "",
             _md(pooled), "",
             f"## Paired CRPS difference by group ({variants[1]} - {variants[0]}, negative = "
             f"{variants[1]} better)", "", _md(paired(res, variants[1], variants[0]).round(4)),
             ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA_DIR)
    ap.add_argument("--rows", type=Path, nargs="*", help="reuse backtest rows (no run)")
    ap.add_argument("--variants", nargs="*", default=["base", LIVE_LOCKED])
    ap.add_argument("--sims", type=int, default=20_000)
    ap.add_argument("--since", default=SINCE)
    ap.add_argument("--intl-only", action="store_true",
                    help="score international series only (for testing a fix)")
    ap.add_argument("--tag", default="intl")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args(argv)
    players = pd.read_parquet(args.data / "players.parquet")
    teams = pd.read_parquet(args.data / "teams.parquet")
    if args.rows:
        res = pd.concat([pd.read_parquet(p) for p in args.rows], ignore_index=True)
    else:
        flt = (lambda s: s.league in INTERNATIONAL_LEAGUES) if args.intl_only else None
        res = run_backtest(players, teams, {v: ALL_VARIANTS[v] for v in args.variants},
                           n_sims=args.sims, eval_start=pd.Timestamp(args.since),
                           score_filter=flt)
        res = res[res["variant"] != "naive_poisson"]
        out = REPORTS_DIR / f"backtest_rows_{args.tag}.parquet"
        res.to_parquet(out, index=False)
        print(f"wrote {out}")
    res = tag_rows(res, players, teams)
    text = report(res, f"International-event backtest (data through "
                       f"{pd.to_datetime(teams['date']).max():%Y-%m-%d}, {args.sims} sims)",
                  args.variants)
    path = args.report or REPORTS_DIR / f"backtest_{args.tag}.md"
    path.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
