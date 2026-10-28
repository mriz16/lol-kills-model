"""Projection study (2026-10-28) -> reports/backtest_form.md.

1. Engine toggles on the live locked config L+O, paired per player-series: R (roster
   weighting), Q (kill share by outcome), F (player form), T (team form). CRPS / log
   score differences with series-clustered se (lower = better).
2. H (recent-form blend, post-sim): walk-forward weekly beta per market on the L+O rows
   (src.backtest.calib_pipeline.apply_form), H vs raw and H+S vs raw, and the slope of
   actual - mean on the last-20 gap before / after H.

    python -m scripts.run_backtest --rows-only --keep-pmf --since 2026-03-01 --tag proj \
        --variants L+O L+O+R1 L+O+R2 L+O+Q10 L+O+Q30 L+O+F20 L+O+F50 L+O+T15 L+O+T30
    python -m scripts.form_report --rows reports/backtest_rows_proj.parquet --data DIR
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_backtest import _md
from src.backtest.calib_pipeline import (apply_form, apply_spread, form_rows, load_rows,
                                         pmf_moments, stack)
from src.config import DATA_DIR, REPORTS_DIR

BASE = "L+O"


def paired(rows: pd.DataFrame, base: str = BASE) -> pd.DataFrame:
    key = ["series_id", "player", "market"]
    b = rows[rows["variant"] == base].drop_duplicates(key).set_index(key)
    out = []
    for v, g in rows.groupby("variant"):
        if v == "naive_poisson":
            continue
        g = g.drop_duplicates(key).set_index(key).reindex(b.index)
        r = {"variant": v, "n": len(g), "crps": g["crps"].mean()}
        for m in ("crps", "logscore"):
            d = g[m] - b[m]
            ds = d.groupby(level=0).sum()
            r[f"d_{m}"] = d.mean()
            r[f"se_{m}"] = np.sqrt(len(ds)) * ds.std() / len(d)
        r["bias"] = (g["y"] - g["mean"]).mean()
        out.append(r)
    return pd.DataFrame(out).sort_values("d_crps")


def crps_rows(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    F = np.cumsum(P, axis=1)
    H = np.arange(P.shape[1])[None, :] >= y[:, None]
    return ((F - H) ** 2).sum(axis=1)


def log_rows(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(np.maximum(P[np.arange(len(y)), np.minimum(y, P.shape[1] - 1)], 1e-6))


def diff_se(a: np.ndarray, b: np.ndarray, sid: np.ndarray) -> tuple[float, float]:
    d = pd.Series(a - b)
    ds = d.groupby(sid).sum()
    return float(d.mean()), float(np.sqrt(len(ds)) * ds.std() / len(d))


def slope(gap: np.ndarray, err: np.ndarray) -> float:
    ok = np.isfinite(gap) & np.isfinite(err)
    return float(np.polyfit(gap[ok], err[ok], 1)[0])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, nargs="+", required=True)
    ap.add_argument("--data", type=Path, default=DATA_DIR)
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "backtest_form.md")
    args = ap.parse_args(argv)
    rows = pd.concat([pd.read_parquet(p, columns=["variant", "series_id", "player", "market",
                                                   "y", "mean", "crps", "logscore"])
                      for p in args.rows], ignore_index=True)
    players = pd.read_parquet(args.data / "players.parquet")
    tab = paired(rows)
    L = ["# Projection study: roster, outcome share, form (2026-10-28)", "",
         f"Walk-forward, series from {pd.read_parquet(args.rows[0], columns=['date'])['date'].min():%Y-%m-%d}"
         f", {tab['n'].iloc[0]:,} player-series (primary market), 10k sims. "
         "Paired differences vs the live locked config L+O; lower is better; se clustered "
         "by series.", "", "## 1. Engine toggles", "",
         "- R1 / R2: team-pool games weighted by (share of today's starters who played) ^ 1 / 2",
         "- Q10 / Q30: kill share in wins / losses, shrunk to overall by n / (n + 10 / 30)",
         "- F20 / F50: kill-share weights x 0.5 ^ (games back / 20 / 50)",
         "- T15 / T30: team-pool weights x 0.5 ^ (team games back / 15 / 30)", "",
         _md(tab), "",
         "None improves on L+O: F (player form in the kill share) is clearly worse, R / Q / T "
         "are within noise. Not locked.", ""]
    base = load_rows(args.rows[:1])[BASE]
    b = form_rows(base, players)
    P0 = stack(b)
    PH, cs = apply_form(b, P0)
    mh, vh = pmf_moments(PH)
    PS, _ = apply_spread(b.assign(mean=mh, var=vh), PH)
    P0S, _ = apply_spread(b, P0)
    y = b["y"].to_numpy().astype(int)
    sid = b["series_id"].to_numpy()
    c0, cH, cS, c0S = (crps_rows(P, y) for P in (P0, PH, PS, P0S))
    l0, lH, lS, l0S = (log_rows(P, y) for P in (P0, PH, PS, P0S))
    res = []
    for name, a, bb, la, lb in (("H vs raw", cH, c0, lH, l0),
                                ("H+S vs S (the app before -> after)", cS, c0S, lS, l0S)):
        d, se = diff_se(a, bb, sid)
        dl, sel = diff_se(la, lb, sid)
        res.append(dict(comparison=name, d_crps=d, se=se, d_log=dl, se_log=sel))
    gap = b["gap"].to_numpy(float)
    by_m = []
    for mk, g in b.assign(cH=cH - c0, mh=mh).groupby("market"):
        idx = g.index.to_numpy()
        by_m.append(dict(market=mk, n=len(g), applied=float((cs[idx] != 1).mean()),
                         d_crps=float(g["cH"].mean()),
                         slope_before=slope(gap[idx], (g["y"] - g["mean"]).to_numpy(float)),
                         slope_after=slope(gap[idx], (g["y"] - g["mh"]).to_numpy(float)),
                         bias_before=float((g["y"] - g["mean"]).mean()),
                         bias_after=float((g["y"] - g["mh"]).mean())))
    months = pd.DataFrame({"month": b["date"].dt.to_period("M").astype(str), "d": cH - c0}) \
        .groupby("month")["d"].agg(["size", "mean"]).reset_index()
    L += ["## 2. H: recent-form blend (post-sim, before S)", "",
          "gap = the player's mean over the previous 20 series of the same market (>= 10) "
          "minus the model mean. Actual minus the model mean rises with the gap, so the "
          "model under-reacts to recent output. H scales the player's simulated total so "
          "its mean moves by beta x gap; beta per market = the slope fitted on rows of "
          "series >= 1 day before, refit weekly (walk-forward).", "",
          _md(pd.DataFrame(res)), "", "By market (H vs raw):", "", _md(pd.DataFrame(by_m)),
          "", "By month (H vs raw CRPS):", "", _md(months), "",
          "slope_after ~ 0 means the recent-form signal is used up. H is applied in the app "
          "before S; S / I / W are refit on the H pmfs (scripts/refit_calibration.py).", ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
