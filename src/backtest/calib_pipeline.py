"""Walk-forward post-sim calibration pipeline: S (spread), I (P(over) map), W (line
shrinkage), on backtest rows that carry predictive pmfs (run_backtest(keep_pmf=True)).

Shared by scripts/spread_report.py (the study), scripts/refit_calibration.py (weekly
refit of the app's calibration data) and the tests. Every fit uses only rows of series
that started at least LAG before the refit date; refits are weekly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PRIZEPICKS_POWER
from src.engine.calibration import (Calibrator, GoblinCal, calibrate_pmf, fit_spread, fit_w,
                                    goblin_pairs, inflate_pmf, pairs_from_rows, select_top)

BE_2L = (1 / PRIZEPICKS_POWER[2]) ** 0.5
N_MAPS = {"Map 1": 1, "Maps 1-2": 2, "Maps 1-3": 3}
PROXY_WINDOW, PROXY_MIN = 10, 5
REFIT = pd.Timedelta(days=7)
LAG = pd.Timedelta(days=1)
CAL_METHOD = "beta"
SPREAD_COLS = ["date", "position", "market", "y", "mean", "var"]
W_COLS = ["date", "slate", "market", "position", "line", "p_more", "p_less", "p_push", "y"]


# ------------------------------------------------------------------ rows ----
def load_rows(paths: list[Path]) -> dict[str, pd.DataFrame]:
    """Backtest rows per variant, aligned on (series_id, player), sorted by date."""
    res = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    res = res[res["variant"] != "naive_poisson"]
    out, order = {}, None
    key = ["series_id", "player"]
    for v, g in res.groupby("variant", sort=False):
        g = g.drop_duplicates(key).set_index(key)
        if order is None:
            order = g.sort_values("date").index
        g = g.reindex(order).reset_index()
        g["date"] = pd.to_datetime(g["date"])
        out[v] = g
    return out


def stack(rows: pd.DataFrame) -> np.ndarray:
    return np.stack(rows["pmf"].to_numpy()).astype(float)


# ------------------------------------------------------------------ S and I ----
def refit_dates(dates: pd.Series) -> pd.DatetimeIndex:
    if len(dates) == 0:
        return pd.DatetimeIndex([])
    d0 = dates.min().normalize()
    return pd.date_range(d0, dates.max().normalize() + REFIT, freq=REFIT)


def apply_spread(base: pd.DataFrame, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """S: walk-forward k per position x market (weekly refit) applied to every pmf."""
    out = P.copy()
    ks = np.ones(len(base))
    d = base["date"]
    fit_rows = base[SPREAD_COLS]
    for t in refit_dates(d):
        sel = ((d >= t) & (d < t + REFIT)).to_numpy()
        if not sel.any():
            continue
        table = fit_spread(fit_rows[d <= t - LAG])
        for i in np.flatnonzero(sel):
            k = table.get((base["position"].iat[i], base["market"].iat[i]), table["_pool"])
            ks[i] = k
            if k != 1.0:
                out[i] = inflate_pmf(P[i], k)
    return out, ks


def apply_calibration(base: pd.DataFrame, P: np.ndarray, method: str) -> np.ndarray:
    """I: walk-forward calibrator (weekly refit) on pairs from the same pmfs."""
    pairs = pairs_from_rows(base.assign(pmf=list(P)))
    pd_dates = pairs["date"]
    out = P.copy()
    d = base["date"]
    for t in refit_dates(d):
        sel = ((d >= t) & (d < t + REFIT)).to_numpy()
        if not sel.any():
            continue
        cal = Calibrator.fit(pairs[pd_dates <= t - LAG], method)
        if cal.pool is None:
            continue
        for i in np.flatnonzero(sel):
            out[i] = calibrate_pmf(P[i], cal, base["position"].iat[i], base["market"].iat[i],
                                   base["line"].iat[i])
    return out


# ------------------------------------------------------------------ proxy lines ----
def proxy_lines(players: pd.DataFrame) -> pd.DataFrame:
    """Per (series_id, player, market): floor(rolling median of the player's prior market
    totals) + 0.5. Prior = series that started strictly earlier; last PROXY_WINDOW
    qualifying series, at least PROXY_MIN; else n_maps x median of the last 20 maps.
    proxy_mean = the same window's mean to the nearest half point (x.5), the PP-like line
    the goblin map is fitted below (reports/goblin_fit.md)."""
    p = players[["series_id", "playername", "map_no", "kills", "date"]].copy()
    p["series_id"] = p["series_id"].astype(str)
    p["playername"] = p["playername"].astype(str)
    p = p.sort_values(["playername", "date", "map_no"])
    p["start"] = p.groupby("series_id")["date"].transform("min")
    out = []
    for pl, g in p.groupby("playername", sort=False):
        ser = g.groupby("series_id", sort=False).agg(
            start=("start", "first"), maps=("kills", list), mapno=("map_no", list))
        ser = ser.sort_values("start")
        hist = {n: [] for n in (1, 2, 3)}       # prior market totals
        maps_hist: list = []                      # prior per-map kills
        prev_start, pending = None, []            # same-start series are not "prior"
        for sid, r in ser.iterrows():
            if prev_start is not None and r["start"] > prev_start:
                for totals, kills in pending:
                    for n, tot in totals.items():
                        hist[n].append(tot)
                    maps_hist.extend(kills)
                pending = []
            for mkt, n in N_MAPS.items():
                h = hist[n][-PROXY_WINDOW:]
                if len(h) >= PROXY_MIN:
                    med, avg = float(np.median(h)), float(np.mean(h))
                elif len(maps_hist) >= PROXY_MIN:
                    med = n * float(np.median(maps_hist[-20:]))
                    avg = n * float(np.mean(maps_hist[-20:]))
                else:
                    med = avg = np.nan
                ok = np.isfinite(med)
                out.append((sid, pl, mkt, np.floor(med) + 0.5 if ok else np.nan,
                            np.round(avg - 0.5) + 0.5 if ok else np.nan))
            by_map = dict(zip(r["mapno"], r["maps"]))
            totals = {n: sum(by_map[m] for m in range(1, n + 1)) for n in (1, 2, 3)
                      if all(m in by_map for m in range(1, n + 1))}
            pending.append((totals, [by_map[m] for m in sorted(by_map)]))
            prev_start = r["start"]
    return pd.DataFrame(out, columns=["series_id", "player", "market", "proxy_line",
                                      "proxy_mean"])


def line_rows(base: pd.DataFrame, P: np.ndarray, line_col: str = "proxy_line") -> pd.DataFrame:
    """W input rows: P(more / less / push) of the pmfs at each row's line (rows without a
    line dropped). slate = UTC date of the series start."""
    ok = base[line_col].notna().to_numpy()
    b = base[ok]
    Pm = P[ok]
    line = b[line_col].to_numpy(float)
    ks = np.arange(P.shape[1])
    return pd.DataFrame({
        "date": b["date"].to_numpy(), "slate": b["date"].dt.normalize().to_numpy(),
        "market": b["market"].to_numpy(), "position": b["position"].to_numpy(),
        "line": line, "p_more": (Pm * (ks[None, :] > line[:, None])).sum(axis=1),
        "p_less": (Pm * (ks[None, :] < line[:, None])).sum(axis=1),
        "p_push": (Pm * (ks[None, :] == line[:, None])).sum(axis=1),
        "y": b["y"].to_numpy()})


# ------------------------------------------------------------------ W ----
def apply_w(wrows: pd.DataFrame, breakeven: float = BE_2L) -> tuple[pd.DataFrame, dict]:
    """Walk-forward lambda per row (weekly refit on picks >= LAG before). Returns the rows
    with a `lam` column and {refit date: lambda dict}."""
    out = wrows.copy()
    out["lam"] = 1.0
    d = pd.to_datetime(out["date"])
    fits = {}
    for t in refit_dates(d):
        sel = ((d >= t) & (d < t + REFIT)).to_numpy()
        if not sel.any():
            continue
        lam = fit_w(wrows[d <= t - LAG], breakeven)
        fits[t] = lam
        out.loc[sel, "lam"] = out.loc[sel, "market"].map(
            lambda m: lam.get(m, lam["_pool"])).to_numpy(float)
    return out, fits


def select_rows(wrows: pd.DataFrame, use_w: bool, top_n: int | None,
                breakeven: float = BE_2L) -> pd.DataFrame:
    """Top-N per slate with (use_w) the row's walk-forward lambda, else lambda = 1."""
    if not use_w or wrows.empty:
        return select_top(wrows, None, breakeven, top_n)
    parts = []
    for lam, g in wrows.groupby("lam"):
        parts.append(select_top(g, {"_pool": float(lam)}, breakeven, None))
    d = pd.concat(parts).sort_values("edge", ascending=False)
    return d.groupby("slate", sort=False).head(top_n) if top_n is not None else d


def sel_summary(c: pd.DataFrame, label: str, p_col: str = "p_w") -> dict:
    out = dict(selection=label, n=len(c), pred=c[p_col].mean(), hit=c["hit"].mean())
    out["diff"] = out["hit"] - out["pred"]
    out["se"] = np.sqrt(out["hit"] * (1 - out["hit"]) / max(1, len(c)))
    for s in ("more", "less"):
        g = c[c["side"] == s]
        out[f"n_{s}"] = len(g)
        out[f"pred_{s}"] = g[p_col].mean() if len(g) else np.nan
        out[f"hit_{s}"] = g["hit"].mean() if len(g) else np.nan
    return out


def derive_min_edge(wrows: pd.DataFrame, breakeven: float = BE_2L) -> tuple[float, pd.DataFrame]:
    """Minimum W-adjusted edge: the residual overstatement of W-adjusted picks plus 2 se,
    rounded up to a whole point, taking the worst of top-1/3/5 per slate and all picks.
    After W the residual is flat in the edge size, so a threshold only has to clear it."""
    rows = []
    for top in (1, 3, 5, None):
        c = select_rows(wrows, True, top, breakeven)
        r = sel_summary(c, f"top {top}" if top else "all edge > 0")
        r["required"] = max(0.0, -r["diff"] + 2 * r["se"])
        rows.append(r)
    tab = pd.DataFrame(rows)
    return float(np.ceil(100 * tab["required"].max() - 1e-9) / 100), tab


# ------------------------------------------------------------------ artifacts ----
def build(base: pd.DataFrame, players: pd.DataFrame) -> dict:
    """Everything the app and the reports need from the live locked variant's rows:
    S pmfs, I-calibrated pmfs, W rows with walk-forward lambda, and the artifact frames."""
    P0 = stack(base)
    PS, ks = apply_spread(base, P0)
    PSI = apply_calibration(base, PS, CAL_METHOD)
    prox = proxy_lines(players)
    b = base.drop(columns=[c for c in ("proxy_line", "proxy_mean") if c in base]).merge(
        prox, on=["series_id", "player", "market"], how="left")
    wrows, fits = apply_w(line_rows(b, PSI))
    pairs = pd.concat([pairs_from_rows(base.assign(pmf=list(P))).assign(model=m)
                       for m, P in (("L+O", P0), ("L+O+S", PS))], ignore_index=True)
    pairs["p"] = pairs["p"].astype("float32")
    pairs["offset"] = pairs["offset"].astype("float32")
    pairs["y"] = pairs["y"].astype("int8")
    # G: goblin lines under the mean-based PP-like line, on the S pmfs (what the app's
    # goblin column reads); rows are aligned with b (same order as base)
    gob = goblin_pairs(b, PS, "proxy_mean")
    return dict(P0=P0, PS=PS, PSI=PSI, ks=ks, base=b, wrows=wrows, w_fits=fits,
                pairs=pairs, spread_rows=base[SPREAD_COLS].copy(), goblin_pairs=gob)


def params_summary(art: dict, cutoff=None) -> dict:
    """The fitted S / I / W parameters as of `cutoff` (default: after the last row)."""
    base = art["base"]
    cutoff = pd.Timestamp(cutoff) if cutoff is not None else \
        base["date"].max().normalize() + pd.Timedelta(days=2)
    lim = cutoff - LAG
    sp = fit_spread(art["spread_rows"][art["spread_rows"]["date"] <= lim])
    pairs = art["pairs"]
    cal = Calibrator.fit(pairs[(pairs["model"] == "L+O+S")
                               & (pd.to_datetime(pairs["date"]) <= lim)], CAL_METHOD)
    w = art["wrows"]
    lam = fit_w(w[pd.to_datetime(w["date"]) <= lim], BE_2L)
    grid = [0.1, 0.3, 0.5, 0.7, 0.9]
    return {
        "fitted_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "cutoff": cutoff.isoformat(), "rows": int(len(base)),
        "last_series": base["date"].max().isoformat(),
        "spread_k": {("_pool" if k == "_pool" else f"{k[0]}|{k[1]}"): round(v, 4)
                     for k, v in sp.items()},
        "calib_pool": [round(float(x), 4) for x in cal(np.array(grid), "_", "_")]
        if cal.pool is not None else None,
        "calib_grid": grid,
        "w_lambda": {k: (int(v) if k == "_n" else round(float(v), 4)) for k, v in lam.items()},
        "min_edge_suggested": derive_min_edge(w)[0],
        "goblin_maps": {pos: [round(float(x), 4) for x in m(np.array(grid))]
                        for pos, m in GoblinCal.fit(
                            gp[pd.to_datetime(gp["date"]) <= lim]).maps.items()}
        if (gp := art.get("goblin_pairs")) is not None else {},
    }


DRIFT_LIMITS = {"spread_k": 0.10, "calib_pool": 0.05, "w_lambda": 0.25}


def compare_params(old: dict | None, new: dict) -> list[str]:
    """Problems that should stop an automatic refit: any S k moving > 0.10, the pooled
    I map moving > 0.05 at any grid point, any W lambda moving > 0.25."""
    if not old:
        return []
    bad = []
    for k, v in new["spread_k"].items():
        o = old.get("spread_k", {}).get(k)
        if o is not None and abs(v - o) > DRIFT_LIMITS["spread_k"]:
            bad.append(f"S k[{k}] {o:.3f} -> {v:.3f}")
    if old.get("calib_pool") and new.get("calib_pool"):
        for g, o, v in zip(new["calib_grid"], old["calib_pool"], new["calib_pool"]):
            if abs(v - o) > DRIFT_LIMITS["calib_pool"]:
                bad.append(f"I map at p={g}: {o:.3f} -> {v:.3f}")
    for k, v in new["w_lambda"].items():
        o = old.get("w_lambda", {}).get(k)
        if k != "_n" and o is not None and abs(v - o) > DRIFT_LIMITS["w_lambda"]:
            bad.append(f"W lambda[{k}] {o:.3f} -> {v:.3f}")
    return bad


def write_artifacts(art: dict, data_dir: Path, params: dict) -> list[Path]:
    """data/calib_pairs.parquet (I), data/spread_rows.parquet (S), data/w_rows.parquet
    (W, calibrated P at proxy lines), data/goblin_pairs.parquet (G) and
    data/calib_params.json (fitted summary)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    out = []
    w = art["wrows"][W_COLS].copy()
    for c in ("p_more", "p_less", "p_push"):
        w[c] = w[c].astype("float32")
    for name, df in (("calib_pairs.parquet", art["pairs"]),
                     ("spread_rows.parquet", art["spread_rows"]), ("w_rows.parquet", w),
                     ("goblin_pairs.parquet", art.get("goblin_pairs"))):
        if df is None:
            continue
        df.to_parquet(data_dir / name, index=False)
        out.append(data_dir / name)
    (data_dir / "calib_params.json").write_text(json.dumps(params, indent=1) + "\n")
    out.append(data_dir / "calib_params.json")
    return out
