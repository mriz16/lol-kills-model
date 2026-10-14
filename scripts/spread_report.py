"""Spread / calibration study on stored backtest pmfs -> reports/backtest_spread.md.

    python scripts/run_backtest.py --rows-only --keep-pmf --tag spread \\
        --variants L+O L+O+K30 L+O+K100
    python scripts/spread_report.py --rows reports/backtest_rows_spread.parquet

(The three variants can also run as separate processes and be passed as several --rows.)

Variants scored (all walk-forward; every fit uses only rows of series that started at
least one day before the refit date, refit weekly):
  L+O          live locked config, raw
  +S           spread inflation, k per position x market
  +I / +Ib     P(over) calibration per position x market, isotonic / beta
  +S+I         I fitted on the S-inflated pmfs
  L+O+K<n0>    per-player share concentration (re-simulated), and K<best>+S+I
Selection-bias check: per slate (UTC date), bet the top-N edges at proxy lines (player's
rolling median of the market total + 0.5) and compare hit rate with predicted.
Also writes data/calib_pairs.parquet (the live app's calibration training set).
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
from src.backtest.walkforward import LIVE_LOCKED, TAIL_OFFSETS  # noqa: E402
from src.config import DATA_DIR, POSITIONS, PRIZEPICKS_POWER, REPORTS_DIR  # noqa: E402
from src.engine.calibration import (OFFSETS, Calibrator, calibrate_pmf,  # noqa: E402
                                    fit_spread, inflate_pmf, pairs_from_rows)

N_SIMS = 20_000
BE_2L = (1 / PRIZEPICKS_POWER[2]) ** 0.5
MARKETS = ["Map 1", "Maps 1-2", "Maps 1-3"]
N_MAPS = {"Map 1": 1, "Maps 1-2": 2, "Maps 1-3": 3}
PROXY_WINDOW, PROXY_MIN = 10, 5
REFIT = pd.Timedelta(days=7)
LAG = pd.Timedelta(days=1)


# ------------------------------------------------------------------ pmf metrics ----
def pmf_metrics(P: np.ndarray, y: np.ndarray, line: np.ndarray) -> pd.DataFrame:
    """Per-row CRPS, log score, mean, PIT pieces, P(over) at the median / half / tail lines."""
    n, m = P.shape
    F = np.cumsum(P, axis=1)
    F[:, -1] = 1.0
    ks = np.arange(m)
    H = (ks[None, :] >= y[:, None]).astype(float)
    crps = np.sum((F - H) ** 2, axis=1)
    py = P[np.arange(n), y]
    last = np.where(P > 0, ks[None, :], 0).max(axis=1)
    upto = np.maximum(last, y) + 1
    logs = -np.log((py * N_SIMS + 0.5) / (N_SIMS + 0.5 * (upto + 1)))
    mean = P @ ks
    Fy = F[np.arange(n), y]
    Fy1 = np.where(y > 0, F[np.arange(n), np.maximum(y - 1, 0)], 0.0)
    surv = 1.0 - F
    out = pd.DataFrame({"crps": crps, "logscore": logs, "mean": mean, "pit_lo": Fy1,
                        "pit_hi": Fy})
    li = line.astype(int)
    out["p_over"] = surv[np.arange(n), np.clip(li, 0, m - 1)]
    for off in OFFSETS:
        j = np.clip(np.floor(line + off).astype(int), 0, m - 1)
        p = surv[np.arange(n), j]
        out[f"h{off:+.1f}_p"] = np.where(line + off > 0, p, np.nan)
    for off in TAIL_OFFSETS:
        j = np.clip(np.floor(line + off).astype(int), 0, m - 1)
        out[f"t{off}_p"] = surv[np.arange(n), j]
    return out


def pit_hist(lo: np.ndarray, hi: np.ndarray, bins: int = 10) -> np.ndarray:
    """Non-randomized PIT histogram for counts (Czado et al. 2009): bin frequencies."""
    edges = np.linspace(0, 1, bins + 1)
    w = np.maximum(hi - lo, 1e-12)
    fu = lambda u: np.clip((u - lo[:, None]) / w[:, None], 0, 1)  # noqa: E731
    cdf = fu(edges[None, :])
    return np.diff(cdf, axis=1).mean(axis=0)


def pit_stats(lo: np.ndarray, hi: np.ndarray) -> dict:
    h = pit_hist(lo, hi)
    n = len(lo)
    m1 = ((lo + hi) / 2).mean()
    m2 = ((hi ** 3 - lo ** 3) / (3 * np.maximum(hi - lo, 1e-12))).mean()
    return dict(pit_mean=m1, pit_var_ratio=(m2 - m1 ** 2) * 12,
                pit_chi2=n * np.sum((h - 0.1) ** 2 / 0.1), pit_lo10=h[0], pit_hi10=h[-1])


def ece(p: np.ndarray, y: np.ndarray, width: float = 0.05) -> float:
    ok = np.isfinite(p)
    p, y = p[ok], y[ok]
    b = np.minimum((p / width).astype(int), int(1 / width) - 1)
    tot = 0.0
    for k in np.unique(b):
        m = b == k
        tot += abs(y[m].mean() - p[m].mean()) * m.sum()
    return tot / len(p)


def score(name: str, P: np.ndarray, base: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    y = base["y"].to_numpy()
    line = base["line"].to_numpy(float)
    m = pmf_metrics(P, y, line)
    hp = np.concatenate([m[f"h{o:+.1f}_p"].to_numpy() for o in OFFSETS])
    hy = np.concatenate([(y > line + o).astype(float) for o in OFFSETS])
    tp = np.concatenate([m[f"t{o}_p"].to_numpy() for o in TAIL_OFFSETS])
    ty = np.concatenate([(y > line + o).astype(float) for o in TAIL_OFFSETS])
    hmat = np.column_stack([m[f"h{o:+.1f}_p"].to_numpy() for o in OFFSETS])
    ymat = np.column_stack([(y > line + o).astype(float) for o in OFFSETS])
    m["brier_h"] = np.nanmean((hmat - ymat) ** 2, axis=1)
    ok = np.isfinite(hp)
    hpc = np.clip(hp[ok], 1e-4, 1 - 1e-4)
    row = dict(variant=name, n=len(y), crps=m["crps"].mean(), logscore=m["logscore"].mean(),
               bias=(m["mean"] - y).mean(),
               brier_half=np.mean((hp[ok] - hy[ok]) ** 2),
               logloss_half=-np.mean(hy[ok] * np.log(hpc) + (1 - hy[ok]) * np.log(1 - hpc)),
               ece_median=ece(m["p_over"].to_numpy(), (y > line).astype(float)),
               ece_half=ece(hp, hy), tail_obs_pred=ty.mean() / tp.mean(),
               **pit_stats(m["pit_lo"].to_numpy(), m["pit_hi"].to_numpy()))
    for pos in POSITIONS:
        sel = (base["position"] == pos).to_numpy()
        row[f"bias_{pos}"] = (m["mean"].to_numpy()[sel] - y[sel]).mean()
    return row, m


# ------------------------------------------------------------------ walk-forward ----
def refit_dates(dates: pd.Series) -> pd.DatetimeIndex:
    d0 = dates.min().normalize()
    return pd.date_range(d0, dates.max().normalize() + REFIT, freq=REFIT)


def apply_spread(base: pd.DataFrame, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """S: walk-forward k per position x market (weekly refit) applied to every pmf."""
    out = P.copy()
    ks = np.ones(len(base))
    d = base["date"]
    fit_rows = base[["date", "position", "market", "y", "mean", "var"]]
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
    qualifying series, at least PROXY_MIN; else n_maps x median of the last 20 maps."""
    p = players[["series_id", "playername", "map_no", "kills", "date"]].copy()
    p["series_id"] = p["series_id"].astype(str)
    p["playername"] = p["playername"].astype(str)
    p = p.sort_values(["playername", "date", "map_no"])
    start = p.groupby("series_id")["date"].transform("min")
    p["start"] = start
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
                    med = float(np.median(h))
                elif len(maps_hist) >= PROXY_MIN:
                    med = n * float(np.median(maps_hist[-20:]))
                else:
                    med = np.nan
                out.append((sid, pl, mkt, np.floor(med) + 0.5 if np.isfinite(med) else np.nan))
            by_map = dict(zip(r["mapno"], r["maps"]))
            totals = {n: sum(by_map[m] for m in range(1, n + 1)) for n in (1, 2, 3)
                      if all(m in by_map for m in range(1, n + 1))}
            pending.append((totals, [by_map[m] for m in sorted(by_map)]))
            prev_start = r["start"]
    return pd.DataFrame(out, columns=["series_id", "player", "market", "proxy_line"])


def selection_table(base: pd.DataFrame, P: np.ndarray, top_ns=(3, 5, 10, None),
                    min_edge: float = 0.0) -> pd.DataFrame:
    """Top-N edges per slate at proxy lines: predicted vs hit, overall and by side."""
    ok = base["proxy_line"].notna().to_numpy()
    b = base[ok].copy()
    Pm = P[ok]
    j = np.clip(np.floor(b["proxy_line"].to_numpy()).astype(int), 0, P.shape[1] - 1)
    surv = 1.0 - np.cumsum(Pm, axis=1)
    b["p_more"] = surv[np.arange(len(b)), j]
    b["side"] = np.where(b["p_more"] >= 0.5, "more", "less")
    b["p_side"] = np.where(b["side"] == "more", b["p_more"], 1 - b["p_more"])
    b["edge"] = b["p_side"] - BE_2L
    b["hit"] = np.where(b["side"] == "more", b["y"] > b["proxy_line"],
                        b["y"] < b["proxy_line"]).astype(float)
    b["slate"] = b["date"].dt.normalize()
    rows = []
    for n in top_ns:
        c = b[b["edge"] > min_edge].sort_values("edge", ascending=False)
        if n is not None:
            c = c.groupby("slate", sort=False).head(n)
        rows.append(_sel_row(f"top {n}" if n else f"all edge > {min_edge:.0%}", c))
    return pd.DataFrame(rows), b


def _sel_row(label: str, c: pd.DataFrame) -> dict:
    out = dict(selection=label, n=len(c), pred=c["p_side"].mean(), hit=c["hit"].mean())
    out["diff"] = out["hit"] - out["pred"]
    out["se"] = np.sqrt(out["hit"] * (1 - out["hit"]) / max(1, len(c)))
    for s in ("more", "less"):
        g = c[c["side"] == s]
        out[f"n_{s}"] = len(g)
        out[f"pred_{s}"] = g["p_side"].mean() if len(g) else np.nan
        out[f"hit_{s}"] = g["hit"].mean() if len(g) else np.nan
    return out


def band_table(b: pd.DataFrame, bands=(0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 1.0)
               ) -> pd.DataFrame:
    """Non-cumulative edge bands: predicted vs hit (shortfall = pred - hit)."""
    rows = []
    for lo, hi in zip(bands[:-1], bands[1:]):
        c = b[(b["edge"] > lo) & (b["edge"] <= hi)]
        rows.append(_sel_row(f"{100 * lo:.0f}-{100 * hi:.0f} pts" if hi < 1
                             else f"> {100 * lo:.0f} pts", c))
    return pd.DataFrame(rows)


def threshold_table(b: pd.DataFrame, cuts=(0.0, 0.02, 0.04, 0.06, 0.08, 0.10)) -> pd.DataFrame:
    rows = []
    for t in cuts:
        c = b[b["edge"] > t]
        r = _sel_row(f"edge > {100 * t:.0f} pts", c)
        r["hit_minus_be"] = r["hit"] - BE_2L
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main ----
def load(paths: list[Path]) -> dict[str, pd.DataFrame]:
    res = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    res = res[res["variant"] != "naive_poisson"]
    out = {}
    key = ["series_id", "player"]
    order = None
    for v, g in res.groupby("variant", sort=False):
        g = g.drop_duplicates(key).set_index(key)
        if order is None:
            order = g.sort_values("date").index
        out[v] = g.reindex(order).reset_index()
    return out


def paired(a: pd.DataFrame, b: pd.DataFrame, col: str) -> tuple[float, float]:
    d = a[col].to_numpy() - b[col].to_numpy()
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, nargs="+", required=True)
    ap.add_argument("--data", type=Path, default=DATA_DIR)
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "backtest_spread.md")
    ap.add_argument("--pairs-out", type=Path, default=DATA_DIR / "calib_pairs.parquet")
    args = ap.parse_args(argv)

    rows = load(args.rows)
    base = rows[LIVE_LOCKED]
    base["date"] = pd.to_datetime(base["date"])
    P0 = np.stack(base["pmf"].to_numpy()).astype(float)
    players = pd.read_parquet(args.data / "players.parquet")
    prox = proxy_lines(players)
    base = base.merge(prox, on=["series_id", "player", "market"], how="left")

    pmfs: dict[str, np.ndarray] = {"L+O": P0}
    print("S ...", flush=True)
    pmfs["L+O+S"], ks = apply_spread(base, P0)
    base["k_spread"] = ks
    print("I ...", flush=True)
    pmfs["L+O+I"] = apply_calibration(base, P0, "iso")
    pmfs["L+O+Ib"] = apply_calibration(base, P0, "beta")
    pmfs["L+O+S+I"] = apply_calibration(base, pmfs["L+O+S"], "iso")
    pmfs["L+O+S+Ib"] = apply_calibration(base, pmfs["L+O+S"], "beta")
    kvars = [v for v in rows if v.startswith("L+O+K")]
    for v in kvars:
        pmfs[v] = np.stack(rows[v]["pmf"].to_numpy()).astype(float)

    print("scoring ...", flush=True)
    table, per_row = [], {}
    for name, P in pmfs.items():
        r, m = score(name, P, base)
        table.append(r)
        per_row[name] = m
    tab = pd.DataFrame(table)
    best_k = None
    if kvars:
        best_k = tab[tab.variant.isin(kvars)].sort_values("crps")["variant"].iloc[0]
        kb = base.copy()
        kb["mean"] = rows[best_k]["mean"].to_numpy()
        kb["var"] = rows[best_k]["var"].to_numpy()
        PkS, _ = apply_spread(kb, pmfs[best_k])
        pmfs[f"{best_k}+S+I"] = apply_calibration(base, PkS, "iso")
        r, m = score(f"{best_k}+S+I", pmfs[f"{best_k}+S+I"], base)
        table.append(r)
        per_row[f"{best_k}+S+I"] = m
        tab = pd.DataFrame(table)
    for c in ("crps", "logscore", "brier_h"):
        d = [paired(per_row[v], per_row["L+O"], c) for v in tab["variant"]]
        tab[f"d{c}"] = [x[0] for x in d]
        tab[f"d{c}_se"] = [x[1] for x in d]

    # selection-bias check
    sel, sel_b = {}, {}
    for name in pmfs:
        sel[name], sel_b[name] = selection_table(base, pmfs[name])
    # calibration pairs for the app (live locked model, all rows)
    # model "L+O": raw locked pmfs; "L+O+S": walk-forward S-inflated pmfs (the app's
    # calibrator picks one by model; "L+O+S" is ready for when S is signed off).
    parts = []
    for model, P in (("L+O", P0), ("L+O+S", pmfs["L+O+S"])):
        pr = pairs_from_rows(base.assign(pmf=list(P)))
        parts.append(pr.assign(model=model))
    pairs = pd.concat(parts, ignore_index=True)
    pairs["p"] = pairs["p"].astype("float32")
    pairs["offset"] = pairs["offset"].astype("float32")
    pairs["y"] = pairs["y"].astype("int8")
    args.pairs_out.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(args.pairs_out, index=False)

    text = render(base, tab, per_row, sel, sel_b, best_k, pmfs)
    args.out.write_text(text)
    print(text[:4000])
    return 0


def pit_tables(base: pd.DataFrame, m: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = []
    for col, keys in (("position", POSITIONS), ("market", MARKETS)):
        for k in keys:
            sel = (base[col] == k).to_numpy()
            if not sel.any():
                continue
            h = pit_hist(m["pit_lo"].to_numpy()[sel], m["pit_hi"].to_numpy()[sel])
            st = pit_stats(m["pit_lo"].to_numpy()[sel], m["pit_hi"].to_numpy()[sel])
            out.append(dict(group=k, n=int(sel.sum()),
                            **{f"{10 * i}-{10 * i + 10}": 100 * h[i] for i in range(10)},
                            mean=st["pit_mean"], var_ratio=st["pit_var_ratio"]))
    df = pd.DataFrame(out)
    return df[df.group.isin(POSITIONS)], df[df.group.isin(MARKETS)]


def render(base, tab, per_row, sel, sel_b, best_k, pmfs) -> str:
    d0, d1 = base["date"].min().date(), base["date"].max().date()
    cols = ["variant", "crps", "dcrps", "dcrps_se", "logscore", "dlogscore", "dlogscore_se",
            "bias", "brier_half", "dbrier_h", "dbrier_h_se", "logloss_half", "ece_median", "ece_half", "tail_obs_pred", "pit_mean", "pit_var_ratio",
            "pit_chi2", "pit_lo10", "pit_hi10"]
    L = [
        "# Spread, calibration and selection-bias backtest (real OE data, walk-forward, "
        "20,000 sims)", "",
        f"{len(base):,} player-series, {d0} -> {d1}. Base = live locked config **L+O** "
        "(shrink30 + A + B50 + O). Paired deltas vs L+O with standard errors.", "",
        "<!-- SUMMARY -->", "",
        "## Scores", "",
        "`pit_var_ratio` = variance of the (non-randomized) PIT x 12: 1 is ideal, > 1 means "
        "the forecast is too narrow. `pit_lo10` / `pit_hi10` = share of PIT mass in the "
        "lowest / highest decile (0.10 ideal). `ece_half` = ECE at half-point lines rounded "
        "median -2.5..+2.5. `tail_obs_pred` = hit / predicted at median +1.5/2.5/3.5.", "",
        _md(tab[cols]), "",
        "### Bias by position (mean - actual)", "",
        _md(tab[["variant"] + [f"bias_{p}" for p in POSITIONS]]), "",
    ]
    for v in ["L+O", "L+O+S", "L+O+S+Ib"] + ([best_k] if best_k else []):
        if v not in per_row:
            continue
        tp, tm = pit_tables(base, per_row[v])
        L += [f"## PIT histogram, {v} (% of mass per decile)", "",
              "### by position", "", _md(tp, 3), "",
              "### by market", "", _md(tm, 3), ""]
    if "k_spread" in base:
        ks = base.groupby(["position", "market"])["k_spread"].last().unstack()
        L += ["## S: latest walk-forward k (position x market)", "",
              _md(ks.reset_index(), 3), ""]
    L += ["## Selection-bias check (proxy lines, PrizePicks 2-leg breakeven "
          f"{BE_2L:.1%})", "",
          "Proxy line = floor(median of the player's last 10 prior market totals) + 0.5 "
          "(fallback n_maps x median of the last 20 maps). Slate = UTC date. Top-N = the N "
          "biggest edges (edge > 0) per slate.", ""]
    show = ["L+O", "L+O+S", "L+O+I", "L+O+Ib", "L+O+S+I", "L+O+S+Ib"] + \
        ([f"{best_k}+S+I"] if best_k else [])
    for v in show:
        if v in sel:
            L += [f"### {v}", "", _md(sel[v]), ""]
    L += ["## Model side at all proxy lines (no selection)", ""]
    rows = []
    for v in show:
        if v not in sel_b:
            continue
        b = sel_b[v]
        rows.append(dict(variant=v, n=len(b), share_less=(b["side"] == "less").mean(),
                         pred_less=b.loc[b.side == "less", "p_side"].mean(),
                         hit_less=b.loc[b.side == "less", "hit"].mean(),
                         pred_more=b.loc[b.side == "more", "p_side"].mean(),
                         hit_more=b.loc[b.side == "more", "hit"].mean(),
                         actual_under_rate=(b["y"] < b["proxy_line"]).mean()))
    L += [_md(pd.DataFrame(rows)), ""]
    for v in ("L+O", "L+O+I", "L+O+Ib", "L+O+S+I", "L+O+S+Ib"):
        L += [f"## Edge bands, {v} (all proxy-line picks, not cumulative)", "",
              _md(band_table(sel_b[v])), ""]
    for v in ("L+O+Ib", "L+O+S+Ib", "L+O"):
        L += [f"## Minimum edge ({v}, every pick above the threshold)", "",
              _md(threshold_table(sel_b[v])), ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
