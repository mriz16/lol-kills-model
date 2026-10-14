"""Walk-forward backtest: every series is projected from strictly prior data."""
from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from src.backtest.elo import Elo
from src.backtest.metrics import (calibration_table, crps_discrete, ece, log_score,
                                  poisson_scores)
from src.config import POSITION_PRIORS, POSITIONS
from src.engine.features import (EngineConfig, History, build_matchup, fit_kappa,
                                 fit_kappa_by_position, fit_nb_dispersion)
from src.engine.sim import simulate_series

S30 = dict(shrink_games=30)
# Every named configuration the runner knows. --variants picks any subset.
ALL_VARIANTS = {
    "base": EngineConfig(),                                   # shrink20
    "shrink30": EngineConfig(**S30),
    "shrink30+A": EngineConfig(**S30, position_kappa=True),
    "shrink30+C": EngineConfig(**S30, smooth_bootstrap=True),
    "shrink30+A+C": EngineConfig(**S30, position_kappa=True, smooth_bootstrap=True),
}
for _f in (15, 25, 35, 50):
    ALL_VARIANTS[f"shrink30+B{_f}"] = EngineConfig(**S30, neff_floor=_f)
    ALL_VARIANTS[f"shrink30+A+B{_f}"] = EngineConfig(**S30, position_kappa=True,
                                                     neff_floor=_f)

# Locked config (user sign-off 2026-10-14): shrink30 + A + B50, game-script off, C off.
LOCKED = "shrink30+A+B50"
LOCKED_CFG = dict(**S30, position_kappa=True, neff_floor=50)
# Mean-bias toggles on top of the locked config. G is reference only (never locked):
# its kill_mult is refit walk-forward by run_backtest from the LOCKED variant's errors.
for _n in (1, 2, 3):
    ALL_VARIANTS[f"L+M{_n}"] = EngineConfig(**LOCKED_CFG, meta_drift_patches=_n)
    ALL_VARIANTS[f"L+M{_n}+P"] = EngineConfig(**LOCKED_CFG, meta_drift_patches=_n,
                                              fit_priors=True)
ALL_VARIANTS["L+P"] = EngineConfig(**LOCKED_CFG, fit_priors=True)
# O (outcome-conditional opponent adjustment) was added after the M/P runs showed the
# bias sits in BO5s / strong-vs-strong matchups; tested alone and with P and M3.
ALL_VARIANTS["L+O"] = EngineConfig(**LOCKED_CFG, opp_by_outcome=True)
ALL_VARIANTS["L+O+P"] = EngineConfig(**LOCKED_CFG, opp_by_outcome=True, fit_priors=True)
ALL_VARIANTS["L+O+M3"] = EngineConfig(**LOCKED_CFG, opp_by_outcome=True,
                                      meta_drift_patches=3)
ALL_VARIANTS["L+G"] = EngineConfig(**LOCKED_CFG)
G_VARIANTS = {"L+G"}
G_MIN_ROWS = 500           # player-series before G departs from 1.0
BIAS_VARIANTS = [LOCKED, "L+M1", "L+M2", "L+M3", "L+P", "L+M1+P", "L+M2+P", "L+M3+P",
                 "L+G", "L+O", "L+O+P", "L+O+M3"]

# Default run (real-data backtest). A is run with every B floor; summarize() keeps the
# one matching the best stand-alone B as "shrink30+A+best-B".
DEFAULT_VARIANTS = ["base", "shrink30", "shrink30+B25", "shrink30+B35", "shrink30+B50",
                    "shrink30+A", "shrink30+A+B25", "shrink30+A+B35", "shrink30+A+B50",
                    "shrink30+C"]
VARIANTS = {k: ALL_VARIANTS[k] for k in DEFAULT_VARIANTS}
TAIL_OFFSETS = (1.5, 2.5, 3.5)


@dataclass
class SeriesInfo:
    series_id: str
    start: pd.Timestamp
    league: str
    best_of: int
    patch: int
    team_a: str
    team_b: str
    lineups: tuple
    maps: pd.DataFrame          # team rows of the series (one per team per map)
    player_kills: dict          # player -> list of kills per map played (None if absent)


def primary_market(best_of: int) -> str:
    """The market actually bet: Map 1 (BO1), Maps 1-2 (BO3), Maps 1-3 (BO5)."""
    return {1: "Map 1", 3: "Maps 1-2", 5: "Maps 1-3"}[best_of]


def market_range(market: str) -> tuple[int, int]:
    from src.engine.sim import parse_market
    return parse_market(market, 5)


def build_series(players: pd.DataFrame, teams: pd.DataFrame) -> list[SeriesInfo]:
    teams = teams.sort_values(["date", "gameid", "side"])
    pk_map = {
        (str(sid), str(pl), int(m)): int(k)
        for sid, pl, m, k in players[["series_id", "playername", "map_no", "kills"]]
        .itertuples(index=False)
    }
    m1p = players[players["map_no"] == 1]
    lineup_map = {
        (str(sid), str(tm), str(pos)): str(pl)
        for sid, tm, pos, pl in m1p[["series_id", "teamname", "position", "playername"]]
        .itertuples(index=False)
    }
    out = []
    for sid, g in teams.groupby("series_id", sort=False, observed=True):
        sid = str(sid)
        g = g.sort_values("map_no")
        m1 = g[g["map_no"] == 1]
        if len(m1) != 2 or ("series_complete" in g and not g["series_complete"].all()):
            continue
        a, b = m1.sort_values("side")["teamname"].astype(str).tolist()
        try:
            lineups = [[lineup_map[(sid, t, pos)] for pos in POSITIONS] for t in (a, b)]
        except KeyError:
            continue
        n_maps = int(g["map_no"].max())
        pk = {pl: [pk_map.get((sid, pl, m)) for m in range(1, n_maps + 1)]
              for pl in lineups[0] + lineups[1]}
        out.append(SeriesInfo(
            series_id=sid, start=m1["date"].iloc[0], league=str(m1["league"].iloc[0]),
            best_of=int(m1["best_of"].iloc[0]), patch=int(m1["patch_ord"].iloc[0]),
            team_a=a, team_b=b, lineups=(lineups[0], lineups[1]), maps=g, player_kills=pk,
        ))
    out.sort(key=lambda s: s.start)
    return out


def _range_actual(kills: list, market: str) -> int | None:
    lo, hi = market_range(market)
    vals = kills[lo - 1:hi]
    if len(vals) < hi - lo + 1 or any(v is None for v in vals):
        return None  # player missed a map in the range: prop would be void
    return int(sum(vals))


def run_backtest(players: pd.DataFrame, teams: pd.DataFrame, variants: dict | None = None,
                 warmup_days: int = 75, n_sims: int = 20_000, seed: int = 2024,
                 refit_days: int = 30, eval_start: pd.Timestamp | None = None,
                 verbose: bool = True, score_filter=None) -> pd.DataFrame:
    """Score every series after the warmup (and after eval_start, if given).

    Elo and history always run over the full period, so eval_start only limits which
    series are scored (used by --quick). score_filter(SeriesInfo) -> bool limits it too.

    G variants (G_VARIANTS) scale team kills by sum(actual)/sum(predicted mean) of the
    LOCKED variant over player-series that started at least one day before the cutoff.
    """
    if G_VARIANTS & set(variants or {}) and LOCKED not in (variants or {}):
        raise ValueError(f"G variants need {LOCKED} in the run")
    g_hist: list = []          # (start, sum y, sum mean) of LOCKED per scored series
    variants = variants or VARIANTS
    hist = History(players, teams)
    series = build_series(players, teams)
    elo = Elo()
    t0 = series[0].start + pd.Timedelta(days=warmup_days) if series else None
    if eval_start is not None:
        t0 = max(t0, pd.Timestamp(eval_start))
    rows = []
    kappa = kappa_at = kappa_pos = nb_r = None
    need_pos = any(v.position_kappa for v in variants.values())
    need_nb = any(v.smooth_bootstrap for v in variants.values())
    tic = time.time()
    for i, s in enumerate(series):
        p_map = elo.p_map(s.team_a, s.team_b, s.league)
        if s.start >= t0 and (score_filter is None or score_filter(s)):
            cutoff = int(s.start.value)
            if kappa is None or (s.start - kappa_at).days >= refit_days:
                kappa, kappa_at = fit_kappa(hist, cutoff), s.start
                kappa_pos = fit_kappa_by_position(hist, cutoff) if need_pos else None
                nb_r = fit_nb_dispersion(hist, cutoff, s.patch) if need_nb else None
            market = primary_market(s.best_of)
            seed_i = seed + i
            naive = {}
            for side, lineup in enumerate(s.lineups):
                for pos, pl in zip(POSITIONS, lineup):
                    mu, n = hist.player_mean_kills(pl, cutoff, s.patch, VARIANTS["base"])
                    if n == 0 or np.isnan(mu):
                        env = hist.env_table(cutoff, s.patch, VARIANTS["base"])
                        mu = POSITION_PRIORS[pos] * env.get(s.league, env["_global"]) * 0.97
                    naive[pl] = mu
            g_mult = 1.0
            if G_VARIANTS & set(variants):
                lim = s.start - pd.Timedelta(days=1)
                done = [(y, m) for st, y, m in g_hist if st <= lim]
                n_rows = sum(len(x) for x in done)
                if n_rows >= G_MIN_ROWS:
                    g_mult = float(sum(sum(y) for y, _ in done)
                                   / sum(sum(m) for _, m in done))
            ys, ms = [], []
            for vname, vcfg in variants.items():
                cfg = replace(vcfg, kappa=kappa, kappa_pos=kappa_pos, nb_r=nb_r)
                if vname in G_VARIANTS:
                    cfg = replace(cfg, kill_mult=g_mult)
                params = build_matchup(hist, s.team_a, s.team_b, s.league, s.best_of, p_map,
                                       s.lineups, cutoff, s.patch, cfg)
                sim = simulate_series(params, n_sims=n_sims, rng=seed_i)
                for side, lineup in enumerate(s.lineups):
                    for pos, pl in zip(POSITIONS, lineup):
                        y = _range_actual(s.player_kills[pl], market)
                        if y is None:
                            continue
                        k = sim.range_kills(pl, market)
                        med = float(np.median(k))
                        line = float(np.floor(med + 0.5))
                        rec = dict(
                            variant=vname, series_id=s.series_id, date=s.start,
                            league=s.league, best_of=s.best_of, market=market, player=pl,
                            position=pos, y=y, mean=float(k.mean()), median=med, line=line,
                            p_over=float((k > line).mean()), p_push=float((k == line).mean()),
                            over=int(y > line), push=int(y == line),
                            crps=crps_discrete(k, y), logscore=log_score(k, y),
                            p_map=p_map, script_mult=params.script_mult, kappa=kappa,
                            kill_mult=cfg.kill_mult,
                        )
                        if vname == LOCKED:
                            ys.append(y)
                            ms.append(rec["mean"])
                        # half-point lines around the median for a wider calibration curve
                        for j, off in enumerate((-1.5, -0.5, 0.5, 1.5)):
                            hl = line + off
                            ok = hl > 0
                            rec[f"hl{j}"] = hl
                            rec[f"hl{j}_p"] = float((k > hl).mean()) if ok else np.nan
                            rec[f"hl{j}_y"] = int(y > hl) if ok else np.nan
                        # upper-tail lines: rounded median + 1.5, 2.5, 3.5
                        for j, off in enumerate(TAIL_OFFSETS):
                            tl = line + off
                            rec[f"tl{j}_p"] = float((k > tl).mean())
                            rec[f"tl{j}_y"] = int(y > tl)
                        rows.append(rec)
                        if vname == next(iter(variants)):
                            lo, hi = market_range(market)
                            nmaps = hi - lo + 1
                            c, lsc = poisson_scores(naive[pl] * nmaps, y)
                            rows.append(dict(variant="naive_poisson", series_id=s.series_id,
                                             date=s.start, league=s.league,
                                             best_of=s.best_of, market=market, player=pl,
                                             position=pos, y=y, mean=naive[pl] * nmaps,
                                             crps=c, logscore=lsc))
            if ys:
                g_hist.append((s.start, ys, ms))
            if verbose and (i % 200 == 0):
                print(f"[{i}/{len(series)}] {s.start.date()} {time.time() - tic:.0f}s",
                      flush=True)
        # Elo update AFTER projecting (walk-forward)
        for _, gm in s.maps.groupby("map_no"):
            ra = gm[gm["teamname"].astype(str) == s.team_a]
            if len(ra):
                elo.update_map(s.team_a, s.team_b, bool(ra["result"].iloc[0]), s.league)
    return pd.DataFrame(rows)


def pick_best_combo(table: pd.DataFrame) -> str | None:
    """The shrink30+A+B<n> variant whose floor matches the best stand-alone B (by CRPS)."""
    cand = table[table["variant"].str.fullmatch(r"shrink30\+B\d+")]
    if cand.empty:
        return None
    best = cand.sort_values("crps")["variant"].iloc[0]
    return "shrink30+A+" + best.split("+", 1)[1]


def summarize(res: pd.DataFrame) -> dict:
    """Variant table, calibration tables (median line, half-point lines) and tail report."""
    out = []
    for v, g in res.groupby("variant", sort=False):
        row = dict(variant=v, n=len(g), crps=g["crps"].mean(), logscore=g["logscore"].mean(),
                   bias=(g["mean"] - g["y"]).mean())
        if "p_over" in g and g["p_over"].notna().any():
            tp, ty = _tail_lines(g)
            row.update(
                brier_median_line=((g["p_over"] - g["over"]) ** 2).mean(),
                ece_median_line=ece(g["p_over"], g["over"]),
                pred_push=g["p_push"].mean(), obs_push=g["push"].mean(),
                ece_multi_line=ece(*_multi_line(g)),
                tail_pred=tp.mean(), tail_obs=ty.mean(),
            )
        out.append(row)
    table = pd.DataFrame(out)

    # paired CRPS difference vs base with standard errors
    base = res[res["variant"] == "base"].set_index(["series_id", "player"])["crps"]
    diffs = []
    for v in table["variant"]:
        other = res[res["variant"] == v].set_index(["series_id", "player"])["crps"]
        d = (other - base).dropna()
        diffs.append((d.mean(), d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan))
    table["crps_vs_base"] = [d[0] for d in diffs]
    table["crps_vs_base_se"] = [d[1] for d in diffs]

    combo = pick_best_combo(table)
    table["note"] = ""
    if combo is not None and combo in set(table["variant"]):
        table.loc[table["variant"] == combo, "note"] = "shrink30+A+best-B"
        others = [v for v in table["variant"] if v.startswith("shrink30+A+") and v != combo]
        table.loc[table["variant"].isin(others), "note"] = "not selected"

    calib, multi = {}, {}
    model = res[res["variant"] != "naive_poisson"]
    for v, g in model.groupby("variant", sort=False):
        calib[v] = calibration_table(g["p_over"], g["over"])
        multi[v] = calibration_table(*_multi_line(g))
    return {"table": table, "calib": calib, "multi": multi, "tail": tail_report(model),
            "combo": combo}


def tail_report(res: pd.DataFrame) -> pd.DataFrame:
    """Calibration of P(over) at lines >= rounded median + 1.5, by variant and position."""
    parts = []
    for j, off in enumerate(TAIL_OFFSETS):
        parts.append(pd.DataFrame({
            "variant": res["variant"].values, "position": res["position"].values,
            "offset": off, "p": res[f"tl{j}_p"].values, "y": res[f"tl{j}_y"].values}))
    t = pd.concat(parts, ignore_index=True)
    rows = []
    for (v, pos), g in t.groupby(["variant", "position"], sort=False):
        pred, obs, n = g["p"].mean(), g["y"].mean(), len(g)
        rows.append(dict(variant=v, position=pos, n=n, pred=pred, obs=obs, diff=obs - pred,
                         se=np.sqrt(obs * (1 - obs) / n), obs_over_pred=obs / pred
                         if pred > 0 else np.nan))
    for v, g in t.groupby("variant", sort=False):
        pred, obs, n = g["p"].mean(), g["y"].mean(), len(g)
        rows.append(dict(variant=v, position="ALL", n=n, pred=pred, obs=obs, diff=obs - pred,
                         se=np.sqrt(obs * (1 - obs) / n), obs_over_pred=obs / pred))
    return pd.DataFrame(rows)


def _tail_lines(g: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    p = pd.concat([g[f"tl{j}_p"] for j in range(len(TAIL_OFFSETS))], ignore_index=True)
    y = pd.concat([g[f"tl{j}_y"] for j in range(len(TAIL_OFFSETS))], ignore_index=True)
    return p, y


def _multi_line(g: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    p = pd.concat([g[f"hl{j}_p"] for j in range(4)], ignore_index=True)
    y = pd.concat([g[f"hl{j}_y"] for j in range(4)], ignore_index=True)
    ok = p.notna()
    return p[ok].reset_index(drop=True), y[ok].reset_index(drop=True)
