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
from src.engine.features import EngineConfig, History, build_matchup, fit_kappa
from src.engine.sim import simulate_series

VARIANTS = {
    "base": EngineConfig(),
    "game_script": EngineConfig(game_script=True),
    "shrink10": EngineConfig(shrink_games=10),
    "shrink30": EngineConfig(shrink_games=30),
}


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
    return "Map 1" if best_of == 1 else "Maps 1-2"


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
    lo, hi = (1, 1) if market == "Map 1" else (1, 2)
    vals = kills[lo - 1:hi]
    if len(vals) < hi - lo + 1 or any(v is None for v in vals):
        return None  # player missed a map in the range: prop would be void
    return int(sum(vals))


def run_backtest(players: pd.DataFrame, teams: pd.DataFrame, variants: dict | None = None,
                 warmup_days: int = 75, n_sims: int = 20_000, seed: int = 2024,
                 kappa_refit_days: int = 30, verbose: bool = True) -> pd.DataFrame:
    variants = variants or VARIANTS
    hist = History(players, teams)
    series = build_series(players, teams)
    elo = Elo()
    t0 = series[0].start + pd.Timedelta(days=warmup_days) if series else None
    rows = []
    kappa, kappa_at = None, None
    tic = time.time()
    for i, s in enumerate(series):
        p_map = elo.p_map(s.team_a, s.team_b, s.league)
        if s.start >= t0:
            cutoff = int(s.start.value)
            if kappa is None or (s.start - kappa_at).days >= kappa_refit_days:
                kappa, kappa_at = fit_kappa(hist, cutoff), s.start
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
            for vname, vcfg in variants.items():
                cfg = replace(vcfg, kappa=kappa)
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
                        )
                        # half-point lines around the median for a wider calibration curve
                        for j, off in enumerate((-1.5, -0.5, 0.5, 1.5)):
                            hl = line + off
                            ok = hl > 0
                            rec[f"hl{j}"] = hl
                            rec[f"hl{j}_p"] = float((k > hl).mean()) if ok else np.nan
                            rec[f"hl{j}_y"] = int(y > hl) if ok else np.nan
                        rows.append(rec)
                        if vname == next(iter(variants)):
                            nmaps = 1 if market == "Map 1" else 2
                            c, lsc = poisson_scores(naive[pl] * nmaps, y)
                            rows.append(dict(variant="naive_poisson", series_id=s.series_id,
                                             date=s.start, league=s.league,
                                             best_of=s.best_of, market=market, player=pl,
                                             position=pos, y=y, mean=naive[pl] * nmaps,
                                             crps=c, logscore=lsc))
            if verbose and (i % 200 == 0):
                print(f"[{i}/{len(series)}] {s.start.date()} {time.time() - tic:.0f}s",
                      flush=True)
        # Elo update AFTER projecting (walk-forward)
        for _, gm in s.maps.groupby("map_no"):
            ra = gm[gm["teamname"].astype(str) == s.team_a]
            if len(ra):
                elo.update_map(s.team_a, s.team_b, bool(ra["result"].iloc[0]), s.league)
    return pd.DataFrame(rows)


def summarize(res: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """Variant table, calibration tables at the median line, and multi-line calibration."""
    out = []
    for v, g in res.groupby("variant", sort=False):
        row = dict(variant=v, n=len(g), crps=g["crps"].mean(), logscore=g["logscore"].mean(),
                   bias=(g["mean"] - g["y"]).mean())
        if "p_over" in g and g["p_over"].notna().any():
            row.update(
                brier_median_line=((g["p_over"] - g["over"]) ** 2).mean(),
                ece_median_line=ece(g["p_over"], g["over"]),
                pred_over=g["p_over"].mean(), obs_over=g["over"].mean(),
                pred_push=g["p_push"].mean(), obs_push=g["push"].mean(),
            )
            ml_p, ml_y = _multi_line(g)
            row["ece_multi_line"] = ece(ml_p, ml_y)
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

    calib, multi = {}, {}
    for v, g in res[res["variant"] != "naive_poisson"].groupby("variant", sort=False):
        calib[v] = calibration_table(g["p_over"], g["over"])
        multi[v] = calibration_table(*_multi_line(g))
    return table, calib, multi


def _multi_line(g: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    p = pd.concat([g[f"hl{j}_p"] for j in range(4)], ignore_index=True)
    y = pd.concat([g[f"hl{j}_y"] for j in range(4)], ignore_index=True)
    ok = p.notna()
    return p[ok].reset_index(drop=True), y[ok].reset_index(drop=True)
