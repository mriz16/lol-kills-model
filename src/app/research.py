"""Player research: real kill distributions, floors / ceilings, boom-bust archetypes.

Everything here is descriptive (what happened), next to the model's simulated
distribution for the slate. No Streamlit here; the Players tab in `streamlit_app.py`
draws the tables and charts.

- market_totals: one row per series with the player's kills over a market range
  (Map 1 / Maps 1-2 / Maps 1-3), with the series result.
- dist_summary: floors (most kills reached in 80% / 90% of series), median, ceilings
  (top 20% / top 10%), mean, spread.
- line_ladder: share of series over every x.5 line, overall and in wins / losses.
- archetypes: per-map kill spread vs position peers -> BOOM/BUST, BALANCED, STEADY,
  plus how much the player's kills depend on winning (WIN-RELIANT / RESULT-PROOF).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C

# market -> (first map, last map, minimum best-of)
MARKETS = {"Map 1": (1, 1, 1), "Maps 1-2": (1, 2, 3), "Maps 1-3": (1, 3, 5)}
FLOOR_QS = (0.80, 0.90)
ARCHETYPE_DAYS = 150         # peer window for archetypes
ARCHETYPE_MIN_MAPS = 20
ARCHETYPE_SPLIT = (1 / 3, 2 / 3)
# Labels are shown only for roles where a player's spread carries over to the next 150
# days (scripts/research_report.py, reports/player_research.md, 2026-10-28): the same
# label again 53% / 45% (bot BOOM/BUST / STEADY), 44% / 44% (mid), 35% / 43% (jng) vs 33%
# by chance; top and support ~ chance, so they get no label. The win / loss ratio
# repeats ~chance too: shown as numbers, never as a label.
LABEL_ROLES = ("bot", "mid", "jng")


def market_totals(players: pd.DataFrame, player: str, market: str,
                  team: str | None = None) -> pd.DataFrame:
    """Per series: date, team, opponent, league, best_of, won (series), kills over the
    market range, maps (played in range), newest first. Series where the player missed a
    map of the range are dropped."""
    lo, hi, min_bo = MARKETS[market]
    p = players[players["playername"].astype(str) == player]
    if team:
        p = p[p["teamname"].astype(str) == team]
    if "series_complete" in p:
        p = p[p["series_complete"].astype(bool)]
    p = p[p["best_of"] >= min_bo]
    if p.empty:
        return pd.DataFrame(columns=["date", "team", "opponent", "league", "best_of",
                                     "won", "kills", "maps"])
    sid = p["series_id"].astype(str)
    res = p.groupby(sid)["result"].agg(["sum", "size"])
    won = (res["sum"] > res["size"] - res["sum"]).rename("won")
    r = p[(p["map_no"] >= lo) & (p["map_no"] <= hi)]
    rs = r["series_id"].astype(str)
    g = r.groupby(rs)
    out = pd.DataFrame({"date": g["date"].min(), "team": g["teamname"].first().astype(str),
                        "league": g["league"].first().astype(str),
                        "best_of": g["best_of"].first().astype(int),
                        "kills": g["kills"].sum().astype(int), "maps": g.size()})
    out = out[out["maps"] == hi - lo + 1]     # every map of the range is always played
    out["won"] = won.reindex(out.index).fillna(False).astype(bool)
    out["opponent"] = _opponents(players, out.index, out["team"])
    return out.sort_values("date", ascending=False).reset_index(names="series_id")


def all_market_totals(players: pd.DataFrame, market: str) -> pd.DataFrame:
    """Every player's series totals of a market: series_id, player, date, kills (same
    rules as market_totals, no opponents / results)."""
    lo, hi, min_bo = MARKETS[market]
    p = players
    if "series_complete" in p:
        p = p[p["series_complete"].astype(bool)]
    p = p[(p["best_of"] >= min_bo) & (p["map_no"] >= lo) & (p["map_no"] <= hi)]
    g = p.groupby([p["series_id"].astype(str).rename("series_id"),
                   p["playername"].astype(str).rename("player")])
    out = pd.DataFrame({"date": g["date"].min(), "kills": g["kills"].sum(), "maps": g.size()})
    out = out[out["maps"] == hi - lo + 1].drop(columns="maps").reset_index()
    return out.sort_values(["player", "date"]).reset_index(drop=True)


def prior_form(players: pd.DataFrame, market: str, n_last: int = 20) -> pd.DataFrame:
    """Per (series_id, player) of the market: n_prev (previous series, capped at n_last)
    and avg_prev (their mean), using only series that started strictly earlier."""
    t = all_market_totals(players, market)
    out = []
    for pl, g in t.groupby("player", sort=False):
        d, k = g["date"].to_numpy(), g["kills"].to_numpy(float)
        cs = np.concatenate([[0.0], np.cumsum(k)])
        j = np.searchsorted(d, d, side="left")           # series strictly before
        lo = np.maximum(j - n_last, 0)
        n = j - lo
        with np.errstate(invalid="ignore", divide="ignore"):
            avg = np.where(n > 0, (cs[j] - cs[lo]) / np.maximum(n, 1), np.nan)
        out.append(pd.DataFrame({"series_id": g["series_id"].to_numpy(), "player": pl,
                                 "n_prev": n, "avg_prev": avg}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["series_id", "player", "n_prev", "avg_prev"])


def recent_avg(players: pd.DataFrame, player: str, market: str,
               n_last: int = 20) -> tuple[float, int]:
    """(mean of the player's latest n_last series of the market, how many)."""
    t = market_totals(players, player, market).head(n_last)
    return (float(t["kills"].mean()) if len(t) else np.nan), int(len(t))


def _opponents(players: pd.DataFrame, sids, teams) -> list[str]:
    s = players[players["series_id"].astype(str).isin(set(sids))]
    names: dict = {}
    for sid, tm in zip(s["series_id"].astype(str), s["teamname"].astype(str)):
        names.setdefault(sid, set()).add(tm)
    out = []
    for sid, tm in zip(sids, teams):
        others = sorted(t for t in names.get(sid, set()) if t != tm)
        out.append(others[0] if others else "")
    return out


def floors_from_counts(values) -> dict:
    """{"floor80", "floor90"}: the most kills reached in >= 80% / 90% of the values."""
    v = np.asarray(values, dtype=np.int64)
    if len(v) == 0:
        return {f"floor{int(100 * q)}": np.nan for q in FLOOR_QS}
    counts = np.bincount(v)
    at_least = 1.0 - np.concatenate([[0.0], np.cumsum(counts)[:-1]]) / len(v)
    return {f"floor{int(round(100 * q))}": int(np.nonzero(at_least >= q - 1e-12)[0].max())
            for q in FLOOR_QS}


def dist_summary(values) -> dict:
    """n, mean, sd, floor90, floor80, median, ceil80 (top-20% mark: reached in 20% of
    values), ceil90 (top 10%), max."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return dict(n=0, mean=np.nan, sd=np.nan, floor90=np.nan, floor80=np.nan,
                    median=np.nan, ceil80=np.nan, ceil90=np.nan, max=np.nan)
    fl = floors_from_counts(v)
    counts = np.bincount(v.astype(np.int64))
    at_least = 1.0 - np.concatenate([[0.0], np.cumsum(counts)[:-1]]) / len(v)
    ce = {"ceil80": int(np.nonzero(at_least >= 0.20 - 1e-12)[0].max()),
          "ceil90": int(np.nonzero(at_least >= 0.10 - 1e-12)[0].max())}
    return dict(n=int(len(v)), mean=float(v.mean()),
                sd=float(v.std(ddof=1)) if len(v) > 1 else np.nan,
                floor90=fl["floor90"], floor80=fl["floor80"], median=float(np.median(v)),
                ceil80=ce["ceil80"], ceil90=ce["ceil90"], max=int(v.max()))


def sim_summary(kills: np.ndarray) -> dict:
    """dist_summary of the model's simulated market totals."""
    return dist_summary(kills)


def line_ladder(values, won=None, lines=None, sim: np.ndarray | None = None) -> pd.DataFrame:
    """Share of values over each x.5 line (all, in wins, in losses) and the model's
    P(over) from `sim` when given."""
    v = np.asarray(values, dtype=float)
    if lines is None:
        top = max(float(np.max(v)) if len(v) else 0.0,
                  float(np.quantile(sim, 0.95)) if sim is not None else 0.0)
        lines = np.arange(0.5, max(top, 1.0) + 0.5, 1.0)
    rows = []
    w = np.asarray(won, dtype=bool) if won is not None else None
    for line in lines:
        r = {"line": float(line), "n": int(len(v)),
             "over": float((v > line).mean()) if len(v) else np.nan}
        if w is not None and len(v):
            r["over_win"] = float((v[w] > line).mean()) if w.any() else np.nan
            r["over_loss"] = float((v[~w] > line).mean()) if (~w).any() else np.nan
        if sim is not None:
            r["model_over"] = float((np.asarray(sim) > line).mean())
        rows.append(r)
    return pd.DataFrame(rows)


def pmf_compare(values, sim: np.ndarray | None, max_k: int | None = None) -> pd.DataFrame:
    """Long frame (kills, source, share) of the real distribution and the model's."""
    v = np.asarray(values, dtype=np.int64)
    hi = max_k if max_k is not None else int(max(v.max() if len(v) else 0,
                                                 np.quantile(sim, 0.99) if sim is not None
                                                 else 0))
    ks = np.arange(hi + 1)
    rows = []
    if len(v):
        c = np.bincount(np.minimum(v, hi), minlength=hi + 1) / len(v)
        rows += [dict(kills=int(k), source="actual", share=float(x)) for k, x in zip(ks, c)]
    if sim is not None:
        s = np.asarray(sim, dtype=np.int64)
        c = np.bincount(np.minimum(s, hi), minlength=hi + 1) / len(s)
        rows += [dict(kills=int(k), source="model", share=float(x)) for k, x in zip(ks, c)]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------- archetypes ----
def map_stats(players: pd.DataFrame, asof=None, days: int = ARCHETYPE_DAYS,
              min_maps: int = ARCHETYPE_MIN_MAPS) -> pd.DataFrame:
    """Per player (latest team, main position) over the last `days` before asof (default:
    the data end): maps, mean kills per map, VMR (variance / mean: spread beyond what
    the average alone implies), floor / ceiling per map, mean in wins / losses and their
    ratio. In-scope leagues only."""
    p = players[players["position"].astype(str).isin(C.POSITIONS)]
    end = pd.Timestamp(asof) if asof is not None else pd.Timestamp(p["date"].max())
    p = p[(p["date"] < end + pd.Timedelta(days=1)) & (p["date"] >= end - pd.Timedelta(days=days))]
    p = p.assign(playername=p["playername"].astype(str), position=p["position"].astype(str),
                 teamname=p["teamname"].astype(str))
    rows = []
    for pl, g in p.groupby("playername"):
        if len(g) < min_maps:
            continue
        k = g["kills"].to_numpy(float)
        res = g["result"].to_numpy(int) == 1
        mean = k.mean()
        fl = floors_from_counts(k)
        mw = k[res].mean() if res.any() else np.nan
        ml = k[~res].mean() if (~res).any() else np.nan
        g = g.sort_values("date")
        rows.append(dict(player=pl, team=g["teamname"].iloc[-1],
                         role=g["position"].value_counts().index[0], maps=len(k),
                         mean=mean, vmr=k.var(ddof=1) / mean if mean > 0 else np.nan,
                         floor80=fl["floor80"], ceil90=int(np.quantile(k, 0.9)),
                         p_zero=float((k == 0).mean()), mean_win=mw, mean_loss=ml,
                         wl_ratio=mw / ml if ml and ml > 0 else np.nan))
    return pd.DataFrame(rows)


def archetypes(stats: pd.DataFrame, label_roles=LABEL_ROLES) -> pd.DataFrame:
    """Adds spread_pct / wl_pct (percentile within role) and type: BOOM/BUST (top third
    of spread in the role), STEADY (bottom third), BALANCED; "" for roles not in
    label_roles (their spread does not carry over)."""
    if stats.empty:
        return stats.assign(spread_pct=[], wl_pct=[], type=[])
    s = stats.copy()
    s["spread_pct"] = s.groupby("role")["vmr"].rank(pct=True)
    s["wl_pct"] = s.groupby("role")["wl_ratio"].rank(pct=True)
    lo, hi = ARCHETYPE_SPLIT
    t = np.where(s["spread_pct"] > hi, "BOOM/BUST",
                 np.where(s["spread_pct"] <= lo, "STEADY", "BALANCED"))
    s["type"] = np.where(s["role"].isin(label_roles), t, "")
    return s


def persistence(players: pd.DataFrame, days: int = ARCHETYPE_DAYS, asof=None) -> dict:
    """Do the measures carry over? Last `days` vs the `days` before: Spearman correlation
    per role of mean / vmr / wl_ratio / p_zero, and P(same type again) per role (all
    roles labelled)."""
    end = pd.Timestamp(asof) if asof is not None else pd.Timestamp(players["date"].max())
    a = archetypes(map_stats(players, asof=end, days=days), label_roles=C.POSITIONS)
    b = archetypes(map_stats(players, asof=end - pd.Timedelta(days=days), days=days),
                   label_roles=C.POSITIONS)
    m = a.merge(b, on="player", suffixes=("", "_prev"))
    corr = {c: m.groupby("role").apply(
        lambda g, c=c: g[c].corr(g[c + "_prev"], method="spearman"), include_groups=False)
        for c in ("mean", "vmr", "wl_ratio", "p_zero")}
    same = (m["type"] == m["type_prev"]).groupby([m["role"], m["type_prev"]]).mean()
    return {"n": len(m), "corr": pd.DataFrame(corr), "same_type": same.unstack()}
