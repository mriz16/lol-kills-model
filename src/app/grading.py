"""Grade logged picks against OE data, settle slips, and summarize the record.

Run by the refresh workflow (`scripts/grade_picks.py`) after new data lands. A pick is
matched to the first series of (team vs opponent) starting between 1 day before and
5 days after it was logged. Kills count only on maps actually played in the market range.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.app.picklog import GRADED_COLUMNS
from src.app.slate import PLATFORMS
from src.config import PRIZEPICKS_FLEX
from src.backtest.metrics import calibration_table
from src.engine.sim import parse_market

WINDOW_BEFORE = pd.Timedelta(days=1)
WINDOW_AFTER = pd.Timedelta(days=5)
VOID_AFTER = pd.Timedelta(days=7)      # still unmatched after this -> void


def _naive_utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo else ts


def grade_picks(picks: pd.DataFrame, players: pd.DataFrame, teams: pd.DataFrame,
                now: datetime | None = None, id_col: str = "pick_id",
                anchor_col: str = "logged_at") -> pd.DataFrame:
    """One graded row per pick: status in win/loss/push/void/pending, actual kills.

    The series is searched around `anchor_col` (the log time for picks, the slate date for
    slate-log rows, so a backfilled slate grades against its own date)."""
    now = _naive_utc(now or datetime.now(timezone.utc))
    graded_at = now.isoformat(timespec="seconds")
    columns = [id_col] + GRADED_COLUMNS[1:]
    if picks.empty:
        return pd.DataFrame(columns=columns)
    t = teams[["gameid", "date", "teamname", "series_id", "map_no", "series_complete"]].copy()
    t["teamname"] = t["teamname"].astype(str)
    t["series_id"] = t["series_id"].astype(str)
    opp = t.groupby("gameid")["teamname"].transform(lambda s: s.iloc[::-1].values)
    t["opponent"] = opp.values
    first = t.groupby(["series_id", "teamname"]).agg(
        start=("date", "min"), opponent=("opponent", "first"),
        complete=("series_complete", "all"), maps=("map_no", "max")).reset_index()
    p = players[["series_id", "map_no", "playername", "teamname", "kills"]].copy()
    for c in ("series_id", "playername", "teamname"):
        p[c] = p[c].astype(str)

    out = []
    for r in picks.to_dict("records"):
        logged = _naive_utc(r[anchor_col])
        row = {id_col: r[id_col], "status": "pending", "actual": np.nan, "series_id": "",
               "graded_at": graded_at}
        cand = first[(first["teamname"] == str(r["team"]))
                     & (first["opponent"] == str(r["opponent"]))
                     & (first["start"] >= logged - WINDOW_BEFORE)
                     & (first["start"] <= logged + WINDOW_AFTER)].sort_values("start")
        if cand.empty:
            if now - logged > VOID_AFTER:
                row["status"] = "void"
            out.append(row)
            continue
        s = cand.iloc[0]
        row["series_id"] = s["series_id"]
        if not s["complete"]:
            if now - logged > VOID_AFTER:
                row["status"] = "void"
            out.append(row)
            continue
        lo, hi = parse_market(str(r["market"]), 5)
        played = list(range(lo, min(hi, int(s["maps"])) + 1))
        mine = p[(p["series_id"] == s["series_id"]) & (p["playername"] == str(r["player"]))
                 & (p["teamname"] == str(r["team"]))]
        by_map = dict(zip(mine["map_no"].astype(int), mine["kills"].astype(int)))
        if not played or any(m not in by_map for m in played):
            row["status"] = "void"            # did not play every map in the range
            out.append(row)
            continue
        actual = int(sum(by_map[m] for m in played))
        row["actual"] = actual
        line = float(r["line"])
        if actual == line:
            row["status"] = "push"
        elif (actual > line) == (str(r["side"]) == "more"):
            row["status"] = "win"
        else:
            row["status"] = "loss"
        out.append(row)
    return pd.DataFrame(out, columns=columns)


def grade_slates(slates: pd.DataFrame, players: pd.DataFrame, teams: pd.DataFrame,
                 now: datetime | None = None) -> pd.DataFrame:
    """Grade every slate-log row (status relative to the logged model side)."""
    return grade_picks(slates, players, teams, now, id_col="row_id", anchor_col="slate_date")


def merge_graded(old: pd.DataFrame, new: pd.DataFrame, id_col: str = "pick_id") -> pd.DataFrame:
    """Keep final grades (win/loss/push/void) from `old`; take everything else from new."""
    final = {"win", "loss", "push", "void"}
    if old is None or old.empty:
        return new.reset_index(drop=True)
    keep = old[old["status"].isin(final)]
    rest = new[~new[id_col].astype(str).isin(set(keep[id_col].astype(str)))]
    return pd.concat([keep, rest], ignore_index=True)[[id_col] + GRADED_COLUMNS[1:]]


def settle_slips(picks: pd.DataFrame, graded: pd.DataFrame) -> pd.DataFrame:
    """Per slip: status and profit. Pushed / void legs drop out; fewer than 2 left ->
    refund. Power: any lost leg loses the slip; a full slip with a goblin / demon leg pays
    its logged payout (what PrizePicks showed), otherwise the table multiplier for the
    remaining legs. Flex (mode "flex"): pays the PrizePicks Flex table for the hits
    among the remaining legs ("partial" when that returns less than the stake).
    (Platform rules for reduced slips vary; this is the approximation.)"""
    if picks.empty:
        return pd.DataFrame(columns=["slip_id", "platform", "mode", "n_legs", "stake",
                                     "status", "profit", "p_joint", "logged_at"])
    df = picks.merge(graded[["pick_id", "status"]], on="pick_id", how="left")
    df["status"] = df["status"].fillna("pending")
    rows = []
    for sid, g in df.groupby("slip_id", sort=False):
        stake = float(g["stake"].iloc[0] or 0)
        platform = str(g["platform"].iloc[0])
        mode = str(g["mode"].iloc[0]) if "mode" in g and pd.notna(g["mode"].iloc[0]) \
            else "power"
        st = set(g["status"])
        live = g[g["status"].isin(["win", "loss"])]
        wins = int((live["status"] == "win").sum())
        if mode != "flex" and "loss" in st:
            status, profit = "loss", -stake
        elif "pending" in st:
            status, profit = "pending", 0.0
        elif len(live) < 2:
            status, profit = "refund", 0.0
        elif mode == "flex":
            mult = PRIZEPICKS_FLEX.get(len(live), {}).get(wins, 0.0)
            status = "win" if mult > 1 else ("partial" if mult > 0 else "loss")
            profit = stake * (mult - 1.0)
        else:
            pay = pd.to_numeric(g["payout"], errors="coerce").iloc[0]
            special = "kind" in g and g["kind"].fillna("standard").ne("standard").any()
            mult = float(pay) if special and len(live) == len(g) and np.isfinite(pay) \
                and pay > 1 else PLATFORMS.get(platform, {}).get(len(live))
            status, profit = "win", stake * (mult - 1.0) if mult else 0.0
        rows.append(dict(slip_id=sid, platform=platform, mode=mode, n_legs=len(g),
                         stake=stake, status=status, profit=profit,
                         p_joint=g["slip_p_joint"].iloc[0], logged_at=g["logged_at"].iloc[0]))
    return pd.DataFrame(rows)


def record_summary(picks: pd.DataFrame, graded: pd.DataFrame) -> dict:
    """Leg record, slip record, ROI and a calibration table of the chosen side's p."""
    legs = picks.merge(graded[["pick_id", "status", "actual"]], on="pick_id", how="left")
    legs["status"] = legs["status"].fillna("pending")
    cnt = legs["status"].value_counts()
    slips = settle_slips(picks, graded)
    done = slips[slips["status"].isin(["win", "loss", "partial", "refund"])]
    staked = float(done["stake"].sum())
    profit = float(done["profit"].sum())
    decided = legs[legs["status"].isin(["win", "loss"])]
    calib = calibration_table(decided["p_side"].astype(float),
                              (decided["status"] == "win").astype(int)) \
        if len(decided) else pd.DataFrame()
    return dict(
        legs_won=int(cnt.get("win", 0)), legs_lost=int(cnt.get("loss", 0)),
        legs_push=int(cnt.get("push", 0)), legs_void=int(cnt.get("void", 0)),
        legs_pending=int(cnt.get("pending", 0)),
        slips_won=int((done["status"] == "win").sum()),
        slips_lost=int((done["status"] == "loss").sum()),
        slips_refund=int((done["status"] == "refund").sum()),
        slips_pending=int((slips["status"] == "pending").sum()),
        staked=staked, profit=profit, roi=profit / staked if staked > 0 else np.nan,
        exp_leg_hit=float(decided["p_side"].astype(float).mean()) if len(decided) else np.nan,
        leg_hit=float((decided["status"] == "win").mean()) if len(decided) else np.nan,
        calib=calib, legs=legs, slips=slips,
    )
