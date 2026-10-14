"""Slate log: every line entered on every slate (played or not) with all model outputs.

Rows live in `logs/slates.csv` on the picks-log branch (same store as the pick log). One
row per (slate date, matchup, platform, market, player, line): re-running a slate replaces
its rows but keeps the `played` flag. Backfilled slates (a past date, projected from data
before that date only) carry `backfill = 1`. The refresh workflow grades every row into
`logs/slates_graded.csv`; that is the primary real-world evaluation set.
"""
from __future__ import annotations

import hashlib
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.app.picklog import ConflictError, _frame

SLATES_PATH = "logs/slates.csv"
SLATES_GRADED_PATH = "logs/slates_graded.csv"

SLATE_COLUMNS = [
    "row_id", "slate_id", "logged_at", "slate_date", "backfill", "data_cutoff",
    "platform", "league", "best_of", "market", "matchup", "team", "opponent", "player",
    "role", "line", "p_map", "calibrated", "side", "p_more", "p_less", "p_push", "p_side",
    "breakeven", "edge", "raw_side", "raw_p_more", "raw_p_less", "raw_p_push", "raw_edge",
    "mean", "median", "fair_line", "n_games", "share", "flags", "played",
]
SLATE_GRADED_COLUMNS = ["row_id", "status", "actual", "series_id", "graded_at"]


def row_key(slate_date: str, r) -> str:
    raw = "|".join(str(x) for x in (slate_date, r["matchup"], r["platform"], r["market"],
                                     r["player"], f"{float(r['line']):g}"))
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def new_slate_id(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return now.strftime("%Y%m%d%H%M") + "-" + uuid.uuid4().hex[:4]


def _r(x, nd=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) \
        else round(float(x), nd)


def make_slate_rows(results: pd.DataFrame, slate_date: str, backfill: bool, cutoff,
                    slate_id: str | None = None, now: datetime | None = None) -> list[dict]:
    """Log rows for every prop of a run slate. `results` = slate.results_table rows."""
    now = now or datetime.now(timezone.utc)
    slate_id = slate_id or new_slate_id(now)
    cut = pd.Timestamp(cutoff).isoformat()
    out = []
    for r in results.to_dict("records"):
        out.append(dict(
            row_id=row_key(slate_date, r), slate_id=slate_id,
            logged_at=now.isoformat(timespec="seconds"), slate_date=slate_date,
            backfill=int(bool(backfill)), data_cutoff=cut, platform=r["platform"],
            league=r["league"], best_of=int(r["best_of"]), market=r["market"],
            matchup=r["matchup"], team=r["team"], opponent=r["opponent"],
            player=r["player"], role=r["role"], line=float(r["line"]),
            p_map=_r(r.get("p_map")), calibrated=int(bool(r.get("calibrated", False))),
            side=r["side"], p_more=_r(r["p_more"]), p_less=_r(r["p_less"]),
            p_push=_r(r["p_push"]), p_side=_r(r["p_side"]), breakeven=_r(r["breakeven"]),
            edge=_r(r["edge"]), raw_side=r.get("raw_side", r["side"]),
            raw_p_more=_r(r.get("raw_p_more", r["p_more"])),
            raw_p_less=_r(r.get("raw_p_less", r["p_less"])),
            raw_p_push=_r(r.get("raw_p_push", r["p_push"])),
            raw_edge=_r(r.get("raw_edge", r["edge"])), mean=_r(r["mean"], 3),
            median=float(r["median"]), fair_line=float(r["fair_line"]),
            n_games=int(r["n_games"]), share=_r(r.get("share")), flags=r.get("flags", ""),
            played=0,
        ))
    return out


def read_slates(store) -> pd.DataFrame:
    df = _frame(store.read(SLATES_PATH)[0], SLATE_COLUMNS)
    df["row_id"] = df["row_id"].astype(str)
    return df


def read_slates_graded(store) -> pd.DataFrame:
    df = _frame(store.read(SLATES_GRADED_PATH)[0], SLATE_GRADED_COLUMNS)
    df["row_id"] = df["row_id"].astype(str)
    return df


def _write(store, fn, msg: str, retries: int = 3) -> pd.DataFrame:
    """Read-modify-write of the slate log, retried on a write conflict."""
    for attempt in range(retries + 1):
        text, sha = store.read(SLATES_PATH)
        df = _frame(text, SLATE_COLUMNS)
        df["row_id"] = df["row_id"].astype(str)
        out = fn(df)
        try:
            store.write(SLATES_PATH, out.to_csv(index=False), msg, sha)
            return out
        except ConflictError:
            if attempt == retries:
                raise
            time.sleep(0.5 * (attempt + 1))
    raise AssertionError("unreachable")


def append_slate(store, rows: list[dict]) -> pd.DataFrame:
    """Add a run slate. Rows with an existing row_id are replaced (played flag kept)."""
    new = pd.DataFrame(rows, columns=SLATE_COLUMNS)

    def fn(df):
        if df.empty:
            return new
        old = df.set_index("row_id")["played"]
        keep = new["row_id"].map(old).fillna(0).astype(int)
        new["played"] = np.maximum(new["played"].astype(int), keep)
        rest = df[~df["row_id"].isin(set(new["row_id"]))]
        return pd.concat([rest, new], ignore_index=True)

    sid = rows[0]["slate_id"] if rows else ""
    return _write(store, fn, f"slates: {len(rows)} line(s), slate {sid}")


def mark_played(store, row_ids, played: bool = True) -> pd.DataFrame:
    ids = {str(x) for x in row_ids}

    def fn(df):
        df = df.copy()
        df.loc[df["row_id"].isin(ids), "played"] = int(played)
        return df

    return _write(store, fn, f"slates: mark {len(ids)} played")


# ------------------------------------------------------------------ evaluation ----
def evaluate(slates: pd.DataFrame, graded: pd.DataFrame) -> dict:
    """Hit rate vs predicted for the model side (calibrated and raw), by side, by played,
    and a calibration table of P(more) vs over at the logged lines."""
    from src.backtest.metrics import calibration_table
    if slates.empty:
        return {"n": 0}
    d = slates.merge(graded[["row_id", "status", "actual"]], on="row_id", how="left")
    d["status"] = d["status"].fillna("pending")
    done = d[d["actual"].notna() & d["status"].isin(["win", "loss", "push"])].copy()
    out = {"n": len(d), "n_graded": len(done),
           "n_pending": int((d["status"] == "pending").sum()),
           "n_void": int((d["status"] == "void").sum()), "rows": d}
    if done.empty:
        return out
    a, line = done["actual"].astype(float), done["line"].astype(float)
    done["over"] = (a > line).astype(int)
    done["under"] = (a < line).astype(int)
    rows = []
    for label, side_col, pm, pl in (("calibrated", "side", "p_more", "p_less"),
                                    ("raw", "raw_side", "raw_p_more", "raw_p_less")):
        p = np.where(done[side_col] == "more", done[pm], done[pl]).astype(float)
        hit = np.where(done[side_col] == "more", done["over"], done["under"])
        for grp, m in (("all", np.ones(len(done), bool)),
                       ("more", (done[side_col] == "more").to_numpy()),
                       ("less", (done[side_col] == "less").to_numpy()),
                       ("played", done["played"].astype(int).to_numpy() == 1),
                       ("backfill", done["backfill"].astype(int).to_numpy() == 1)):
            if m.sum() == 0:
                continue
            rows.append(dict(model=label, group=grp, n=int(m.sum()), pred=p[m].mean(),
                             hit=hit[m].mean(), diff=hit[m].mean() - p[m].mean()))
    out["table"] = pd.DataFrame(rows)
    out["calib"] = calibration_table(done["p_more"].astype(float), done["over"], 0.1)
    out["bias"] = float((done["mean"].astype(float) - a).mean())
    return out
