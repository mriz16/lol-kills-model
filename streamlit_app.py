"""LoL player-kills props: slate entry -> Monte Carlo -> edges, slips and a pick log.

Run locally:  streamlit run streamlit_app.py
Deployed from `main` on Streamlit Community Cloud (see README.md).
"""
from __future__ import annotations

import json
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402
from src.app import grading, picklog, slatelog  # noqa: E402
from src.data import databranch  # noqa: E402
from src.data.upload import UploadError, process_upload  # noqa: E402
from src.app.names import search_teams  # noqa: E402
from src.app.ocr import ocr_image, parse_board_text, props_to_lines  # noqa: E402
from src.app.slate import (MARKET_BY_FORMAT, MIN_EDGE, PLATFORMS, SAFE_FLOOR_P,  # noqa: E402
                           THIN_EDGE, DataStore, MatchupInput, check_matchup, lineups_for,
                           per_leg_breakeven, rebreakeven, results_table, run_matchup,
                           safest_overs, side_balance, slip_summary)
from src.engine.features import build_matchup  # noqa: E402
from src.engine.sim import SeriesSim, simulate_series  # noqa: E402

st.set_page_config(page_title="LoL Kills Props", page_icon="🎯", layout="centered",
                   initial_sidebar_state="collapsed")

FIELDS = ["q_a", "team_a", "q_b", "team_b", "bo", "platform", "league", "ml_a", "ml_b",
          "m1_a", "m1_b", "lines"]
LEAGUE_AUTO = "auto"


# ============================================================== data (cached) ====
# Match data: the latest commit of the `data` branch (uploads from this app, via the
# GitHub API), else an upload kept on this container's disk, else the parquet bundled in
# main. VERSION = data key + calibration files, the cache key of the store and the sims.
LOCAL_DATA = ROOT / ".data_cache"


def _secret(section: str, key: str, default=None):
    try:
        return st.secrets[section][key]
    except Exception:
        return default


@st.cache_resource
def get_data_branch():
    """(GitHubDataBranch, True) with the [github] secrets, else (LocalDataBranch, False)."""
    token, repo = _secret("github", "token"), _secret("github", "repo")
    if token and repo:
        return databranch.GitHubDataBranch(
            token, repo, _secret("github", "data_branch", databranch.DATA_BRANCH)), True
    return databranch.LocalDataBranch(LOCAL_DATA), False


@st.cache_data(ttl=300, show_spinner=False)
def data_head(nonce: int) -> tuple[str | None, str | None]:
    """(commit sha of the data branch, error). Re-checked every 5 min so an upload from
    another device shows up without a restart."""
    branch, remote = get_data_branch()
    if not remote:
        return None, None
    try:
        return branch.head(), None
    except Exception as exc:
        return None, f"Could not reach the data branch: {exc}"


def calib_version() -> str:
    files = [C.DATA_DIR / "calib_pairs.parquet", C.DATA_DIR / "spread_rows.parquet",
             C.DATA_DIR / "w_rows.parquet", C.DATA_DIR / "w_lambda_real.json"]
    return "-".join(str(int(f.stat().st_mtime)) + str(f.stat().st_size) for f in files
                    if f.exists())


def data_key() -> str:
    """branch:<sha> | local:<hash> | bundled:<mtime>."""
    local = databranch.LocalDataBranch(LOCAL_DATA)
    if ss().get("data_override") == "local" and local.head():
        return f"local:{local.head()}"
    sha, err = data_head(ss()["data_nonce"])
    if err:
        ss()["data_error"] = err
    if sha:
        return f"branch:{sha}"
    if local.head():
        return f"local:{local.head()}"
    f = C.DATA_DIR / "teams.parquet"
    return f"bundled:{int(f.stat().st_mtime)}{f.stat().st_size}"


def data_version() -> str:
    return data_key() + "|" + calib_version()


@st.cache_data(show_spinner="Loading match data...", max_entries=2)
def load_frames(version: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    kind, _, ref = version.split("|")[0].partition(":")
    if kind == "branch":
        frames = databranch.load_frames(get_data_branch()[0], ref)
        if frames is not None:
            return frames
    if kind == "local":
        frames = databranch.load_frames(databranch.LocalDataBranch(LOCAL_DATA))
        if frames is not None:
            return frames
    return (pd.read_parquet(C.DATA_DIR / "players.parquet"),
            pd.read_parquet(C.DATA_DIR / "teams.parquet"))


@st.cache_resource(show_spinner="Building team pools...", max_entries=4)
def get_store(version: str, asof: str | None = None) -> DataStore:
    """asof (YYYY-MM-DD): backfill store built from games before that date only."""
    return DataStore(*load_frames(version), asof=asof)


@st.cache_data(show_spinner=False, max_entries=128)
def cached_sim(version: str, asof: str | None, team_a: str, team_b: str, best_of: int,
               league: str, p_map: float, lineup_a: tuple, lineup_b: tuple, n_sims: int,
               seed: int):
    """Matchup pools + 20k-sim arrays, cached per (data version, asof, matchup inputs)."""
    store = get_store(version, asof)
    params = build_matchup(store.hist, team_a, team_b, league, best_of, p_map,
                           (list(lineup_a), list(lineup_b)), store.cutoff, store.patch,
                           store.config())
    sim = simulate_series(params, n_sims=n_sims, rng=seed)
    # int16 keeps the cached arrays small (kills never exceed 32k)
    return SeriesSim(params, sim.a_wins, sim.played, sim.team_kills.astype(np.int16),
                     sim.player_kills.astype(np.int16)), params


def app_sim_fn(store, m, lines, odds, league, n_sims, seed):
    lu = lineups_for(store, m, lines)
    return cached_sim(VERSION, ASOF, m.team_a, m.team_b, int(m.best_of), league,
                      round(float(odds["p_map_a"]), 6), tuple(lu[0]), tuple(lu[1]),
                      n_sims, seed)


# ============================================================== pick log store ====
@st.cache_resource
def get_log_store():
    token = _secret("github", "token")
    repo = _secret("github", "repo")
    if token and repo:
        return picklog.GitHubStore(token, repo, _secret("github", "branch",
                                                        picklog.LOG_BRANCH)), True
    return picklog.LocalStore(ROOT), False


@st.cache_data(ttl=120, show_spinner="Reading pick log...")
def load_log(nonce: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    store, _ = get_log_store()
    return picklog.read_picks(store), picklog.read_graded(store)


@st.cache_data(ttl=120, show_spinner="Reading slate log...")
def load_slate_log(nonce: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    store, _ = get_log_store()
    return slatelog.read_slates(store), slatelog.read_slates_graded(store)


def log_slate(store: DataStore, df: pd.DataFrame):
    """Every line of a run slate goes to the slate log (played or not)."""
    meta = ss()["slate_meta"]
    rows = slatelog.make_slate_rows(df, meta["slate_date"], store.backfill, store.cutoff,
                                    slate_id=meta["slate_id"])
    log_store, remote = get_log_store()
    try:
        slatelog.append_slate(log_store, rows)
        ss()["log_nonce"] += 1
        if remote:
            st.caption(f"Slate log: {len(rows)} line(s) saved to GitHub for "
                       f"{meta['slate_date']}.")
        else:
            st.error(f"Slate NOT saved to GitHub ({len(rows)} lines kept on the app's "
                     "temporary disk, wiped on restart): no GitHub secrets. Use the "
                     "download below, and see Record → Log connection.")
    except Exception as exc:
        st.error(f"Could not save the slate log: {exc}. Use the download below, and see "
                 "Record → Log connection.")
    st.download_button("Download this slate (CSV)", pd.DataFrame(rows).to_csv(index=False),
                       file_name=f"slate_{meta['slate_date']}_{meta['slate_id']}.csv",
                       mime="text/csv", key=f"dl_slate_{meta['slate_id']}")


def mark_played(row_ids: list[str]) -> bool:
    log_store, _ = get_log_store()
    try:
        slatelog.mark_played(log_store, row_ids)
        ss()["log_nonce"] += 1
        return True
    except Exception as exc:
        st.error(f"Could not update the slate log: {exc}")
        return False


# ============================================================== slate state ====
def ss():
    return st.session_state


def init_state():
    ss().setdefault("blocks", [0])
    ss().setdefault("next_id", 1)
    ss().setdefault("saved", {})
    ss().setdefault("results", None)
    ss().setdefault("sims", {})
    ss().setdefault("log_nonce", 0)
    ss().setdefault("slate_meta", None)
    ss().setdefault("data_nonce", 0)
    ss().setdefault("upload_summary", None)


def block_values(bid: int) -> dict:
    return {f: ss().get(f"{f}_{bid}") for f in FIELDS}


def set_block(bid: int, vals: dict):
    for f in FIELDS:
        if f in vals and vals[f] is not None:
            ss()[f"{f}_{bid}"] = vals[f]


def block_input(bid: int) -> MatchupInput:
    v = block_values(bid)
    league = v["league"] if v["league"] and v["league"] != LEAGUE_AUTO else ""
    return MatchupInput(team_a=v["team_a"] or "", team_b=v["team_b"] or "",
                        best_of=int(v["bo"] or 3), platform=v["platform"] or "PrizePicks",
                        ml_a=v["ml_a"] or "", ml_b=v["ml_b"] or "", map1_a=v["m1_a"] or "",
                        map1_b=v["m1_b"] or "", lines=v["lines"] or "", league=league)


def slate_dicts() -> list[dict]:
    return [block_values(b) for b in ss()["blocks"]]


def load_slate(blocks: list[dict]):
    for b in list(ss()["blocks"]):
        for f in FIELDS:
            ss().pop(f"{f}_{b}", None)
    ids = []
    for vals in blocks:
        bid = ss()["next_id"]
        ss()["next_id"] += 1
        set_block(bid, vals)
        ids.append(bid)
    ss()["blocks"] = ids or [0]


def add_block():
    bid = ss()["next_id"]
    ss()["next_id"] += 1
    ss()["blocks"].append(bid)


def remove_block(bid: int):
    ss()["blocks"] = [b for b in ss()["blocks"] if b != bid]
    if not ss()["blocks"]:
        add_block()


def pct(x) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.1%}"


# ============================================================== slate tab ====
def team_picker(store: DataStore, bid: int, side: str, label: str):
    q = st.text_input(f"{label} search", key=f"q_{side}_{bid}",
                      placeholder="e.g. GEN, HLE, Bilibili")
    opts = search_teams(q, store.team_names) if q else store.team_names
    key = f"team_{side}_{bid}"
    if ss().get(key) not in opts:
        ss().pop(key, None)
    if not opts:
        st.caption(f"No team matches '{q}'.")
        return
    st.selectbox(label, opts, key=key, label_visibility="collapsed")


def render_block(store: DataStore, bid: int, n: int):
    m = block_input(bid)
    title = m.key if m.team_a and m.team_b else f"Matchup {n}"
    with st.expander(title, expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            team_picker(store, bid, "a", "Team A")
        with c2:
            team_picker(store, bid, "b", "Team B")
        c1, c2 = st.columns(2)
        with c1:
            st.radio("Format", [1, 3, 5], index=1, key=f"bo_{bid}", horizontal=True,
                     format_func=lambda x: f"BO{x}")
        with c2:
            st.radio("Platform", list(PLATFORMS), key=f"platform_{bid}", horizontal=True)
        m = block_input(bid)
        leagues = [LEAGUE_AUTO] + C.LEAGUES
        auto = store.infer_league(m.team_a, m.team_b) if m.team_a and m.team_b else ""
        st.selectbox("League / event", leagues, key=f"league_{bid}",
                     format_func=lambda x: f"auto ({auto})" if x == LEAGUE_AUTO and auto
                     else x)

        st.markdown("**Odds (American)**")
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("Series ML Team A", key=f"ml_a_{bid}", placeholder="-150")
            st.text_input("Map 1 ML Team A (optional)", key=f"m1_a_{bid}")
        with c2:
            st.text_input("Series ML Team B", key=f"ml_b_{bid}", placeholder="+120")
            st.text_input("Map 1 ML Team B (optional)", key=f"m1_b_{bid}")
        m = block_input(bid)
        lines, odds, problems = check_matchup(store, m)
        if odds["error"]:
            if any(block_values(bid)[f] for f in ("ml_a", "ml_b", "m1_a", "m1_b")):
                st.warning(odds["error"])
        else:
            src = "Map 1 ML" if odds["source"] == "map1" else "series ML"
            st.caption(f"De-vigged series: {m.team_a or 'A'} {pct(odds['p_series_a'])} / "
                       f"{m.team_b or 'B'} {pct(odds['p_series_b'])} · per map: "
                       f"{pct(odds['p_map_a'])} / {pct(odds['p_map_b'])} (from {src})")

        st.text_area(f"Lines ({m.market}): one per line, \"player line\"", key=f"lines_{bid}",
                     placeholder="Chovy 10.5\nCanyon 8.5\nChovy 7.5 g", height=140,
                     help="Add \"g\" after a goblin line and \"d\" after a demon line "
                          "(MORE only). A player can have a standard and a goblin line.")
        if lines:
            ok = [lm for lm in lines if lm.ok]
            bad = [lm for lm in lines if not lm.ok]
            if ok:
                st.caption("Matched: " + " · ".join(
                    f"{lm.player} ({lm.team}, {lm.role}) {lm.line:g}"
                    + ("" if lm.kind == "standard" else f" {lm.kind}") for lm in ok))
            for lm in bad:
                cands = ", ".join(f"{c[0]} ({c[1]}, {c[2]})" for c in lm.candidates[:3])
                if lm.status == "bad line":
                    st.error(f"Can't read \"{lm.raw.strip()}\": use \"player 10.5\".")
                elif lm.status == "ambiguous":
                    st.warning(f"Ambiguous \"{lm.query}\": {cands}. Type the full name.")
                else:
                    st.error(f"No match for \"{lm.query}\" on these rosters."
                             + (f" Closest: {cands}." if cands else ""))
        st.button("Remove matchup", key=f"rm_{bid}", on_click=remove_block, args=(bid,))


def slate_tab(store: DataStore):
    for i, bid in enumerate(list(ss()["blocks"]), 1):
        render_block(store, bid, i)
    c1, c2 = st.columns(2)
    c1.button("+ Add matchup", on_click=add_block, width="stretch", key="add_block")
    run = c2.button("Run slate", type="primary", width="stretch", key="run_slate")
    if run:
        run_slate(store)

    with st.expander("Save / load slates"):
        name = st.text_input("Slate name", value=time.strftime("Slate %H:%M"))
        if st.button("Save current slate"):
            ss()["saved"][name] = slate_dicts()
            st.success(f"Saved \"{name}\" ({len(ss()['blocks'])} matchups) for this session.")
        if ss()["saved"]:
            pick = st.selectbox("Saved slates", list(ss()["saved"]))
            st.button("Load", on_click=load_slate, args=(ss()["saved"][pick],))
        st.download_button("Download slate (.json)", json.dumps(slate_dicts(), indent=1),
                           file_name="slate.json", mime="application/json")
        up = st.file_uploader("Load slate from .json", type=["json"], key="slate_upload")
        if up is not None:
            st.button("Load uploaded slate", on_click=load_slate,
                      args=(json.loads(up.getvalue()),))


def run_slate(store: DataStore):
    legs = ss().get("be_legs", 2)
    results, skipped = [], []
    tic = time.time()
    with st.spinner("Simulating 20,000 series per matchup..."):
        for bid in ss()["blocks"]:
            m = block_input(bid)
            _, _, problems = check_matchup(store, m)
            if problems:
                skipped.append(f"{m.key if m.team_a else 'empty block'}: {'; '.join(problems)}")
                continue
            results.append(run_matchup(store, m, legs, sim_fn=app_sim_fn))
    for s in skipped:
        st.error(f"Not run: {s}")
    if results:
        df = results_table(results)
        slate_date = (store.asof or pd.Timestamp.now(tz="UTC").tz_localize(None)).date()
        ss()["slate_meta"] = dict(slate_date=slate_date.isoformat(),
                                  slate_id=slatelog.new_slate_id(), backfill=store.backfill)
        df["row_id"] = [slatelog.row_key(slate_date.isoformat(), r)
                        for r in df.to_dict("records")]
        ss()["results"] = df
        ss()["sims"] = {r.inp.key: r.sim for r in results}
        st.success(f"Ran {len(results)} matchup(s), {len(ss()['results'])} props in "
                   f"{time.time() - tic:.1f}s. Open the Results tab.")
        log_slate(store, df)


# ============================================================== screenshot tab ====
def apply_ocr_lines(target: int, text: str, replace: bool):
    key = f"lines_{target}"
    cur = ss().get(key) or ""
    ss()[key] = text if replace or not cur.strip() else cur.rstrip() + "\n" + text


def screenshot_tab(store: DataStore):
    st.caption("Beta. Reads PrizePicks board cards into a matchup's lines box. Check the "
               "result: nothing runs until you press Run slate.")
    up = st.file_uploader("Board screenshot", type=["png", "jpg", "jpeg", "webp"])
    cols = st.radio("Card columns in the screenshot", [1, 2], index=1, horizontal=True)
    if up is not None and st.button("Read screenshot"):
        try:
            raw = ocr_image(up.getvalue(), columns=cols)
        except Exception as exc:  # tesseract missing, bad image ...
            st.error(f"OCR failed: {exc}")
            raw = ""
        props = parse_board_text(raw)
        ss()["ocr_raw"] = raw
        ss()["ocr_lines"] = props_to_lines(props)
        ss()["ocr_meta"] = [p.__dict__ for p in props]
    if "ocr_lines" not in ss():
        return
    meta = ss().get("ocr_meta", [])
    if meta:
        mk = sorted({p["market"] for p in meta})
        teams = sorted({t for p in meta for t in (p["team"], p["opponent"]) if t})
        st.caption(f"Found {len(meta)} card(s). Markets: {', '.join(mk)}. "
                   f"Team tags: {', '.join(teams) or '–'}.")
    else:
        st.warning("No cards recognized. Check the raw text below, or type the lines.")
    st.text_area("Parsed lines (edit before using)", key="ocr_lines", height=160)
    labels = {b: (block_input(b).key if block_input(b).team_a else f"Matchup {i}")
              for i, b in enumerate(ss()["blocks"], 1)}
    target = st.selectbox("Send to matchup", list(labels), format_func=labels.get)
    if meta:
        want = MARKET_BY_FORMAT[block_input(target).best_of]
        other = [p["market"] for p in meta if p["market"] != want]
        if other:
            st.warning(f"Cards show {', '.join(sorted(set(other)))} but this matchup's "
                       f"format prices {want}. Check the format.")
    replace = st.checkbox("Replace existing lines", value=False)
    st.button("Use these lines", on_click=apply_ocr_lines,
              args=(target, ss().get("ocr_lines", ""), replace))
    with st.expander("Raw OCR text"):
        st.text(ss().get("ocr_raw", ""))


# ============================================================== results tab ====
FLAG_HELP = {
    "BELOW MIN": f"shown (W-adjusted) edge under the {100 * MIN_EDGE:.0f}-point minimum: "
                 "after W the top edges per slate still carry ~1 point of noise (se); "
                 f"{100 * MIN_EDGE:.0f} points clears it by 2 se",
    "THIN EDGE": f"edge under {100 * THIN_EDGE:.0f} points vs the per-leg breakeven "
                 "(uncalibrated: no minimum edge applied)",
    "LOW SAMPLE": "player has fewer than 20 games in the model window",
    "CAUTION": "'less' at a whole-number line with P(more) < 25%: in the backtest those "
               "overs hit about 9 points more often than predicted",
    "GOBLIN": "goblin line: MORE only, and PrizePicks pays less than the standard power "
              "multiplier, so the edge shown (vs the standard breakeven) is too high and it "
              "is never a PLAY. Judge it on P(more) and the floor; set the real payout in "
              "the Slip tab",
    "DEMON": "demon line: MORE only, boosted payout not in the edge (never a PLAY). Set "
             "the real payout in the Slip tab",
    "SAFE FLOOR": f"shown P(more) ≥ {100 * SAFE_FLOOR_P:.0f}% (not a support): in the "
                  "backtest P(more) at lines 2+ kills under the market line was calibrated "
                  "within 1-3 pts at 85-95%, and the shown P sits below the raw model. A "
                  "goblin only 1 kill under the standard line ran ~7 pts hot: trust the "
                  "flag on goblins 2+ kills under",
    "SUP MORE": "support MORE: at PrizePicks-like lines support overs hit 55.8% when "
                "shown at 67.3% (backtest, n=267), below the 2-pick breakeven. Skip unless "
                "it is a deep goblin",
}


def results_tab():
    df = ss().get("results")
    if df is None or df.empty:
        st.info("Enter a slate and press Run slate.")
        return
    legs = st.radio("Breakeven for slip size", [2, 3, 4, 5, 6], horizontal=True,
                    key="be_legs", format_func=lambda x: f"{x}L")
    df = rebreakeven(df, legs)
    ss()["results"] = df
    plats = sorted(df["platform"].unique())
    st.caption("Per-leg breakeven = (1/multiplier)^(1/legs): " + ", ".join(
        f"{p} {legs}L {PLATFORMS[p][legs]:g}x → {per_leg_breakeven(p, legs):.1%}"
        for p in plats))
    calibrated = bool(df["calibrated"].all())
    views = {"Shown": "", "Calibrated": "cal_", "Raw": "raw_"}
    view_name = st.radio("Probabilities", list(views), horizontal=True, key="prob_view",
                         disabled=not calibrated,
                         help="Shown = calibrated (I) then shrunk toward the line (W); "
                              "Calibrated = I only; Raw = the model's sims (with S).")
    if not calibrated:
        st.warning("No calibration data for this cutoff: raw model probabilities, and no "
                   "PLAY marks (the minimum edge applies only after calibration).")
    bal = side_balance(df)
    if bal:
        st.warning(f"Side balance: {bal['share']:.0%} of this slate's {bal['n']} plays are "
                   f"{bal['side'].upper()}. One-sided slates were where the backtest's "
                   "false edges clustered; check the lines before stacking them.")
    pre = views[view_name] if calibrated else ""
    view = pd.DataFrame({
        "play": np.where(df["play"], "✅", ""),
        "player": df["player"], "team": df["team"], "role": df["role"],
        "line": df["line"], "pick": df[f"{pre}side"].str.upper(),
        "edge": 100 * df[f"{pre}edge"], "P(more)": 100 * df[f"{pre}p_more"],
        "P(less)": 100 * df[f"{pre}p_less"], "P(push)": 100 * df[f"{pre}p_push"],
        "mean": df["mean"], "fair": df["fair_line"],
        "floor80": df["floor80"], "floor90": df["floor90"], "flags": df["flags"],
    })
    if "kind" in df:
        view["line"] = [f"{l:g}" + ("" if k == "standard" else f" {k[0].upper()}")
                        for l, k in zip(df["line"], df["kind"])]
    st.caption({"": f"Shown probabilities (calibrated + W). ✅ = shown edge ≥ "
                    f"{100 * MIN_EDGE:.0f} pts.",
                "cal_": "Calibrated (I) probabilities, before W. ✅ / flags follow Shown.",
                "raw_": "Raw model probabilities (sims with S, before I and W). ✅ / flags "
                        "follow Shown."}[pre]
               if calibrated else "Raw model probabilities.")
    pc = st.column_config.NumberColumn(format="%.1f%%")
    ev = st.dataframe(
        view, hide_index=True, width="stretch", on_select="rerun",
        selection_mode="single-row", key="res_table",
        column_config={"edge": st.column_config.NumberColumn("edge", format="%+.1f%%"),
                       "P(more)": pc, "P(less)": pc, "P(push)": pc,
                       "mean": st.column_config.NumberColumn(format="%.2f"),
                       "fair": st.column_config.NumberColumn(format="%g"),
                       "floor80": st.column_config.NumberColumn(
                           "floor 80", help="kills reached in 80% of sims"),
                       "floor90": st.column_config.NumberColumn(
                           "floor 90", help="kills reached in 90% of sims")})
    rows = ev.selection.rows if ev and ev.selection else []
    if rows:
        r = df.iloc[rows[0]]
        st.info(f"**{r.player} {r.side.upper()} {r.line:g}** ({r.market}, {r.matchup}): "
                + r.why)
        flags = [f for f in FLAG_HELP if f in str(r.flags)]
        for f in flags:
            st.caption(f"{f}: {FLAG_HELP[f]}")
    else:
        st.caption("Tap a row to see why (raw and calibrated probabilities).")
    safest_overs_ui(df)
    mark_played_ui(df)


def safest_overs_ui(df: pd.DataFrame):
    """The slate's overs ranked by shown P(more), with each player's floor."""
    top = safest_overs(df)
    if top.empty:
        return
    with st.expander("Safest overs & floors", expanded=False):
        st.caption("Every line ranked by shown P(more), goblins included. floor 80 / 90 = "
                   "kills the player reaches in 80% / 90% of this matchup's sims (series "
                   "length, win odds and opponent included). A goblin line below floor 90 "
                   "(3.5 with floor 90 = 4) hits in 90%+ of sims. SAFE FLOOR = shown P(more) ≥ "
                   f"{100 * SAFE_FLOOR_P:.0f}%; supports never get it (their floors ran "
                   "3-5 pts hot in the backtest).")
        st.dataframe(pd.DataFrame({
            "player": top["player"], "role": top["role"],
            "line": [f"{l:g}" + ("" if k == "standard" else f" {k}")
                     for l, k in zip(top["line"], top.get("kind", ["standard"] * len(top)))],
            "P(more)": 100 * top["p_more"], "floor 90": top["floor90"],
            "floor 80": top["floor80"], "median": top["median"], "mean": top["mean"],
            "flags": top["flags"],
        }), hide_index=True, width="stretch",
            column_config={"P(more)": st.column_config.NumberColumn(format="%.1f%%"),
                           "mean": st.column_config.NumberColumn(format="%.2f"),
                           "median": st.column_config.NumberColumn(format="%g")})


def mark_played_ui(df: pd.DataFrame):
    """Set the slate log's played flag for lines played outside a logged slip."""
    with st.expander("Mark lines as played (slate log)"):
        opts = {f"{r.player} {r.side.upper()} {r.line:g} ({r.matchup})": r.row_id
                for r in df.itertuples()}
        picked = st.multiselect("Played", list(opts), key="played_pick")
        if st.button("Mark played", disabled=not picked) and mark_played(
                [opts[k] for k in picked]):
            st.success(f"Marked {len(picked)} line(s) played.")


# ============================================================== slip tab ====
def leg_options(df: pd.DataFrame) -> dict[str, dict]:
    opts = {}
    for side_first in (True, False):
        for r in df.itertuples():
            side = r.side if side_first else ("less" if r.side == "more" else "more")
            if side == "less" and getattr(r, "kind", "standard") != "standard":
                continue                        # goblins / demons are MORE only
            p = r.p_more if side == "more" else r.p_less
            label = f"{r.player} {side.upper()} {r.line:g} · {p:.0%} ({r.matchup})"
            opts[label] = dict(r._asdict(), side=side, p_side=p)
    return opts


def slip_tab():
    df = ss().get("results")
    if df is None or df.empty:
        st.info("Run a slate first.")
        return
    opts = leg_options(df)
    picked = st.multiselect("Legs", list(opts), key="slip_legs",
                            placeholder="Pick 2-6 legs")
    legs = [opts[k] for k in picked]
    if not legs:
        return
    players = [l["player"] for l in legs]
    if len(set(players)) < len(players):
        st.error("The same player is in the slip twice.")
        return
    default = legs[0]["platform"]
    platform = st.radio("Platform", list(PLATFORMS), horizontal=True,
                        index=list(PLATFORMS).index(default), key="slip_platform")
    special = [l for l in legs if l.get("kind", "standard") != "standard"]
    payout = None
    if special:
        std = PLATFORMS[platform].get(len(legs))
        payout = st.number_input(
            "Payout (x) shown by PrizePicks for this slip", min_value=1.0,
            value=float(std or 3.0), step=0.25, key="slip_payout",
            help="Goblin legs lower the payout, demon legs raise it. Enter the multiplier "
                 "PrizePicks shows for this exact slip; EV uses it.")
        st.caption(f"{len(special)} goblin / demon leg(s): EV uses the payout above, not "
                   f"the standard {std:g}x." if std else "")
    s = slip_summary(ss()["sims"], legs, platform, payout)
    c1, c2, c3 = st.columns(3)
    c1.metric("Joint P", pct(s["p_joint"]))
    c2.metric("Payout", f"{s['payout']:g}x" if s["payout"] else "–")
    c3.metric("EV", f"{s['ev']:+.1%}" if s["ev"] is not None else "–")
    if s["p_joint_raw"] != s["p_joint"]:
        with st.expander("Calibrated and raw slip"):
            for lab, pk, ek in (("Calibrated (I, before W)", "p_joint_cal", "ev_cal"),
                                ("Raw model", "p_joint_raw", "ev_raw")):
                st.caption(f"{lab}: joint P {pct(s[pk])}"
                           + (f", EV {s[ek]:+.1%}" if s[ek] is not None else ""))
            st.caption("Shown / calibrated joint = raw shared-sim joint x each leg's shown "
                       "/ calibrated over raw probability (capped at the weakest leg).")
    note = f"Raw independent product {pct(s['p_independent'])}."
    if s["payout"]:
        note += f" Breakeven joint P {1 / s['payout']:.1%}."
    else:
        note += f" {platform} pays {min(PLATFORMS[platform])}-{max(PLATFORMS[platform])} legs."
    st.caption(note + " Same-matchup legs use shared sims.")

    _, remote = get_log_store()
    if (ss().get("slate_meta") or {}).get("backfill"):
        st.caption("Backfill slate: the pick log is for live slips. This marks the legs "
                   "played in the slate log only.")
        if st.button("Mark these legs played", type="primary") and mark_played(
                [l["row_id"] for l in legs]):
            st.success(f"Marked {len(legs)} leg(s) played in the slate log.")
        return
    stake = st.number_input("Entry ($)", min_value=0.0, value=10.0, step=5.0)
    if st.button("Log this slip as played", type="primary", disabled=s["payout"] is None):
        rows = picklog.make_pick_rows(legs, s, platform, stake)
        store, _ = get_log_store()
        try:
            picklog.append_picks(store, rows)
            ss()["log_nonce"] += 1
            mark_played([l["row_id"] for l in legs])
            if remote:
                st.success(f"Logged slip {rows[0]['slip_id']} ({len(rows)} legs) to GitHub.")
            else:
                st.error(f"Slip {rows[0]['slip_id']} NOT saved to GitHub (temporary disk "
                         "only): no GitHub secrets. Download it below.")
        except Exception as exc:
            st.error(f"Could not save the slip: {exc}. Download it below, and see "
                     "Record → Log connection.")
        st.download_button("Download this slip (CSV)", pd.DataFrame(rows).to_csv(index=False),
                           file_name=f"slip_{rows[0]['slip_id']}.csv", mime="text/csv")


# ============================================================== record tab ====
def record_tab(store: DataStore):
    log_store, remote = get_log_store()
    if not remote:
        st.error("No GitHub secrets: slates and slips are NOT being saved (the app's disk "
                 "is wiped on restart). Add the [github] block under the app's Settings → "
                 "Secrets on share.streamlit.io (README step 2).")
    with st.expander("Log connection", expanded=not remote):
        if not remote:
            st.caption("Secrets missing: `[github] token`, `repo` not found.")
        elif st.button("Test log connection"):
            ok, msg = picklog.check_store(log_store)
            (st.success if ok else st.error)(msg)
    if st.button("Refresh"):
        ss()["log_nonce"] += 1
    slate_log_section(store)
    try:
        picks, graded = load_log(ss()["log_nonce"])
    except Exception as exc:
        st.error(f"Could not read the log: {exc}")
        return
    st.markdown("### Played slips")
    if picks.empty:
        st.info("No logged picks yet. Log a slip from the Slip tab.")
        return
    # Grades from the workflow are final; anything still pending is graded live here.
    live = grading.grade_picks(picks, store.players, store.teams)
    graded = grading.merge_graded(graded, live)
    s = grading.record_summary(picks, graded)
    c1, c2, c3 = st.columns(3)
    c1.metric("Legs W-L-P", f"{s['legs_won']}-{s['legs_lost']}-{s['legs_push']}")
    c2.metric("Slips W-L", f"{s['slips_won']}-{s['slips_lost']}")
    c3.metric("ROI", f"{s['roi']:+.1%}" if np.isfinite(s["roi"]) else "–",
              f"${s['profit']:+.2f} on ${s['staked']:.0f}")
    st.caption(f"Leg hit rate {pct(s['leg_hit'])} vs model {pct(s['exp_leg_hit'])}. "
               f"Pending: {s['legs_pending']} legs / {s['slips_pending']} slips. "
               f"Void: {s['legs_void']}.")
    if len(s["calib"]):
        st.markdown("**Calibration of logged picks** (model P of the chosen side)")
        cal = s["calib"].copy()
        cal["bucket"] = cal["bucket"].astype(str)
        st.dataframe(cal, hide_index=True, width="stretch")
    st.markdown("**Picks**")
    legs = s["legs"].sort_values("logged_at", ascending=False)
    st.dataframe(legs[["date", "player", "line", "side", "p_side", "status", "actual",
                       "platform", "slip_id"]], hide_index=True, width="stretch")
    st.markdown("**Slips**")
    st.dataframe(s["slips"], hide_index=True, width="stretch")


def slate_log_section(store: DataStore):
    """Every logged line (played or not), graded: the real-world evaluation set."""
    st.markdown("### Slate log (every entered line)")
    try:
        slates, sgraded = load_slate_log(ss()["log_nonce"])
    except Exception as exc:
        st.error(f"Could not read the slate log: {exc}")
        return
    if slates.empty:
        st.info("No slates logged yet. Every Run slate is logged here.")
        return
    live = grading.grade_slates(slates, store.players, store.teams)
    sgraded = grading.merge_graded(sgraded, live, id_col="row_id")
    e = slatelog.evaluate(slates, sgraded)
    st.caption(f"{e['n']} lines: {e.get('n_graded', 0)} graded, {e.get('n_pending', 0)} "
               f"pending, {e.get('n_void', 0)} void. Hit rate of the model side vs its "
               "predicted P: shown, calibrated and raw.")
    if "table" in e:
        t = e["table"].copy()
        for c in ("pred", "hit", "diff"):
            t[c] = 100 * t[c]
        pc = st.column_config.NumberColumn(format="%.1f%%")
        st.dataframe(t, hide_index=True, width="stretch",
                     column_config={"pred": pc, "hit": pc, "diff": pc})
        st.caption(f"Mean bias (model mean - actual): {e['bias']:+.2f} kills.")
    n_real = len(slatelog.graded_real_rows(slates, sgraded))
    need = slatelog.REAL_W_MIN_ROWS
    st.progress(min(1.0, n_real / need),
                text=f"Graded PrizePicks lines toward the real-line W refit: {n_real} / {need}"
                + (" (ready: run scripts/refit_w_real.py)" if n_real >= need else ""))


# ============================================================== data upload ====
def league_banner(teams: pd.DataFrame) -> str:
    """'Data through' per in-scope league, most recent first; leagues with no games last."""
    last = teams.astype({"league": str}).groupby("league")["date"].max()
    last = last[last.index.isin(C.LEAGUES)].sort_values(ascending=False)
    parts = [f"**{lg}** {d:%Y-%m-%d}" for lg, d in last.items()]
    none = [lg for lg in C.LEAGUES if lg not in last.index and lg != "Worlds"]
    return " · ".join(parts) + (f" · no games: {', '.join(none)}" if none else "")


def data_section(players: pd.DataFrame, teams: pd.DataFrame):
    """Top of the page: data freshness per league, and the OE file uploader."""
    last = pd.Timestamp(teams["date"].max())
    age = (pd.Timestamp.now(tz="UTC").tz_localize(None) - last).days
    src = VERSION.split("|")[0].split(":")[0]
    where = {"branch": "data branch", "local": "this server's disk only (not saved to "
             "GitHub)", "bundled": "bundled with the app (no upload yet)"}[src]
    box = st.warning if age >= 2 else st.info
    box(f"Data through {last:%Y-%m-%d} ({age} day{'s' if age != 1 else ''} ago) · "
        f"source: {where}\n\n{league_banner(teams)}")
    if ss().get("data_error"):
        st.error(ss().pop("data_error") + " Using the last data this server loaded.")
    up = st.file_uploader("Upload Oracle's Elixir match data (.csv or .zip)",
                          type=["csv", "zip"], key="oe_upload",
                          help="The yearly OE file. New games are added and games already "
                               "stored are replaced (by gameid); nothing older is dropped.")
    if up is not None and st.button("Merge into data", type="primary", key="oe_merge"):
        merge_upload(up, players, teams)
    summary = ss().get("upload_summary")
    if summary:
        show_upload_summary(summary)


def merge_upload(up, players: pd.DataFrame, teams: pd.DataFrame):
    branch, remote = get_data_branch()
    # the branch commit these frames came from: a newer one (another upload) is a conflict.
    # Frames from an unsaved local merge already contain the branch data they were merged
    # on, so they are written on top of whatever the branch holds now.
    kind, _, ref = VERSION.split("|")[0].partition(":")
    expected = ref if kind == "branch" else ("unchecked" if kind == "local" else None)
    try:
        with st.spinner(f"Parsing {up.name}..."):
            res = process_upload(up, up.name, players, teams)
    except UploadError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Could not process {up.name}: {exc}")
        return
    summary = {"source": res.source, "added": res.games_added,
               "updated": res.games_updated, "player_rows": res.player_rows_added,
               "ranges": res.ranges, "warnings": list(res.warnings), "audit": res.audit_md,
               "saved": None}
    try:
        with st.spinner("Saving to the data branch..." if remote else "Saving..."):
            if remote:
                databranch.save_frames(branch, res.players, res.teams, res.audit_md,
                                       res.source, expected_head=expected)
                ss().pop("data_override", None)
                summary["saved"] = "Saved to the `data` branch on GitHub."
            else:
                databranch.save_frames(branch, res.players, res.teams, res.audit_md,
                                       res.source)
                ss()["data_override"] = "local"
                summary["warnings"].insert(0, "No GitHub secrets: the merged data is on "
                                              "this server's disk only and is lost on "
                                              "restart.")
    except databranch.ConflictError:
        st.error("The data branch changed while this file was processed (another upload?). "
                 "Nothing was saved: press Merge again to merge on top of it.")
        data_head.clear()
        return
    except Exception as exc:
        databranch.save_frames(databranch.LocalDataBranch(LOCAL_DATA), res.players,
                               res.teams, res.audit_md, res.source)
        ss()["data_override"] = "local"
        summary["warnings"].insert(0, f"NOT saved to GitHub ({exc}). This server uses the "
                                      "merged data until it restarts; upload again later.")
    ss()["upload_summary"] = summary
    # new data: drop every cached frame / pool / sim and rebuild from the new version
    ss()["data_nonce"] += 1
    for fn in (data_head, load_frames, get_store, cached_sim):
        fn.clear()
    ss()["results"], ss()["sims"] = None, {}
    st.rerun()


def show_upload_summary(s: dict):
    msg = (f"{s['source']}: {s['added']} new game(s), {s['updated']} already stored "
           f"(replaced), {s['player_rows']} player rows added.")
    (st.success if s["saved"] else st.warning)(msg + (f" {s['saved']}" if s["saved"] else ""))
    for w in s["warnings"]:
        st.warning(w)
    with st.expander("Upload audit"):
        r = s["ranges"].copy()
        for c in ("before_last", "first", "last"):
            r[c] = pd.to_datetime(r[c]).dt.strftime("%Y-%m-%d")
        st.dataframe(r.rename(columns={"before_last": "last before", "in_upload": "in file"}),
                     hide_index=True, width="stretch")
        st.markdown(s["audit"])
    st.button("Dismiss", key="dismiss_upload",
              on_click=lambda: ss().update(upload_summary=None))


# ============================================================== main ====
init_state()
VERSION = data_version()
st.title("🎯 LoL Kills Props")
data_section(*load_frames(VERSION))
with st.expander("Mode: live or backfill", expanded=False):
    backfill = st.toggle("Backfill a past slate", key="backfill",
                         help="Project a past slate from data before its date only.")
    bf_date = st.date_input("Slate date", key="backfill_date",
                            value=pd.Timestamp.now(tz="UTC").date() - timedelta(days=1),
                            max_value=pd.Timestamp.now(tz="UTC").date(),
                            disabled=not backfill)
ASOF = pd.Timestamp(bf_date).date().isoformat() if backfill else None
try:
    store = get_store(VERSION, ASOF)
except ValueError as exc:
    st.error(f"Backfill: {exc}")
    st.stop()
cal_note = (f"S + I (beta) + W ({store.w_source})" if store.calibrated
            else "uncalibrated") if store.spread_k else \
    ("I (beta) + W, no S data" if store.calibrated else "uncalibrated, no S data")
if store.backfill:
    st.warning(f"BACKFILL {ASOF}: projections use only games before {ASOF} "
               f"(last game in window {store.last_date:%Y-%m-%d}).")
age = (pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timestamp(store.last_date)).days
if not store.backfill and age >= 2:
    st.warning(f"Match data ends {store.last_date:%Y-%m-%d} ({age} days ago). If games were "
               "played since, upload the latest Oracle's Elixir file above: recent form "
               "and rosters are missing from these projections.")
st.caption(f"OE data through {store.last_date:%Y-%m-%d} · patch {store.patch_name} · "
           f"locked model (shrink30 + A + B50 + O + S), 20,000 sims · {cal_note}")
tabs = st.tabs(["Slate", "Screenshot (beta)", "Results", "Slip", "Record"])
with tabs[0]:
    slate_tab(store)
with tabs[1]:
    screenshot_tab(store)
with tabs[2]:
    results_tab()
with tabs[3]:
    slip_tab()
with tabs[4]:
    # grading needs every game, also in backfill mode
    record_tab(get_store(VERSION) if store.backfill else store)
