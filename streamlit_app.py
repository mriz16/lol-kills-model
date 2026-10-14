"""LoL player-kills props: slate entry -> Monte Carlo -> edges, slips and a pick log.

Run locally:  streamlit run streamlit_app.py
Deployed from `main` on Streamlit Community Cloud (see README.md).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402
from src.app import grading, picklog  # noqa: E402
from src.app.names import search_teams  # noqa: E402
from src.app.ocr import ocr_image, parse_board_text, props_to_lines  # noqa: E402
from src.app.slate import (MARKET_BY_FORMAT, PLATFORMS, THIN_EDGE, DataStore,  # noqa: E402
                           MatchupInput,
                           check_matchup, lineups_for, per_leg_breakeven, rebreakeven,
                           results_table, run_matchup, slip_summary)
from src.engine.features import build_matchup  # noqa: E402
from src.engine.sim import SeriesSim, simulate_series  # noqa: E402

st.set_page_config(page_title="LoL Kills Props", page_icon="🎯", layout="centered",
                   initial_sidebar_state="collapsed")

FIELDS = ["q_a", "team_a", "q_b", "team_b", "bo", "platform", "league", "ml_a", "ml_b",
          "m1_a", "m1_b", "lines"]
LEAGUE_AUTO = "auto"


# ============================================================== data (cached) ====
def data_version() -> str:
    files = [C.DATA_DIR / "players.parquet", C.DATA_DIR / "teams.parquet"]
    return "-".join(str(int(f.stat().st_mtime)) + str(f.stat().st_size) for f in files)


@st.cache_data(show_spinner="Loading match data...")
def load_frames(version: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (pd.read_parquet(C.DATA_DIR / "players.parquet"),
            pd.read_parquet(C.DATA_DIR / "teams.parquet"))


@st.cache_resource(show_spinner="Building team pools...")
def get_store(version: str) -> DataStore:
    return DataStore(*load_frames(version))


@st.cache_data(show_spinner=False, max_entries=128)
def cached_sim(version: str, team_a: str, team_b: str, best_of: int, league: str,
               p_map: float, lineup_a: tuple, lineup_b: tuple, n_sims: int, seed: int):
    """Matchup pools + 20k-sim arrays, cached per (data version, matchup inputs)."""
    store = get_store(version)
    params = build_matchup(store.hist, team_a, team_b, league, best_of, p_map,
                           (list(lineup_a), list(lineup_b)), store.cutoff, store.patch,
                           store.config())
    sim = simulate_series(params, n_sims=n_sims, rng=seed)
    # int16 keeps the cached arrays small (kills never exceed 32k)
    return SeriesSim(params, sim.a_wins, sim.played, sim.team_kills.astype(np.int16),
                     sim.player_kills.astype(np.int16)), params


def app_sim_fn(store, m, lines, odds, league, n_sims, seed):
    lu = lineups_for(store, m, lines)
    return cached_sim(VERSION, m.team_a, m.team_b, int(m.best_of), league,
                      round(float(odds["p_map_a"]), 6), tuple(lu[0]), tuple(lu[1]),
                      n_sims, seed)


# ============================================================== pick log store ====
def _secret(section: str, key: str, default=None):
    try:
        return st.secrets[section][key]
    except Exception:
        return default


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
                     placeholder="Chovy 10.5\nCanyon 8.5", height=140)
        if lines:
            ok = [lm for lm in lines if lm.ok]
            bad = [lm for lm in lines if not lm.ok]
            if ok:
                st.caption("Matched: " + " · ".join(
                    f"{lm.player} ({lm.team}, {lm.role}) {lm.line:g}" for lm in ok))
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
        ss()["results"] = results_table(results)
        ss()["sims"] = {r.inp.key: r.sim for r in results}
        st.success(f"Ran {len(results)} matchup(s), {len(ss()['results'])} props in "
                   f"{time.time() - tic:.1f}s. Open the Results tab.")


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
    "THIN EDGE": f"edge under {100 * THIN_EDGE:.0f} points vs the per-leg breakeven",
    "LOW SAMPLE": "player has fewer than 20 games in the model window",
    "CAUTION": "'less' at a whole-number line with P(more) < 25%: in the backtest those "
               "overs hit about 9 points more often than predicted",
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
    view = pd.DataFrame({
        "player": df["player"], "team": df["team"], "role": df["role"],
        "line": df["line"], "pick": df["side"].str.upper(),
        "edge": 100 * df["edge"], "P(more)": 100 * df["p_more"],
        "P(less)": 100 * df["p_less"], "P(push)": 100 * df["p_push"],
        "mean": df["mean"], "fair": df["fair_line"], "flags": df["flags"],
    })
    pc = st.column_config.NumberColumn(format="%.1f%%")
    ev = st.dataframe(
        view, hide_index=True, width="stretch", on_select="rerun",
        selection_mode="single-row", key="res_table",
        column_config={"edge": st.column_config.NumberColumn("edge", format="%+.1f%%"),
                       "P(more)": pc, "P(less)": pc, "P(push)": pc,
                       "mean": st.column_config.NumberColumn(format="%.2f"),
                       "line": st.column_config.NumberColumn(format="%g"),
                       "fair": st.column_config.NumberColumn(format="%g")})
    rows = ev.selection.rows if ev and ev.selection else []
    if rows:
        r = df.iloc[rows[0]]
        st.info(f"**{r.player} {r.side.upper()} {r.line:g}** ({r.market}, {r.matchup}): "
                + r.why)
        flags = [f for f in ("THIN EDGE", "LOW SAMPLE", "CAUTION") if f in str(r.flags)]
        for f in flags:
            st.caption(f"{f}: {FLAG_HELP[f]}")
    else:
        st.caption("Tap a row to see why.")


# ============================================================== slip tab ====
def leg_options(df: pd.DataFrame) -> dict[str, dict]:
    opts = {}
    for side_first in (True, False):
        for r in df.itertuples():
            side = r.side if side_first else ("less" if r.side == "more" else "more")
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
    s = slip_summary(ss()["sims"], legs, platform)
    c1, c2, c3 = st.columns(3)
    c1.metric("Joint P", pct(s["p_joint"]))
    c2.metric("Payout", f"{s['payout']:g}x" if s["payout"] else "–")
    c3.metric("EV", f"{s['ev']:+.1%}" if s["ev"] is not None else "–")
    note = f"Independent product {pct(s['p_independent'])}."
    if s["payout"]:
        note += f" Breakeven joint P {1 / s['payout']:.1%}."
    else:
        note += f" {platform} pays {min(PLATFORMS[platform])}-{max(PLATFORMS[platform])} legs."
    st.caption(note + " Same-matchup legs use shared sims.")

    _, remote = get_log_store()
    stake = st.number_input("Entry ($)", min_value=0.0, value=10.0, step=5.0)
    if st.button("Log this slip as played", type="primary", disabled=s["payout"] is None):
        rows = picklog.make_pick_rows(legs, s, platform, stake)
        store, _ = get_log_store()
        try:
            picklog.append_picks(store, rows)
            ss()["log_nonce"] += 1
            st.success(f"Logged slip {rows[0]['slip_id']} ({len(rows)} legs)"
                       + ("" if remote else " locally. Set GitHub secrets to keep it."))
        except Exception as exc:
            st.error(f"Could not write the log: {exc}")


# ============================================================== record tab ====
def record_tab(store: DataStore):
    _, remote = get_log_store()
    if not remote:
        st.warning("No GitHub secrets: the log is read from the app's local disk, which "
                   "Streamlit Cloud wipes on restart. See README.")
    if st.button("Refresh"):
        ss()["log_nonce"] += 1
    try:
        picks, graded = load_log(ss()["log_nonce"])
    except Exception as exc:
        st.error(f"Could not read the log: {exc}")
        return
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


# ============================================================== main ====
init_state()
VERSION = data_version()
store = get_store(VERSION)
st.title("🎯 LoL Kills Props")
st.caption(f"OE data through {store.last_date:%Y-%m-%d} · patch {store.patch_name} · "
           f"locked model (shrink30 + A + B50 + O), 20,000 sims")
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
    record_tab(store)
