"""KILLCAST: LoL player-kills props. Slate box -> Monte Carlo -> board, players, slips, log.

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
from src.app import grading, picklog, research, slatelog, theme  # noqa: E402
from src.app.ocr import ocr_image, parse_board_text, props_to_lines  # noqa: E402
from src.app.quick import insert_lines, parse_quick_slate, to_text  # noqa: E402
from src.app.slate import (MARKET_BY_FORMAT, MIN_EDGE, PLATFORMS, DataStore,  # noqa: E402
                           MatchupInput, check_matchup, flex_summary, lineups_for,
                           per_leg_breakeven, rebreakeven, results_table, run_matchup,
                           slip_by_winner, slip_conflicts, slip_summary)
from src.data import databranch  # noqa: E402
from src.data.upload import UploadError, process_upload  # noqa: E402
from src.engine.features import build_matchup  # noqa: E402
from src.engine.sim import SeriesSim, simulate_series  # noqa: E402

st.set_page_config(page_title="KILLCAST · LoL kills projections", page_icon="🎯",
                   layout="centered", initial_sidebar_state="collapsed")

SLATE_EXAMPLE = ("HLE vs GEN -150 +120\nZeka 3.5\nChovy 4.5\n\n"
                 "DK vs T1 +130 -160 bo3\nFaker 3.5\nShowMaker 3.5")


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


# ============================================================== logs ====
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


def saved_note(n: int, what: str, remote: bool) -> None:
    if remote:
        st.success(f"✓ {n} {what} saved to GitHub and read back.")
    else:
        st.error(f"{n} {what} kept on this server's temporary disk only (no GitHub "
                 "secrets): Record → Log connection.")


def log_slate(store: DataStore, df: pd.DataFrame):
    """Every line of a run slate goes to the slate log (played or not)."""
    meta = ss()["slate_meta"]
    rows = slatelog.make_slate_rows(df, meta["slate_date"], store.backfill, store.cutoff,
                                    slate_id=meta["slate_id"])
    log_store, remote = get_log_store()
    try:
        slatelog.append_slate(log_store, rows)
        ss()["log_nonce"] += 1
        miss = slatelog.missing_rows(log_store, [r["row_id"] for r in rows])
        if miss:
            st.error(f"Slate log write returned OK but {len(miss)} line(s) are not in the "
                     "log. Use the download below.")
        else:
            ss()["slate_saved"] = (len(rows), remote)
    except Exception as exc:
        st.error(f"Could not save the slate log: {exc}. Use the download below, and see "
                 "Record → Log connection.")
    ss()["slate_csv"] = pd.DataFrame(rows).to_csv(index=False)


def save_played(rows: list[dict]) -> bool:
    """Played flag on the legs' slate-log rows, verified by reading the log back."""
    log_store, remote = get_log_store()
    try:
        ids = [r["row_id"] for r in rows]
        if ids:
            slatelog.mark_played(log_store, ids)
        ss()["log_nonce"] += 1
        all_ids = ids
        miss = slatelog.missing_rows(log_store, all_ids, played=True)
        if miss:
            st.error(f"{len(miss)} line(s) did not save as played (write returned OK, "
                     "read-back did not find them). Try again or check Record → Log "
                     "connection.")
            return False
        saved_note(len(all_ids), "played line(s)", remote)
        return True
    except Exception as exc:
        st.error(f"Could not update the slate log: {exc}")
        return False


# ============================================================== state ====
def ss():
    return st.session_state


def init_state():
    ss().setdefault("slate_text", "")
    ss().setdefault("saved", {})
    ss().setdefault("results", None)
    ss().setdefault("sims", {})
    ss().setdefault("log_nonce", 0)
    ss().setdefault("slate_meta", None)
    ss().setdefault("data_nonce", 0)
    ss().setdefault("upload_summary", None)
    ss().setdefault("be_legs", 2)


def be_legs() -> int:
    return int(ss().get("be_legs") or 2)


def html(s: str):
    st.markdown(s, unsafe_allow_html=True)


# ============================================================== slate tab ====
def parsed_slate(store: DataStore):
    """[(MatchupInput, QuickMatchup, line matches, odds, problems)], loose problems."""
    qms, loose = parse_quick_slate(ss()["slate_text"], store.team_names)
    out = []
    for qm in qms:
        m = MatchupInput(team_a=qm.team_a, team_b=qm.team_b,
                         best_of=int(qm.best_of or ss().get("default_bo") or 3),
                         platform=ss().get("default_platform") or "PrizePicks",
                         ml_a=qm.ml_a, ml_b=qm.ml_b, map1_a=qm.m1_a, map1_b=qm.m1_b,
                         lines="\n".join(qm.lines))
        if qm.team_a and qm.team_b:
            lines, odds, problems = check_matchup(store, m)
        else:
            lines, odds, problems = [], {"error": None}, []
        out.append((m, qm, lines, odds, list(dict.fromkeys(qm.problems + problems))))
    return out, loose


def slate_tab(store: DataStore):
    c1, c2 = st.columns(2)
    with c1:
        st.segmented_control("Default format", [1, 3, 5], key="default_bo", default=3,
                             format_func=lambda x: f"BO{x}",
                             help="Used when a matchup line has no bo1 / bo3 / bo5.")
    with c2:
        st.segmented_control("Platform", list(PLATFORMS), key="default_platform",
                             default="PrizePicks")
    st.text_area("Slate", key="slate_text", height=260, placeholder=SLATE_EXAMPLE,
                 label_visibility="collapsed",
                 help="One matchup line (TEAM vs TEAM, then the series moneylines; "
                      "optional bo1/bo3/bo5 and m1 + Map 1 moneylines), then its player "
                      "lines. Add g after a goblin line (MORE only).")
    st.caption("`TEAM vs TEAM -150 +120` then `player line` under it · optional `bo1` · "
               "`m1 -130 +110` for a Map 1 price.")
    parsed, loose = parsed_slate(store)
    ready = [p for p in parsed if not p[4] and p[2]]
    for msg in loose:
        html(theme.matchup_preview("lines without a matchup", "", [], [msg]))
    for m, qm, lines, odds, problems in parsed:
        title = (f"{m.team_a} vs {m.team_b}" if m.team_a and m.team_b else qm.header)
        note = f"BO{m.best_of} · {m.market}"
        if odds.get("p_map_a") is not None and m.team_a:
            note += f" · map {theme.pct(odds['p_map_a'], 0)} / {theme.pct(odds['p_map_b'], 0)}"
        ok = [f"{lm.player} {lm.line:g}" + ("" if lm.kind == "standard" else f" {lm.kind[0]}")
              for lm in lines if lm.ok]
        if m.team_a and m.team_b and not lines and not problems:
            problems = ["no player lines under this matchup"]
        html(theme.matchup_preview(title, note, ok, problems))
    n_lines = sum(len(p[2]) for p in ready)
    run = st.button(f"Run projections · {len(ready)} matchup(s) · {n_lines} line(s)",
                    type="primary", width="stretch", key="run_slate",
                    disabled=not ready)
    if run:
        run_slate(store, [p[0] for p in ready])
    if ss().get("slate_saved"):
        n, remote = ss()["slate_saved"]
        saved_note(n, "line(s) of this slate", remote)
    if ss().get("slate_csv"):
        meta = ss()["slate_meta"] or {}
        st.download_button("Download slate log rows (CSV)", ss()["slate_csv"],
                           file_name=f"slate_{meta.get('slate_id', '')}.csv",
                           mime="text/csv", key="dl_slate_rows")

    with st.expander("Save / load slates"):
        name = st.text_input("Slate name", value=time.strftime("Slate %H:%M"))
        if st.button("Save for this session"):
            ss()["saved"][name] = ss()["slate_text"]
            st.success(f"Saved \"{name}\".")
        if ss()["saved"]:
            pick = st.selectbox("Saved slates", list(ss()["saved"]))
            st.button("Load", on_click=lambda: ss().update(slate_text=ss()["saved"][pick]))
        st.download_button("Download slate (.txt)", ss()["slate_text"] or "",
                           file_name="slate.txt", mime="text/plain")
        up = st.file_uploader("Load a slate (.txt, or an old .json)", type=["txt", "json"],
                              key="slate_upload")
        if up is not None:
            raw = up.getvalue().decode("utf-8", errors="replace")
            text = to_text(json.loads(raw)) if up.name.endswith(".json") else raw
            st.button("Load uploaded slate", on_click=lambda: ss().update(slate_text=text))


def run_slate(store: DataStore, inputs: list[MatchupInput]):
    legs = be_legs()
    results = []
    tic = time.time()
    with st.spinner(f"Simulating {C.N_SIMS:,} series per matchup..."):
        for m in inputs:
            try:
                results.append(run_matchup(store, m, legs, sim_fn=app_sim_fn))
            except ValueError as exc:
                st.error(str(exc))
    if not results:
        return
    df = results_table(results)
    slate_date = (store.asof or pd.Timestamp.now(tz="UTC").tz_localize(None)).date()
    ss()["slate_meta"] = dict(slate_date=slate_date.isoformat(),
                              slate_id=slatelog.new_slate_id(), backfill=store.backfill)
    df["row_id"] = [slatelog.row_key(slate_date.isoformat(), r) for r in df.to_dict("records")]
    ss()["results"] = df
    ss()["sims"] = {r.inp.key: r.sim for r in results}
    ss()["slate_saved"] = None
    st.success(f"{len(results)} matchup(s), {len(df)} lines projected in "
               f"{time.time() - tic:.1f}s. Board and Players are ready.")
    log_slate(store, df)


# ============================================================== board tab ====
VIEWS = {"Shown": "", "Calibrated": "cal_", "Raw": "raw_"}


def board_df() -> pd.DataFrame | None:
    df = ss().get("results")
    if df is None or df.empty:
        return None
    df = rebreakeven(df, be_legs())
    ss()["results"] = df
    return df


def board_tab():
    df = board_df()
    if df is None:
        html(theme.grid([]).replace("nothing clears the bar on this slate",
                                    "enter a slate and run projections"))
        return
    pre = VIEWS[ss().get("prob_view", "Shown")] if bool(df["calibrated"].all()) else ""
    plats = sorted(df["platform"].unique())
    n = be_legs()
    be_txt = " · ".join(f"{p} {n}-pick {PLATFORMS[p][n]:g}x → "
                        f"{per_leg_breakeven(p, n):.1%}" for p in plats)
    plays = df[df["play"].astype(bool)]
    html(theme.section(f"Plays · {len(plays)}", f"edge ≥ {100 * MIN_EDGE:.0f} pts vs {be_txt}"))
    html(theme.grid([theme.prop_card(r, pre) for r in plays.to_dict("records")]))
    std = df[df["kind"] == "standard"] if "kind" in df else df
    html(theme.section("All lines", f"{len(df)} lines · tap a row for the breakdown"))
    view = pd.DataFrame({
        "": np.where(df["play"], "▶", ""), "player": df["player"], "role": df["role"],
        "line": [f"{l:g}" + ("" if k == "standard" else f" {k[0].upper()}")
                 for l, k in zip(df["line"], df.get("kind", ["standard"] * len(df)))],
        "pick": df[f"{pre}side"].str.upper(), "P": 100 * df[f"{pre}p_side"],
        "edge": 100 * df[f"{pre}edge"], "proj": df["mean"], "fl80": df["floor80"],
        "if W": 100 * df["p_if_win"], "if L": 100 * df["p_if_loss"],
        "needs": df["needs"].map({"win": "W", "loss": "L"}).fillna(""),
        "team": df["team"],
    })
    ev = st.dataframe(
        view, hide_index=True, width="stretch", on_select="rerun",
        selection_mode="single-row", key="res_table",
        column_config={"P": st.column_config.ProgressColumn("P", format="%.1f%%",
                                                            min_value=0, max_value=100),
                       "edge": st.column_config.NumberColumn("edge", format="%+.1f"),
                       "proj": st.column_config.NumberColumn("proj", format="%.1f"),
                       "fl80": st.column_config.NumberColumn("fl80", help="kills reached "
                                                             "in 80% of sims"),
                       "if W": st.column_config.NumberColumn(
                           "if W", format="%.0f%%",
                           help="model P(pick hits) if the player's team wins the series"),
                       "if L": st.column_config.NumberColumn(
                           "if L", format="%.0f%%",
                           help="model P(pick hits) if the player's team loses the series"),
                       "needs": st.column_config.TextColumn(
                           "needs", help="W: the pick needs the team to win the series "
                                         "(20+ pts better if it does); L: needs it to lose")})
    rows = ev.selection.rows if ev and ev.selection else []
    if rows:
        r = df.iloc[rows[0]].to_dict()
        html(theme.grid([theme.prop_card(r, pre)]))
        st.caption(r["why"])
    with st.expander("Model layers and slip size"):
        st.segmented_control("Breakeven slip size", [2, 3, 4, 5, 6], key="be_legs",
                             format_func=lambda x: f"{x}-pick")
        st.radio("Probabilities", list(VIEWS), horizontal=True, key="prob_view",
                 disabled=not bool(df["calibrated"].all()),
                 help="Shown = calibrated (I) then shrunk toward the line (W); "
                      "Calibrated = I only; Raw = the model's sims (with S).")
        n_less = int((std["side"] == "less").sum())
        st.caption(f"{n_less} of {len(std)} standard lines lean LESS. At lines near a "
                   "player's average that is expected (kill counts are right-skewed); the "
                   "walk-forward check in reports/goblin_fit.md finds shown LESS plays "
                   "68.4% → hit 68.5%, MORE 67.1% → 67.3%. if W / if L are raw model "
                   "P(pick hits) given the series winner.")
    mark_played_ui(df)


def mark_played_ui(df: pd.DataFrame):
    with st.expander("Log lines as played"):
        opts = {f"{r.player} {r.side.upper()} {r.line:g} · {r.matchup}": r.Index
                for r in df.itertuples()}
        picked = st.multiselect("Played", list(opts), key="played_pick")
        if st.button("Save as played", disabled=not picked, key="mark_played"):
            save_played([df.loc[opts[k]].to_dict() for k in picked])


# ============================================================== players tab ====
# validated categorical pair on the dark panel (dataviz validate_palette.js: lightness
# band, CVD dE 25, normal-vision dE 28, >= 3:1 contrast)
C_MODEL, C_ACTUAL = "#2a9be0", "#cf7f16"
N_SERIES = {"last 10": 10, "last 20": 20, "last 40": 40, "all": None}


@st.cache_data(show_spinner="Measuring every player's spread...", max_entries=4)
def archetype_table(version: str, asof: str | None) -> pd.DataFrame:
    store_ = get_store(version, asof)
    return research.archetypes(research.map_stats(store_.players))


@st.cache_data(show_spinner=False, max_entries=256)
def series_totals(version: str, asof: str | None, player: str, market: str,
                  team: str | None) -> pd.DataFrame:
    return research.market_totals(get_store(version, asof).players, player, market, team)


def model_kills(r: dict) -> np.ndarray | None:
    sim = ss()["sims"].get(r["matchup"])
    return None if sim is None else sim.range_kills(r["player"], r["market"])


def slate_research(df: pd.DataFrame, n_last: int | None, arch: pd.DataFrame) -> pd.DataFrame:
    """One row per slate line: model floors / ceilings (sims with S) next to the real
    last-N series of the same market, over-rate at the line, in wins / losses."""
    types = arch.set_index("player")["type"].to_dict() if len(arch) else {}
    rows = []
    for r in df.to_dict("records"):
        k = model_kills(r)
        m = research.dist_summary(k) if k is not None else {}
        t = series_totals(VERSION, ASOF, r["player"], r["market"], None)
        t = t.head(n_last) if n_last else t
        a = research.dist_summary(t["kills"])
        v, w = t["kills"].to_numpy(), t["won"].to_numpy(bool)
        line = float(r["line"])
        rows.append({
            "player": r["player"], "role": r["role"], "line": line,
            "type": types.get(r["player"], ""),
            "m fl90": m.get("floor90"), "m fl80": m.get("floor80"),
            "m med": m.get("median"), "m top20": m.get("ceil80"), "m top10": m.get("ceil90"),
            "P over": 100 * float((k > line).mean()) if k is not None else np.nan,
            "n": a["n"], "a fl80": a["floor80"], "a med": a["median"],
            "a top10": a["ceil90"],
            "over%": 100 * float((v > line).mean()) if len(v) else np.nan,
            "over% W": 100 * float((v[w] > line).mean()) if w.any() else np.nan,
            "over% L": 100 * float((v[~w] > line).mean()) if (~w).any() else np.nan,
        })
    return pd.DataFrame(rows)


def dist_chart(t: pd.DataFrame, k: np.ndarray | None, line: float | None):
    import altair as alt
    d = research.pmf_compare(t["kills"], k)
    if d.empty:
        return None
    d["share"] = 100 * d["share"]
    bars = alt.Chart(d).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=9).encode(
        x=alt.X("kills:O", title="kills"),
        xOffset=alt.XOffset("source:N", sort=["actual", "model"]),
        y=alt.Y("share:Q", title="% of series"),
        color=alt.Color("source:N", sort=["actual", "model"],
                        scale=alt.Scale(domain=["actual", "model"],
                                        range=[C_ACTUAL, C_MODEL]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("source:N"), alt.Tooltip("kills:O"),
                 alt.Tooltip("share:Q", format=".1f", title="% of series")])
    out = bars
    if line is not None:
        rule = alt.Chart(pd.DataFrame({"kills": [str(int(np.ceil(line)))]})).mark_rule(
            strokeDash=[4, 3], strokeWidth=2, color="#dce7ff").encode(x="kills:O")
        out = bars + rule
    return out.properties(height=230)


def timeline_chart(t: pd.DataFrame, line: float | None):
    import altair as alt
    if t.empty:
        return None
    d = t.iloc[::-1].reset_index(drop=True)
    d["n"] = np.arange(1, len(d) + 1)
    d["result"] = np.where(d["won"], "won series", "lost series")
    d["day"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    bars = alt.Chart(d).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=10).encode(
        x=alt.X("n:O", title="series (oldest → newest)", axis=alt.Axis(labels=False)),
        y=alt.Y("kills:Q", title="kills"),
        color=alt.Color("result:N", scale=alt.Scale(domain=["won series", "lost series"],
                                                    range=[C_MODEL, C_ACTUAL]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["day:N", "opponent:N", "result:N", "kills:Q", "league:N"])
    out = bars
    if line is not None:
        out = bars + alt.Chart(pd.DataFrame({"y": [line]})).mark_rule(
            strokeDash=[4, 3], strokeWidth=2, color="#dce7ff").encode(y="y:Q")
    return out.properties(height=200)


def players_tab(store: DataStore):
    df = ss().get("results")
    arch = archetype_table(VERSION, ASOF)
    c1, c2 = st.columns(2)
    with c1:
        win = st.segmented_control("Real series", list(N_SERIES), default="last 20",
                                   key="res_n") or "last 20"
    n_last = N_SERIES[win]
    html(theme.chips([
        theme.chip("m = model sims · a = actual " + win),
        theme.chip("fl80 = reached in 80%"), theme.chip("top10 = reached in 10%"),
    ]))
    if df is not None and not df.empty:
        html(theme.section("Slate distributions", "model vs what actually happened"))
        t = slate_research(df, n_last, arch)
        pc = st.column_config.NumberColumn(format="%.0f%%")
        st.dataframe(t, hide_index=True, width="stretch",
                     column_config={"P over": pc, "over%": pc, "over% W": pc,
                                    "over% L": pc,
                                    "line": st.column_config.NumberColumn(format="%g"),
                                    "m med": st.column_config.NumberColumn(format="%g"),
                                    "a med": st.column_config.NumberColumn(format="%g")})
        st.caption("Model floors are the tested ones: in the walk-forward backtest P(kills ≥ "
                   "floor) is calibrated within ~1 pt from 80% to 98% (supports 3-5 pts "
                   "hot, reports/audit_unders.md). Actual columns are the player's real "
                   "series of this market: small samples, every opponent and patch mixed.")
        choices = list(dict.fromkeys(df["player"]))
    else:
        st.caption("Run a slate to compare the model with each player's real series. Any "
                   "player can be looked up below.")
        choices = []
    html(theme.section("Player deep dive", "distribution, floors, every series"))
    names = sorted(set(map(str, store.players["playername"].unique())))
    # keyed per slate run, so a new slate starts on its own first player and market
    run_key = (ss().get("slate_meta") or {}).get("slate_id", "none")
    c1, c2 = st.columns(2)
    with c1:
        player = st.selectbox("Player", choices + [n for n in names if n not in choices],
                              key=f"res_player_{run_key}",
                              index=0 if choices or names else None)
    row = df[df["player"] == player].iloc[0].to_dict() if choices and player in choices \
        else None
    with c2:
        market = st.segmented_control("Market", list(research.MARKETS),
                                      key=f"res_market_{run_key}",
                                      default=row["market"] if row else "Map 1") or "Map 1"
    if not player:
        return
    t_all = series_totals(VERSION, ASOF, player, market, None)
    t = t_all.head(n_last) if n_last else t_all
    k = model_kills(row) if row and row["market"] == market else None
    line = float(row["line"]) if row and row["market"] == market else None
    a = research.dist_summary(t["kills"])
    w = t["won"].to_numpy(bool)
    v = t["kills"].to_numpy()
    me = arch[arch["player"] == player]
    chips = [theme.chip(f"{a['n']} series · {market}")]
    if len(me):
        x = me.iloc[0]
        if x["type"]:
            chips.append(theme.chip(f"type <b>{theme.escape(x['type'])}</b>",
                                    "warn" if x["type"] == "BOOM/BUST" else "ok"))
        chips.append(theme.chip(f"per map {x['mean']:.1f} · wins {x['mean_win']:.1f} · "
                                f"losses {x['mean_loss']:.1f}"))
    if w.any() and (~w).any():
        chips.append(theme.chip(f"series avg won <b>{v[w].mean():.1f}</b> · lost "
                                f"<b>{v[~w].mean():.1f}</b>"))
    html(theme.chips(chips))
    if t.empty:
        st.caption("No series of this market for this player.")
        return
    summ = [dict(source=f"actual ({win})", **a)]
    if k is not None:
        summ.append(dict(source="model", **research.dist_summary(k)))
    st.dataframe(pd.DataFrame(summ)[["source", "n", "floor90", "floor80", "median", "mean",
                                     "ceil80", "ceil90", "max"]].rename(
        columns={"ceil80": "top20", "ceil90": "top10"}), hide_index=True, width="stretch",
        column_config={"mean": st.column_config.NumberColumn(format="%.1f"),
                       "median": st.column_config.NumberColumn(format="%g")})
    ch = dist_chart(t, k, line)
    if ch is not None:
        st.altair_chart(ch, width="stretch")
    st.caption("Share of series at each kill count: actual (orange) vs the model's sims "
               "(blue); dashed line = the line." if k is not None else
               "Share of series at each kill count (run the slate to add the model).")
    tl = timeline_chart(t, line)
    if tl is not None:
        st.altair_chart(tl, width="stretch")
    lad = research.line_ladder(v, w, sim=k)
    if line is not None:
        lad = lad[(lad["line"] >= line - 4) & (lad["line"] <= line + 3)]
    for c in ("over", "over_win", "over_loss", "model_over"):
        if c in lad:
            lad[c] = 100 * lad[c]
    st.dataframe(lad.rename(columns={"over": "over %", "over_win": "in wins",
                                     "over_loss": "in losses", "model_over": "model"}),
                 hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.0f%%")
                                for c in ("over %", "in wins", "in losses", "model")})
    with st.expander("Every series"):
        st.dataframe(t[["date", "team", "opponent", "league", "best_of", "won", "kills"]],
                     hide_index=True, width="stretch")
    with st.expander("Boom / bust types by role"):
        roles = st.multiselect("Roles", C.POSITIONS, default=C.POSITIONS, key="arch_roles")
        show = arch[arch["role"].isin(roles)].sort_values(["role", "spread_pct"],
                                                         ascending=[True, False])
        st.dataframe(show[["player", "team", "role", "maps", "mean", "vmr", "floor80",
                           "ceil90", "p_zero", "mean_win", "mean_loss", "type"]],
                     hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.2f")
                                    for c in ("mean", "vmr", "p_zero", "mean_win",
                                              "mean_loss")})
        st.caption(f"Last {research.ARCHETYPE_DAYS} days, players with ≥ "
                   f"{research.ARCHETYPE_MIN_MAPS} maps. vmr = variance / mean of kills per "
                   "map (higher = more boom and bust for the same average). Types only for "
                   "bot / mid / jng: there a player's spread carries over to the next 150 "
                   "days (same type again 35-53% vs 33% by chance); for top and support it "
                   "does not, so no type is shown (reports/player_research.md).")


# ============================================================== slip tab ====
def leg_options(df: pd.DataFrame) -> dict[str, dict]:
    opts = {}
    for side_first in (True, False):
        for r in df.to_dict("records"):
            side = r["side"] if side_first else ("less" if r["side"] == "more" else "more")
            if side == "less" and r.get("kind", "standard") != "standard":
                continue                        # goblins / demons are MORE only
            p = r["p_more"] if side == "more" else r["p_less"]
            label = f"{r['player']} {side.upper()} {r['line']:g} · {p:.0%}"
            opts.setdefault(label, dict(r, side=side, p_side=p))
    return opts


def slip_tab():
    df = board_df()
    if df is None:
        st.info("Run projections first.")
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
    c1, c2 = st.columns(2)
    with c1:
        platform = st.segmented_control("Platform", list(PLATFORMS), key="slip_platform",
                                        default=legs[0]["platform"]) or legs[0]["platform"]
    with c2:
        mode = st.segmented_control("Type", ["Power", "Flex"], key="slip_mode",
                                    default="Power") or "Power"
    if mode == "Flex" and (platform != "PrizePicks" or len(legs) < 3):
        st.info("Flex: PrizePicks, 3-6 legs.")
        return
    special = any(l.get("kind", "standard") != "standard" for l in legs)
    payout = None
    if mode == "Power" and special:
        # the user enters what PrizePicks shows for a slip with goblin / demon legs
        guess = float(PLATFORMS[platform].get(len(legs), 1.0))
        payout = st.number_input("Payout PrizePicks shows (x)", min_value=1.0,
                                 value=round(guess, 2), step=0.25, key="slip_payout",
                                 help="Goblin legs lower the payout: enter the multiplier "
                                      "PrizePicks shows for this exact slip.")
    script_ui(legs)
    if mode == "Power":
        s = slip_summary(ss()["sims"], legs, platform, payout)
        c1, c2, c3 = st.columns(3)
        c1.metric("Hit all", theme.pct(s["p_joint"]))
        c2.metric("Payout", f"{s['payout']:g}x" if s["payout"] else "–")
        c3.metric("EV", f"{s['ev']:+.1%}" if s["ev"] is not None else "–")
        st.caption(f"Raw model {theme.pct(s['p_joint_raw'])} · breakeven "
                   f"{theme.pct(1 / s['payout']) if s['payout'] else '–'} · same-matchup "
                   "legs share sims.")
        slip, ok = s, s["payout"] is not None
    else:
        f = flex_summary(ss()["sims"], legs)
        d = f["dist"]
        c1, c2, c3 = st.columns(3)
        n = len(legs)
        c1.metric(f"{n}/{n}", theme.pct(d[n]))
        c2.metric(f"{n - 1}/{n}", theme.pct(d[n - 1]))
        c3.metric("EV", f"{f['ev']:+.1%}")
        st.caption("Pays " + " · ".join(f"{k}/{n} {m:g}x" for k, m in
                                         sorted(f["table"].items(), reverse=True))
                   + f" · raw-sim EV {f['ev_raw']:+.1%}.")
        slip = {"payout": f["table"][n], "p_joint": float(d[n]), "ev": f["ev"]}
        ok = True

    if (ss().get("slate_meta") or {}).get("backfill"):
        st.caption("Backfill slate: legs are saved as played in the slate log only.")
        if st.button("Save legs as played", type="primary"):
            save_played(legs)
        return
    stake = st.number_input("Entry ($)", min_value=0.0, value=10.0, step=5.0)
    if st.button("Log this slip", type="primary", disabled=not ok, width="stretch"):
        log_slip(legs, slip, platform, stake, mode.lower())


def script_ui(legs: list[dict]):
    """Which series results the slip needs: a conflict chip when legs in one matchup need
    different winners, and P(all of a matchup's legs hit) by series winner."""
    bad = slip_conflicts(legs)
    html(theme.chips([theme.chip(f"conflict · <b>{theme.escape(c)}</b>", "bad") for c in bad]
                     or [theme.chip("no conflicting series results", "ok")]))
    rows = slip_by_winner(ss()["sims"], legs)
    if rows:
        st.dataframe(pd.DataFrame([{
            "matchup": r["matchup"], "legs": r["legs"],
            f"if A wins": 100 * r["hit_if_a"], f"if B wins": 100 * r["hit_if_b"],
            "A series %": 100 * r["p_a"]} for r in rows]),
            hide_index=True, width="stretch",
            column_config={c: st.column_config.NumberColumn(c, format="%.0f%%")
                           for c in ("if A wins", "if B wins", "A series %")})
        st.caption("P(all of that matchup's legs hit) if team A (first team named) / team B "
                   "wins the series, raw model. Legs that agree on one winner hit together.")


def log_slip(legs: list[dict], slip: dict, platform: str, stake: float, mode: str):
    rows = picklog.make_pick_rows(legs, slip, platform, stake, mode=mode)
    store_, remote = get_log_store()
    try:
        picklog.append_picks(store_, rows)
        ss()["log_nonce"] += 1
        miss = picklog.missing_picks(store_, [r["pick_id"] for r in rows])
        if miss:
            st.error(f"Slip write returned OK but {len(miss)} leg(s) are not in the log. "
                     "Download it below and see Record → Log connection.")
        else:
            saved_note(len(rows), f"leg(s) of slip {rows[0]['slip_id']}", remote)
        save_played(legs)
    except Exception as exc:
        st.error(f"Could not save the slip: {exc}. Download it below, and see Record → "
                 "Log connection.")
    st.download_button("Download this slip (CSV)", pd.DataFrame(rows).to_csv(index=False),
                       file_name=f"slip_{rows[0]['slip_id']}.csv", mime="text/csv")


# ============================================================== record tab ====
def record_tab(store: DataStore):
    log_store, remote = get_log_store()
    html(theme.chips([theme.chip("log: <b>GitHub</b>", "ok") if remote else
                      theme.chip("log: <b>temporary disk</b> (no GitHub secrets)", "bad")]))
    c1, c2 = st.columns(2)
    if c1.button("Refresh", width="stretch"):
        ss()["log_nonce"] += 1
    if remote and c2.button("Test log connection", width="stretch"):
        ok, msg = picklog.check_store(log_store)
        (st.success if ok else st.error)(msg)
    try:
        slates, sgraded = load_slate_log(ss()["log_nonce"])
        picks, graded = load_log(ss()["log_nonce"])
    except Exception as exc:
        st.error(f"Could not read the log: {exc}")
        return
    live = grading.grade_slates(slates, store.players, store.teams) if len(slates) else \
        sgraded
    sgraded = grading.merge_graded(sgraded, live, id_col="row_id")

    played = slatelog.played_rows(slates, sgraded)
    html(theme.section(f"My plays · {len(played)}", "every line saved as played"))
    if played.empty:
        st.caption("Nothing saved as played yet: Board → Log lines as played, or Slip → "
                   "Log this slip.")
    else:
        res = played["result"].value_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("Hit-Miss", f"{res.get('hit', 0)}-{res.get('miss', 0)}")
        c2.metric("Pending", int(res.get("pending", 0)))
        dec = played[played["result"].isin(["hit", "miss"])]
        c3.metric("Model said", theme.pct(dec["p_side"].astype(float).mean())
                  if len(dec) else "–")
        st.dataframe(played[["slate_date", "player", "kind", "side", "line", "p_side",
                             "actual", "result"]], hide_index=True, width="stretch",
                     column_config={"p_side": st.column_config.NumberColumn(
                         "P", format="%.3f")})

    html(theme.section("Slips", "logged from the Slip tab"))
    if picks.empty:
        st.caption("No slips logged yet.")
    else:
        lg = grading.grade_picks(picks, store.players, store.teams)
        graded = grading.merge_graded(graded, lg)
        s = grading.record_summary(picks, graded)
        c1, c2, c3 = st.columns(3)
        c1.metric("Legs W-L", f"{s['legs_won']}-{s['legs_lost']}")
        c2.metric("Slips W-L", f"{s['slips_won']}-{s['slips_lost']}")
        c3.metric("ROI", f"{s['roi']:+.1%}" if np.isfinite(s["roi"]) else "–",
                  f"${s['profit']:+.2f} on ${s['staked']:.0f}")
        st.dataframe(s["slips"], hide_index=True, width="stretch")

    html(theme.section("Model check", "every logged line, graded"))
    if slates.empty:
        st.caption("No slates logged yet.")
        return
    e = slatelog.evaluate(slates, sgraded)
    st.caption(f"{e['n']} lines: {e.get('n_graded', 0)} graded, {e.get('n_pending', 0)} "
               f"pending, {e.get('n_void', 0)} void.")
    pc = st.column_config.NumberColumn(format="%.1f%%")
    for key in ("table", "goblin"):
        t = e.get(key)
        if t is not None and len(t):
            t = t.copy()
            for c in ("pred", "hit", "diff"):
                t[c] = 100 * t[c]
            st.dataframe(t, hide_index=True, width="stretch",
                         column_config={"pred": pc, "hit": pc, "diff": pc})
    n_real = len(slatelog.graded_real_rows(slates, sgraded))
    need = slatelog.REAL_W_MIN_ROWS
    st.progress(min(1.0, n_real / need),
                text=f"Graded PrizePicks lines toward the real-line refit: {n_real} / {need}")


# ============================================================== scan tab ====
def scan_tab(store: DataStore):
    st.caption("Beta. Reads PrizePicks board cards into the slate box. Check the result: "
               "nothing runs until you press Run.")
    up = st.file_uploader("Board screenshot", type=["png", "jpg", "jpeg", "webp"])
    cols = st.segmented_control("Card columns", [1, 2], default=2, key="ocr_cols")
    if up is not None and st.button("Read screenshot"):
        try:
            raw = ocr_image(up.getvalue(), columns=int(cols or 2))
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
    st.caption(f"Found {len(meta)} card(s)." if meta else "No cards recognized: check the "
               "raw text, or type the lines.")
    st.text_area("Parsed lines (edit before using)", key="ocr_lines", height=160)
    qms, _ = parse_quick_slate(ss()["slate_text"], store.team_names)
    labels = {i: f"{q.team_a or '?'} vs {q.team_b or '?'}" for i, q in enumerate(qms)}
    if not labels:
        st.caption("Add the matchup line (TEAM vs TEAM -150 +120) to the slate box first.")
        return
    target = st.selectbox("Add under matchup", list(labels), format_func=labels.get)
    st.button("Use these lines", on_click=lambda: ss().update(slate_text=insert_lines(
        ss()["slate_text"], target, ss().get("ocr_lines", ""))))
    with st.expander("Raw OCR text"):
        st.text(ss().get("ocr_raw", ""))


# ============================================================== data upload ====
def data_section(players: pd.DataFrame, teams: pd.DataFrame):
    """Data freshness per league and the OE file uploader."""
    last = teams.astype({"league": str}).groupby("league")["date"].max()
    last = last[last.index.isin(C.LEAGUES)].sort_values(ascending=False)
    html(theme.chips([theme.chip(f"{lg} <b>{d:%m-%d}</b>") for lg, d in last.items()]))
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
html(theme.CSS)
VERSION = data_version()
players_df, teams_df = load_frames(VERSION)
backfill = bool(ss().get("backfill"))
bf_date = ss().get("backfill_date")
ASOF = pd.Timestamp(bf_date).date().isoformat() if backfill and bf_date else None
try:
    store = get_store(VERSION, ASOF)
except ValueError as exc:
    st.error(f"Backfill: {exc}")
    st.stop()

age = (pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timestamp(store.last_date)).days
src_kind = VERSION.split("|")[0].split(":")[0]
cal = ("S · I · W" if store.calibrated else "uncalibrated") if store.spread_k else \
    ("I · W" if store.calibrated else "uncalibrated")
html(theme.hero(f"LoL KILL PROJECTION ENGINE · <b>{C.N_SIMS:,}</b> SIMS / SERIES"))
html(theme.chips([
    theme.chip(f"data <b>{store.last_date:%Y-%m-%d}</b>",
               "warn" if age >= 2 and not store.backfill else "ok"),
    theme.chip(f"patch <b>{store.patch_name}</b>"),
    theme.chip("model <b>locked</b>"),
    theme.chip(f"calibration <b>{cal}</b>", "ok" if store.calibrated else "warn"),
    theme.chip({"branch": "source <b>data branch</b>", "local": "source <b>server disk</b>",
                "bundled": "source <b>bundled</b>"}[src_kind],
               "" if src_kind == "branch" else "warn"),
] + ([theme.chip(f"BACKFILL <b>{ASOF}</b>", "warn")] if store.backfill else [])))
with st.expander("Data, mode and settings"):
    data_section(players_df, teams_df)
    st.toggle("Backfill a past slate", key="backfill",
              help="Project a past slate from data before its date only.")
    st.date_input("Slate date", key="backfill_date",
                  value=pd.Timestamp.now(tz="UTC").date() - timedelta(days=1),
                  max_value=pd.Timestamp.now(tz="UTC").date(), disabled=not backfill)
tabs = st.tabs(["Slate", "Board", "Players", "Slip", "Record", "Scan"])
with tabs[0]:
    slate_tab(store)
with tabs[1]:
    board_tab()
with tabs[2]:
    players_tab(store)
with tabs[3]:
    slip_tab()
with tabs[4]:
    # grading needs every game, also in backfill mode
    record_tab(get_store(VERSION) if store.backfill else store)
with tabs[5]:
    scan_tab(store)
