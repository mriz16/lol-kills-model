"""Look of the app: one stylesheet plus small HTML builders (cards, chips, bars).

Pure functions returning HTML strings, rendered with st.markdown(unsafe_allow_html=True)
by streamlit_app.py. Every value that comes from data is escaped.
"""
from __future__ import annotations

from html import escape

import numpy as np

CYAN = "#00e5ff"
MAGENTA = "#ff2bd6"
LIME = "#9dff3a"
AMBER = "#ffb020"
RED = "#ff4d6d"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
:root {
  --bg: #060911; --panel: #0c1220; --panel2: #101a2e; --line: #1b2a48;
  --text: #dce7ff; --muted: #7d8bb0; --cyan: #00e5ff; --magenta: #ff2bd6;
  --lime: #9dff3a; --amber: #ffb020; --red: #ff4d6d;
}
.stApp {
  background:
    radial-gradient(1200px 500px at 50% -120px, rgba(0,229,255,.13), transparent 70%),
    radial-gradient(900px 400px at 100% 0%, rgba(255,43,214,.08), transparent 70%),
    linear-gradient(rgba(0,229,255,.035) 1px, transparent 1px) 0 0 / 100% 28px,
    var(--bg);
  color: var(--text);
}
html, body, .stApp, .stMarkdown, p, label, input, textarea, button, li {
  font-family: 'Rajdhani', 'Segoe UI', sans-serif !important; font-size: 1.02rem;
}
h1, h2, h3, h4 { font-family: 'Orbitron', sans-serif !important; letter-spacing: .06em; }
code, pre, textarea { font-family: 'JetBrains Mono', monospace !important; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.2rem; max-width: 820px; }

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] {
  font-family: 'Orbitron', sans-serif; font-size: .72rem; letter-spacing: .12em;
  text-transform: uppercase; color: var(--muted); padding: 10px 12px;
  border-radius: 8px 8px 0 0;
}
.stTabs [aria-selected="true"] {
  color: var(--cyan) !important; background: rgba(0,229,255,.07);
  box-shadow: inset 0 -2px 0 var(--cyan);
}
.stTabs [data-baseweb="tab-highlight"] { background: transparent; }

/* inputs and buttons */
textarea, input, [data-baseweb="select"] > div {
  background: var(--panel) !important; border-color: var(--line) !important;
  border-radius: 10px !important;
}
textarea:focus, input:focus { border-color: var(--cyan) !important;
  box-shadow: 0 0 0 1px var(--cyan), 0 0 18px rgba(0,229,255,.25) !important; }
.stButton > button, .stDownloadButton > button {
  border-radius: 10px; border: 1px solid var(--line); background: var(--panel2);
  font-family: 'Orbitron', sans-serif !important; font-size: .74rem; letter-spacing: .1em;
  text-transform: uppercase;
}
.stButton > button[kind="primary"] {
  background: linear-gradient(90deg, #00b8d4, #7c4dff 55%, #ff2bd6);
  border: 0; color: #fff; box-shadow: 0 0 22px rgba(0,229,255,.35);
}
.stButton > button[kind="primary"]:hover { filter: brightness(1.12); }
[data-testid="stExpander"] details {
  background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
}
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; }
[data-testid="stMetric"] {
  background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
  padding: 10px 14px;
}
[data-testid="stMetricValue"] { font-family: 'Orbitron', sans-serif; color: var(--cyan); }

/* hero */
.kc-hero { display: flex; align-items: center; gap: 14px; margin: 4px 0 10px; }
.kc-logo {
  width: 46px; height: 46px; border-radius: 12px; display: grid; place-items: center;
  background: conic-gradient(from 200deg, var(--cyan), #7c4dff, var(--magenta), var(--cyan));
  box-shadow: 0 0 24px rgba(0,229,255,.45); font-family: 'Orbitron'; font-weight: 900;
  color: #050810; font-size: 1.15rem;
}
.kc-title { font-family: 'Orbitron', sans-serif; font-weight: 900; font-size: 1.55rem;
  letter-spacing: .14em; line-height: 1;
  background: linear-gradient(90deg, #e9f6ff, var(--cyan) 60%, var(--magenta));
  -webkit-background-clip: text; background-clip: text; color: transparent; }
.kc-sub { font-family: 'JetBrains Mono', monospace; font-size: .72rem; color: var(--muted);
  letter-spacing: .08em; margin-top: 5px; }
.kc-sub b { color: var(--lime); font-weight: 600; }
.kc-chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 6px 0 12px; }
.kc-chip { font-family: 'JetBrains Mono', monospace; font-size: .68rem; letter-spacing: .04em;
  padding: 3px 8px; border-radius: 999px; border: 1px solid var(--line);
  background: rgba(16,26,46,.8); color: var(--muted); white-space: nowrap; }
.kc-chip b { color: var(--text); font-weight: 600; }
.kc-chip.ok { border-color: rgba(157,255,58,.45); color: var(--lime); }
.kc-chip.warn { border-color: rgba(255,176,32,.5); color: var(--amber); }
.kc-chip.bad { border-color: rgba(255,77,109,.55); color: var(--red); }

/* section header */
.kc-sec { display: flex; flex-direction: column; gap: 2px;
  margin: 18px 0 8px; border-bottom: 1px solid var(--line); padding-bottom: 6px; }
.kc-sec h4 { margin: 0; font-size: .86rem; color: var(--cyan); }
.kc-sec span { font-family: 'JetBrains Mono', monospace; font-size: .7rem; color: var(--muted); }

/* prop cards */
.kc-grid { display: grid; grid-template-columns: 1fr; gap: 10px; }
@media (min-width: 640px) { .kc-grid { grid-template-columns: 1fr 1fr; } }
.kc-card { position: relative; background: linear-gradient(180deg, var(--panel2), var(--panel));
  border: 1px solid var(--line); border-radius: 14px; padding: 12px 14px 12px;
  overflow: hidden; }
.kc-card::before { content: ""; position: absolute; inset: 0 auto 0 0; width: 3px;
  background: var(--accent, var(--cyan)); box-shadow: 0 0 14px var(--accent, var(--cyan)); }
.kc-card.more { --accent: var(--cyan); }
.kc-card.less { --accent: var(--magenta); }
.kc-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; }
.kc-name { font-family: 'Orbitron', sans-serif; font-weight: 700; font-size: 1.02rem;
  letter-spacing: .05em; color: #fff; }
.kc-meta { font-family: 'JetBrains Mono', monospace; font-size: .68rem; color: var(--muted);
  margin-top: 2px; }
.kc-role { display: inline-block; padding: 0 6px; border-radius: 6px; margin-right: 6px;
  border: 1px solid var(--line); color: var(--text); text-transform: uppercase; }
.kc-pick { text-align: right; }
.kc-badge { display: inline-block; font-family: 'Orbitron', sans-serif; font-weight: 700;
  font-size: .78rem; letter-spacing: .1em; padding: 4px 9px; border-radius: 8px;
  color: #050810; background: var(--accent, var(--cyan)); }
.kc-line { font-family: 'Orbitron', sans-serif; font-size: 1.35rem; font-weight: 900;
  color: #fff; margin-top: 4px; }
.kc-line small { font-size: .62rem; color: var(--muted); font-weight: 500; }
.kc-bar { position: relative; height: 8px; background: #0a0f1b; border-radius: 99px;
  margin: 10px 0 4px; border: 1px solid var(--line); }
.kc-bar i { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 99px;
  background: linear-gradient(90deg, rgba(255,255,255,.15), var(--accent, var(--cyan)));
  box-shadow: 0 0 10px var(--accent, var(--cyan)); }
.kc-bar u { position: absolute; top: -4px; bottom: -4px; width: 2px; background: #fff;
  opacity: .85; text-decoration: none; }
.kc-stats { display: flex; justify-content: space-between; gap: 6px;
  font-family: 'JetBrains Mono', monospace; font-size: .7rem; color: var(--muted); }
.kc-stats b { color: var(--text); font-weight: 600; }
.kc-stats .pos { color: var(--lime); }
.kc-stats .neg { color: var(--red); }
.kc-tags { margin-top: 7px; display: flex; flex-wrap: wrap; gap: 4px; }
.kc-tag { font-family: 'JetBrains Mono', monospace; font-size: .6rem; padding: 1px 6px;
  border-radius: 5px; border: 1px solid var(--line); color: var(--muted); }
.kc-tag.play { color: #050810; background: var(--lime); border-color: var(--lime);
  font-weight: 600; }
.kc-empty { font-family: 'JetBrains Mono', monospace; color: var(--muted); font-size: .8rem;
  padding: 14px; border: 1px dashed var(--line); border-radius: 12px; text-align: center; }

/* matchup preview */
.kc-mu { background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
  padding: 9px 12px; margin-bottom: 8px; }
.kc-mu.bad { border-color: rgba(255,77,109,.55); }
.kc-mu-h { display: flex; justify-content: space-between; gap: 8px;
  font-family: 'Orbitron', sans-serif; font-size: .8rem; letter-spacing: .06em; }
.kc-mu-h span { font-family: 'JetBrains Mono', monospace; font-size: .68rem;
  color: var(--muted); letter-spacing: 0; }
.kc-mu-p { font-family: 'JetBrains Mono', monospace; font-size: .68rem; color: var(--muted);
  margin-top: 4px; line-height: 1.5; }
.kc-mu-p .x { color: var(--red); }
.kc-mu-p .ok { color: var(--text); }
</style>
"""


def pct(x, nd: int = 1) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{100 * float(x):.{nd}f}%"


def chip(text: str, kind: str = "") -> str:
    """text may contain <b>; callers escape data values themselves."""
    return f'<span class="kc-chip {kind}">{text}</span>'


def hero(subtitle: str) -> str:
    return ('<div class="kc-hero"><div class="kc-logo">K</div><div>'
            '<div class="kc-title">KILLCAST</div>'
            f'<div class="kc-sub">{subtitle}</div></div></div>')


def chips(items: list[str]) -> str:
    return '<div class="kc-chips">' + "".join(items) + "</div>"


def section(title: str, note: str = "") -> str:
    return (f'<div class="kc-sec"><h4>{escape(title)}</h4>'
            f'<span>{escape(note)}</span></div>')


def _bar(p: float, mark: float | None = None) -> str:
    w = max(0.0, min(1.0, float(p))) * 100
    m = "" if mark is None else f'<u style="left:{max(0, min(1, mark)) * 100:.1f}%"></u>'
    return f'<div class="kc-bar"><i style="width:{w:.1f}%"></i>{m}</div>'


def prop_card(r: dict, pre: str = "") -> str:
    """A standard line: model side, its P vs the per-leg breakeven, edge, floors."""
    side = str(r.get(f"{pre}side") or r["side"])
    p = float(r[f"{pre}p_more"] if side == "more" else r[f"{pre}p_less"])
    edge = float(r[f"{pre}edge"])
    tags = []
    if r.get("play"):
        tags.append('<span class="kc-tag play">PLAY</span>')
    for f in str(r.get("flags") or "").split():
        if f not in ("BELOW", "MIN"):
            tags.append(f'<span class="kc-tag">{escape(f)}</span>')
    return (
        f'<div class="kc-card {side}"><div class="kc-top"><div>'
        f'<div class="kc-name">{escape(str(r["player"]))}</div>'
        f'<div class="kc-meta"><span class="kc-role">{escape(str(r["role"]))}</span>'
        f'{escape(str(r["team"]))} vs {escape(str(r["opponent"]))}</div></div>'
        f'<div class="kc-pick"><span class="kc-badge">{side.upper()}</span>'
        f'<div class="kc-line">{float(r["line"]):g}<small> {escape(str(r["market"]))}'
        f'</small></div></div></div>'
        + _bar(p, r.get("breakeven")) +
        f'<div class="kc-stats"><span>P <b>{pct(p)}</b></span>'
        f'<span>BE {pct(r.get("breakeven"))}</span>'
        f'<span class="{"pos" if edge >= 0 else "neg"}">{100 * edge:+.1f} pts</span>'
        f'<span>proj <b>{float(r["mean"]):.1f}</b></span></div>'
        f'<div class="kc-stats" style="margin-top:3px"><span>floor80 <b>{r.get("floor80", "–")}'
        f'</b></span><span>floor90 <b>{r.get("floor90", "–")}</b></span>'
        f'<span>median <b>{float(r["median"]):g}</b></span>'
        f'<span>{int(r.get("n_games") or 0)} gms</span></div>'
        + _script_row(r) +
        (f'<div class="kc-tags">{"".join(tags)}</div>' if tags else "") + "</div>")


def _script_row(r: dict) -> str:
    """P(the pick hits) if the player's team wins / loses the series (raw model)."""
    w, l = r.get("p_if_win"), r.get("p_if_loss")
    if w is None or l is None:
        return ""
    team = escape(str(r["team"]))
    return (f'<div class="kc-stats" style="margin-top:3px"><span>hits <b>{pct(w, 0)}</b> if '
            f'{team} wins</span><span><b>{pct(l, 0)}</b> if loses</span>'
            f'<span>{team} series {pct(r.get("p_team_series"), 0)}</span></div>')


def grid(cards: list[str]) -> str:
    if not cards:
        return '<div class="kc-empty">nothing clears the bar on this slate</div>'
    return '<div class="kc-grid">' + "".join(cards) + "</div>"


def matchup_preview(title: str, note: str, ok_lines: list[str], problems: list[str]) -> str:
    body = ""
    if ok_lines:
        body += '<div class="ok">' + " · ".join(escape(x) for x in ok_lines) + "</div>"
    for p in problems:
        body += f'<div class="x">✕ {escape(p)}</div>'
    return (f'<div class="kc-mu{" bad" if problems else ""}"><div class="kc-mu-h">'
            f'{escape(title)}<span>{escape(note)}</span></div>'
            f'<div class="kc-mu-p">{body}</div></div>')
