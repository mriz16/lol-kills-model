"""Name matching for slate entry: team search on the alias table, player lines vs rosters.

Pure functions (no Streamlit) so they can be unit tested.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from src.data.teams import ALIASES, normalize, resolve

try:
    from rapidfuzz import fuzz as _fuzz
except ImportError:  # pragma: no cover
    _fuzz = None

# A fuzzy player match needs this score, and must beat the runner-up by AMBIGUOUS_GAP.
PLAYER_MIN_SCORE = 80.0
AMBIGUOUS_GAP = 6.0


def _ratio(a: str, b: str) -> float:
    if _fuzz is not None:
        return float(_fuzz.ratio(a, b))
    return 100.0 * SequenceMatcher(None, a, b).ratio()


# ------------------------------------------------------------------------- teams ----
def search_teams(query: str, known: list[str], k: int = 6) -> list[str]:
    """Ranked team candidates for a typed label ("gen", "HLE", "bilibili").

    The alias-table / exact resolution comes first, then fuzzy matches on the names and
    on the aliases that point at a known team.
    """
    q = normalize(query or "")
    if not q:
        return []
    known = list(known)
    out: list[str] = []
    hit = resolve(query, known)
    if hit:
        out.append(hit)
    known_set = set(known)
    scores: dict[str, float] = {}
    for name in known:
        n = normalize(name)
        sc = _ratio(q, n)
        if n.startswith(q):
            sc = max(sc, 90.0 + min(len(q), 10) / 2)
        elif q in n and len(q) >= 3:
            sc = max(sc, 85.0)
        scores[name] = max(scores.get(name, 0.0), sc)
    for alias, canon in ALIASES.items():
        if canon in known_set:
            sc = 100.0 if alias == q else _ratio(q, alias)
            scores[canon] = max(scores.get(canon, 0.0), sc)
    for name, sc in sorted(scores.items(), key=lambda kv: -kv[1]):
        if sc < 55 or len(out) >= k:
            break
        if name not in out:
            out.append(name)
    return out[:k]


# ------------------------------------------------------------------------- lines ----
LINE_RE = re.compile(r"^\s*(?P<name>.+?)[\s,:;=\t]+(?P<line>\d+(?:\.\d+)?)\s*$")


@dataclass
class LineMatch:
    raw: str                     # the text line as typed
    query: str                   # name part
    line: float | None
    status: str                  # "ok" | "ambiguous" | "unmatched" | "bad line"
    player: str | None = None    # canonical OE player name
    team: str | None = None
    role: str | None = None
    candidates: list = field(default_factory=list)  # [(player, team, role, score)]

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def parse_line(text: str) -> tuple[str, float] | None:
    """'Loki 10.5' -> ('Loki', 10.5). Accepts 'Loki: 10.5', 'Loki, 10.5', tabs."""
    m = LINE_RE.match(text)
    if not m:
        return None
    return m.group("name").strip(), float(m.group("line"))


def match_player(query: str, roster: list[dict]) -> tuple[str, list]:
    """Match a typed name against roster entries {player, team, role}.

    Returns (status, ranked candidates). Exact (normalized) match wins; then a unique
    prefix of >= 3 characters; then fuzzy ratio >= PLAYER_MIN_SCORE with a clear gap.
    """
    q = normalize(query)
    if not q:
        return "unmatched", []
    scored = []
    for r in roster:
        n = normalize(r["player"])
        if n == q:
            sc = 100.0
        elif len(q) >= 3 and n.startswith(q):
            sc = 95.0
        else:
            sc = _ratio(q, n)
        scored.append((r["player"], r["team"], r["role"], sc))
    scored.sort(key=lambda x: -x[3])
    exact = [c for c in scored if c[3] == 100.0]
    if len(exact) == 1:
        return "ok", exact
    if len(exact) > 1:
        return "ambiguous", exact
    prefix = [c for c in scored if c[3] == 95.0]
    if len(prefix) == 1:
        return "ok", prefix
    if len(prefix) > 1:
        return "ambiguous", prefix
    good = [c for c in scored if c[3] >= PLAYER_MIN_SCORE]
    if not good:
        return "unmatched", scored[:3]
    if len(good) > 1 and good[0][3] - good[1][3] < AMBIGUOUS_GAP:
        return "ambiguous", good[:3]
    return "ok", good[:1]


def match_lines(text: str, roster: list[dict]) -> list[LineMatch]:
    """Parse the lines box and match every name against the two teams' rosters."""
    out = []
    for raw in (text or "").splitlines():
        if not raw.strip():
            continue
        parsed = parse_line(raw)
        if parsed is None:
            out.append(LineMatch(raw=raw, query=raw.strip(), line=None, status="bad line"))
            continue
        name, line = parsed
        status, cands = match_player(name, roster)
        lm = LineMatch(raw=raw, query=name, line=line, status=status, candidates=cands)
        if status == "ok":
            lm.player, lm.team, lm.role = cands[0][0], cands[0][1], cands[0][2]
        out.append(lm)
    # The same player twice is almost always a typo: flag the later copies.
    seen = set()
    for lm in out:
        if lm.ok:
            if lm.player in seen:
                lm.status = "ambiguous"
                lm.candidates = [(lm.player, lm.team, lm.role, 100.0)]
            seen.add(lm.player)
    return out
