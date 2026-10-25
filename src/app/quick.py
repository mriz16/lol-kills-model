"""Quick slate: the whole slate in one text box.

    HLE vs GEN -150 +120
    Zeka 3.5
    Chovy 4.5
    DK vs T1 +130 -160 bo3
    Faker 3.5
    T1 vs BLG -120 +100 m1 -110 -110

A matchup header is "team vs team" (also "v" or "@") with the series moneylines after the
second team (or one after each team: "HLE -150 vs GEN +120"). Optional on the header: a
format token (bo1 / bo3 / bo5, else the slate default) and "m1" followed by the Map 1
moneylines. The player lines under a header belong to that matchup ("player line", with
an optional g / d tag, exactly like the per-matchup box). Pure functions, unit tested.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.app.names import search_teams

HEADER_RE = re.compile(r"^(?P<a>.+?)\s+(?:vs\.?|v\.?|@)\s+(?P<b>.+)$", flags=re.I)
ODDS_RE = re.compile(r"^(?:[+\-−]\d{3,}(?:\.\d+)?|\d{3,}(?:\.\d+)?|ev|even|evens|pk)$",
                     flags=re.I)
BO_RE = re.compile(r"^bo([135])$", flags=re.I)
M1_RE = re.compile(r"^(?:m1|map1|map ?1:?)$", flags=re.I)


@dataclass
class QuickMatchup:
    header: str
    team_a: str = ""
    team_b: str = ""
    best_of: int | None = None
    ml_a: str = ""
    ml_b: str = ""
    m1_a: str = ""
    m1_b: str = ""
    lines: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def _split_tail(text: str) -> tuple[str, list[str]]:
    """'Gen.G -150 +120 bo3' -> ('Gen.G', ['-150', '+120', 'bo3'])."""
    toks = text.split()
    tail = []
    while toks and (ODDS_RE.match(toks[-1]) or BO_RE.match(toks[-1]) or M1_RE.match(toks[-1])):
        tail.insert(0, toks.pop())
    return " ".join(toks), tail


def is_header(line: str) -> bool:
    return HEADER_RE.match(line.strip()) is not None


def parse_header(line: str, team_names: list[str]) -> QuickMatchup:
    qm = QuickMatchup(header=line.strip())
    m = HEADER_RE.match(line.strip())
    name_a, tail_a = _split_tail(m.group("a"))
    name_b, tail_b = _split_tail(m.group("b"))
    for name, attr in ((name_a, "team_a"), (name_b, "team_b")):
        hits = search_teams(name, team_names, k=1)
        if hits:
            setattr(qm, attr, hits[0])
        else:
            qm.problems.append(f"no team matches \"{name}\"")
    series, map1, in_m1 = [], [], False
    for t in [x for x in tail_a if ODDS_RE.match(x)] + tail_b:
        if BO_RE.match(t):
            qm.best_of = int(BO_RE.match(t).group(1))
        elif M1_RE.match(t):
            in_m1 = True
        else:
            (map1 if in_m1 else series).append(t.replace("−", "-"))
    if len(series) == 2:
        qm.ml_a, qm.ml_b = series
    elif series:
        qm.problems.append("give both series moneylines (\"-150 +120\")")
    if len(map1) == 2:
        qm.m1_a, qm.m1_b = map1
    elif in_m1:
        qm.problems.append("give both Map 1 moneylines after m1")
    if not series and not map1:
        qm.problems.append("add the moneylines: \"HLE vs GEN -150 +120\"")
    if qm.team_a and qm.team_a == qm.team_b:
        qm.problems.append("both teams resolve to the same team")
    return qm


def parse_quick_slate(text: str, team_names: list[str]) -> tuple[list[QuickMatchup], list[str]]:
    """(matchups in order, problems with lines outside any matchup)."""
    out: list[QuickMatchup] = []
    loose = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if is_header(line):
            out.append(parse_header(line, team_names))
        elif out:
            out[-1].lines.append(line)
        else:
            loose.append(line)
    problems = []
    if loose:
        problems.append(f"{len(loose)} line(s) before the first matchup header "
                        f"(\"{loose[0]}\"): put \"TEAM vs TEAM -150 +120\" above them")
    return out, problems


def to_text(blocks: list[dict]) -> str:
    """Inverse of parse_quick_slate for saved per-matchup slates (block dicts)."""
    parts = []
    for b in blocks:
        if not b.get("team_a") or not b.get("team_b"):
            continue
        head = f"{b['team_a']} vs {b['team_b']} {b.get('ml_a') or ''} {b.get('ml_b') or ''}"
        if b.get("m1_a") and b.get("m1_b"):
            head += f" m1 {b['m1_a']} {b['m1_b']}"
        head += f" bo{int(b.get('bo') or 3)}"
        parts.append(" ".join(head.split()))
        parts.extend(x for x in (b.get("lines") or "").splitlines() if x.strip())
    return "\n".join(parts)


def insert_lines(text: str, matchup_index: int, new_lines: str) -> str:
    """Append lines at the end of the matchup_index-th matchup (0-based) of a quick slate
    (the screenshot reader's "use these lines")."""
    rows = (text or "").splitlines()
    heads = [i for i, r in enumerate(rows) if r.strip() and is_header(r.strip())]
    add = [x for x in (new_lines or "").splitlines() if x.strip()]
    if not heads or matchup_index >= len(heads):
        return "\n".join(rows + add)
    end = heads[matchup_index + 1] if matchup_index + 1 < len(heads) else len(rows)
    while end > heads[matchup_index] + 1 and not rows[end - 1].strip():
        end -= 1
    return "\n".join(rows[:end] + add + rows[end:])
