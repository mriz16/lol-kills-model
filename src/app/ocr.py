"""Beta: read PrizePicks board screenshots into the editable lines box.

A card reads (one text line each, OCR may merge or split them):

    GEN • MID
    Chovy
    vs T1 Sun 4:00am
    10.5 MAPS 1-2 Kills

The output is only ever used to pre-fill the lines text box; nothing auto-runs on it.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass

POS_WORDS = {"TOP": "top", "JNG": "jng", "JG": "jng", "JUNGLE": "jng", "MID": "mid",
             "ADC": "bot", "BOT": "bot", "SUP": "sup", "SUPP": "sup", "SUPPORT": "sup"}
_POS = "|".join(sorted(POS_WORDS, key=len, reverse=True))
# "GEN • MID" ; the bullet may come out as *, ·, +, -, e, o or vanish
HEADER_RE = re.compile(
    rf"^\s*(?P<team>[A-Za-z0-9.]{{1,6}})\s*(?:[^\sA-Za-z0-9]{{1,2}}|[eo])?\s*(?P<pos>{_POS})\b",
    re.I)
VS_RE = re.compile(r"^\s*vs\.?\s+(?P<opp>[A-Za-z0-9.]{1,8})", re.I)
# "10.5 MAPS 1-2 Kills" ; OCR may read the dash as ~ or a long dash, or drop "Kills".
# The line is the FIRST number; OCR sometimes adds a stray digit before "MAPS".
PROP_RE = re.compile(
    r"^\W*(?P<line>\d{1,3}(?:[.,]\d)?)\b[^A-Za-z]{0,6}MAPS?\s*(?P<lo>\d)\s*"
    r"(?:[-\u2013\u2014~]\s*(?P<hi>\d))?(?:\s*KILLS?)?", re.I)
NUM_ONLY_RE = re.compile(r"^\s*(?P<line>\d{1,3}(?:[.,]\d)?)\s*$")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._'-]{0,22}$")
MAPS_ONLY_RE = re.compile(
    r"^\s*MAPS?\s*(?P<lo>\d)\s*(?:[-–—~]\s*(?P<hi>\d))?(?:\s*KILLS?)?\s*$", re.I)
NOISE_RE = re.compile(
    r"^\W*$|^\s*(?:(?:more|less|trending|demon|goblin|\d+(?:\.\d+)?k)\W*\s*)+$", re.I)


@dataclass
class OcrProp:
    name: str
    line: float
    market: str
    team: str | None = None
    role: str | None = None
    opponent: str | None = None

    def as_line(self) -> str:
        return f"{self.name} {self.line:g}"


def _market(lo: str, hi: str | None) -> str:
    return f"Maps {lo}-{hi}" if hi and hi != lo else f"Map {lo}"


def _num(s: str) -> float:
    return float(s.replace(",", "."))


def parse_board_text(text: str) -> list[OcrProp]:
    """Parse OCR text of one or more cards into props (name, line, market, ...)."""
    lines = [ln.strip() for ln in (text or "").splitlines()]
    lines = [ln for ln in lines if ln]
    out: list[OcrProp] = []
    card: dict = {}
    pending_line: float | None = None

    def emit(line_val: float, market: str):
        nonlocal card
        if card.get("name"):
            out.append(OcrProp(name=card["name"], line=line_val, market=market,
                               team=card.get("team"), role=card.get("role"),
                               opponent=card.get("opp")))
        card = {}

    for ln in lines:
        h = HEADER_RE.match(ln)
        if h and not PROP_RE.match(ln):
            card = {"team": h.group("team").upper(), "role": POS_WORDS[h.group("pos").upper()]}
            pending_line = None
            continue
        v = VS_RE.match(ln)
        if v:
            card["opp"] = v.group("opp").upper()
            continue
        p = PROP_RE.match(ln)
        if p:
            emit(_num(p.group("line")), _market(p.group("lo"), p.group("hi")))
            pending_line = None
            continue
        n = NUM_ONLY_RE.match(ln)
        if n and card.get("name"):
            pending_line = _num(n.group("line"))
            continue
        mo = MAPS_ONLY_RE.match(ln)
        if mo and pending_line is not None:
            emit(pending_line, _market(mo.group("lo"), mo.group("hi")))
            pending_line = None
            continue
        if NOISE_RE.match(ln):
            continue
        # The name sits right above "vs OPP": keep the LAST name-like line before it, so
        # a garbled header read as text is overwritten by the real name below it.
        if "opp" not in card and NAME_RE.match(ln) and re.search(r"[A-Za-z]", ln) \
                and len(ln.split()) <= 2:
            card["name"] = ln
    return out


def props_to_lines(props: list[OcrProp]) -> str:
    return "\n".join(p.as_line() for p in props)


def ocr_image(data: bytes, columns: int = 2) -> str:
    """OCR a screenshot with pytesseract. The board is a grid, so each of `columns`
    vertical strips is read separately (top to bottom) to keep cards together."""
    import pytesseract
    from PIL import Image, ImageOps

    img = Image.open(io.BytesIO(data)).convert("L")
    if sum(img.resize((32, 32)).getdata()) / 1024 < 128:  # dark theme -> dark text
        img = ImageOps.invert(img)
    img = img.resize((img.width * 2, img.height * 2))
    w = img.width
    parts = []
    for c in range(max(1, columns)):
        strip = img.crop((c * w // columns, 0, (c + 1) * w // columns, img.height))
        parts.append(pytesseract.image_to_string(strip, config="--psm 4"))
    return "\n".join(parts)
