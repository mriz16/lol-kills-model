"""Team-name alias table and fuzzy matcher.

`resolve(name, known)` maps a user / sportsbook / DFS label (e.g. "GEN", "Gen.G Esports",
"hle") to the canonical OE `teamname` among `known`.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

try:  # rapidfuzz is optional; difflib fallback keeps tests dependency-light.
    from rapidfuzz import fuzz as _fuzz
except ImportError:  # pragma: no cover
    _fuzz = None

# alias (normalized) -> canonical OE teamname
ALIASES: dict[str, str] = {
    # LCK
    "t1": "T1", "skt": "T1", "sktt1": "T1",
    "gen": "Gen.G", "geng": "Gen.G", "gengesports": "Gen.G",
    "hle": "Hanwha Life Esports", "hanwha": "Hanwha Life Esports",
    "dk": "Dplus Kia", "dpluskia": "Dplus Kia", "damwon": "Dplus Kia",
    "kt": "KT Rolster", "ktrolster": "KT Rolster",
    "ns": "Nongshim RedForce", "nongshim": "Nongshim RedForce",
    "drx": "Kiwoom DRX", "kiwoomdrx": "Kiwoom DRX",
    "bro": "HANJIN BRION", "brion": "HANJIN BRION", "hanjinbrion": "HANJIN BRION",
    "oksavingsbankbrion": "HANJIN BRION",
    "fox": "BNK FEARX", "fearx": "BNK FEARX", "bfx": "BNK FEARX",
    "dns": "DN SOOPers", "soopers": "DN SOOPers", "dnsoopers": "DN SOOPers",
    "kdf": "DN SOOPers", "dnf": "DN SOOPers", "dnfreecs": "DN SOOPers", "kwangdong": "DN SOOPers",
    # LPL
    "blg": "Bilibili Gaming", "bilibili": "Bilibili Gaming",
    "jdg": "JD Gaming", "tes": "Top Esports", "topesports": "Top Esports",
    "wbg": "Weibo Gaming", "weibo": "Weibo Gaming",
    "al": "Anyone's Legend", "anyoneslegend": "Anyone's Legend",
    "ig": "Invictus Gaming", "invictus": "Invictus Gaming",
    "lng": "LNG Esports", "nip": "Ninjas in Pyjamas", "edg": "EDward Gaming",
    "fpx": "FunPlus Phoenix", "omg": "Oh My God", "we": "Team WE", "ra": "Rare Atom",
    "lgd": "LGD Gaming", "tt": "ThunderTalk Gaming", "up": "Ultra Prime",
    # LEC
    "g2": "G2 Esports", "fnc": "Fnatic", "fnatic": "Fnatic",
    "kc": "Karmine Corp", "karmine": "Karmine Corp",
    "mkoi": "Movistar KOI", "koi": "Movistar KOI", "vit": "Team Vitality",
    "vitality": "Team Vitality", "th": "Team Heretics", "heretics": "Team Heretics",
    "bds": "Team BDS", "gx": "GiantX", "sk": "SK Gaming", "lr": "Los Ratones",
    "ratones": "Los Ratones",
    "rge": "Rogue", "navi": "Natus Vincere",
    # LCS / LTA North
    "tl": "Team Liquid", "liquid": "Team Liquid", "c9": "Cloud9", "cloud9": "Cloud9",
    "fly": "FlyQuest", "flyquest": "FlyQuest", "100": "100 Thieves", "100t": "100 Thieves",
    "dig": "Dignitas", "sr": "Shopify Rebellion", "imt": "Immortals", "nrg": "NRG",
    "lyon": "LYON", "dsg": "Disguised", "sen": "Sentinels", "sentinels": "Sentinels",
}


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _score(a: str, b: str) -> float:
    if _fuzz is not None:
        return float(_fuzz.WRatio(a, b))
    return 100.0 * SequenceMatcher(None, a, b).ratio()


def resolve(name: str, known: list[str] | None = None, min_score: float = 85.0) -> str | None:
    """Resolve a team label to a canonical name (restricted to `known` if given)."""
    if not name:
        return None
    known = list(known) if known is not None else sorted(set(ALIASES.values()))
    norm_known = {normalize(k): k for k in known}
    n = normalize(name)
    if n in norm_known:
        return norm_known[n]
    alias = ALIASES.get(n)
    if alias is not None and (alias in known):
        return alias
    best, best_score = None, 0.0
    for nk, k in norm_known.items():
        sc = _score(n, nk)
        if sc > best_score:
            best, best_score = k, sc
    return best if best_score >= min_score else None
