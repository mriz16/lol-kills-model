"""Project-wide configuration and locked model constants (see CLAUDE.md)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

# --- Leagues -------------------------------------------------------------------------
DOMESTIC_LEAGUES = ["LCK", "LPL", "LEC", "LCS", "LCKC", "EM", "LFL"]
# OE codes for international events. Worlds is "WLDs" in OE; accept "Worlds" too.
INTERNATIONAL_LEAGUES = ["WLDs", "Worlds", "MSI", "FST"]
LEAGUES = DOMESTIC_LEAGUES + INTERNATIONAL_LEAGUES

# OE league code -> canonical code used by the model. Applied in src.data.oe.filter_raw
# before the league filter. Fill from reports/data_audit.md ("League codes found").
# 2025: the LCS was replaced by the LTA with a North conference ("LTA N"); OE files those
# games as "LTA N" and the 2026 LCS as "LCS". Same teams and environment -> one code.
# Not aliased (out of scope): "LTA S" (CBLOL/LLA successor) and "LTA" (cross-conference
# N vs S games), "EWC" (Esports World Cup), "KeSPA Cup".
LEAGUE_CODE_ALIASES: dict[str, str] = {"LTA N": "LCS", "LTA North": "LCS"}

# Prior multiplier applied to the participants' home-league average when an international
# event has little measured history. FST is structurally low-kill.
EVENT_ENV_PRIOR = {"WLDs": 1.00, "Worlds": 1.00, "MSI": 1.00, "FST": 0.88}
EVENT_ENV_FULL_WEIGHT_GAMES = 40

# --- Positions -----------------------------------------------------------------------
POSITIONS = ["top", "jng", "mid", "bot", "sup"]
POSITION_PRIORS = {"top": 0.168, "jng": 0.230, "mid": 0.243, "bot": 0.312, "sup": 0.047}

# --- Engine constants ----------------------------------------------------------------
N_SIMS = 20_000
PATCH_DECAY = 0.65
PATCH_LOOKBACK = 12
OPP_ADJ_FULL_GAMES = 10
OPP_ADJ_CLAMP = (0.80, 1.20)
SHRINK_GAMES = 20
MIN_OUTCOME_POOL = 5
GAME_SCRIPT_CLAMP = (0.85, 1.15)
DEFAULT_KAPPA = 25.0

# --- Payouts ------------------------------------------------------------------------
PRIZEPICKS_POWER = {2: 3.0, 3: 5.0, 4: 10.0, 5: 20.0, 6: 40.0}
UNDERDOG = {2: 3.0, 3: 6.0, 4: 10.0, 5: 20.0, 6: 40.0}

# --- Elo (backtest p_map) ------------------------------------------------------------
ELO_K = 24.0
ELO_LEAGUE_INIT = {
    "LCK": 1600, "LPL": 1600, "LEC": 1500, "LCS": 1450,
    "LCKC": 1400, "LFL": 1375, "EM": 1375,
}
ELO_DEFAULT_INIT = 1400
