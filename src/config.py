"""Project-wide configuration and locked model constants (see CLAUDE.md)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

# --- Leagues -------------------------------------------------------------------------
# Every league that sends teams to Worlds / MSI / FST, plus the feeder leagues their
# players come up from (so rosters and player histories are complete):
#  - LCK, LPL, LEC, LCS (NA; 2026 also the LATAM teams), CBLOL (Brazil), LCP (Pacific:
#    Vietnam / Taiwan / Japan / Oceania teams, e.g. GAM, MVK, TSW, CFO, DCG)
#  - feeders: LCKC (LCK CL), EM (EMEA Masters), LFL, VCS (Vietnam), LJL (Japan), PCS
DOMESTIC_LEAGUES = ["LCK", "LPL", "LEC", "LCS", "CBLOL", "LCP",
                    "LCKC", "EM", "LFL", "VCS", "LJL", "PCS"]
# OE codes for international events. Worlds is "WLDs" in OE; accept "Worlds" too.
# EWC = Esports World Cup; LTA = 2025 cross-conference (North vs South) games, which are
# priced like an event between two home leagues; WLRQ = Worlds regional qualifier games.
INTERNATIONAL_LEAGUES = ["WLDs", "Worlds", "MSI", "FST", "EWC", "LTA", "WLRQ"]
LEAGUES = DOMESTIC_LEAGUES + INTERNATIONAL_LEAGUES

# OE league code -> canonical code used by the model. Applied in src.data.oe.filter_raw
# before the league filter. Fill from reports/data_audit.md ("League codes found").
# 2025: the LCS was replaced by the LTA with a North conference ("LTA N"); OE files those
# games as "LTA N" and the 2026 LCS as "LCS". Same teams and environment -> one code.
# The 2025 South conference ("LTA S") was mostly the CBLOL teams (LOUD, FURIA, Vivo Keyd,
# paiN, ...) and CBLOL returned under its own code in 2026 -> one code, CBLOL.
# Out of scope: "KeSPA Cup" (LCK off-season cup, mostly substitutes) and regional
# second divisions (LFL2, NACL, ...).
LEAGUE_CODE_ALIASES: dict[str, str] = {"LTA N": "LCS", "LTA North": "LCS",
                                       "LTA S": "CBLOL", "LTA South": "CBLOL"}

# Leagues whose datacompleteness == "partial" games are kept when the kill data checks out
# (src.data.oe.partial_check): kills/deaths/assists/result/gamelength (and playername)
# non-null for all 10 players and sum of player kills == team kills on both sides.
# OE marks all of LPL 2025 "partial" (missing non-kill stats); LPL signed off 2026-10-14.
# Extended 2026-10-15 to every in-scope league: Worlds 2026 games with non-LPL teams
# (e.g. the play-in) were also "partial" and were being dropped. Same kill-consistency
# rule; the audit reports pass / fail per league and year.
PARTIAL_OK_LEAGUES = list(LEAGUES)

# Prior multiplier applied to the participants' home-league average when an international
# event has little measured history. FST is structurally low-kill.
EVENT_ENV_PRIOR = {"WLDs": 1.00, "Worlds": 1.00, "MSI": 1.00, "FST": 0.88, "EWC": 1.00,
                   "LTA": 1.00, "WLRQ": 1.00}
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
# Toggle M (meta drift): min games in the "latest N patches" window (extended patch by
# patch until reached) and clamp on the drift ratio.
DRIFT_MIN_GAMES = 20
DRIFT_CLAMP = (0.85, 1.20)
# Toggle P (fitted priors): league games for full weight on the league's own fit.
PRIOR_FULL_GAMES = 200

# --- Payouts ------------------------------------------------------------------------
PRIZEPICKS_POWER = {2: 3.0, 3: 5.0, 4: 10.0, 5: 20.0, 6: 40.0}
UNDERDOG = {2: 3.0, 3: 6.0, 4: 10.0, 5: 20.0, 6: 40.0}

# --- Elo (backtest p_map) ------------------------------------------------------------
ELO_K = 24.0
ELO_LEAGUE_INIT = {
    "LCK": 1600, "LPL": 1600, "LEC": 1500, "LCS": 1450,
    "LCKC": 1400, "LFL": 1375, "EM": 1375,
    "LCP": 1425, "CBLOL": 1400, "VCS": 1325, "PCS": 1325, "LJL": 1300,
}
ELO_DEFAULT_INIT = 1400
