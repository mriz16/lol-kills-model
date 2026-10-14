# LoL Player-Kills Prop Model

Goal: price League of Legends player **kills** props (PrizePicks / Underdog style) with a
Monte Carlo series simulator built on Oracle's Elixir (OE) match data, and prove it with a
walk-forward backtest before any UI work.

Stack: Python 3.11+, pandas, numpy (vectorized), pyarrow parquet. Streamlit UI comes later.
Run tests with `pytest`. Code lives in `src/` and is imported as `src.<pkg>`.

## Phases
1. Phase 1: data + engine + backtest. Gate passed (real-data backtest, locked config).
2. **Phase 2 (current)**: Streamlit app on top of `src.engine` (`streamlit_app.py`,
   `src/app/`), deployed from `main` to Streamlit Community Cloud, Python 3.11.

## Locked decisions (do not change without the user's sign-off)

### Data
- Source: OE yearly match CSV from Google Drive, downloaded with `gdown` by
  `scripts/refresh_data.py`. Keep `datacompleteness == "complete"`, plus **LPL "partial"**
  games (`PARTIAL_OK_LEAGUES`, signed off 2026-10-14) only if kills, deaths, assists, result
  and gamelength are non-null for all 10 players (and playername, needed for lineups) AND
  sum of player kills == team kills on each side (`src.data.oe.partial_check`). Team-level
  kills/deaths/result/length of those games are filled from the player rows; parquet
  carries a `partial` flag; the audit logs pass/fail counts per year. Split player rows
  (positions top/jng/mid/bot/sup) from team rows (`position == "team"`). Write compact
  parquet to `data/` (`players.parquet`, `teams.parquet`).
- `.github/workflows/refresh.yml` runs the refresh daily and commits updated parquet.
- The dev sandbox may block Google Drive: tests must never depend on it. Tests use the
  synthetic OE-schema fixture in `tests/fixtures/`.
- Leagues: LCK, LPL, LEC, LCS, LCKC, EM, LFL, FST + international events (Worlds = OE code
  `WLDs`, MSI). Config in `src/config.py`.
- Team-name alias table + fuzzy matcher (`src/data/teams.py`).
- OE has no series id; series are inferred from consecutive games of the same team pair
  (new series when game number resets or > 1 day gap). Best-of inferred as 2*max_wins-1.

### Locked engine config (user sign-off 2026-10-14)
- **shrink30 + A + B50**: `EngineConfig(shrink_games=30, position_kappa=True,
  neff_floor=50)` = backtest variant `shrink30+A+B50` (`LOCKED` in
  `src/backtest/walkforward.py`). Game-script **OFF**, smooth bootstrap (C) **OFF**.
  New toggles are tested on top of this config (variants named `L+...`).

### Engine (`src/engine/`)
- Monte Carlo, **20,000 sims per series**, vectorized numpy.
- Per map:
  a) Draw map winner from `p_map` (input).
  b) Team kills: weighted bootstrap from that team's **WIN or LOSS pool** (by drawn
     outcome). Patch decay weight **0.65 per patch back**. **No extra recency weighting.**
  c) Opponent adjustment, **outcome-conditional (O, locked 2026-10-14)**: when side s
     draws a WIN, `adj = (opp kills allowed in its LOSSES) / (avg kills of winning teams)`;
     on a LOSS, `(opp kills allowed in its WINS) / (avg kills of losing teams)` (both
     env-normalized). Faded by `min(1, nDef/10)` with nDef = opponent games of that result,
     clamped **[0.80, 1.20]**. (Replaced the plain `opp kills-allowed / league avg`, which
     double-counted team strength.) Softer fades /15, /20 tested and rejected
     (`reports/backtest_ofade_quick.md`).
  d) Cross-league scaling from measured team-kill averages per league. International
     events use scaled environments; FST is structurally low-kill.
  e) Game-script multiplier (combined KPM x projected game length vs league baseline) is a
     **TOGGLE**; it stays on only if the backtest proves it helps.
  f) Player allocation: kill share from **OVERALL** pools (not W/L split). Shrink players
     with < 20 games toward position priors via `min(1, n/20)`:
     top .168, jng .230, mid .243, bot .312, sup .047. Allocate team kills to the 5 players
     via **Dirichlet-multinomial**.
- Series: BO1/BO3/BO5, simulate maps until clinch. Markets: "Maps 1-2", "Maps 1-3",
  "Map N". Count kills only on maps actually played within the range.
- Odds: series moneylines for both teams (American) + optional Map 1 moneyline. De-vig with
  the **power method**. Invert series price to `p_map` (BO3: `P = p^2(3-2p)`, BO5 analog
  `P = p^3(10-15p+6p^2)`). If a map ML exists, use it.
- Output per prop: P(over), P(under), P(push) as **separate counters**; mean, median, fair
  line; edge vs breakeven for PrizePicks power (2L 3x, 3L 5x, 4L 10x, 5L 20x, 6L 40x) and
  Underdog (2L 3x, 3L 6x, 4L 10x, 5L 20x, 6L 40x).
- Slip joint probability: same-matchup legs from shared sim arrays; cross-matchup legs
  multiplied.

### Backtest (`src/backtest/`)
- Walk-forward: each historical series is projected using ONLY prior data.
- No historical odds: `p_map` comes from a simple Elo trained walk-forward.
- Metrics per player-series: CRPS and log score of actual kills; calibration of P(over) at
  line = rounded player median. Calibration table in 5% buckets.
- Variants: base vs +game-script vs shrink thresholds 10/20/30.

## Implementation choices made in Phase 1 (not dictated by the spec; revisit freely)
- Pool values are normalized by the league environment of the game they came from, then
  rescaled to the target environment (this *is* the cross-league scaling, step d).
- International env = blend of measured event average (weight `min(1, n/40)`) and the mean of
  the two teams' home-league averages times an event prior (FST 0.88, Worlds/MSI 1.0).
- Teams with < 5 games in an outcome pool get topped up from the global outcome pool.
- Scaled team kills are stochastically rounded to integers before allocation.
- Dirichlet concentration `kappa` is fit by method of moments on prior data only.
- Patch lookback capped at 12 patches (0.65^12 ~ 0.6% weight).
- Primary backtest market: BO1 -> "Map 1", BO3/BO5 -> "Maps 1-2".

## Layout
```
scripts/refresh_data.py     download OE CSV(s) -> data/*.parquet
scripts/make_synthetic.py   synthetic OE-schema data (fixture + backtest sandbox data)
scripts/run_backtest.py     walk-forward backtest -> reports/
src/config.py               leagues, priors, constants
src/data/                   OE processing, series inference, team aliases, synthetic gen
src/engine/                 odds, history/features, sim, props/slips
src/backtest/               elo, metrics, walk-forward runner
src/app/                    app logic: names, odds_input, slate, ocr, picklog, grading
streamlit_app.py            Streamlit app (Phase 2)
scripts/grade_picks.py      grade logs/picks.csv on the picks-log branch (refresh workflow)
packages.txt                apt packages for Streamlit Cloud (tesseract-ocr)
tests/                      pytest (fixture in tests/fixtures/)
```

## Locked config
`src.engine.features.locked_config()` = shrink30 + A (per-position kappa) + B50 (n_eff
floor 50) + O (outcome-conditional opp adj, fade nDef/10). Game-script off, C off, M off,
P off. `locked_config() == ALL_VARIANTS["L+O"]` (`LIVE_LOCKED` in walkforward.py). The app
uses only `locked_config()`.

## Phase 2 app (`streamlit_app.py`, logic in `src/app/`)
- Tabs: Slate, Screenshot (beta), Results, Slip, Record. Mobile-first (centered layout).
- Slate block: team search (alias table + fuzzy, `names.search_teams`), BO1/3/5,
  platform, league (auto: shared home league, else latest international event, else WLDs),
  series ML both teams + optional Map 1 ML (`odds_input`), lines box "player line".
  Names are fuzzy-matched to the two teams' rosters (players seen in the team's last 60
  days); team/role auto-filled; unmatched/ambiguous/bad lines block that matchup from
  running. Market from format: BO1 Map 1, BO3 Maps 1-2, BO5 Maps 1-3. Lineup = latest
  starters with matched line players swapped into their role.
- Screenshot: pytesseract on 1 or 2 vertical strips; `ocr.parse_board_text` fills an
  editable box; "Use these lines" copies into a chosen matchup. Never auto-runs.
- Results: one table sorted by edge = P(model side) - (1/mult)^(1/legs) for the matchup's
  platform; slip size for the breakeven is selectable (2-6, default 2). Flags: THIN EDGE
  (edge < 3 pts), LOW SAMPLE (< 20 games in window), CAUTION ("less" with P(more) < 25%
  at a whole line). Tap a row -> one-line explanation.
- Slip: joint P from shared sims within a matchup, multiplied across; payout; EV.
- Pick log: `picklog.GitHubStore` writes `logs/picks.csv` to branch `picks-log` via the
  contents API (secrets `[github] token/repo/branch`); `LocalStore` without secrets.
  `refresh.yml` grades into `logs/graded.csv` on that branch; Record tab merges those
  final grades with live grading (record, ROI, calibration of logged picks).
- Caching: parquet via `st.cache_data`, `DataStore` via `st.cache_resource`, matchup
  pools + sims via `st.cache_data` keyed on the inputs. 5 matchups run in ~1 s.

## Phase 1.5 additions
- `refresh.yml`: manual dispatch (optional `years` input), `OE_DRIVE_FOLDER_ID` from repo
  variable `vars.OE_DRIVE_FOLDER_ID`, Python 3.11. Fails the job on HTML downloads, 0 rows,
  missing OE columns, no in-scope games, or parquet > 50 MB. Writes `reports/data_audit.md`
  (also printed to the log and the job summary) and commits it with the parquet.
- Committed parquet is slim: only the columns in `PLAYER_KEEP` / `TEAM_KEEP` (`src/data/oe.py`).
- `ci.yml`: pytest on Python 3.11 on every push / PR.
- Upper-tail toggles on `EngineConfig`, all off by default; **A and B50 are locked** (via
  `locked_config()`, together with O), C rejected:
  - A `position_kappa`: concentration fitted per position (method of moments); shares
    drawn as independent Gamma(k_j*s_j, 1/k_j), normalized, so means stay at s_j.
  - B `neff_floor` (25 / 35 / 50 tested; 50 locked): if Kish n_eff of a team's decay-weighted W or L pool
    is below the floor, mix with the league W/L pool (international -> global pool) at the
    largest team weight that reaches the floor.
  - C `smooth_bootstrap`: each drawn team-kill value x becomes NB(mean x, r_league), r
    fitted per league by method of moments on within-(team, result) variance.
- Backtest primary market = what is bet: Map 1 (BO1), Maps 1-2 (BO3), Maps 1-3 (BO5).
  Tail report: P(over) vs hit rate at lines rounded median + 1.5/2.5/3.5, by position.
  Variants: base(shrink20), shrink30, shrink30+A, +B15, +B25, +C, and shrink30+A+X where X
  is the best stand-alone B/C by CRPS (all three A+X combos run; the others are marked
  "not selected"). `--quick` scores only the latest 6 months; `--variants` picks a subset.

## Bias toggles (tested on the locked config; `reports/backtest_bias.md`)
- Variants `L+<toggles>` in `src/backtest/walkforward.py`, all off by default:
  - M `meta_drift_patches` (1/2/3): pool values x (league kills/game over the latest N
    patches, min 20 games) / (decay-weighted league kills/game) of each pool game's league.
    REJECTED: worse CRPS for every N, bias unchanged.
  - P `fit_priors`: position priors refit walk-forward per league (decay-weighted, blended
    with the all-league fit by min(1, n/200)). NOT LOCKED: score-neutral, only moves bias
    between positions.
  - G `kill_mult`: reference only, never lock. Global multiplier = sum(actual)/sum(mean)
    of the locked variant over series started >= 1 day before the cutoff (after 500 rows).
  - O `opp_by_outcome`: outcome-conditional opponent adjustment (opponent kills allowed in
    its losses when side s wins, vs the average winner; mirror for losses). Same fade and
    clamp. **LOCKED** (user sign-off 2026-10-14). Softer fade (nDef/15, nDef/20) tested
    --quick: +0.0006 (se 0.0004) / +0.0008 (se 0.0007) CRPS vs O -> not locked.
- Run with `run_backtest.py --rows-only --tag ... --variants ...` (+ `--drop-partial 2025`),
  then `scripts/bias_report.py`.

## Status
- Phase 1 built; synthetic gate report in `reports/backtest_synthetic.md`.
- Real-data backtest: `reports/backtest_real.md`. Locked shrink30 + A + B50 (2026-10-14).
- Bias backtest: `reports/backtest_bias.md` (24,979 player-series incl. LPL 2025 partial).
  The ~-0.13 under-prediction came from step (c) double-counting team strength (BO5 bias
  -0.49). L+O: CRPS -0.0268 (se 0.0027), bias -0.129 -> -0.069. M rejected, P not locked.
- League codes: OE 2025 `LTA N` aliased to LCS (`LEAGUE_CODE_ALIASES`). LPL partial games
  kept under the kill-consistency rule (LPL 2025: 805 pass, 0 fail). This improves LPL 2026
  CRPS by 0.021 (se 0.009).
- O locked 2026-10-14; softer fades rejected (`reports/backtest_ofade_quick.md`).
- Phase 2 app built (see above); deploy steps in README.md.
- Open: residual bias -0.07 (top -0.15); tails ~6% thin (CAUTION flag in the app covers
  the worst bucket).
