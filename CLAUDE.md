# LoL Player-Kills Prop Model

Goal: price League of Legends player **kills** props (PrizePicks / Underdog style) with a
Monte Carlo series simulator built on Oracle's Elixir (OE) match data, and prove it with a
walk-forward backtest before any UI work.

Stack: Python 3.11+, pandas, numpy (vectorized), pyarrow parquet. Streamlit UI comes later.
Run tests with `pytest`. Code lives in `src/` and is imported as `src.<pkg>`.

## Phases
1. **Phase 1 (current)**: data + engine + backtest. Gate: report backtest table, calibration
   buckets, winning variant and data issues. **Do not build the UI until the gate is passed.**
2. Phase 2: Streamlit UI on top of `src.engine`.

## Locked decisions (do not change without the user's sign-off)

### Data
- Source: OE yearly match CSV from Google Drive, downloaded with `gdown` by
  `scripts/refresh_data.py`. Keep `datacompleteness == "complete"` only. Split player rows
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

### Engine (`src/engine/`)
- Monte Carlo, **20,000 sims per series**, vectorized numpy.
- Per map:
  a) Draw map winner from `p_map` (input).
  b) Team kills: weighted bootstrap from that team's **WIN or LOSS pool** (by drawn
     outcome). Patch decay weight **0.65 per patch back**. **No extra recency weighting.**
  c) Opponent adjustment: `adjRatio = opp kills-allowed / league avg`, faded by
     `min(1, nDef/10)`, clamped **[0.80, 1.20]**.
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
tests/                      pytest (fixture in tests/fixtures/)
```
