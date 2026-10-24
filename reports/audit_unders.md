# Audit: why so many unders, and are they real? (2026-10-24)

Question: the app recommends LESS far more often than MORE. Is that a model bug, or real?
Everything below is walk-forward (each series projected from earlier data only) on the
data through 2026-10-23. Tables: `reports/audit_unders_tables.md`, reproduce with
`python scripts/audit_unders.py` (~3 min). International toggle test:
`reports/backtest_intl_E.md`.

**Caveat:** no real PrizePicks line has been logged yet. The `picks-log` branch does not
exist on GitHub, so the slate log has never saved (see section 6). "PP-like" lines are
proxies: the nearest half point to each player's average over their last 10 series in that
market, before the series. Real PrizePicks lines also use matchup information, so the real
edges are smaller than the proxy ones. The *direction* of each finding is what carries over.

## Verdict

1. **The unders are real, and they come mostly from the market, not the model.** With
   no model at all, a line set at a player's recent average goes UNDER 54.7% of the time
   in Maps 1-2, 56.5% in Map 1 and 56.7% in Maps 1-3. Supports go under 57-62%. Kill
   counts are right-skewed: a few big games pull the average about 0.3 kills above the
   typical (median) game. So a line set at the average sits above most outcomes. A
   calibrated model has to lean LESS on those lines.
2. **The model is not biased low.** The mean error since March 2026 is +0.002 kills per
   player-series (se 0.03), flat month to month. The old -0.07 bias was 2025 data.
3. **The LESS plays are the more reliable side.** These are the app's shown probabilities
   (S + I + W, exactly what the Results tab displays) at PP-like lines, for every line
   that would get a ✅ PLAY (shown P ≥ 2-pick breakeven + 3 pts):

   | side | n | shown P | actual hit | gap |
   |---|---|---|---|---|
   | LESS | 4,533 | 69.8% | 67.5% | -2.3 |
   | MORE | 1,656 | 68.2% | 63.4% | -4.8 |

   Both sides clear the 57.7% 2-pick breakeven by a wide margin on proxy lines. The model
   **overstates its MORE plays twice as much as its LESS plays.** If you distrust one
   side, distrust the overs.
4. **Support MORE is the one losing spot.** Support overs at PP-like lines hit 55.8% when
   shown at 67.3% (n=267), below breakeven. Support unders are fine (68.2% at 70.5%).
   The app now flags these **SUP MORE**.
5. **Worlds 2026 leans the model toward OVERS, not unders.** Worlds 2026 is running 12.9
   team kills per map. The model's Worlds environment is 14.6, because the event average
   it uses still contains the 2025 edition (15.0). The model's Worlds 2026 projections are
   0.43 kills per player-series too high (n=69). So at this Worlds, a LESS from the model
   is, if anything, conservative.
   A fix that measures only the current edition (toggle E) was tested and **rejected**. It
   helped Worlds 2026 (CRPS -0.022, se 0.046, only 7 series) but hurt MSI 2026 (+0.083,
   se 0.034) and FST 2026 (+0.056, se 0.038), because early in an event too few games
   exist. It stays off (`event_env_days = 0`).

## Floors and goblins (new in the app)

- **The lower tail is calibrated.** P(kills ≥ f), for f from the model median -7 to +4:
  predicted 82.5 / 87.6 / 92.6 / 96.5% hit 82.6 / 88.0 / 93.4 / 97.4%. The model is
  slightly conservative on floors. **Supports are the exception:** their floors run 4 pts
  hot (86.4% predicted, 82.3% hit).
- **Goblin proxies** (the PP-like line minus g kills, raw model):
  - g = 2 to 4 kills under: P(more) of 85-95% hit within -3 to +2 pts of the prediction.
  - g = 1 kill under: 85-90% ran 7 pts hot (87% predicted, 80% hit). A 1-kill goblin is
    nearly the standard line, where the model is overconfident.
- The app now:
  - reads `g` / `goblin` and `d` / `demon` after a line (`Chovy 7.5 g`). Those rows are
    MORE only and never ✅ PLAY, because the edge is computed against the standard
    multiplier and goblins pay less.
  - shows **floor 80 / floor 90**: kills the player reaches in 80% / 90% of this matchup's
    sims. Series length, win odds and the opponent are all built in.
  - flags **SAFE FLOOR** when shown P(more) ≥ 85% and the player is not a support. The
    shown P is below the raw model, so this is conservative.
  - adds a **Safest overs & floors** list (Results tab), ranked by shown P(more).
  - takes a **payout override** in the Slip tab when a slip has a goblin or demon leg, so
    the EV uses what PrizePicks actually shows.

## What was checked in the code (no under-bias found)

- Market by format: BO1 = Map 1, BO3 = Maps 1-2 (both maps always played), BO5 = Maps 1-3.
  Kills count only on maps played.
- A push counts as a loss for **both** sides (symmetric, conservative).
- S (spread inflation) is symmetric about the mean. The clip at 0 nudges the mean *up*
  (pro-over, negligible).
- The I calibration map is close to identity (0.30 → 0.31, 0.70 → 0.71).
- W shrinks both sides equally toward the line. After W, the top-5 picks per slate at
  median proxy lines: LESS shown 80.1% hit 79.6%, MORE shown 79.2% hit 80.3%.
- Opponent adjustment (O): outcome-conditional, clamped [0.80, 1.20]. It removed the old
  -0.13 low bias (`reports/backtest_bias.md`).
- Kill share: patch-weighted over all of a player's games, shrunk below 30 games. Lineup =
  latest starters, with the line's player swapped in.
- Team names: no team at Worlds or in a major league is split across names with a thin
  pool.

## Open items for you

1. **Minimum edge (done 2026-10-24: raised to 7 pts).** The weekly refit (`reports/calibration_fit.md`, 2026-10-19) now
   derives **7 pts**. The app still uses the signed-off 3 pts. The 7 comes from the
   top-1-per-slate sample (n=470, gap -3.2 ± 1.7). Top-3 / top-5 need 3.6 / 3.1 pts.
   Raising `MIN_EDGE` needs your sign-off.
2. **Turn on logging (section 6).** Real PrizePicks lines are the only way to be sure. Set
   the `[github]` secrets on Streamlit Cloud, open Record → Log connection (it creates
   `picks-log`), then run every slate. At 300 graded lines, W is refit on real lines
   (`scripts/refit_w_real.py`).
3. SUP MORE stays a flag only (user, 2026-10-24).

## 6. Logging is not reaching GitHub

`git ls-remote` shows no `picks-log` and no `data` branch. So:
- no slate or slip has ever been saved to GitHub,
- `grade.yml` has nothing to grade,
- in-app uploads stay on the app's temporary disk and are lost on restart. The app then
  falls back to the parquet bundled in `main` (through 2026-10-23).

This points to missing or invalid `[github] token / repo` secrets on Streamlit Cloud.
Record → Log connection tests the token.
