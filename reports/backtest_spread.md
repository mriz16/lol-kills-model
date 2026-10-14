# Spread, calibration and selection-bias backtest (real OE data, walk-forward, 20,000 sims)

24,979 player-series, 2025-03-28 -> 2026-10-13. Base = live locked config **L+O** (shrink30 + A + B50 + O). Paired deltas vs L+O with standard errors.

## Recommendation

| | CRPS vs L+O (se) | log vs L+O (se) | bias | ECE half-pt | tail obs/pred | PIT var ratio | top-5 shortfall |
|---|---|---|---|---|---|---|---|
| L+O (live locked) | – | – | -0.069 | 0.0093 | 1.058 | 1.047 | -4.1 pts |
| **+S** | **-0.0012 (0.0003)** | -0.0008 (0.0006) | -0.064 | 0.0083 | 1.031 | **0.997** | -3.1 |
| +I iso | +0.0005 (0.0004) | +0.0236 (0.0017) | -0.017 | 0.0061 | 1.007 | 1.033 | -3.4 |
| **+I beta** | +0.0003 (0.0005) | +0.0097 (0.0012) | -0.028 | **0.0045** | 1.006 | 1.029 | -3.6 |
| +S+I beta | -0.0004 (0.0006) | +0.0146 (0.0019) | -0.019 | **0.0037** | 0.997 | 0.992 | -3.0 |
| +K30 / +K100 | +0.0002 / -0.0002 (0.0003) | +0.0003 / -0.0005 | -0.069 | 0.0086 / 0.0087 | 1.058 | 1.048 | – |

("top-5 shortfall" = hit rate minus predicted P for the 5 biggest edges per slate at proxy
lines; negative = the model overstated its picks.)

1. **Lock S (spread inflation). Needs your sign-off.** It adds a step after the simulation:
   each sim's player total becomes `mean + k (x - mean)` (stochastically rounded, clipped at
   0), with k fitted walk-forward per position x market as sqrt(sum (y - mean)^2 / sum var)
   on series that started at least a day earlier, shrunk to the pooled k (n / (n + 300)).
   Locked steps a-f do not change. Gains: CRPS -0.0012 (4 se), the PIT variance goes from
   1.047 to 0.997, PIT chi^2 halves (103 -> 50), the tail ratio drops from 1.058 to 1.031,
   and the top-5 shortfall shrinks by 1 pt. Current k: 1.01-1.09 (bot and Maps 1-3 are the
   widest). **Not wired into the app.** The pairs for the S+I calibrator are already in
   `data/calib_pairs.parquet` (model `L+O+S`).
2. **I (probability calibration) is a line-probability layer, not a distribution fix. Beta
   over isotonic.** At the half-point lines the model was already close to calibrated:
   Brier is unchanged (+0.0000, se 0.0001), and I cuts ECE from 0.0093 to 0.0045 (beta) or
   0.0061 (iso). It also removes the under asymmetry described below. Applied to the whole
   pmf, it makes the log score worse (+0.010 beta, +0.024 iso), because a monotone remap of
   the CDF makes the pmf lumpy. So use I only for P(more) / P(less) at a line, never for
   the sims. Beta is smooth, has the lowest ECE and the smallest log-score cost. **The app
   now shows I-beta (on L+O) by default**, as asked. Once S is locked, switch
   `CALIB_MODEL` to `L+O+S` (S+I beta: ECE 0.0037, PIT var 0.992, tails 0.997).
3. **Reject K (per-player share concentration).** The change is ±0.0002 CRPS (se 0.0003)
   with n0 = 30 or 100, and PIT is unchanged. Players' kill-share dispersion is not where
   the spread error is.
4. **Minimum edge: 5 points, after calibration** (`MIN_EDGE` in `src/app/slate.py`). With
   I-beta, the top 3 / 5 / 10 calibrated edges per slate still hit 3.4 / 3.6 / 3.3 points
   (se ~1) under their predicted P. Across every pick with edge > 0 it is 1.9 points. A
   5-point calibrated edge therefore leaves about 1.5-3 points of real edge on the plays you
   would actually pick. Uncalibrated rows never get a PLAY.

## What the backtest says about the under lean

- **The spread was only a minor problem.** The PIT is close to flat (mean 0.507). The
  variance ratio is 1.047, so the model is slightly too narrow: bot 1.11, Maps 1-3 1.07,
  top-decile mass 11.7% vs 10%. S fixes that. It is not the main reason for a bad slate.
- **False unders exist, but they are small.** At proxy lines (rolling median + 0.5), the
  model picks LESS 63% of the time, while unders actually hit 54% (the +0.5 puts the line
  above the median, so some under lean is expected). The raw model overstates its unders
  more than its overs: LESS predicted 65.8% and hit 63.2% (-2.6 pts), MORE predicted 62.5%
  and hit 61.3% (-1.2 pts). After I-beta the gaps are -1.5 / -1.0. With isotonic they are
  -1.8 / -1.7. The remaining mean bias (-0.07, top -0.15) still leans the same way.
- **Selection bias is the big effect, and it hits both sides.** Taking the biggest edges on
  a slate means taking the rows where the model disagrees most with the line, and those are
  partly model error. Raw top-5: predicted 82.2%, hit 78.1%. MORE and LESS are overstated by
  similar amounts (top-5: -3.6 / -4.3 pts). Calibration fitted on every line cannot remove
  this. S trims it by about 1 pt, and the minimum edge and the side-balance warning cover
  the rest. A one-sided slate is where these errors stack.
- **Caveat: proxy lines are softer than real PrizePicks lines.** Hit rates of 67-80% would
  not survive real lines. Only the shortfall (hit - predicted) carries over, not hit minus
  breakeven. The real-world check is the slate log (below), which grades every line you
  enter.

## What changed in the app and the pipeline

- **Slate log**: every Run slate writes every line (played or not) to `logs/slates.csv` on
  the `picks-log` branch. Each row has the date, matchup, player, line, platform, p_map, the
  calibrated and raw P(more/less/push), side, edge, mean, median, fair line, games, share,
  flags and `played`. Re-running a slate replaces its rows and keeps `played`. Logging a
  slip, or "Mark lines as played" in Results, sets `played = 1`.
- **Backfill mode** ("Mode: live or backfill" at the top): pick a past date. The whole data
  store (pools, rosters, lineups, kappas, calibrator) is rebuilt from games before 00:00 UTC
  of that date, and the rows are logged with `backfill = 1`. Slips can only be marked
  played; backfill never writes to the pick log.
- **Grading**: `refresh.yml` → `scripts/grade_picks.py` also grades every slate-log row
  (matched to the series around its slate date) into `logs/slates_graded.csv`. The job
  summary and the Record tab show hit rate vs predicted, calibrated and raw, split by side,
  played and backfill.
- **Results**: calibrated P and edge by default, "Show raw model probabilities" toggle, and
  raw vs calibrated in the tap-a-row explanation. ✅ PLAY = calibrated edge >= 5 pts,
  otherwise the BELOW MIN flag. A side-balance warning appears when more than 70% of the
  slate's plays (at least 3) are on one side. Slip joint P = shared-sim joint x each leg's
  calibrated / raw ratio, with the raw slip one tap away.

Reproduce:
`python scripts/run_backtest.py --rows-only --keep-pmf --tag spread --variants L+O L+O+K30 L+O+K100`
(or one process per variant), then
`python scripts/spread_report.py --rows reports/backtest_rows_spread*.parquet`. The second
step also rewrites `data/calib_pairs.parquet`, the app's calibration set.

## Scores

`pit_var_ratio` = variance of the (non-randomized) PIT x 12: 1 is ideal, > 1 means the forecast is too narrow. `pit_lo10` / `pit_hi10` = share of PIT mass in the lowest / highest decile (0.10 ideal). `ece_half` = ECE at half-point lines rounded median -2.5..+2.5. `tail_obs_pred` = hit / predicted at median +1.5/2.5/3.5.

| variant | crps | dcrps | dcrps_se | logscore | dlogscore | dlogscore_se | bias | brier_half | dbrier_h | dbrier_h_se | logloss_half | ece_median | ece_half | tail_obs_pred | pit_mean | pit_var_ratio | pit_chi2 | pit_lo10 | pit_hi10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L+O | 1.8651 | 0.0000 | 0.0000 | 2.4636 | 0.0000 | 0.0000 | -0.0690 | 0.2092 | 0.0000 | 0.0000 | 0.6053 | 0.0151 | 0.0093 | 1.0576 | 0.5074 | 1.0468 | 102.5108 | 0.1042 | 0.1174 |
| L+O+S | 1.8640 | -0.0012 | 0.0003 | 2.4628 | -0.0008 | 0.0006 | -0.0635 | 0.2092 | -0.0000 | 0.0000 | 0.6053 | 0.0151 | 0.0083 | 1.0311 | 0.5094 | 0.9967 | 50.1385 | 0.0895 | 0.1084 |
| L+O+I | 1.8656 | 0.0005 | 0.0004 | 2.4872 | 0.0236 | 0.0017 | -0.0166 | 0.2093 | 0.0001 | 0.0001 | 0.6058 | 0.0142 | 0.0061 | 1.0066 | 0.5021 | 1.0331 | 128.9576 | 0.1055 | 0.1164 |
| L+O+Ib | 1.8654 | 0.0003 | 0.0005 | 2.4734 | 0.0097 | 0.0012 | -0.0283 | 0.2092 | 0.0000 | 0.0001 | 0.6052 | 0.0153 | 0.0045 | 1.0058 | 0.5032 | 1.0287 | 118.9415 | 0.1048 | 0.1153 |
| L+O+S+I | 1.8646 | -0.0005 | 0.0005 | 2.4860 | 0.0223 | 0.0018 | 0.0000 | 0.2093 | 0.0001 | 0.0001 | 0.6056 | 0.0148 | 0.0055 | 0.9930 | 0.5028 | 0.9989 | 61.3813 | 0.0928 | 0.1078 |
| L+O+S+Ib | 1.8647 | -0.0004 | 0.0006 | 2.4782 | 0.0146 | 0.0019 | -0.0191 | 0.2093 | 0.0001 | 0.0001 | 0.6054 | 0.0164 | 0.0037 | 0.9973 | 0.5043 | 0.9918 | 63.2407 | 0.0921 | 0.1066 |
| L+O+K30 | 1.8653 | 0.0002 | 0.0003 | 2.4639 | 0.0003 | 0.0004 | -0.0690 | 0.2092 | 0.0000 | 0.0000 | 0.6054 | 0.0132 | 0.0086 | 1.0575 | 0.5072 | 1.0483 | 103.4288 | 0.1041 | 0.1176 |
| L+O+K100 | 1.8650 | -0.0002 | 0.0002 | 2.4631 | -0.0005 | 0.0004 | -0.0690 | 0.2092 | -0.0000 | 0.0000 | 0.6053 | 0.0138 | 0.0087 | 1.0574 | 0.5073 | 1.0479 | 102.5396 | 0.1040 | 0.1175 |
| L+O+K100+S+I | 1.8647 | -0.0004 | 0.0006 | 2.4877 | 0.0241 | 0.0020 | 0.0002 | 0.2093 | 0.0001 | 0.0001 | 0.6057 | 0.0172 | 0.0052 | 0.9932 | 0.5027 | 0.9987 | 62.7724 | 0.0932 | 0.1081 |

### Bias by position (mean - actual)

| variant | bias_top | bias_jng | bias_mid | bias_bot | bias_sup |
|---|---|---|---|---|---|
| L+O | -0.1479 | -0.0369 | -0.0933 | -0.0790 | 0.0123 |
| L+O+S | -0.1391 | -0.0315 | -0.0890 | -0.0750 | 0.0174 |
| L+O+I | -0.0517 | 0.0008 | -0.0580 | -0.0040 | 0.0301 |
| L+O+Ib | -0.0613 | 0.0011 | -0.0686 | -0.0497 | 0.0370 |
| L+O+S+I | -0.0261 | 0.0198 | -0.0405 | 0.0118 | 0.0350 |
| L+O+S+Ib | -0.0419 | 0.0203 | -0.0643 | -0.0518 | 0.0424 |
| L+O+K30 | -0.1477 | -0.0361 | -0.0956 | -0.0791 | 0.0137 |
| L+O+K100 | -0.1482 | -0.0363 | -0.0950 | -0.0784 | 0.0131 |
| L+O+K100+S+I | -0.0268 | 0.0205 | -0.0418 | 0.0129 | 0.0361 |

## PIT histogram, L+O (% of mass per decile)

### by position

| group | n | 0-10 | 10-20 | 20-30 | 30-40 | 40-50 | 50-60 | 60-70 | 70-80 | 80-90 | 90-100 | mean | var_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top | 4991 | 9.739 | 9.765 | 9.533 | 9.702 | 9.409 | 9.356 | 9.433 | 9.985 | 10.477 | 12.601 | 0.516 | 1.053 |
| jng | 4999 | 10.405 | 9.534 | 9.552 | 9.913 | 10.047 | 10.517 | 9.812 | 9.464 | 9.255 | 11.500 | 0.504 | 1.022 |
| mid | 4999 | 9.974 | 10.527 | 9.531 | 9.386 | 9.665 | 9.508 | 9.710 | 10.210 | 10.012 | 11.418 | 0.507 | 1.030 |
| bot | 5002 | 11.808 | 10.487 | 9.417 | 9.106 | 8.343 | 8.951 | 9.546 | 10.090 | 9.718 | 12.534 | 0.503 | 1.114 |
| sup | 4988 | 10.178 | 9.677 | 9.615 | 9.753 | 9.826 | 10.044 | 10.104 | 10.095 | 10.053 | 10.655 | 0.506 | 1.015 |

### by market

| group | n | 0-10 | 10-20 | 20-30 | 30-40 | 40-50 | 50-60 | 60-70 | 70-80 | 80-90 | 90-100 | mean | var_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Map 1 | 5040 | 10.191 | 9.554 | 9.962 | 9.559 | 9.482 | 9.693 | 10.317 | 10.480 | 9.747 | 11.015 | 0.507 | 1.020 |
| Maps 1-2 | 15579 | 10.192 | 10.164 | 9.369 | 9.442 | 9.562 | 9.802 | 9.659 | 9.798 | 10.085 | 11.923 | 0.509 | 1.048 |
| Maps 1-3 | 4360 | 11.507 | 9.921 | 9.605 | 10.052 | 9.059 | 9.203 | 9.252 | 9.989 | 9.431 | 11.936 | 0.500 | 1.072 |

## PIT histogram, L+O+S (% of mass per decile)

### by position

| group | n | 0-10 | 10-20 | 20-30 | 30-40 | 40-50 | 50-60 | 60-70 | 70-80 | 80-90 | 90-100 | mean | var_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top | 4991 | 8.056 | 9.802 | 10.005 | 10.320 | 9.863 | 9.545 | 9.838 | 10.527 | 10.698 | 11.346 | 0.519 | 0.988 |
| jng | 4999 | 8.972 | 9.864 | 9.758 | 10.385 | 10.396 | 10.669 | 10.138 | 9.794 | 9.280 | 10.743 | 0.506 | 0.974 |
| mid | 4999 | 8.167 | 10.672 | 10.127 | 9.909 | 10.129 | 9.710 | 10.200 | 10.617 | 9.997 | 10.472 | 0.510 | 0.978 |
| bot | 5002 | 9.491 | 10.713 | 10.254 | 9.760 | 8.855 | 9.207 | 10.186 | 10.531 | 9.856 | 11.148 | 0.506 | 1.031 |
| sup | 4988 | 10.070 | 9.690 | 9.673 | 9.812 | 9.849 | 10.028 | 10.115 | 10.148 | 10.119 | 10.496 | 0.505 | 1.010 |

### by market

| group | n | 0-10 | 10-20 | 20-30 | 30-40 | 40-50 | 50-60 | 60-70 | 70-80 | 80-90 | 90-100 | mean | var_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Map 1 | 5040 | 9.203 | 9.602 | 10.224 | 9.942 | 9.709 | 9.799 | 10.536 | 10.716 | 9.862 | 10.407 | 0.509 | 0.985 |
| Maps 1-2 | 15579 | 8.643 | 10.393 | 9.836 | 9.871 | 9.931 | 9.956 | 10.031 | 10.152 | 10.198 | 10.989 | 0.511 | 0.996 |
| Maps 1-3 | 4360 | 9.760 | 9.907 | 10.117 | 10.742 | 9.542 | 9.427 | 9.817 | 10.483 | 9.393 | 10.812 | 0.503 | 1.011 |

## PIT histogram, L+O+S+Ib (% of mass per decile)

### by position

| group | n | 0-10 | 10-20 | 20-30 | 30-40 | 40-50 | 50-60 | 60-70 | 70-80 | 80-90 | 90-100 | mean | var_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top | 4991 | 8.810 | 11.118 | 9.536 | 9.521 | 9.569 | 10.115 | 11.125 | 9.486 | 9.444 | 11.276 | 0.508 | 1.000 |
| jng | 4999 | 9.194 | 10.553 | 10.092 | 9.941 | 10.130 | 10.609 | 10.384 | 9.192 | 9.161 | 10.744 | 0.501 | 0.986 |
| mid | 4999 | 8.408 | 11.193 | 10.012 | 9.291 | 9.842 | 9.979 | 11.133 | 10.036 | 9.645 | 10.460 | 0.507 | 0.983 |
| bot | 5002 | 9.559 | 10.989 | 10.669 | 9.020 | 8.960 | 9.759 | 10.305 | 9.909 | 9.681 | 11.148 | 0.503 | 1.001 |
| sup | 4988 | 10.059 | 9.662 | 9.733 | 9.948 | 10.077 | 10.218 | 10.351 | 10.348 | 9.926 | 9.679 | 0.502 | 0.989 |

### by market

| group | n | 0-10 | 10-20 | 20-30 | 30-40 | 40-50 | 50-60 | 60-70 | 70-80 | 80-90 | 90-100 | mean | var_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Map 1 | 5040 | 9.783 | 10.023 | 10.262 | 9.762 | 9.635 | 9.927 | 10.796 | 10.317 | 9.270 | 10.226 | 0.501 | 0.991 |
| Maps 1-2 | 15579 | 8.835 | 11.050 | 9.829 | 9.537 | 9.962 | 10.425 | 10.694 | 9.111 | 9.760 | 10.797 | 0.505 | 0.994 |
| Maps 1-3 | 4360 | 9.865 | 10.253 | 10.358 | 9.317 | 8.929 | 9.343 | 10.378 | 11.631 | 9.244 | 10.681 | 0.505 | 0.986 |

## PIT histogram, L+O+K100 (% of mass per decile)

### by position

| group | n | 0-10 | 10-20 | 20-30 | 30-40 | 40-50 | 50-60 | 60-70 | 70-80 | 80-90 | 90-100 | mean | var_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top | 4991 | 9.700 | 9.729 | 9.587 | 9.756 | 9.388 | 9.385 | 9.397 | 10.012 | 10.436 | 12.610 | 0.516 | 1.054 |
| jng | 4999 | 10.368 | 9.595 | 9.574 | 9.828 | 10.049 | 10.534 | 9.842 | 9.512 | 9.178 | 11.500 | 0.504 | 1.022 |
| mid | 4999 | 9.969 | 10.472 | 9.583 | 9.320 | 9.717 | 9.495 | 9.762 | 10.198 | 10.043 | 11.401 | 0.508 | 1.033 |
| bot | 5002 | 11.759 | 10.580 | 9.384 | 9.062 | 8.360 | 8.994 | 9.543 | 10.090 | 9.696 | 12.532 | 0.503 | 1.113 |
| sup | 4988 | 10.195 | 9.745 | 9.655 | 9.737 | 9.850 | 10.003 | 10.051 | 10.075 | 9.984 | 10.705 | 0.505 | 1.017 |

### by market

| group | n | 0-10 | 10-20 | 20-30 | 30-40 | 40-50 | 50-60 | 60-70 | 70-80 | 80-90 | 90-100 | mean | var_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Map 1 | 5040 | 10.190 | 9.559 | 9.970 | 9.554 | 9.500 | 9.680 | 10.356 | 10.436 | 9.709 | 11.047 | 0.507 | 1.023 |
| Maps 1-2 | 15579 | 10.164 | 10.177 | 9.395 | 9.429 | 9.569 | 9.811 | 9.654 | 9.811 | 10.058 | 11.932 | 0.509 | 1.049 |
| Maps 1-3 | 4360 | 11.478 | 10.016 | 9.656 | 9.923 | 9.096 | 9.225 | 9.214 | 10.042 | 9.369 | 11.912 | 0.500 | 1.072 |

## S: latest walk-forward k (position x market)

| position | Map 1 | Maps 1-2 | Maps 1-3 |
|---|---|---|---|
| bot | 1.056 | 1.086 | 1.091 |
| jng | 1.040 | 1.025 | 1.076 |
| mid | 1.036 | 1.065 | 1.063 |
| sup | 1.066 | 1.020 | 1.009 |
| top | 1.046 | 1.061 | 1.065 |

## Selection-bias check (proxy lines, PrizePicks 2-leg breakeven 57.7%)

Proxy line = floor(median of the player's last 10 prior market totals) + 0.5 (fallback n_maps x median of the last 20 maps). Slate = UTC date. Top-N = the N biggest edges (edge > 0) per slate.

### L+O

| selection | n | pred | hit | diff | se | n_more | pred_more | hit_more | n_less | pred_less | hit_less |
|---|---|---|---|---|---|---|---|---|---|---|---|
| top 3 | 1341 | 0.8448 | 0.8054 | -0.0395 | 0.0108 | 340 | 0.8355 | 0.8029 | 1001 | 0.8480 | 0.8062 |
| top 5 | 2228 | 0.8220 | 0.7810 | -0.0410 | 0.0088 | 587 | 0.8110 | 0.7751 | 1641 | 0.8259 | 0.7831 |
| top 10 | 4304 | 0.7870 | 0.7472 | -0.0398 | 0.0066 | 1176 | 0.7716 | 0.7364 | 3128 | 0.7928 | 0.7513 |
| all edge > 0% | 16171 | 0.6941 | 0.6666 | -0.0275 | 0.0037 | 5280 | 0.6792 | 0.6640 | 10891 | 0.7013 | 0.6678 |

### L+O+S

| selection | n | pred | hit | diff | se | n_more | pred_more | hit_more | n_less | pred_less | hit_less |
|---|---|---|---|---|---|---|---|---|---|---|---|
| top 3 | 1341 | 0.8350 | 0.8069 | -0.0281 | 0.0108 | 315 | 0.8209 | 0.8032 | 1026 | 0.8393 | 0.8080 |
| top 5 | 2228 | 0.8125 | 0.7819 | -0.0307 | 0.0087 | 551 | 0.7964 | 0.7786 | 1677 | 0.8178 | 0.7829 |
| top 10 | 4301 | 0.7786 | 0.7466 | -0.0320 | 0.0066 | 1118 | 0.7599 | 0.7415 | 3183 | 0.7852 | 0.7484 |
| all edge > 0% | 16033 | 0.6897 | 0.6670 | -0.0227 | 0.0037 | 5146 | 0.6730 | 0.6654 | 10887 | 0.6976 | 0.6678 |

### L+O+I

| selection | n | pred | hit | diff | se | n_more | pred_more | hit_more | n_less | pred_less | hit_less |
|---|---|---|---|---|---|---|---|---|---|---|---|
| top 3 | 1340 | 0.8428 | 0.7970 | -0.0457 | 0.0110 | 382 | 0.8315 | 0.7853 | 958 | 0.8472 | 0.8017 |
| top 5 | 2226 | 0.8184 | 0.7848 | -0.0336 | 0.0087 | 649 | 0.8061 | 0.7673 | 1577 | 0.8235 | 0.7920 |
| top 10 | 4289 | 0.7815 | 0.7498 | -0.0317 | 0.0066 | 1287 | 0.7707 | 0.7350 | 3002 | 0.7861 | 0.7562 |
| all edge > 0% | 15699 | 0.6910 | 0.6697 | -0.0213 | 0.0038 | 5404 | 0.6793 | 0.6621 | 10295 | 0.6971 | 0.6736 |

### L+O+Ib

| selection | n | pred | hit | diff | se | n_more | pred_more | hit_more | n_less | pred_less | hit_less |
|---|---|---|---|---|---|---|---|---|---|---|---|
| top 3 | 1340 | 0.8408 | 0.8067 | -0.0341 | 0.0108 | 371 | 0.8314 | 0.7951 | 969 | 0.8445 | 0.8111 |
| top 5 | 2226 | 0.8171 | 0.7817 | -0.0355 | 0.0088 | 641 | 0.8083 | 0.7660 | 1585 | 0.8207 | 0.7880 |
| top 10 | 4298 | 0.7804 | 0.7473 | -0.0331 | 0.0066 | 1282 | 0.7711 | 0.7418 | 3016 | 0.7844 | 0.7497 |
| all edge > 0% | 15733 | 0.6886 | 0.6698 | -0.0188 | 0.0037 | 5251 | 0.6795 | 0.6639 | 10482 | 0.6931 | 0.6728 |

### L+O+S+I

| selection | n | pred | hit | diff | se | n_more | pred_more | hit_more | n_less | pred_less | hit_less |
|---|---|---|---|---|---|---|---|---|---|---|---|
| top 3 | 1341 | 0.8345 | 0.8024 | -0.0322 | 0.0109 | 358 | 0.8192 | 0.7877 | 983 | 0.8401 | 0.8077 |
| top 5 | 2226 | 0.8113 | 0.7830 | -0.0283 | 0.0087 | 626 | 0.7950 | 0.7732 | 1600 | 0.8177 | 0.7869 |
| top 10 | 4285 | 0.7764 | 0.7515 | -0.0249 | 0.0066 | 1278 | 0.7637 | 0.7379 | 3007 | 0.7818 | 0.7572 |
| all edge > 0% | 15724 | 0.6891 | 0.6699 | -0.0192 | 0.0038 | 5428 | 0.6765 | 0.6623 | 10296 | 0.6958 | 0.6740 |

### L+O+S+Ib

| selection | n | pred | hit | diff | se | n_more | pred_more | hit_more | n_less | pred_less | hit_less |
|---|---|---|---|---|---|---|---|---|---|---|---|
| top 3 | 1341 | 0.8329 | 0.8076 | -0.0253 | 0.0108 | 352 | 0.8206 | 0.8068 | 989 | 0.8372 | 0.8079 |
| top 5 | 2228 | 0.8104 | 0.7805 | -0.0299 | 0.0088 | 614 | 0.7992 | 0.7769 | 1614 | 0.8146 | 0.7819 |
| top 10 | 4311 | 0.7756 | 0.7453 | -0.0303 | 0.0066 | 1250 | 0.7656 | 0.7408 | 3061 | 0.7797 | 0.7471 |
| all edge > 0% | 15923 | 0.6869 | 0.6688 | -0.0181 | 0.0037 | 5226 | 0.6776 | 0.6648 | 10697 | 0.6914 | 0.6707 |

### L+O+K100+S+I

| selection | n | pred | hit | diff | se | n_more | pred_more | hit_more | n_less | pred_less | hit_less |
|---|---|---|---|---|---|---|---|---|---|---|---|
| top 3 | 1341 | 0.8340 | 0.8009 | -0.0331 | 0.0109 | 349 | 0.8179 | 0.7679 | 992 | 0.8397 | 0.8125 |
| top 5 | 2227 | 0.8108 | 0.7845 | -0.0263 | 0.0087 | 602 | 0.7939 | 0.7691 | 1625 | 0.8170 | 0.7902 |
| top 10 | 4291 | 0.7755 | 0.7544 | -0.0211 | 0.0066 | 1273 | 0.7614 | 0.7455 | 3018 | 0.7814 | 0.7581 |
| all edge > 0% | 15612 | 0.6892 | 0.6689 | -0.0203 | 0.0038 | 5478 | 0.6749 | 0.6588 | 10134 | 0.6969 | 0.6744 |

## Model side at all proxy lines (no selection)

| variant | n | share_less | pred_less | hit_less | pred_more | hit_more | actual_under_rate |
|---|---|---|---|---|---|---|---|
| L+O | 23395 | 0.6336 | 0.6584 | 0.6319 | 0.6247 | 0.6131 | 0.5421 |
| L+O+S | 23395 | 0.6335 | 0.6557 | 0.6319 | 0.6188 | 0.6130 | 0.5421 |
| L+O+I | 23395 | 0.6105 | 0.6544 | 0.6368 | 0.6231 | 0.6063 | 0.5421 |
| L+O+Ib | 23395 | 0.6295 | 0.6489 | 0.6337 | 0.6234 | 0.6135 | 0.5421 |
| L+O+S+I | 23395 | 0.6109 | 0.6534 | 0.6367 | 0.6217 | 0.6065 | 0.5421 |
| L+O+S+Ib | 23395 | 0.6307 | 0.6497 | 0.6329 | 0.6223 | 0.6128 | 0.5421 |
| L+O+K100+S+I | 23395 | 0.6177 | 0.6518 | 0.6355 | 0.6238 | 0.6088 | 0.5421 |

## Edge bands, L+O (all proxy-line picks, not cumulative)

| selection | n | pred | hit | diff | se | n_more | pred_more | hit_more | n_less | pred_less | hit_less |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-2 pts | 1747 | 0.5872 | 0.5753 | -0.0119 | 0.0118 | 721 | 0.5871 | 0.6006 | 1026 | 0.5872 | 0.5575 |
| 2-4 pts | 1731 | 0.6073 | 0.5945 | -0.0129 | 0.0118 | 668 | 0.6072 | 0.6123 | 1063 | 0.6074 | 0.5833 |
| 4-6 pts | 1616 | 0.6275 | 0.5996 | -0.0279 | 0.0122 | 617 | 0.6272 | 0.5948 | 999 | 0.6277 | 0.6026 |
| 6-8 pts | 1586 | 0.6472 | 0.6148 | -0.0324 | 0.0122 | 554 | 0.6474 | 0.6119 | 1032 | 0.6471 | 0.6163 |
| 8-10 pts | 1378 | 0.6673 | 0.6459 | -0.0214 | 0.0129 | 466 | 0.6675 | 0.6481 | 912 | 0.6672 | 0.6447 |
| 10-15 pts | 3017 | 0.7013 | 0.6672 | -0.0341 | 0.0086 | 952 | 0.7007 | 0.6880 | 2065 | 0.7015 | 0.6576 |
| > 15 pts | 5096 | 0.7989 | 0.7649 | -0.0340 | 0.0059 | 1302 | 0.7938 | 0.7688 | 3794 | 0.8006 | 0.7636 |

## Edge bands, L+O+I (all proxy-line picks, not cumulative)

| selection | n | pred | hit | diff | se | n_more | pred_more | hit_more | n_less | pred_less | hit_less |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-2 pts | 1458 | 0.5875 | 0.5748 | -0.0127 | 0.0129 | 655 | 0.5872 | 0.5985 | 803 | 0.5877 | 0.5554 |
| 2-4 pts | 1642 | 0.6075 | 0.6023 | -0.0052 | 0.0121 | 707 | 0.6077 | 0.6096 | 935 | 0.6074 | 0.5968 |
| 4-6 pts | 1805 | 0.6277 | 0.6188 | -0.0088 | 0.0114 | 674 | 0.6271 | 0.5935 | 1131 | 0.6280 | 0.6340 |
| 6-8 pts | 1972 | 0.6475 | 0.6182 | -0.0294 | 0.0109 | 541 | 0.6476 | 0.6229 | 1431 | 0.6475 | 0.6164 |
| 8-10 pts | 1560 | 0.6668 | 0.6462 | -0.0207 | 0.0121 | 589 | 0.6673 | 0.6570 | 971 | 0.6665 | 0.6395 |
| 10-15 pts | 2777 | 0.7020 | 0.6770 | -0.0250 | 0.0089 | 925 | 0.6993 | 0.6724 | 1852 | 0.7034 | 0.6793 |
| > 15 pts | 4485 | 0.8013 | 0.7719 | -0.0293 | 0.0063 | 1313 | 0.7951 | 0.7685 | 3172 | 0.8038 | 0.7733 |

## Edge bands, L+O+Ib (all proxy-line picks, not cumulative)

| selection | n | pred | hit | diff | se | n_more | pred_more | hit_more | n_less | pred_less | hit_less |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-2 pts | 1825 | 0.5873 | 0.5825 | -0.0049 | 0.0115 | 728 | 0.5870 | 0.6030 | 1097 | 0.5876 | 0.5688 |
| 2-4 pts | 1794 | 0.6072 | 0.6120 | 0.0048 | 0.0115 | 653 | 0.6069 | 0.6049 | 1141 | 0.6074 | 0.6161 |
| 4-6 pts | 1765 | 0.6271 | 0.5887 | -0.0384 | 0.0117 | 631 | 0.6271 | 0.5880 | 1134 | 0.6271 | 0.5891 |
| 6-8 pts | 1605 | 0.6473 | 0.6393 | -0.0080 | 0.0120 | 533 | 0.6472 | 0.6229 | 1072 | 0.6473 | 0.6474 |
| 8-10 pts | 1363 | 0.6674 | 0.6735 | 0.0061 | 0.0127 | 452 | 0.6676 | 0.6881 | 911 | 0.6673 | 0.6663 |
| 10-15 pts | 2896 | 0.7005 | 0.6699 | -0.0306 | 0.0087 | 945 | 0.7005 | 0.6667 | 1951 | 0.7006 | 0.6715 |
| > 15 pts | 4485 | 0.7999 | 0.7701 | -0.0298 | 0.0063 | 1309 | 0.7946 | 0.7701 | 3176 | 0.8021 | 0.7702 |

## Edge bands, L+O+S+I (all proxy-line picks, not cumulative)

| selection | n | pred | hit | diff | se | n_more | pred_more | hit_more | n_less | pred_less | hit_less |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-2 pts | 1411 | 0.5876 | 0.5726 | -0.0149 | 0.0132 | 691 | 0.5877 | 0.5890 | 720 | 0.5874 | 0.5569 |
| 2-4 pts | 1864 | 0.6076 | 0.6121 | 0.0045 | 0.0113 | 777 | 0.6072 | 0.6165 | 1087 | 0.6079 | 0.6090 |
| 4-6 pts | 1627 | 0.6268 | 0.6023 | -0.0245 | 0.0121 | 564 | 0.6262 | 0.5833 | 1063 | 0.6272 | 0.6124 |
| 6-8 pts | 1908 | 0.6481 | 0.6394 | -0.0087 | 0.0110 | 573 | 0.6486 | 0.6562 | 1335 | 0.6479 | 0.6322 |
| 8-10 pts | 1667 | 0.6661 | 0.6323 | -0.0339 | 0.0118 | 580 | 0.6665 | 0.6345 | 1087 | 0.6659 | 0.6311 |
| 10-15 pts | 2819 | 0.7022 | 0.6765 | -0.0257 | 0.0088 | 964 | 0.6986 | 0.6691 | 1855 | 0.7040 | 0.6803 |
| > 15 pts | 4428 | 0.7968 | 0.7733 | -0.0235 | 0.0063 | 1279 | 0.7892 | 0.7748 | 3149 | 0.7998 | 0.7726 |

## Edge bands, L+O+S+Ib (all proxy-line picks, not cumulative)

| selection | n | pred | hit | diff | se | n_more | pred_more | hit_more | n_less | pred_less | hit_less |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-2 pts | 1821 | 0.5872 | 0.5805 | -0.0068 | 0.0116 | 711 | 0.5870 | 0.6006 | 1110 | 0.5874 | 0.5676 |
| 2-4 pts | 1834 | 0.6073 | 0.6140 | 0.0066 | 0.0114 | 665 | 0.6072 | 0.6120 | 1169 | 0.6074 | 0.6151 |
| 4-6 pts | 1792 | 0.6272 | 0.5882 | -0.0391 | 0.0116 | 638 | 0.6271 | 0.5831 | 1154 | 0.6273 | 0.5910 |
| 6-8 pts | 1581 | 0.6473 | 0.6344 | -0.0129 | 0.0121 | 501 | 0.6474 | 0.6367 | 1080 | 0.6472 | 0.6333 |
| 8-10 pts | 1445 | 0.6673 | 0.6706 | 0.0033 | 0.0124 | 470 | 0.6672 | 0.6681 | 975 | 0.6673 | 0.6718 |
| 10-15 pts | 2970 | 0.7008 | 0.6710 | -0.0297 | 0.0086 | 961 | 0.7003 | 0.6660 | 2009 | 0.7010 | 0.6735 |
| > 15 pts | 4480 | 0.7949 | 0.7694 | -0.0254 | 0.0063 | 1280 | 0.7884 | 0.7773 | 3200 | 0.7974 | 0.7662 |

## Minimum edge (L+O+Ib, every pick above the threshold)

| selection | n | pred | hit | diff | se | n_more | pred_more | hit_more | n_less | pred_less | hit_less | hit_minus_be |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge > 0 pts | 15733 | 0.6886 | 0.6698 | -0.0188 | 0.0037 | 5251 | 0.6795 | 0.6639 | 10482 | 0.6931 | 0.6728 | 0.0925 |
| edge > 2 pts | 13908 | 0.7018 | 0.6813 | -0.0206 | 0.0040 | 4523 | 0.6944 | 0.6737 | 9385 | 0.7054 | 0.6849 | 0.1039 |
| edge > 4 pts | 12114 | 0.7159 | 0.6915 | -0.0243 | 0.0042 | 3870 | 0.7092 | 0.6853 | 8244 | 0.7190 | 0.6944 | 0.1142 |
| edge > 6 pts | 10349 | 0.7310 | 0.7091 | -0.0219 | 0.0045 | 3239 | 0.7252 | 0.7042 | 7110 | 0.7336 | 0.7113 | 0.1317 |
| edge > 8 pts | 8744 | 0.7464 | 0.7219 | -0.0245 | 0.0048 | 2706 | 0.7405 | 0.7203 | 6038 | 0.7490 | 0.7226 | 0.1445 |
| edge > 10 pts | 7381 | 0.7609 | 0.7308 | -0.0301 | 0.0052 | 2254 | 0.7552 | 0.7267 | 5127 | 0.7635 | 0.7326 | 0.1534 |

## Minimum edge (L+O+S+Ib, every pick above the threshold)

| selection | n | pred | hit | diff | se | n_more | pred_more | hit_more | n_less | pred_less | hit_less | hit_minus_be |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge > 0 pts | 15923 | 0.6869 | 0.6688 | -0.0181 | 0.0037 | 5226 | 0.6776 | 0.6648 | 10697 | 0.6914 | 0.6707 | 0.0914 |
| edge > 2 pts | 14102 | 0.6997 | 0.6802 | -0.0195 | 0.0039 | 4515 | 0.6919 | 0.6749 | 9587 | 0.7034 | 0.6827 | 0.1028 |
| edge > 4 pts | 12268 | 0.7136 | 0.6901 | -0.0235 | 0.0042 | 3850 | 0.7065 | 0.6857 | 8418 | 0.7168 | 0.6921 | 0.1127 |
| edge > 6 pts | 10476 | 0.7283 | 0.7075 | -0.0208 | 0.0044 | 3212 | 0.7223 | 0.7061 | 7264 | 0.7310 | 0.7081 | 0.1302 |
| edge > 8 pts | 8895 | 0.7427 | 0.7205 | -0.0222 | 0.0048 | 2711 | 0.7362 | 0.7189 | 6184 | 0.7456 | 0.7212 | 0.1432 |
| edge > 10 pts | 7450 | 0.7573 | 0.7302 | -0.0271 | 0.0051 | 2241 | 0.7506 | 0.7296 | 5209 | 0.7602 | 0.7305 | 0.1529 |

## Minimum edge (L+O, every pick above the threshold)

| selection | n | pred | hit | diff | se | n_more | pred_more | hit_more | n_less | pred_less | hit_less | hit_minus_be |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge > 0 pts | 16171 | 0.6941 | 0.6666 | -0.0275 | 0.0037 | 5280 | 0.6792 | 0.6640 | 10891 | 0.7013 | 0.6678 | 0.0892 |
| edge > 2 pts | 14424 | 0.7070 | 0.6776 | -0.0294 | 0.0039 | 4559 | 0.6938 | 0.6741 | 9865 | 0.7131 | 0.6793 | 0.1003 |
| edge > 4 pts | 12693 | 0.7206 | 0.6890 | -0.0317 | 0.0041 | 3891 | 0.7087 | 0.6847 | 8802 | 0.7259 | 0.6909 | 0.1116 |
| edge > 6 pts | 11077 | 0.7342 | 0.7020 | -0.0322 | 0.0043 | 3274 | 0.7240 | 0.7016 | 7803 | 0.7385 | 0.7022 | 0.1246 |
| edge > 8 pts | 9491 | 0.7487 | 0.7166 | -0.0322 | 0.0046 | 2720 | 0.7396 | 0.7199 | 6771 | 0.7524 | 0.7153 | 0.1392 |
| edge > 10 pts | 8113 | 0.7626 | 0.7286 | -0.0340 | 0.0049 | 2254 | 0.7545 | 0.7347 | 5859 | 0.7657 | 0.7262 | 0.1512 |
