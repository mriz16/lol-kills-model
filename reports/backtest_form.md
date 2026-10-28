# Projection study: roster, outcome share, form (2026-10-28)

Walk-forward, series from 2026-03-01, 16,004 player-series (primary market), 10k sims. Paired differences vs the live locked config L+O; lower is better; se clustered by series.

## 1. Engine toggles

- R1 / R2: team-pool games weighted by (share of today's starters who played) ^ 1 / 2
- Q10 / Q30: kill share in wins / losses, shrunk to overall by n / (n + 10 / 30)
- F20 / F50: kill-share weights x 0.5 ^ (games back / 20 / 50)
- T15 / T30: team-pool weights x 0.5 ^ (team games back / 15 / 30)

| variant | n | crps | d_crps | se_crps | d_logscore | se_logscore | bias |
|---|---|---|---|---|---|---|---|
| L+O | 16004 | 1.9175 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0003 |
| L+O+Q30 | 16004 | 1.9176 | 0.0001 | 0.0006 | -0.0004 | 0.0007 | 0.0003 |
| L+O+R1 | 16004 | 1.9176 | 0.0001 | 0.0002 | -0.0002 | 0.0006 | -0.0003 |
| L+O+T30 | 16004 | 1.9177 | 0.0003 | 0.0003 | 0.0004 | 0.0006 | 0.0048 |
| L+O+R2 | 16004 | 1.9178 | 0.0004 | 0.0003 | -0.0001 | 0.0006 | -0.0006 |
| L+O+T15 | 16004 | 1.9180 | 0.0005 | 0.0005 | -0.0004 | 0.0006 | 0.0077 |
| L+O+Q10 | 16004 | 1.9182 | 0.0007 | 0.0009 | -0.0006 | 0.0009 | 0.0002 |
| L+O+F50 | 16004 | 1.9222 | 0.0048 | 0.0004 | 0.0026 | 0.0006 | 0.0003 |
| L+O+F20 | 16004 | 1.9303 | 0.0129 | 0.0009 | 0.0067 | 0.0007 | 0.0004 |

None improves on L+O: F (player form in the kill share) is clearly worse, R / Q / T are within noise. Not locked.

## 2. H: recent-form blend (post-sim, before S)

gap = the player's mean over the previous 20 series of the same market (>= 10) minus the model mean. Actual minus the model mean rises with the gap, so the model under-reacts to recent output. H scales the player's simulated total so its mean moves by beta x gap; beta per market = the slope fitted on rows of series >= 1 day before, refit weekly (walk-forward).

| comparison | d_crps | se | d_log | se_log |
|---|---|---|---|---|
| H vs raw | -0.0077 | 0.0016 | -0.0052 | 0.0013 |
| H+S vs S (the app before -> after) | -0.0072 | 0.0015 | -0.0033 | 0.0011 |

By market (H vs raw):

| market | n | applied | d_crps | slope_before | slope_after | bias_before | bias_after |
|---|---|---|---|---|---|---|---|
| Map 1 | 3170 | 0.5656 | -0.0006 | 0.1460 | 0.0658 | -0.0204 | -0.0199 |
| Maps 1-2 | 9946 | 0.7618 | -0.0097 | 0.2691 | 0.0374 | 0.0347 | 0.0210 |
| Maps 1-3 | 2888 | 0.4830 | -0.0086 | 0.2742 | 0.0702 | -0.0951 | -0.1115 |

By month (H vs raw CRPS):

| month | size | mean |
|---|---|---|
| 2026-03 | 1584 | -0.0004 |
| 2026-04 | 3706 | -0.0098 |
| 2026-05 | 2833 | -0.0082 |
| 2026-06 | 1216 | -0.0037 |
| 2026-07 | 1389 | -0.0071 |
| 2026-08 | 2683 | -0.0106 |
| 2026-09 | 1805 | -0.0132 |
| 2026-10 | 788 | 0.0042 |

slope_after ~ 0 means the recent-form signal is used up. H is applied in the app before S; S / I / W are refit on the H pmfs (scripts/refit_calibration.py).

