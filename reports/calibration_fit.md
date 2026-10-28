# Calibration fit (S, I, W)

Fitted 2026-10-28T21:32:37+00:00 on 37,230 backtest player-series through 2026-10-26 (walk-forward, weekly refits, fits use series >= 1 day before). Base = live locked config L+O; S and I-beta on top; W on the S+I probabilities at proxy lines (rolling median of the player's last 10 market totals, floored, + 0.5).

## Drift check

Passed. Limits: S k ±0.1, I map ±0.05, W lambda ±0.25.

## W: lambda by market (current fit)

Displayed P = p_line + lambda (p_cal - p_line), p_line = (1 - P(push)) / 2.

| _pool | Map 1 | Maps 1-2 | Maps 1-3 |
|---|---|---|---|
| 0.9395 | 0.9347 | 0.9912 | 0.8681 |

Selected picks in the fit: 2374.

### Weekly fits (latest 8)

| refit | _pool | Map 1 | Maps 1-2 | Maps 1-3 | _n |
|---|---|---|---|---|---|
| 2026-09-04 | 0.9563 | 0.9333 | 1.0013 | 0.8991 | 2125 |
| 2026-09-11 | 0.9540 | 0.9324 | 1.0009 | 0.8944 | 2160 |
| 2026-09-18 | 0.9471 | 0.9276 | 0.9925 | 0.8894 | 2194 |
| 2026-09-25 | 0.9432 | 0.9301 | 0.9888 | 0.8817 | 2229 |
| 2026-10-02 | 0.9407 | 0.9350 | 0.9864 | 0.8743 | 2264 |
| 2026-10-09 | 0.9480 | 0.9343 | 0.9927 | 0.8902 | 2299 |
| 2026-10-16 | 0.9427 | 0.9339 | 0.9914 | 0.8783 | 2334 |
| 2026-10-23 | 0.9379 | 0.9390 | 0.9905 | 0.8620 | 2348 |

## Top edges per slate, before vs after W (walk-forward)

`pred` = mean shown P of the picked side (before: calibrated S+I; after: W), `hit` = hit rate, `diff` = hit - pred.

| selection | W | n | pred | hit | diff | se | n_more | pred_more | hit_more | n_less | pred_less | hit_less |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top 1 | before | 478 | 0.8711 | 0.8285 | -0.0426 | 0.0172 | 124 | 0.8661 | 0.8387 | 354 | 0.8728 | 0.8249 |
| top 1 | after | 478 | 0.8556 | 0.8347 | -0.0208 | 0.0170 | 123 | 0.8505 | 0.8374 | 355 | 0.8573 | 0.8338 |
| top 3 | before | 1433 | 0.8372 | 0.8186 | -0.0187 | 0.0102 | 360 | 0.8343 | 0.8250 | 1073 | 0.8382 | 0.8164 |
| top 3 | after | 1432 | 0.8235 | 0.8191 | -0.0044 | 0.0102 | 363 | 0.8197 | 0.8237 | 1069 | 0.8249 | 0.8176 |
| top 5 | before | 2379 | 0.8154 | 0.7945 | -0.0209 | 0.0083 | 621 | 0.8121 | 0.7907 | 1758 | 0.8165 | 0.7958 |
| top 5 | after | 2374 | 0.8030 | 0.7965 | -0.0065 | 0.0083 | 623 | 0.7997 | 0.7929 | 1751 | 0.8042 | 0.7978 |

## Minimum edge on W-adjusted edges

**6 pts** = worst of (-diff + 2 se) below, rounded up.

| selection | n | pred | hit | diff | se | n_more | pred_more | hit_more | n_less | pred_less | hit_less | required |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top 1 | 478 | 0.8556 | 0.8347 | -0.0208 | 0.0170 | 123 | 0.8505 | 0.8374 | 355 | 0.8573 | 0.8338 | 0.0548 |
| top 3 | 1432 | 0.8235 | 0.8191 | -0.0044 | 0.0102 | 363 | 0.8197 | 0.8237 | 1069 | 0.8249 | 0.8176 | 0.0248 |
| top 5 | 2374 | 0.8030 | 0.7965 | -0.0065 | 0.0083 | 623 | 0.7997 | 0.7929 | 1751 | 0.8042 | 0.7978 | 0.0230 |
| all edge > 0 | 22498 | 0.6782 | 0.6761 | -0.0021 | 0.0031 | 7143 | 0.6692 | 0.6689 | 15355 | 0.6824 | 0.6795 | 0.0084 |

## S: spread k (position|market)

| position|market | k |
|---|---|
| _pool | 1.056 |
| bot|Map 1 | 1.009 |
| bot|Maps 1-2 | 1.075 |
| bot|Maps 1-3 | 1.089 |
| jng|Map 1 | 1.023 |
| jng|Maps 1-2 | 1.048 |
| jng|Maps 1-3 | 1.070 |
| mid|Map 1 | 1.001 |
| mid|Maps 1-2 | 1.052 |
| mid|Maps 1-3 | 1.079 |
| sup|Map 1 | 0.989 |
| sup|Maps 1-2 | 1.034 |
| sup|Maps 1-3 | 1.009 |
| top|Map 1 | 1.020 |
| top|Maps 1-2 | 1.059 |
| top|Maps 1-3 | 1.065 |

## I: pooled beta map

| p=0.1 | p=0.3 | p=0.5 | p=0.7 | p=0.9 |
|---|---|---|---|---|
| 0.1131 | 0.3051 | 0.5008 | 0.7064 | 0.9130 |
