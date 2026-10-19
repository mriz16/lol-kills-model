# Calibration fit (S, I, W)

Fitted 2026-10-19T17:32:36+00:00 on 36,941 backtest player-series through 2026-10-15 (walk-forward, weekly refits, fits use series >= 1 day before). Base = live locked config L+O; S and I-beta on top; W on the S+I probabilities at proxy lines (rolling median of the player's last 10 market totals, floored, + 0.5).

## Drift check

Passed. Limits: S k ±0.1, I map ±0.05, W lambda ±0.25.

## W: lambda by market (current fit)

Displayed P = p_line + lambda (p_cal - p_line), p_line = (1 - P(push)) / 2.

| _pool | Map 1 | Maps 1-2 | Maps 1-3 |
|---|---|---|---|
| 0.8953 | 0.8630 | 0.9413 | 0.8425 |

Selected picks in the fit: 2342.

### Weekly fits (latest 8)

| refit | _pool | Map 1 | Maps 1-2 | Maps 1-3 | _n |
|---|---|---|---|---|---|
| 2026-08-21 | 0.9063 | 0.8712 | 0.9489 | 0.8525 | 2057 |
| 2026-08-28 | 0.9073 | 0.8771 | 0.9511 | 0.8485 | 2092 |
| 2026-09-04 | 0.9002 | 0.8616 | 0.9561 | 0.8224 | 2126 |
| 2026-09-11 | 0.9055 | 0.8711 | 0.9535 | 0.8440 | 2162 |
| 2026-09-18 | 0.9017 | 0.8723 | 0.9446 | 0.8485 | 2197 |
| 2026-09-25 | 0.8969 | 0.8687 | 0.9411 | 0.8414 | 2232 |
| 2026-10-02 | 0.8944 | 0.8627 | 0.9371 | 0.8431 | 2267 |
| 2026-10-09 | 0.8993 | 0.8729 | 0.9437 | 0.8446 | 2302 |

## Top edges per slate, before vs after W (walk-forward)

`pred` = mean shown P of the picked side (before: calibrated S+I; after: W), `hit` = hit rate, `diff` = hit - pred.

| selection | W | n | pred | hit | diff | se | n_more | pred_more | hit_more | n_less | pred_less | hit_less |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top 1 | before | 470 | 0.8861 | 0.8234 | -0.0627 | 0.0176 | 97 | 0.8742 | 0.8144 | 373 | 0.8892 | 0.8257 |
| top 1 | after | 470 | 0.8594 | 0.8277 | -0.0317 | 0.0174 | 100 | 0.8489 | 0.8100 | 370 | 0.8622 | 0.8324 |
| top 3 | before | 1410 | 0.8541 | 0.8135 | -0.0407 | 0.0104 | 304 | 0.8458 | 0.8289 | 1106 | 0.8565 | 0.8092 |
| top 3 | after | 1410 | 0.8297 | 0.8149 | -0.0148 | 0.0103 | 305 | 0.8204 | 0.8295 | 1105 | 0.8323 | 0.8109 |
| top 5 | before | 2344 | 0.8329 | 0.7944 | -0.0385 | 0.0083 | 548 | 0.8226 | 0.8029 | 1796 | 0.8360 | 0.7918 |
| top 5 | after | 2344 | 0.8100 | 0.7956 | -0.0143 | 0.0083 | 553 | 0.7998 | 0.8065 | 1791 | 0.8131 | 0.7923 |

## Minimum edge on W-adjusted edges

**7 pts** = worst of (-diff + 2 se) below, rounded up.

| selection | n | pred | hit | diff | se | n_more | pred_more | hit_more | n_less | pred_less | hit_less | required |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top 1 | 470 | 0.8594 | 0.8277 | -0.0317 | 0.0174 | 100 | 0.8489 | 0.8100 | 370 | 0.8622 | 0.8324 | 0.0666 |
| top 3 | 1410 | 0.8297 | 0.8149 | -0.0148 | 0.0103 | 305 | 0.8204 | 0.8295 | 1105 | 0.8323 | 0.8109 | 0.0355 |
| top 5 | 2344 | 0.8100 | 0.7956 | -0.0143 | 0.0083 | 553 | 0.7998 | 0.8065 | 1791 | 0.8131 | 0.7923 | 0.0310 |
| all edge > 0 | 22967 | 0.6830 | 0.6708 | -0.0122 | 0.0031 | 7369 | 0.6714 | 0.6656 | 15598 | 0.6885 | 0.6732 | 0.0184 |

## S: spread k (position|market)

| position|market | k |
|---|---|
| _pool | 1.067 |
| bot|Map 1 | 1.017 |
| bot|Maps 1-2 | 1.085 |
| bot|Maps 1-3 | 1.099 |
| jng|Map 1 | 1.032 |
| jng|Maps 1-2 | 1.055 |
| jng|Maps 1-3 | 1.078 |
| mid|Map 1 | 1.006 |
| mid|Maps 1-2 | 1.064 |
| mid|Maps 1-3 | 1.087 |
| sup|Map 1 | 1.022 |
| sup|Maps 1-2 | 1.056 |
| sup|Maps 1-3 | 1.023 |
| top|Map 1 | 1.033 |
| top|Maps 1-2 | 1.071 |
| top|Maps 1-3 | 1.073 |

## I: pooled beta map

| p=0.1 | p=0.3 | p=0.5 | p=0.7 | p=0.9 |
|---|---|---|---|---|
| 0.1199 | 0.3085 | 0.5010 | 0.7069 | 0.9154 |
