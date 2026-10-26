# Calibration fit (S, I, W)

Fitted 2026-10-26T17:22:29+00:00 on 37,190 backtest player-series through 2026-10-25 (walk-forward, weekly refits, fits use series >= 1 day before). Base = live locked config L+O; S and I-beta on top; W on the S+I probabilities at proxy lines (rolling median of the player's last 10 market totals, floored, + 0.5).

## Drift check

Passed. Limits: S k ±0.1, I map ±0.05, W lambda ±0.25.

## W: lambda by market (current fit)

Displayed P = p_line + lambda (p_cal - p_line), p_line = (1 - P(push)) / 2.

| _pool | Map 1 | Maps 1-2 | Maps 1-3 |
|---|---|---|---|
| 0.8988 | 0.8766 | 0.9457 | 0.8399 |

Selected picks in the fit: 2377.

### Weekly fits (latest 8)

| refit | _pool | Map 1 | Maps 1-2 | Maps 1-3 | _n |
|---|---|---|---|---|---|
| 2026-09-04 | 0.9002 | 0.8616 | 0.9561 | 0.8224 | 2126 |
| 2026-09-11 | 0.9055 | 0.8711 | 0.9535 | 0.8440 | 2162 |
| 2026-09-18 | 0.9017 | 0.8723 | 0.9446 | 0.8485 | 2197 |
| 2026-09-25 | 0.8969 | 0.8687 | 0.9411 | 0.8414 | 2232 |
| 2026-10-02 | 0.8929 | 0.8673 | 0.9382 | 0.8344 | 2267 |
| 2026-10-09 | 0.8993 | 0.8729 | 0.9437 | 0.8446 | 2302 |
| 2026-10-16 | 0.8972 | 0.8751 | 0.9425 | 0.8405 | 2337 |
| 2026-10-23 | 0.8967 | 0.8735 | 0.9432 | 0.8384 | 2357 |

## Top edges per slate, before vs after W (walk-forward)

`pred` = mean shown P of the picked side (before: calibrated S+I; after: W), `hit` = hit rate, `diff` = hit - pred.

| selection | W | n | pred | hit | diff | se | n_more | pred_more | hit_more | n_less | pred_less | hit_less |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top 1 | before | 477 | 0.8851 | 0.8239 | -0.0612 | 0.0174 | 97 | 0.8742 | 0.8144 | 380 | 0.8879 | 0.8263 |
| top 1 | after | 477 | 0.8581 | 0.8281 | -0.0300 | 0.0173 | 100 | 0.8489 | 0.8100 | 377 | 0.8605 | 0.8329 |
| top 3 | before | 1431 | 0.8529 | 0.8134 | -0.0395 | 0.0103 | 307 | 0.8447 | 0.8241 | 1124 | 0.8552 | 0.8105 |
| top 3 | after | 1431 | 0.8283 | 0.8155 | -0.0128 | 0.0103 | 307 | 0.8194 | 0.8274 | 1124 | 0.8307 | 0.8123 |
| top 5 | before | 2379 | 0.8317 | 0.7936 | -0.0381 | 0.0083 | 553 | 0.8218 | 0.8011 | 1826 | 0.8346 | 0.7913 |
| top 5 | after | 2379 | 0.8086 | 0.7949 | -0.0137 | 0.0083 | 559 | 0.7988 | 0.8050 | 1820 | 0.8116 | 0.7918 |

## Minimum edge on W-adjusted edges

**7 pts** = worst of (-diff + 2 se) below, rounded up.

| selection | n | pred | hit | diff | se | n_more | pred_more | hit_more | n_less | pred_less | hit_less | required |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top 1 | 477 | 0.8581 | 0.8281 | -0.0300 | 0.0173 | 100 | 0.8489 | 0.8100 | 377 | 0.8605 | 0.8329 | 0.0645 |
| top 3 | 1431 | 0.8283 | 0.8155 | -0.0128 | 0.0103 | 307 | 0.8194 | 0.8274 | 1124 | 0.8307 | 0.8123 | 0.0333 |
| top 5 | 2379 | 0.8086 | 0.7949 | -0.0137 | 0.0083 | 559 | 0.7988 | 0.8050 | 1820 | 0.8116 | 0.7918 | 0.0303 |
| all edge > 0 | 23127 | 0.6828 | 0.6706 | -0.0123 | 0.0031 | 7409 | 0.6712 | 0.6647 | 15718 | 0.6883 | 0.6733 | 0.0185 |

## S: spread k (position|market)

| position|market | k |
|---|---|
| _pool | 1.066 |
| bot|Map 1 | 1.017 |
| bot|Maps 1-2 | 1.085 |
| bot|Maps 1-3 | 1.099 |
| jng|Map 1 | 1.032 |
| jng|Maps 1-2 | 1.055 |
| jng|Maps 1-3 | 1.077 |
| mid|Map 1 | 1.008 |
| mid|Maps 1-2 | 1.064 |
| mid|Maps 1-3 | 1.087 |
| sup|Map 1 | 1.021 |
| sup|Maps 1-2 | 1.056 |
| sup|Maps 1-3 | 1.021 |
| top|Map 1 | 1.032 |
| top|Maps 1-2 | 1.071 |
| top|Maps 1-3 | 1.076 |

## I: pooled beta map

| p=0.1 | p=0.3 | p=0.5 | p=0.7 | p=0.9 |
|---|---|---|---|---|
| 0.1197 | 0.3081 | 0.5006 | 0.7066 | 0.9154 |
