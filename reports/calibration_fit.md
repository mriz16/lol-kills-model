# Calibration fit (S, I, W)

Fitted 2026-10-15T03:16:12+00:00 on 36,911 backtest player-series through 2026-10-13 (walk-forward, weekly refits, fits use series >= 1 day before). Base = live locked config L+O; S and I-beta on top; W on the S+I probabilities at proxy lines (rolling median of the player's last 10 market totals, floored, + 0.5).

## Drift check

Passed. Limits: S k ±0.1, I map ±0.05, W lambda ±0.25.

## W: lambda by market (current fit)

Displayed P = p_line + lambda (p_cal - p_line), p_line = (1 - P(push)) / 2.

| _pool | Map 1 | Maps 1-2 | Maps 1-3 |
|---|---|---|---|
| 0.8984 | 0.8653 | 0.9411 | 0.8503 |

Selected picks in the fit: 2332.

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
| top 1 | before | 468 | 0.8865 | 0.8269 | -0.0595 | 0.0175 | 96 | 0.8752 | 0.8229 | 372 | 0.8894 | 0.8280 |
| top 1 | after | 468 | 0.8598 | 0.8312 | -0.0286 | 0.0173 | 99 | 0.8501 | 0.8182 | 369 | 0.8624 | 0.8347 |
| top 3 | before | 1404 | 0.8545 | 0.8155 | -0.0390 | 0.0104 | 302 | 0.8462 | 0.8311 | 1102 | 0.8568 | 0.8113 |
| top 3 | after | 1404 | 0.8302 | 0.8170 | -0.0132 | 0.0103 | 303 | 0.8210 | 0.8317 | 1101 | 0.8327 | 0.8129 |
| top 5 | before | 2334 | 0.8332 | 0.7956 | -0.0376 | 0.0083 | 546 | 0.8228 | 0.8040 | 1788 | 0.8364 | 0.7931 |
| top 5 | after | 2334 | 0.8104 | 0.7969 | -0.0135 | 0.0083 | 551 | 0.8001 | 0.8076 | 1783 | 0.8136 | 0.7936 |

## Minimum edge on W-adjusted edges

**7 pts** = worst of (-diff + 2 se) below, rounded up.

| selection | n | pred | hit | diff | se | n_more | pred_more | hit_more | n_less | pred_less | hit_less | required |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top 1 | 468 | 0.8598 | 0.8312 | -0.0286 | 0.0173 | 99 | 0.8501 | 0.8182 | 369 | 0.8624 | 0.8347 | 0.0632 |
| top 3 | 1404 | 0.8302 | 0.8170 | -0.0132 | 0.0103 | 303 | 0.8210 | 0.8317 | 1101 | 0.8327 | 0.8129 | 0.0339 |
| top 5 | 2334 | 0.8104 | 0.7969 | -0.0135 | 0.0083 | 551 | 0.8001 | 0.8076 | 1783 | 0.8136 | 0.7936 | 0.0301 |
| all edge > 0 | 22942 | 0.6831 | 0.6712 | -0.0119 | 0.0031 | 7361 | 0.6714 | 0.6659 | 15581 | 0.6886 | 0.6737 | 0.0181 |

## S: spread k (position|market)

| position|market | k |
|---|---|
| _pool | 1.066 |
| bot|Map 1 | 1.017 |
| bot|Maps 1-2 | 1.085 |
| bot|Maps 1-3 | 1.099 |
| jng|Map 1 | 1.032 |
| jng|Maps 1-2 | 1.055 |
| jng|Maps 1-3 | 1.076 |
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
| 0.1197 | 0.3084 | 0.5010 | 0.7069 | 0.9154 |
