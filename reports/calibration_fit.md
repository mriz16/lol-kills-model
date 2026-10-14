# Calibration fit (S, I, W)

Fitted 2026-10-14T21:35:02+00:00 on 24,979 backtest player-series through 2026-10-13 (walk-forward, weekly refits, fits use series >= 1 day before). Base = live locked config L+O; S and I-beta on top; W on the S+I probabilities at proxy lines (rolling median of the player's last 10 market totals, floored, + 0.5).

## Drift check

No previous fit. Limits: S k ±0.1, I map ±0.05, W lambda ±0.25.

## W: lambda by market (current fit)

Displayed P = p_line + lambda (p_cal - p_line), p_line = (1 - P(push)) / 2.

| _pool | Map 1 | Maps 1-2 | Maps 1-3 |
|---|---|---|---|
| 0.9052 | 0.8878 | 0.9345 | 0.8681 |

Selected picks in the fit: 2225.

### Weekly fits (latest 8)

| refit | _pool | Map 1 | Maps 1-2 | Maps 1-3 | _n |
|---|---|---|---|---|---|
| 2026-08-21 | 0.9091 | 0.8761 | 0.9300 | 0.8891 | 1966 |
| 2026-08-28 | 0.9120 | 0.8825 | 0.9348 | 0.8864 | 2001 |
| 2026-09-04 | 0.9079 | 0.8806 | 0.9338 | 0.8763 | 2036 |
| 2026-09-11 | 0.9060 | 0.8797 | 0.9340 | 0.8712 | 2071 |
| 2026-09-18 | 0.9069 | 0.8802 | 0.9341 | 0.8750 | 2105 |
| 2026-09-25 | 0.9034 | 0.8786 | 0.9303 | 0.8711 | 2139 |
| 2026-10-02 | 0.9007 | 0.8858 | 0.9283 | 0.8634 | 2169 |
| 2026-10-09 | 0.9069 | 0.8868 | 0.9347 | 0.8723 | 2200 |

## Top edges per slate, before vs after W (walk-forward)

`pred` = mean shown P of the picked side (before: calibrated S+I; after: W), `hit` = hit rate, `diff` = hit - pred.

| selection | W | n | pred | hit | diff | se | n_more | pred_more | hit_more | n_less | pred_less | hit_less |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top 1 | before | 448 | 0.8671 | 0.8415 | -0.0255 | 0.0173 | 102 | 0.8573 | 0.8333 | 346 | 0.8699 | 0.8439 |
| top 1 | after | 448 | 0.8424 | 0.8482 | 0.0058 | 0.0170 | 103 | 0.8338 | 0.8447 | 345 | 0.8450 | 0.8493 |
| top 3 | before | 1341 | 0.8329 | 0.8076 | -0.0253 | 0.0108 | 352 | 0.8206 | 0.8068 | 989 | 0.8372 | 0.8079 |
| top 3 | after | 1340 | 0.8109 | 0.8045 | -0.0064 | 0.0108 | 349 | 0.8007 | 0.7994 | 991 | 0.8144 | 0.8063 |
| top 5 | before | 2228 | 0.8104 | 0.7805 | -0.0299 | 0.0088 | 614 | 0.7992 | 0.7769 | 1614 | 0.8146 | 0.7819 |
| top 5 | after | 2225 | 0.7901 | 0.7811 | -0.0090 | 0.0088 | 611 | 0.7801 | 0.7741 | 1614 | 0.7940 | 0.7838 |

## Minimum edge on W-adjusted edges

**3 pts** = worst of (-diff + 2 se) below, rounded up.

| selection | n | pred | hit | diff | se | n_more | pred_more | hit_more | n_less | pred_less | hit_less | required |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top 1 | 448 | 0.8424 | 0.8482 | 0.0058 | 0.0170 | 103 | 0.8338 | 0.8447 | 345 | 0.8450 | 0.8493 | 0.0281 |
| top 3 | 1340 | 0.8109 | 0.8045 | -0.0064 | 0.0108 | 349 | 0.8007 | 0.7994 | 991 | 0.8144 | 0.8063 | 0.0281 |
| top 5 | 2225 | 0.7901 | 0.7811 | -0.0090 | 0.0088 | 611 | 0.7801 | 0.7741 | 1614 | 0.7940 | 0.7838 | 0.0265 |
| all edge > 0 | 15476 | 0.6786 | 0.6723 | -0.0063 | 0.0038 | 5049 | 0.6704 | 0.6673 | 10427 | 0.6826 | 0.6747 | 0.0139 |

## S: spread k (position|market)

| position|market | k |
|---|---|
| _pool | 1.065 |
| bot|Map 1 | 1.052 |
| bot|Maps 1-2 | 1.087 |
| bot|Maps 1-3 | 1.103 |
| jng|Map 1 | 1.035 |
| jng|Maps 1-2 | 1.025 |
| jng|Maps 1-3 | 1.078 |
| mid|Map 1 | 1.033 |
| mid|Maps 1-2 | 1.064 |
| mid|Maps 1-3 | 1.072 |
| sup|Map 1 | 1.062 |
| sup|Maps 1-2 | 1.021 |
| sup|Maps 1-3 | 1.014 |
| top|Map 1 | 1.033 |
| top|Maps 1-2 | 1.061 |
| top|Maps 1-3 | 1.066 |

## I: pooled beta map

| p=0.1 | p=0.3 | p=0.5 | p=0.7 | p=0.9 |
|---|---|---|---|---|
| 0.1220 | 0.3100 | 0.5015 | 0.7068 | 0.9156 |
