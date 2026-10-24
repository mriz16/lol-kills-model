# International-event backtest (data through 2026-10-23, 20000 sims)

Walk-forward (each series projected from prior data only, Elo p_map), primary market = what is bet (BO1 Map 1, BO3 Maps 1-2, BO5 Maps 1-3). `base` vs `L+O`. crps_rel = CRPS / mean actual kills (CRPS grows with the kill total, so compare markets on crps_rel). ECE at the line = rounded median (`ece_median`) and at half-point lines median +/- 0.5, 1.5 (`ece_half`), 5% buckets. `ece_median_null` = the ECE a perfectly calibrated model would show at the same sample size (outcomes drawn from its own P(over), 200 draws): ECE below it is noise, not miscalibration.

## Per event

| group | variant | series | n | crps | crps_rel | logscore | bias | mean_y | ece_median | ece_median_null | ece_half |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Worlds 2025 | L+O | 41 | 407 | 1.5403 | 0.2999 | 2.2571 | 0.0169 | 5.1351 | 0.0427 | 0.0447 | 0.0344 |
| Worlds 2025 | base | 41 | 407 | 1.5689 | 0.3055 | 2.2832 | -0.4059 | 5.1351 | 0.0679 | 0.0414 | 0.0640 |
| MSI 2026 | L+O | 20 | 200 | 2.7267 | 0.2757 | 2.9302 | -0.1047 | 9.8900 | 0.0683 | 0.0524 | 0.0513 |
| MSI 2026 | base | 20 | 200 | 2.8168 | 0.2848 | 2.9978 | -1.0554 | 9.8900 | 0.1124 | 0.0525 | 0.0863 |
| FST 2026 | L+O | 13 | 130 | 2.2014 | 0.2609 | 2.7105 | -0.1128 | 8.4385 | 0.0908 | 0.0714 | 0.0601 |
| FST 2026 | base | 13 | 130 | 2.2338 | 0.2647 | 2.7241 | -0.7660 | 8.4385 | 0.1344 | 0.0730 | 0.0962 |
| EWC 2026 | L+O | 91 | 910 | 1.8308 | 0.2955 | 2.4895 | -0.0039 | 6.1967 | 0.0174 | 0.0262 | 0.0195 |
| EWC 2026 | base | 91 | 910 | 1.8538 | 0.2992 | 2.5004 | -0.1227 | 6.1967 | 0.0176 | 0.0262 | 0.0214 |
| LTA 2025 | L+O | 6 | 60 | 2.4201 | 0.2635 | 2.8294 | 0.1508 | 9.1833 | 0.0990 | 0.0996 | 0.0789 |
| LTA 2025 | base | 6 | 60 | 2.4459 | 0.2663 | 2.8939 | -0.7078 | 9.1833 | 0.1593 | 0.1041 | 0.0986 |
| WLRQ 2026 | L+O | 6 | 60 | 1.3440 | 0.3992 | 2.0799 | -0.2629 | 3.3667 | 0.0934 | 0.0811 | 0.1161 |
| WLRQ 2026 | base | 6 | 60 | 1.3174 | 0.3913 | 2.0622 | -0.2787 | 3.3667 | 0.2176 | 0.0917 | 0.1455 |
| Worlds 2026 (so far) | L+O | 7 | 69 | 2.1538 | 0.3033 | 2.6214 | 0.4218 | 7.1014 | 0.1251 | 0.0952 | 0.0784 |
| Worlds 2026 (so far) | base | 7 | 69 | 2.1984 | 0.3096 | 2.6470 | 0.0930 | 7.1014 | 0.0676 | 0.0950 | 0.0674 |
| regional qualifiers filed as intl | L+O | 18 | 180 | 2.3166 | 0.3255 | 2.6842 | -0.1105 | 7.1167 | 0.0524 | 0.0630 | 0.0579 |
| regional qualifiers filed as intl | base | 18 | 180 | 2.3017 | 0.3234 | 2.6701 | -0.1313 | 7.1167 | 0.0370 | 0.0606 | 0.0419 |
| domestic | L+O | 2170 | 21606 | 1.8729 | 0.3180 | 2.4599 | -0.0414 | 5.8893 | 0.0128 | 0.0056 | 0.0108 |
| domestic | base | 2170 | 21606 | 1.9129 | 0.3248 | 2.4872 | -0.1024 | 5.8893 | 0.0234 | 0.0057 | 0.0160 |

## International (Worlds 2025 + MSI 2026 + FST 2026) vs domestic

| group | variant | series | n | crps | crps_rel | logscore | bias | mean_y | ece_median | ece_median_null | ece_half |
|---|---|---|---|---|---|---|---|---|---|---|---|
| international (3 events) | base | 74 | 737 | 2.0248 | 0.2889 | 2.5549 | -0.6457 | 7.0081 | 0.0854 | 0.0312 | 0.0741 |
| domestic, same window | base | 2170 | 21606 | 1.9129 | 0.3248 | 2.4872 | -0.1024 | 5.8893 | 0.0234 | 0.0057 | 0.0160 |
| international (3 events) | L+O | 74 | 737 | 1.9788 | 0.2824 | 2.5197 | -0.0390 | 7.0081 | 0.0408 | 0.0313 | 0.0315 |
| domestic, same window | L+O | 2170 | 21606 | 1.8729 | 0.3180 | 2.4599 | -0.0414 | 5.8893 | 0.0128 | 0.0056 | 0.0108 |
| international / Map 1 | base | 20 | 200 | 1.1566 | 0.4065 | 1.9611 | -0.0297 | 2.8450 | 0.0761 | 0.0599 | 0.0467 |
| domestic / Map 1 | base | 636 | 6360 | 1.3737 | 0.4175 | 2.1407 | -0.0412 | 3.2903 | 0.0199 | 0.0110 | 0.0135 |
| international / Maps 1-2 | base | 13 | 128 | 1.5708 | 0.2699 | 2.3917 | -0.1967 | 5.8203 | 0.0902 | 0.0714 | 0.0740 |
| domestic / Maps 1-2 | base | 1109 | 11033 | 1.9906 | 0.3235 | 2.5596 | -0.0177 | 6.1542 | 0.0258 | 0.0078 | 0.0182 |
| international / Maps 1-3 | base | 41 | 409 | 2.5915 | 0.2752 | 2.8963 | -1.0874 | 9.4156 | 0.1294 | 0.0382 | 0.1030 |
| domestic / Maps 1-3 | base | 425 | 4213 | 2.5235 | 0.2767 | 2.8207 | -0.4166 | 9.1194 | 0.0421 | 0.0121 | 0.0341 |
| international / Map 1 | L+O | 20 | 200 | 1.1535 | 0.4054 | 1.9555 | 0.1933 | 2.8450 | 0.0315 | 0.0641 | 0.0414 |
| domestic / Map 1 | L+O | 636 | 6360 | 1.3554 | 0.4119 | 2.1197 | -0.0520 | 3.2903 | 0.0146 | 0.0108 | 0.0120 |
| international / Maps 1-2 | L+O | 13 | 128 | 1.5599 | 0.2680 | 2.3697 | 0.2065 | 5.8203 | 0.0937 | 0.0732 | 0.0589 |
| domestic / Maps 1-2 | L+O | 1109 | 11033 | 1.9481 | 0.3165 | 2.5334 | -0.0376 | 6.1542 | 0.0209 | 0.0080 | 0.0140 |
| international / Maps 1-3 | L+O | 41 | 409 | 2.5136 | 0.2670 | 2.8426 | -0.2295 | 9.4156 | 0.0574 | 0.0380 | 0.0450 |
| domestic / Maps 1-3 | L+O | 425 | 4213 | 2.4575 | 0.2695 | 2.7810 | -0.0353 | 9.1194 | 0.0209 | 0.0115 | 0.0152 |

## Paired CRPS difference by group (L+O - base, negative = L+O better)

| group | n | diff | se |
|---|---|---|---|
| EWC 2026 | 910 | -0.0229 | 0.0138 |
| FST 2026 | 130 | -0.0324 | 0.0487 |
| LTA 2025 | 60 | -0.0258 | 0.1071 |
| MSI 2026 | 200 | -0.0901 | 0.0568 |
| WLRQ 2026 | 60 | 0.0266 | 0.0076 |
| Worlds 2025 | 407 | -0.0286 | 0.0216 |
| Worlds 2026 (so far) | 69 | -0.0446 | 0.0438 |
| domestic | 21606 | -0.0400 | 0.0033 |
| regional qualifiers filed as intl | 180 | 0.0149 | 0.0305 |

## Reading (hand-written 2026-10-24; rerunning the script drops this section)

- **Locked (L+O) on Worlds 2025 + MSI 2026 + FST 2026 (74 series, 737 player-series):**
  CRPS 1.979 (crps_rel 0.282), bias -0.04 kills (series-clustered se 0.15), ECE 0.041 at
  the median line against a noise floor of 0.031 at that sample size; half-line ECE 0.032.
- **Domestic, same window (2,170 series, 21,606 rows):** crps_rel 0.318, bias -0.04
  (se 0.03), ECE 0.013 against a floor of 0.006.
- Like with like (same market): Maps 1-3 crps_rel 0.267 international vs 0.270 domestic;
  Map 1 0.405 vs 0.412. ECE excess over the noise floor is about 0.01 in both. The only
  gap is Maps 1-3 bias, -0.23 international vs -0.04 domestic, with a clustered se of
  0.25, so it is not significant.
- L+O beats base on every headline event (paired CRPS -0.03 / -0.09 / -0.03). Base
  under-predicts international BO5s by about 1 kill; O removes most of that.
- **Verdict: international events are not clearly worse than domestic. No fix was built
  and nothing is locked.**
- Things to watch (not tested, they would need sign-off):
  1. Once an event code has 40 or more games, the event env is just the plain mean of
     its last 120 games, across years. At Worlds that window is full (n = 120, weight
     1.0), so the two teams' home leagues currently have no effect on the target (every
     Round 1 matchup gets 14.58). Restricting the measured window to the current edition
     would let the home-league blend matter early in an event.
  2. OE files the LPL regional finals (Sept 2025 and Sept 2026, 6 Bo5 series) under WLDs
     and the LCS EWC qualifier (April 2026, 12 series) under EWC. They count toward the
     event env. Relabelling them to their home league would be a data-rule change.
