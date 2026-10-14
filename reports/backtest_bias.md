# Backtest: mean-bias toggles on the locked config (real OE data, walk-forward, 20,000 sims)

Reference = **locked config `shrink30+A+B50`** (game-script off, C off). Every variant
below is that config plus one or more toggles, scored on the same 24,979 player-series
(2025-03-28 -> 2026-10-13). All `*_vs_locked` numbers are paired, with standard errors.
Data: OE 2025 + 2026 refreshed 2026-10-14, now **including LPL partial games** (rule below).

## Recommendation

**Lock O (outcome-conditional opponent adjustment) on top of the locked config. Do not
lock M or P. G is reference only.** O changes locked engine step (c), so it needs your
sign-off. It is implemented as `EngineConfig(opp_by_outcome=True)` and is off by default.

| | CRPS vs locked (se) | log score vs locked (se) | bias | ECE median / half-pt | tail obs/pred |
|---|---|---|---|---|---|
| locked | - | - | -0.129 | 0.0193 / 0.0131 | 1.091 |
| **L+O** | **-0.0268 (0.0027)** | **-0.0160 (0.0014)** | -0.069 | 0.0151 / 0.0101 | 1.058 |
| L+O+P | -0.0274 (0.0028) | -0.0158 (0.0014) | -0.069 | 0.0151 / 0.0095 | 1.056 |
| L+P | -0.0008 (0.0006) | -0.0004 (0.0004) | -0.129 | 0.0195 / 0.0128 | 1.090 |
| L+M1 / M2 / M3 | +0.0045 / +0.0026 / +0.0013 | +0.0024 / +0.0014 / +0.0007 | -0.128 / -0.127 / -0.127 | ~unchanged | ~1.09 |
| L+G (reference) | +0.0006 (0.0006) | -0.0015 (0.0004) | -0.007 | 0.0173 / 0.0135 | 1.041 |

1. **The bias was not meta drift. It came from the opponent adjustment.** The locked
   step (c) scales a team's kills by the opponent's overall kills-allowed. But a strong
   opponent allows few kills mostly *because it usually wins*, and the W/L pools already
   condition on the drawn outcome, so team strength gets counted twice. In BO5s (strong vs
   strong: playoffs, MSI, Worlds) both teams get adj < 1. Average adj was 0.945 in BO5s,
   so the model under-predicted. Bias by market under the locked config: Map 1 -0.06,
   Maps 1-2 -0.05, **Maps 1-3 -0.49**. Per-map kills in BO5s are no lower than in BO3s
   in the data (14.9 vs 15.2 per team-map). With the realized outcome held fixed, team-kill
   error ran from -1.84/map at adj <= 0.90 to +1.65/map at adj >= 1.10. That is also why
   the bias peaked in 2025Q4 / 2026Q4 (Worlds) at about -0.45.
2. **O fixes it at the source.** When side s draws a WIN, O uses the opponent's kills allowed
   in its *losses*, relative to the average kills of winning teams (and the mirror for a
   LOSS). Same fade (`min(1, n/10)`, n = opponent games with that result) and clamp
   [0.80, 1.20]. Results: BO5 bias -0.49 -> +0.02, Worlds quarters -0.42/-0.47 -> -0.14/+0.08,
   CRPS better in **all 10 leagues** (LCK -0.039, LCKC -0.036, LPL -0.028, MSI -0.092,
   FST -0.054). This gain is larger than all of B's (-0.013 vs shrink30), it lowers ECE
   at both line types, and it cuts the thin-tail ratio from 1.091 to 1.058.
3. **M (meta drift) rejected.** It is worse for every N (N=1 +0.0045, se 0.0008) and barely
   moves the bias (-0.129 -> -0.127). On top of O, M3 is also worse (+0.0018 vs L+O,
   se 0.0004). The kill environment is tracked well enough by the 0.65 decay.
4. **P (fitted position priors) not locked.** It is score-neutral: L+P vs locked -0.0008
   (se 0.0006), and L+O+P vs L+O -0.0007 (se 0.0006, log score +0.0002). It only moves bias
   between positions. Top improves (-0.15 -> -0.07 with O), while jng and bot get worse
   (-0.04 -> -0.11, -0.08 -> -0.11). The total absolute position bias is slightly higher
   with P (0.40 vs 0.37). The fitted priors give top more and jng less than the hard-coded
   ones, which only matters for players with < 30 games.
5. **G (reference): a naive global multiplier removes the mean bias but not the error.**
   Its walk-forward multiplier settles near 1.02-1.03. Bias goes to -0.007, but CRPS gets
   slightly *worse* (+0.0006) because it inflates BO1/BO3 props (which were nearly
   unbiased) to pay for BO5. L+O beats L+G by -0.027 CRPS (se 0.003). Fixing the mechanism
   is worth far more than fixing the average.

Remaining after O: bias -0.069 (se 0.023), mostly **top -0.15** and Maps 1-2 -0.10. The
under-25% bucket at the median line is still optimistic for unders (pred 0.221, hit 0.315,
n=406, vs 0.338 hit for the locked config). Tails are still ~6% thin (top 1.12, bot 1.08).
Possible next steps (not built): O tends to over-correct at its extremes (residual
+/-1.2-1.5 kills/map in the outer adj buckets), so a softer fade or a tighter clamp is
worth a test. Top's remaining bias looks like a share problem, not a team-kill problem.

## LPL partial data (2025)

Rule (signed off): an LPL `partial` game is kept only if kills, deaths, assists, result
and gamelength are non-null for all 10 players (playername too, since lineups need it) AND
player kills sum to team kills on both sides. **LPL 2025: 805 pass, 0 fail.** LPL 2026
also has 126 partial games, and all of them pass. Kept games: 5,700 -> 6,631 (2,804 series).

The comparison run drops the 805 LPL 2025 partial games entirely (no history, not scored):
- LPL 2026 series (paired, n=2,457): CRPS **-0.0209 (se 0.0092)** with the 2025 data under
  the locked config, and -0.0207 (0.0086) under L+O+P. Bias -0.104 -> -0.073 (locked). The
  extra history helps the 2026 projections.
- LPL 2025 series are scored only when the data is included (n=2,426). Their CRPS (1.974
  locked, 1.945 L+O+P) is in line with LPL 2026, so the partial rows look clean.
- Non-LPL: -0.0017 (se 0.0011). That is within noise, as expected (LPL teams at MSI/Worlds
  get a little more history).

## Reproduce

```
python scripts/run_backtest.py --rows-only --tag g  --variants shrink30+A+B50 L+G
python scripts/run_backtest.py --rows-only --tag m1 --variants L+M1 L+M1+P    # m2, m3 likewise (+L+P)
python scripts/run_backtest.py --rows-only --tag o  --variants L+O            # op: L+O+P, om: L+O+M3
python scripts/run_backtest.py --rows-only --drop-partial 2025 --tag npg ...  # same splits, prefix np
python scripts/bias_report.py --rows reports/backtest_rows_{g,m1,m2,m3,o,op,om}.parquet \
    --nopartial reports/backtest_rows_np{g,m1,m2,m3,o}.parquet --intro intro.md
```

## Generated tables

Data: 6,631 games, 2,804 series, 2025-01-12 -> 2026-10-13, leagues: EM, FST, LCK, LCKC, LCS, LEC, LFL, LPL, MSI, WLDs (incl. 931 LPL partial games)  
Player-series per variant: 24979. Reference: **shrink30+A+B50** (locked). Paired differences are variant minus locked (negative CRPS / log score = better). Bias = mean predicted - actual kills per player-series.

## Variants vs locked

| variant | n | crps | crps_vs_locked | crps_se | logscore | log_vs_locked | log_se | bias | bias_se | ece_median | ece_half | tail_pred | tail_obs | tail_obs_over_pred | kill_mult |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| shrink30+A+B50 | 24979 | 1.8919 | 0.0000 | 0.0000 | 2.4796 | 0.0000 | 0.0000 | -0.1291 | 0.0232 | 0.0193 | 0.0131 | 0.2238 | 0.2441 | 1.0906 | 1.0000 |
| L+G | 24979 | 1.8925 | 0.0006 | 0.0006 | 2.4781 | -0.0015 | 0.0004 | -0.0067 | 0.0232 | 0.0173 | 0.0135 | 0.2267 | 0.2361 | 1.0414 | 1.0206 |
| L+M1 | 24979 | 1.8964 | 0.0045 | 0.0008 | 2.4820 | 0.0024 | 0.0005 | -0.1282 | 0.0232 | 0.0188 | 0.0130 | 0.2238 | 0.2443 | 1.0914 | 1.0000 |
| L+M1+P | 24979 | 1.8956 | 0.0037 | 0.0010 | 2.4814 | 0.0017 | 0.0006 | -0.1283 | 0.0232 | 0.0173 | 0.0131 | 0.2240 | 0.2443 | 1.0907 | 1.0000 |
| L+M2 | 24979 | 1.8945 | 0.0026 | 0.0005 | 2.4810 | 0.0014 | 0.0004 | -0.1268 | 0.0232 | 0.0187 | 0.0131 | 0.2239 | 0.2445 | 1.0919 | 1.0000 |
| L+M2+P | 24979 | 1.8935 | 0.0016 | 0.0008 | 2.4802 | 0.0006 | 0.0006 | -0.1269 | 0.0232 | 0.0178 | 0.0124 | 0.2239 | 0.2444 | 1.0916 | 1.0000 |
| L+M3 | 24979 | 1.8932 | 0.0013 | 0.0004 | 2.4803 | 0.0007 | 0.0004 | -0.1272 | 0.0232 | 0.0199 | 0.0132 | 0.2238 | 0.2445 | 1.0927 | 1.0000 |
| L+M3+P | 24979 | 1.8924 | 0.0005 | 0.0008 | 2.4794 | -0.0002 | 0.0005 | -0.1273 | 0.0232 | 0.0190 | 0.0133 | 0.2237 | 0.2442 | 1.0917 | 1.0000 |
| L+O | 24979 | 1.8651 | -0.0268 | 0.0027 | 2.4636 | -0.0160 | 0.0014 | -0.0690 | 0.0229 | 0.0151 | 0.0101 | 0.2265 | 0.2396 | 1.0576 | 1.0000 |
| L+O+M3 | 24979 | 1.8669 | -0.0250 | 0.0028 | 2.4643 | -0.0153 | 0.0014 | -0.0667 | 0.0229 | 0.0132 | 0.0089 | 0.2266 | 0.2397 | 1.0578 | 1.0000 |
| L+O+P | 24979 | 1.8645 | -0.0274 | 0.0028 | 2.4639 | -0.0158 | 0.0014 | -0.0691 | 0.0228 | 0.0151 | 0.0095 | 0.2266 | 0.2393 | 1.0559 | 1.0000 |
| L+P | 24979 | 1.8911 | -0.0008 | 0.0006 | 2.4792 | -0.0004 | 0.0004 | -0.1292 | 0.0232 | 0.0195 | 0.0128 | 0.2237 | 0.2439 | 1.0904 | 1.0000 |

## Bias by position

| variant | top | jng | mid | bot | sup |
|---|---|---|---|---|---|
| shrink30+A+B50 | -0.1979 | -0.1036 | -0.1695 | -0.1720 | -0.0023 |
| L+G | -0.0876 | 0.0311 | -0.0201 | 0.0148 | 0.0283 |
| L+M1 | -0.1954 | -0.1023 | -0.1695 | -0.1717 | -0.0019 |
| L+M1+P | -0.1189 | -0.1797 | -0.1473 | -0.2072 | 0.0122 |
| L+M2 | -0.1948 | -0.1006 | -0.1677 | -0.1690 | -0.0018 |
| L+M2+P | -0.1182 | -0.1780 | -0.1459 | -0.2043 | 0.0123 |
| L+M3 | -0.1957 | -0.1022 | -0.1676 | -0.1685 | -0.0017 |
| L+M3+P | -0.1193 | -0.1794 | -0.1462 | -0.2034 | 0.0122 |
| L+O | -0.1479 | -0.0369 | -0.0933 | -0.0790 | 0.0123 |
| L+O+M3 | -0.1458 | -0.0343 | -0.0909 | -0.0751 | 0.0128 |
| L+O+P | -0.0717 | -0.1144 | -0.0708 | -0.1148 | 0.0265 |
| L+P | -0.1215 | -0.1810 | -0.1473 | -0.2075 | 0.0118 |
| (se, locked) | 0.0475 | 0.0529 | 0.0570 | 0.0683 | 0.0208 |

## Bias by market (BO1 Map 1, BO3 Maps 1-2, BO5 Maps 1-3)

| variant | Map 1 | Maps 1-2 | Maps 1-3 |
|---|---|---|---|
| shrink30+A+B50 | -0.0612 | -0.0510 | -0.4864 |
| L+G | 0.0022 | 0.0700 | -0.2910 |
| L+M1 | -0.0563 | -0.0407 | -0.5239 |
| L+M1+P | -0.0563 | -0.0408 | -0.5239 |
| L+M2 | -0.0627 | -0.0505 | -0.4737 |
| L+M2+P | -0.0627 | -0.0506 | -0.4738 |
| L+M3 | -0.0770 | -0.0516 | -0.4552 |
| L+M3+P | -0.0770 | -0.0518 | -0.4552 |
| L+O | -0.0531 | -0.0980 | 0.0161 |
| L+O+M3 | -0.0688 | -0.0985 | 0.0494 |
| L+O+P | -0.0531 | -0.0981 | 0.0160 |
| L+P | -0.0612 | -0.0511 | -0.4865 |
| (se, locked) | 0.0357 | 0.0293 | 0.0700 |

## Bias by quarter (and G multiplier)

| quarter | n | shrink30+A+B50 | L+G | L+M1 | L+M1+P | L+M2 | L+M2+P | L+M3 | L+M3+P | L+O | L+O+M3 | L+O+P | L+P | G kill_mult |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025Q1 | 260 | 0.1506 | 0.1506 | 0.1993 | 0.1993 | 0.2155 | 0.2155 | 0.1964 | 0.1964 | -0.1370 | -0.0949 | -0.1370 | 0.1506 | 1.0000 |
| 2025Q2 | 5308 | -0.1473 | -0.0944 | -0.1023 | -0.1021 | -0.1260 | -0.1259 | -0.1280 | -0.1279 | -0.1306 | -0.1108 | -0.1304 | -0.1471 | 1.0080 |
| 2025Q3 | 4166 | -0.1630 | -0.0200 | -0.1710 | -0.1714 | -0.1444 | -0.1449 | -0.1530 | -0.1535 | -0.1289 | -0.1181 | -0.1294 | -0.1635 | 1.0224 |
| 2025Q4 | 1217 | -0.4201 | -0.2814 | -0.4194 | -0.4200 | -0.4000 | -0.4005 | -0.4343 | -0.4349 | -0.1434 | -0.1574 | -0.1440 | -0.4207 | 1.0263 |
| 2026Q1 | 4283 | -0.1377 | 0.0160 | -0.1765 | -0.1766 | -0.1446 | -0.1446 | -0.1279 | -0.1279 | -0.0621 | -0.0524 | -0.0621 | -0.1377 | 1.0277 |
| 2026Q2 | 4822 | -0.0476 | 0.0915 | -0.0893 | -0.0894 | -0.0699 | -0.0700 | -0.0570 | -0.0570 | 0.0371 | 0.0282 | 0.0370 | -0.0477 | 1.0242 |
| 2026Q3 | 4524 | -0.0631 | 0.0734 | -0.0225 | -0.0225 | -0.0674 | -0.0674 | -0.0818 | -0.0818 | -0.0507 | -0.0684 | -0.0507 | -0.0631 | 1.0229 |
| 2026Q4 | 399 | -0.4675 | -0.3042 | -0.4991 | -0.4989 | -0.5143 | -0.5140 | -0.4762 | -0.4759 | 0.0824 | 0.0731 | 0.0828 | -0.4673 | 1.0212 |

## Under-25% buckets (P(over) < 0.25)

`median_*`: at line = rounded median; `half_*`: half-point lines median +/- 0.5, 1.5. gap = observed - predicted.

| variant | median_n | median_pred | median_obs | median_gap | half_n | half_pred | half_obs | half_gap |
|---|---|---|---|---|---|---|---|---|
| shrink30+A+B50 | 447 | 0.2210 | 0.3378 | 0.1168 | 5856 | 0.1757 | 0.2148 | 0.0391 |
| L+G | 431 | 0.2222 | 0.3248 | 0.1026 | 5735 | 0.1766 | 0.2063 | 0.0296 |
| L+M1 | 461 | 0.2203 | 0.3406 | 0.1203 | 5867 | 0.1752 | 0.2132 | 0.0380 |
| L+M1+P | 463 | 0.2209 | 0.3283 | 0.1074 | 5876 | 0.1762 | 0.2107 | 0.0345 |
| L+M2 | 447 | 0.2207 | 0.3221 | 0.1015 | 5884 | 0.1758 | 0.2146 | 0.0388 |
| L+M2+P | 452 | 0.2210 | 0.3142 | 0.0932 | 5878 | 0.1762 | 0.2084 | 0.0322 |
| L+M3 | 436 | 0.2203 | 0.3280 | 0.1077 | 5837 | 0.1757 | 0.2143 | 0.0386 |
| L+M3+P | 448 | 0.2206 | 0.3237 | 0.1031 | 5882 | 0.1764 | 0.2096 | 0.0333 |
| L+O | 406 | 0.2213 | 0.3153 | 0.0940 | 5611 | 0.1761 | 0.1968 | 0.0206 |
| L+O+M3 | 408 | 0.2220 | 0.3137 | 0.0917 | 5642 | 0.1765 | 0.2005 | 0.0239 |
| L+O+P | 423 | 0.2227 | 0.3026 | 0.0799 | 5612 | 0.1766 | 0.1951 | 0.0185 |
| L+P | 459 | 0.2205 | 0.3290 | 0.1085 | 5887 | 0.1761 | 0.2088 | 0.0326 |

## Upper tail: hit rate / predicted at median + 1.5/2.5/3.5

| variant | ALL | top | jng | mid | bot | sup |
|---|---|---|---|---|---|---|
| shrink30+A+B50 | 1.0906 | 1.1520 | 1.0425 | 1.0959 | 1.1012 | 1.0244 |
| L+G | 1.0414 | 1.1017 | 0.9966 | 1.0402 | 1.0529 | 0.9841 |
| L+M1 | 1.0914 | 1.1566 | 1.0461 | 1.0915 | 1.1029 | 1.0198 |
| L+M1+P | 1.0907 | 1.1190 | 1.0777 | 1.0851 | 1.1123 | 1.0073 |
| L+M2 | 1.0919 | 1.1548 | 1.0448 | 1.0938 | 1.1047 | 1.0222 |
| L+M2+P | 1.0916 | 1.1173 | 1.0772 | 1.0878 | 1.1144 | 1.0088 |
| L+M3 | 1.0927 | 1.1509 | 1.0451 | 1.0949 | 1.1080 | 1.0282 |
| L+M3+P | 1.0917 | 1.1166 | 1.0761 | 1.0888 | 1.1144 | 1.0122 |
| L+O | 1.0576 | 1.1221 | 1.0102 | 1.0503 | 1.0795 | 0.9825 |
| L+O+M3 | 1.0578 | 1.1210 | 1.0105 | 1.0527 | 1.0792 | 0.9812 |
| L+O+P | 1.0559 | 1.0816 | 1.0412 | 1.0442 | 1.0885 | 0.9670 |
| L+P | 1.0904 | 1.1166 | 1.0729 | 1.0899 | 1.1107 | 1.0135 |

## Calibration at line = rounded median (shrink30+A+B50)

| bucket | n | pred | obs | diff | se |
|---|---|---|---|---|---|
| [0.0, 0.05) | 1 | 0.0454 | 1.0000 | 0.9546 | 0.0000 |
| [0.05, 0.1) | 2 | 0.0909 | 0.5000 | 0.4091 | 0.3536 |
| [0.1, 0.15) | 3 | 0.1156 | 0.3333 | 0.2177 | 0.2722 |
| [0.15, 0.2) | 42 | 0.1908 | 0.3095 | 0.1187 | 0.0713 |
| [0.2, 0.25) | 399 | 0.2261 | 0.3383 | 0.1123 | 0.0237 |
| [0.25, 0.3) | 652 | 0.2791 | 0.3006 | 0.0215 | 0.0180 |
| [0.3, 0.35) | 1258 | 0.3273 | 0.3680 | 0.0407 | 0.0136 |
| [0.35, 0.4) | 3458 | 0.3807 | 0.4291 | 0.0485 | 0.0084 |
| [0.4, 0.45) | 9127 | 0.4267 | 0.4471 | 0.0204 | 0.0052 |
| [0.45, 0.5) | 10037 | 0.4747 | 0.4757 | 0.0010 | 0.0050 |

## Calibration, half-point lines (shrink30+A+B50)

| bucket | n | pred | obs | diff | se |
|---|---|---|---|---|---|
| [0.0, 0.05) | 26 | 0.0347 | 0.1923 | 0.1576 | 0.0773 |
| [0.05, 0.1) | 567 | 0.0820 | 0.1552 | 0.0732 | 0.0152 |
| [0.1, 0.15) | 1192 | 0.1276 | 0.1560 | 0.0284 | 0.0105 |
| [0.15, 0.2) | 1822 | 0.1758 | 0.1981 | 0.0224 | 0.0093 |
| [0.2, 0.25) | 2249 | 0.2263 | 0.2748 | 0.0485 | 0.0094 |
| [0.25, 0.3) | 4276 | 0.2785 | 0.3248 | 0.0464 | 0.0072 |
| [0.3, 0.35) | 8533 | 0.3273 | 0.3584 | 0.0311 | 0.0052 |
| [0.35, 0.4) | 10838 | 0.3751 | 0.3908 | 0.0158 | 0.0047 |
| [0.4, 0.45) | 10418 | 0.4247 | 0.4333 | 0.0086 | 0.0049 |
| [0.45, 0.5) | 10037 | 0.4747 | 0.4757 | 0.0010 | 0.0050 |
| [0.5, 0.55) | 9702 | 0.5248 | 0.5264 | 0.0016 | 0.0051 |
| [0.55, 0.6) | 9159 | 0.5748 | 0.5741 | -0.0007 | 0.0052 |
| [0.6, 0.65) | 9495 | 0.6250 | 0.6233 | -0.0018 | 0.0050 |
| [0.65, 0.7) | 7547 | 0.6734 | 0.6706 | -0.0028 | 0.0054 |
| [0.7, 0.75) | 4539 | 0.7224 | 0.7513 | 0.0289 | 0.0064 |
| [0.75, 0.8) | 2490 | 0.7726 | 0.7968 | 0.0242 | 0.0081 |
| [0.8, 0.85) | 1376 | 0.8240 | 0.8314 | 0.0074 | 0.0101 |
| [0.85, 0.9) | 887 | 0.8693 | 0.8501 | -0.0192 | 0.0120 |
| [0.9, 0.95) | 39 | 0.9043 | 0.8462 | -0.0582 | 0.0578 |

## Calibration at line = rounded median (L+O)

| bucket | n | pred | obs | diff | se |
|---|---|---|---|---|---|
| [0.0, 0.05) | 1 | 0.0411 | 1.0000 | 0.9589 | 0.0000 |
| [0.05, 0.1) | 3 | 0.0955 | 0.3333 | 0.2378 | 0.2722 |
| [0.1, 0.15) | 2 | 0.1310 | 0.5000 | 0.3690 | 0.3536 |
| [0.15, 0.2) | 39 | 0.1904 | 0.3077 | 0.1173 | 0.0739 |
| [0.2, 0.25) | 361 | 0.2267 | 0.3130 | 0.0864 | 0.0244 |
| [0.25, 0.3) | 659 | 0.2789 | 0.3171 | 0.0382 | 0.0181 |
| [0.3, 0.35) | 1163 | 0.3269 | 0.3371 | 0.0102 | 0.0139 |
| [0.35, 0.4) | 3327 | 0.3806 | 0.4139 | 0.0332 | 0.0085 |
| [0.4, 0.45) | 9345 | 0.4272 | 0.4440 | 0.0168 | 0.0051 |
| [0.45, 0.5) | 10079 | 0.4749 | 0.4716 | -0.0033 | 0.0050 |

## Calibration, half-point lines (L+O)

| bucket | n | pred | obs | diff | se |
|---|---|---|---|---|---|
| [0.0, 0.05) | 19 | 0.0302 | 0.2105 | 0.1803 | 0.0935 |
| [0.05, 0.1) | 513 | 0.0831 | 0.1481 | 0.0651 | 0.0157 |
| [0.1, 0.15) | 1167 | 0.1279 | 0.1517 | 0.0237 | 0.0105 |
| [0.15, 0.2) | 1807 | 0.1763 | 0.1799 | 0.0035 | 0.0090 |
| [0.2, 0.25) | 2105 | 0.2267 | 0.2480 | 0.0212 | 0.0094 |
| [0.25, 0.3) | 4165 | 0.2785 | 0.3148 | 0.0363 | 0.0072 |
| [0.3, 0.35) | 8457 | 0.3275 | 0.3501 | 0.0226 | 0.0052 |
| [0.35, 0.4) | 10955 | 0.3750 | 0.3842 | 0.0092 | 0.0046 |
| [0.4, 0.45) | 10691 | 0.4252 | 0.4345 | 0.0093 | 0.0048 |
| [0.45, 0.5) | 10079 | 0.4749 | 0.4716 | -0.0033 | 0.0050 |
| [0.5, 0.55) | 9793 | 0.5248 | 0.5295 | 0.0047 | 0.0050 |
| [0.55, 0.6) | 9399 | 0.5752 | 0.5698 | -0.0053 | 0.0051 |
| [0.6, 0.65) | 9308 | 0.6248 | 0.6205 | -0.0043 | 0.0050 |
| [0.65, 0.7) | 7749 | 0.6736 | 0.6739 | 0.0003 | 0.0053 |
| [0.7, 0.75) | 4575 | 0.7220 | 0.7357 | 0.0138 | 0.0065 |
| [0.75, 0.8) | 2243 | 0.7727 | 0.7878 | 0.0151 | 0.0086 |
| [0.8, 0.85) | 1327 | 0.8240 | 0.8350 | 0.0110 | 0.0102 |
| [0.85, 0.9) | 875 | 0.8694 | 0.8526 | -0.0168 | 0.0120 |
| [0.9, 0.95) | 39 | 0.9055 | 0.8462 | -0.0593 | 0.0578 |

## CRPS by league (variant - locked, paired)

| league | n | crps_locked | L+G | L+G se | L+M1 | L+M1 se | L+M1+P | L+M1+P se | L+M2 | L+M2 se | L+M2+P | L+M2+P se | L+M3 | L+M3 se | L+M3+P | L+M3+P se | L+O | L+O se | L+O+M3 | L+O+M3 se | L+O+P | L+O+P se | L+P | L+P se |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LPL | 4883 | 1.9738 | 0.0025 | 0.0013 | 0.0107 | 0.0025 | 0.0104 | 0.0026 | 0.0008 | 0.0010 | 0.0005 | 0.0012 | 0.0016 | 0.0010 | 0.0012 | 0.0012 | -0.0284 | 0.0059 | -0.0259 | 0.0061 | -0.0285 | 0.0060 | -0.0002 | 0.0006 |
| EM | 4358 | 1.8663 | -0.0043 | 0.0014 | 0.0037 | 0.0010 | 0.0019 | 0.0027 | 0.0003 | 0.0010 | -0.0018 | 0.0027 | -0.0002 | 0.0009 | -0.0022 | 0.0027 | -0.0185 | 0.0068 | -0.0189 | 0.0068 | -0.0198 | 0.0072 | -0.0018 | 0.0025 |
| LCK | 3565 | 1.9421 | 0.0041 | 0.0016 | 0.0032 | 0.0020 | 0.0032 | 0.0020 | 0.0036 | 0.0013 | 0.0036 | 0.0013 | 0.0012 | 0.0010 | 0.0013 | 0.0011 | -0.0385 | 0.0078 | -0.0370 | 0.0079 | -0.0386 | 0.0078 | 0.0001 | 0.0004 |
| LCKC | 3467 | 2.1199 | 0.0012 | 0.0017 | -0.0007 | 0.0028 | -0.0006 | 0.0031 | 0.0049 | 0.0019 | 0.0049 | 0.0023 | 0.0050 | 0.0016 | 0.0052 | 0.0020 | -0.0360 | 0.0075 | -0.0311 | 0.0076 | -0.0358 | 0.0076 | 0.0002 | 0.0012 |
| LFL | 3230 | 1.6061 | -0.0002 | 0.0013 | 0.0055 | 0.0019 | 0.0064 | 0.0024 | 0.0050 | 0.0016 | 0.0058 | 0.0022 | -0.0009 | 0.0010 | 0.0001 | 0.0019 | -0.0192 | 0.0056 | -0.0193 | 0.0058 | -0.0178 | 0.0057 | 0.0010 | 0.0016 |
| LEC | 2700 | 1.6886 | 0.0016 | 0.0015 | 0.0026 | 0.0022 | 0.0002 | 0.0025 | 0.0028 | 0.0018 | 0.0004 | 0.0021 | 0.0030 | 0.0012 | 0.0007 | 0.0017 | -0.0194 | 0.0071 | -0.0168 | 0.0072 | -0.0217 | 0.0072 | -0.0022 | 0.0011 |
| LCS | 1825 | 1.9250 | 0.0041 | 0.0022 | 0.0034 | 0.0027 | -0.0005 | 0.0051 | 0.0034 | 0.0025 | -0.0003 | 0.0050 | -0.0002 | 0.0020 | -0.0040 | 0.0047 | -0.0110 | 0.0105 | -0.0098 | 0.0110 | -0.0152 | 0.0114 | -0.0036 | 0.0042 |
| WLDs | 437 | 1.6357 | -0.0040 | 0.0048 | 0.0023 | 0.0032 | 0.0024 | 0.0035 | -0.0054 | 0.0026 | -0.0053 | 0.0030 | 0.0012 | 0.0018 | 0.0019 | 0.0023 | -0.0243 | 0.0188 | -0.0201 | 0.0197 | -0.0240 | 0.0188 | 0.0001 | 0.0015 |
| MSI | 384 | 2.4938 | -0.0131 | 0.0071 | 0.0129 | 0.0058 | 0.0082 | 0.0068 | 0.0060 | 0.0035 | 0.0021 | 0.0051 | -0.0033 | 0.0031 | -0.0077 | 0.0048 | -0.0918 | 0.0396 | -0.0869 | 0.0411 | -0.0954 | 0.0391 | -0.0041 | 0.0039 |
| FST | 130 | 2.1563 | -0.0130 | 0.0128 | -0.0072 | 0.0069 | -0.0097 | 0.0083 | -0.0071 | 0.0032 | -0.0100 | 0.0056 | 0.0045 | 0.0031 | 0.0027 | 0.0055 | -0.0542 | 0.0431 | -0.0534 | 0.0411 | -0.0561 | 0.0430 | -0.0031 | 0.0043 |

## LPL with and without 2025 partial data

`without` = same run with the 805 LPL 2025 partial games removed from the data (they are neither history nor scored). LPL 2026 series are scored in both runs, so that comparison is paired; LPL 2025 series exist only with the partial data.

| variant | slice | n | crps_with | crps_without | diff | se | bias_with | bias_without |
|---|---|---|---|---|---|---|---|---|
| shrink30+A+B50 | LPL 2025 (with only) | 2426 | 1.9736 |  |  |  | -0.1089 |  |
| shrink30+A+B50 | LPL 2026 (paired) | 2457 | 1.9740 | 1.9949 | -0.0209 | 0.0092 | -0.0732 | -0.1035 |
| shrink30+A+B50 | non-LPL (paired) | 20096 | 1.8720 | 1.8737 | -0.0017 | 0.0011 |  |  |
| shrink30+A+B50 | all common (paired) | 22553 | 1.8831 | 1.8869 | -0.0038 | 0.0014 |  |  |
| L+O | LPL 2025 (with only) | 2426 | 1.9452 |  |  |  | -0.1277 |  |
| L+O | LPL 2026 (paired) | 2457 | 1.9457 | 1.9663 | -0.0206 | 0.0087 | -0.0532 | -0.0643 |
| L+O | non-LPL (paired) | 20096 | 1.8456 | 1.8475 | -0.0019 | 0.0011 |  |  |
| L+O | all common (paired) | 22553 | 1.8565 | 1.8605 | -0.0040 | 0.0014 |  |  |
