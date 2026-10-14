# Data audit

Sources: 2025_LoL_esports_match_data_from_OraclesElixir.csv, 2026_LoL_esports_match_data_from_OraclesElixir.csv  
Raw rows: 236,292; raw games: 19,691  
Date range: 2025-01-11 11:11:24 -> 2026-10-13 17:35:36  
Patch range: 15.01 -> 16.2 (43 patches)  
Kept (complete + passing partial, in-scope): 6,631 games, 2,804 series, 66,310 player rows  
Incomplete in-scope games: 943 (kept as passing partial: 931)  
Series with a missing map (excluded from backtest): 4  
Best-of inferred: {1: 613, 3: 1683, 5: 508}

## Expected league codes

All expected league codes found.

## League codes found in the file

AC, AL, AM, ASI, CBLOL, CCWS, CD, CT, DCGI, DCup, EBL, EM, EWC, FST, HC, HLL, HM, HW, IC, KeSPA Cup, LAS, LCK, LCKC, LCP, LCS, LEC, LES, LFL, LFL2, LIT, LJL, LPL, LPLOL, LRN, LRS, LTA, LTA N, LTA S, LVP SL, MSI, NACL, NEXO, NL, NLC, PCS, PRM, PRMP, RL, ROL, TCL, VCS, WLDs, WLRQ, WSCI

## League code aliases

LTA N -> LCS, LTA North -> LCS

## Per league

| league | raw_games | incomplete_games | raw_rows | kept_games | kept_series | maps_to | in_scope |
|---|---|---|---|---|---|---|---|
| LPL | 1479 | 931 | 17748 | 1479 | 562 |  | True |
| LCK | 1057 | 0 | 12684 | 1057 | 397 |  | True |
| LCKC | 1033 | 0 | 12396 | 1033 | 386 |  | True |
| EM | 954 | 0 | 11448 | 954 | 466 |  | True |
| LEC | 691 | 0 | 8292 | 691 | 329 |  | True |
| LFL | 593 | 0 | 7116 | 593 | 360 |  | True |
| LCS | 283 | 0 | 3396 | 283 | 102 |  | True |
| LTA N | 214 | 0 | 2568 | 214 | 93 | LCS | True |
| MSI | 151 | 0 | 1812 | 151 | 39 |  | True |
| WLDs | 108 | 12 | 1296 | 96 | 44 |  | True |
| FST | 80 | 0 | 960 | 80 | 26 |  | True |
| LAS | 949 | 0 | 11388 | 0 | 0 |  | False |
| LJL | 737 | 0 | 8844 | 0 | 0 |  | False |
| AL | 643 | 0 | 7716 | 0 | 0 |  | False |
| PRM | 623 | 0 | 7476 | 0 | 0 |  | False |
| LCP | 574 | 0 | 6888 | 0 | 0 |  | False |
| CD | 559 | 0 | 6708 | 0 | 0 |  | False |
| NACL | 556 | 0 | 6672 | 0 | 0 |  | False |
| ROL | 469 | 0 | 5628 | 0 | 0 |  | False |
| NLC | 460 | 0 | 5520 | 0 | 0 |  | False |
| TCL | 459 | 0 | 5508 | 0 | 0 |  | False |
| HLL | 456 | 0 | 5472 | 0 | 0 |  | False |
| RL | 450 | 0 | 5400 | 0 | 0 |  | False |
| LIT | 419 | 0 | 5028 | 0 | 0 |  | False |
| EBL | 404 | 0 | 4848 | 0 | 0 |  | False |
| HC | 401 | 0 | 4812 | 0 | 0 |  | False |
| PCS | 388 | 0 | 4656 | 0 | 0 |  | False |
| HM | 364 | 0 | 4368 | 0 | 0 |  | False |
| LRN | 327 | 0 | 3924 | 0 | 0 |  | False |
| LRS | 323 | 0 | 3876 | 0 | 0 |  | False |
| LPLOL | 320 | 0 | 3840 | 0 | 0 |  | False |
| CBLOL | 299 | 0 | 3588 | 0 | 0 |  | False |
| VCS | 296 | 0 | 3552 | 0 | 0 |  | False |
| EWC | 265 | 0 | 3180 | 0 | 0 |  | False |
| LVP SL | 264 | 0 | 3168 | 0 | 0 |  | False |
| AM | 240 | 0 | 2880 | 0 | 0 |  | False |
| LFL2 | 233 | 0 | 2796 | 0 | 0 |  | False |
| LTA S | 222 | 0 | 2664 | 0 | 0 |  | False |
| HW | 217 | 0 | 2604 | 0 | 0 |  | False |
| LES | 208 | 0 | 2496 | 0 | 0 |  | False |
| NEXO | 154 | 0 | 1848 | 0 | 0 |  | False |
| NL | 149 | 0 | 1788 | 0 | 0 |  | False |
| KeSPA Cup | 117 | 0 | 1404 | 0 | 0 |  | False |
| WSCI | 85 | 0 | 1020 | 0 | 0 |  | False |
| DCup | 81 | 6 | 972 | 0 | 0 |  | False |
| PRMP | 73 | 0 | 876 | 0 | 0 |  | False |
| CT | 50 | 0 | 600 | 0 | 0 |  | False |
| ASI | 42 | 0 | 504 | 0 | 0 |  | False |
| CCWS | 40 | 0 | 480 | 0 | 0 |  | False |
| LTA | 40 | 0 | 480 | 0 | 0 |  | False |
| DCGI | 39 | 0 | 468 | 0 | 0 |  | False |
| IC | 28 | 0 | 336 | 0 | 0 |  | False |
| AC | 19 | 0 | 228 | 0 | 0 |  | False |
| WLRQ | 6 | 6 | 72 | 0 | 0 |  | False |

## Warnings

- LPL: 931/1479 games not 'complete' (see partial games below)

## Partial games (LPL)

Kept only if kills/deaths/assists/result/gamelength and playername are non-null for all 10 players and player kills sum to team kills on both sides.

| league | year | partial_games | passed | failed |
|---|---|---|---|---|
| LPL | 2025 | 805 | 805 | 0 |
| LPL | 2026 | 126 | 126 | 0 |

**LPL 2025 partial games: 805 pass, 0 fail.**
