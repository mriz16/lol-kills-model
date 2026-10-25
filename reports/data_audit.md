# Data audit

Sources: 2026_LoL_esports_match_data_from_OraclesElixir (1).csv  
Raw rows: 117,660; raw games: 9,805  
Date range: 2026-01-01 09:14:49 -> 2026-10-25 07:17:26  
Patch range: 15.24 -> 16.20 (20 patches)  
Kept (complete + passing partial, in-scope): 4,722 games, 2,121 series, 47,220 player rows  
Incomplete in-scope games: 132 (kept as passing partial: 132)  
Series with a missing map (excluded from backtest): 3  
Best-of inferred: {1: 603, 3: 1157, 5: 361}

## Expected league codes

All expected league codes found.

## League codes found in the file

AC, AL, AM, CBLOL, CCWS, CD, DCGI, DCup, EBL, EM, EWC, FST, HC, HLL, HM, HW, KeSPA Cup, LAS, LCK, LCKC, LCP, LCS, LEC, LES, LFL, LIT, LJL, LPL, LPLOL, LRN, LRS, MSI, NACL, NL, NLC, PCS, PRM, RL, ROL, TCL, VCS, WLDs, WLRQ, WSCI

## League code aliases

LTA N -> LCS, LTA North -> LCS, LTA S -> CBLOL, LTA South -> CBLOL

## Per league

| league | raw_games | incomplete_games | raw_rows | kept_games | kept_series | maps_to | in_scope |
|---|---|---|---|---|---|---|---|
| LPL | 674 | 126 | 8088 | 674 | 246 |  | True |
| EM | 524 | 0 | 6288 | 524 | 266 |  | True |
| LCK | 502 | 0 | 6024 | 502 | 188 |  | True |
| LCKC | 491 | 0 | 5892 | 491 | 183 |  | True |
| LEC | 383 | 0 | 4596 | 383 | 186 |  | True |
| LJL | 363 | 0 | 4356 | 363 | 270 |  | True |
| CBLOL | 299 | 0 | 3588 | 299 | 124 |  | True |
| LCP | 283 | 0 | 3396 | 283 | 97 |  | True |
| LCS | 283 | 0 | 3396 | 283 | 102 |  | True |
| LFL | 275 | 0 | 3300 | 275 | 201 |  | True |
| EWC | 232 | 0 | 2784 | 232 | 103 |  | True |
| VCS | 155 | 0 | 1860 | 155 | 58 |  | True |
| PCS | 86 | 0 | 1032 | 86 | 33 |  | True |
| MSI | 71 | 0 | 852 | 71 | 20 |  | True |
| WLDs | 50 | 0 | 600 | 50 | 25 |  | True |
| FST | 45 | 0 | 540 | 45 | 13 |  | True |
| WLRQ | 6 | 6 | 72 | 6 | 6 |  | True |
| LAS | 541 | 0 | 6492 | 0 | 0 |  | False |
| AL | 350 | 0 | 4200 | 0 | 0 |  | False |
| PRM | 322 | 0 | 3864 | 0 | 0 |  | False |
| NACL | 281 | 0 | 3372 | 0 | 0 |  | False |
| LIT | 280 | 0 | 3360 | 0 | 0 |  | False |
| ROL | 261 | 0 | 3132 | 0 | 0 |  | False |
| CD | 258 | 0 | 3096 | 0 | 0 |  | False |
| HLL | 235 | 0 | 2820 | 0 | 0 |  | False |
| TCL | 219 | 0 | 2628 | 0 | 0 |  | False |
| EBL | 213 | 0 | 2556 | 0 | 0 |  | False |
| RL | 211 | 0 | 2532 | 0 | 0 |  | False |
| NLC | 210 | 0 | 2520 | 0 | 0 |  | False |
| LES | 208 | 0 | 2496 | 0 | 0 |  | False |
| HC | 201 | 0 | 2412 | 0 | 0 |  | False |
| LRS | 187 | 0 | 2244 | 0 | 0 |  | False |
| LRN | 176 | 0 | 2112 | 0 | 0 |  | False |
| HM | 174 | 0 | 2088 | 0 | 0 |  | False |
| LPLOL | 162 | 0 | 1944 | 0 | 0 |  | False |
| NL | 149 | 0 | 1788 | 0 | 0 |  | False |
| HW | 103 | 0 | 1236 | 0 | 0 |  | False |
| WSCI | 85 | 0 | 1020 | 0 | 0 |  | False |
| DCGI | 71 | 32 | 852 | 0 | 0 |  | False |
| AM | 63 | 0 | 756 | 0 | 0 |  | False |
| KeSPA Cup | 61 | 0 | 732 | 0 | 0 |  | False |
| CCWS | 40 | 0 | 480 | 0 | 0 |  | False |
| AC | 19 | 0 | 228 | 0 | 0 |  | False |
| DCup | 3 | 3 | 36 | 0 | 0 |  | False |

## Warnings

- WLRQ: 6/6 games not 'complete' (see partial games below)

## Partial games (all in-scope leagues)

Kept only if kills/deaths/assists/result/gamelength and playername are non-null for all 10 players and player kills sum to team kills on both sides.

| league | year | partial_games | passed | failed |
|---|---|---|---|---|
| LPL | 2026 | 126 | 126 | 0 |
| WLRQ | 2026 | 6 | 6 | 0 |

**LPL 2025 partial games: 0 pass, 0 fail.**

## International events

| league | year | teams | names |
|---|---|---|---|
| EWC | 2026 | 50 | Anyone's Legend, BNK FEARX, Bilibili Gaming, CTBC Flying Oyster, Cloud9, DN SOOPers, Deep Cross Gaming, DetonatioN FocusMe, Dignitas, Disguised, Dplus Kia, FURIA, Fluxo W7M, FlyQuest, Fnatic, Fukuoka SoftBank HAWKS gaming, G2 Esports, GAM Esports, Galions, Gen.G, GiantX, Ground Zero Gaming, HANJIN BRION, Hanwha Life Esports, JD Gaming, KT Rolster, Karmine Corp, Kiwoom DRX, LOUD, LYON, Leviatan, LØS, MVK Esports, Movistar KOI, Natus Vincere, Nongshim RedForce, RED Canids, SK Gaming, Sentinels, Shifters, Shopify Rebellion, Solary, T1, Team Heretics, Team Liquid, Team Secret, Team Secret Whales, Team Vitality, Vivo Keyd Stars, paiN Gaming |
| FST | 2026 | 8 | BNK FEARX, Bilibili Gaming, G2 Esports, Gen.G, JD Gaming, LOUD, LYON, Team Secret Whales |
| MSI | 2026 | 11 | Bilibili Gaming, Deep Cross Gaming, FURIA, G2 Esports, Hanwha Life Esports, Karmine Corp, LYON, T1, Team Liquid, Team Secret Whales, Top Esports |
| WLDs | 2026 | 21 | Anyone's Legend, Bilibili Gaming, CTBC Flying Oyster, Cloud9, Dplus Kia, FURIA, G2 Esports, Gen.G, Hanwha Life Esports, Invictus Gaming, JD Gaming, Karmine Corp, LYON, LØS, MVK Esports, Movistar KOI, T1, Team Liquid, Team Secret Whales, Team WE, Top Esports |
| WLRQ | 2026 | 8 | Bilibili Gaming Junior, EDG Youth Team, Ocean & Kimi Tournament Alliance, Top Esports Challenger, Weibo Gaming Youth Team, ZSK, 小爬虫, 我超威 |

**Teams at international events with no domestic games in the data:** Bilibili Gaming Junior, EDG Youth Team, Ocean & Kimi Tournament Alliance, Team Secret, Top Esports Challenger, Weibo Gaming Youth Team, ZSK, 小爬虫, 我超威
