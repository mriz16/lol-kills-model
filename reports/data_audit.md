# Data audit

Sources: 2025_LoL_esports_match_data_from_OraclesElixir.csv, 2026_LoL_esports_match_data_from_OraclesElixir.csv  
Raw rows: 236,328; raw games: 19,694  
Date range: 2025-01-11 11:11:24 -> 2026-10-14 18:23:21  
Patch range: 15.01 -> 16.2 (43 patches)  
Kept (complete + passing partial, in-scope): 9,473 games, 4,240 series, 94,730 player rows  
Incomplete in-scope games: 949 (kept as passing partial: 949)  
Series with a missing map (excluded from backtest): 12  
Best-of inferred: {1: 1229, 3: 2294, 5: 717}

## Expected league codes

All expected league codes found.

## League codes found in the file

AC, AL, AM, ASI, CBLOL, CCWS, CD, CT, DCGI, DCup, EBL, EM, EWC, FST, HC, HLL, HM, HW, IC, KeSPA Cup, LAS, LCK, LCKC, LCP, LCS, LEC, LES, LFL, LFL2, LIT, LJL, LPL, LPLOL, LRN, LRS, LTA, LTA N, LTA S, LVP SL, MSI, NACL, NEXO, NL, NLC, PCS, PRM, PRMP, RL, ROL, TCL, VCS, WLDs, WLRQ, WSCI

## League code aliases

LTA N -> LCS, LTA North -> LCS, LTA S -> CBLOL, LTA South -> CBLOL

## Per league

| league | raw_games | incomplete_games | raw_rows | kept_games | kept_series | maps_to | in_scope |
|---|---|---|---|---|---|---|---|
| LPL | 1479 | 931 | 17748 | 1479 | 562 |  | True |
| LCK | 1057 | 0 | 12684 | 1057 | 397 |  | True |
| LCKC | 1033 | 0 | 12396 | 1033 | 386 |  | True |
| EM | 957 | 0 | 11484 | 957 | 467 |  | True |
| LJL | 737 | 0 | 8844 | 737 | 596 |  | True |
| LEC | 691 | 0 | 8292 | 691 | 329 |  | True |
| LFL | 593 | 0 | 7116 | 593 | 360 |  | True |
| LCP | 574 | 0 | 6888 | 574 | 208 |  | True |
| PCS | 388 | 0 | 4656 | 388 | 155 |  | True |
| CBLOL | 299 | 0 | 3588 | 299 | 124 |  | True |
| VCS | 296 | 0 | 3552 | 296 | 116 |  | True |
| LCS | 283 | 0 | 3396 | 283 | 102 |  | True |
| EWC | 265 | 0 | 3180 | 265 | 121 |  | True |
| LTA S | 222 | 0 | 2664 | 222 | 93 | CBLOL | True |
| LTA N | 214 | 0 | 2568 | 214 | 93 | LCS | True |
| MSI | 151 | 0 | 1812 | 151 | 39 |  | True |
| WLDs | 108 | 12 | 1296 | 108 | 47 |  | True |
| FST | 80 | 0 | 960 | 80 | 26 |  | True |
| LTA | 40 | 0 | 480 | 40 | 13 |  | True |
| WLRQ | 6 | 6 | 72 | 6 | 6 |  | True |
| LAS | 949 | 0 | 11388 | 0 | 0 |  | False |
| AL | 643 | 0 | 7716 | 0 | 0 |  | False |
| PRM | 623 | 0 | 7476 | 0 | 0 |  | False |
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
| HM | 364 | 0 | 4368 | 0 | 0 |  | False |
| LRN | 327 | 0 | 3924 | 0 | 0 |  | False |
| LRS | 323 | 0 | 3876 | 0 | 0 |  | False |
| LPLOL | 320 | 0 | 3840 | 0 | 0 |  | False |
| LVP SL | 264 | 0 | 3168 | 0 | 0 |  | False |
| AM | 240 | 0 | 2880 | 0 | 0 |  | False |
| LFL2 | 233 | 0 | 2796 | 0 | 0 |  | False |
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
| DCGI | 39 | 0 | 468 | 0 | 0 |  | False |
| IC | 28 | 0 | 336 | 0 | 0 |  | False |
| AC | 19 | 0 | 228 | 0 | 0 |  | False |

## Warnings

- LPL: 931/1479 games not 'complete' (see partial games below)
- WLRQ: 6/6 games not 'complete' (see partial games below)

## Partial games (all in-scope leagues)

Kept only if kills/deaths/assists/result/gamelength and playername are non-null for all 10 players and player kills sum to team kills on both sides.

| league | year | partial_games | passed | failed |
|---|---|---|---|---|
| LPL | 2025 | 805 | 805 | 0 |
| LPL | 2026 | 126 | 126 | 0 |
| WLDs | 2025 | 12 | 12 | 0 |
| WLRQ | 2026 | 6 | 6 | 0 |

**LPL 2025 partial games: 805 pass, 0 fail.**

## International events

| league | year | teams | names |
|---|---|---|---|
| EWC | 2025 | 12 | Anyone's Legend, Bilibili Gaming, CTBC Flying Oyster, Cloud9, FURIA, FlyQuest, G2 Esports, GAM Esports, Gen.G, Hanwha Life Esports, Movistar KOI, T1 |
| EWC | 2026 | 50 | Anyone's Legend, BNK FEARX, Bilibili Gaming, CTBC Flying Oyster, Cloud9, DN SOOPers, Deep Cross Gaming, DetonatioN FocusMe, Dignitas, Disguised, Dplus Kia, FURIA, Fluxo W7M, FlyQuest, Fnatic, Fukuoka SoftBank HAWKS gaming, G2 Esports, GAM Esports, Galions, Gen.G, GiantX, Ground Zero Gaming, HANJIN BRION, Hanwha Life Esports, JD Gaming, KT Rolster, Karmine Corp, Kiwoom DRX, LOUD, LYON, Leviatan, LØS, MVK Esports, Movistar KOI, Natus Vincere, Nongshim RedForce, RED Canids, SK Gaming, Sentinels, Shifters, Shopify Rebellion, Solary, T1, Team Heretics, Team Liquid, Team Secret, Team Secret Whales, Team Vitality, Vivo Keyd Stars, paiN Gaming |
| FST | 2025 | 5 | CTBC Flying Oyster, Hanwha Life Esports, Karmine Corp, Team Liquid, Top Esports |
| FST | 2026 | 8 | BNK FEARX, Bilibili Gaming, G2 Esports, Gen.G, JD Gaming, LOUD, LYON, Team Secret Whales |
| LTA | 2025 | 11 | 100 Thieves, Cloud9, FlyQuest, Isurus Estral, LOUD, Leviatan, RED Canids, Shopify Rebellion, Team Liquid, Vivo Keyd Stars, paiN Gaming |
| MSI | 2025 | 10 | Anyone's Legend, Bilibili Gaming, CTBC Flying Oyster, FURIA, FlyQuest, G2 Esports, GAM Esports, Gen.G, Movistar KOI, T1 |
| MSI | 2026 | 11 | Bilibili Gaming, Deep Cross Gaming, FURIA, G2 Esports, Hanwha Life Esports, Karmine Corp, LYON, T1, Team Liquid, Team Secret Whales, Top Esports |
| WLDs | 2025 | 19 | 100 Thieves, Anyone's Legend, Bilibili Gaming, CTBC Flying Oyster, FlyQuest, Fnatic, G2 Esports, Gen.G, Hanwha Life Esports, Invictus Gaming, JD Gaming, KT Rolster, Movistar KOI, PSG Talon, T1, Team Secret Whales, Top Esports, Vivo Keyd Stars, Weibo Gaming |
| WLDs | 2026 | 4 | Invictus Gaming, JD Gaming, Team WE, Top Esports |
| WLRQ | 2026 | 8 | Bilibili Gaming Junior, EDG Youth Team, Ocean & Kimi Tournament Alliance, Top Esports Challenger, Weibo Gaming Youth Team, ZSK, 小爬虫, 我超威 |

**Teams at international events with no domestic games in the data:** Bilibili Gaming Junior, EDG Youth Team, Ocean & Kimi Tournament Alliance, Team Secret, Top Esports Challenger, Weibo Gaming Youth Team, ZSK, 小爬虫, 我超威
