# Worlds 2026 modeling check (data through 2026-10-23, patch 16.2)

## Team labels

39 / 39 labels put the right team first in the slate's team search (`names.search_teams`).

| label | expected | resolve | search_top | ok |
|---|---|---|---|---|
| Gen.G | Gen.G | Gen.G | Gen.G | yes |
| GEN | Gen.G | Gen.G | Gen.G | yes |
| Hanwha Life | Hanwha Life Esports | Hanwha Life Esports | Hanwha Life Esports | yes |
| HLE | Hanwha Life Esports | Hanwha Life Esports | Hanwha Life Esports | yes |
| T1 | T1 | T1 | T1 | yes |
| Dplus KIA | Dplus Kia | Dplus Kia | Dplus Kia | yes |
| DK | Dplus Kia | Dplus Kia | Dplus Kia | yes |
| Anyone's Legend | Anyone's Legend | Anyone's Legend | Anyone's Legend | yes |
| AL | Anyone's Legend | Anyone's Legend | Anyone's Legend | yes |
| Bilibili Gaming | Bilibili Gaming | Bilibili Gaming | Bilibili Gaming | yes |
| BLG | Bilibili Gaming | Bilibili Gaming | Bilibili Gaming | yes |
| Top Esports | Top Esports | Top Esports | Top Esports | yes |
| TES | Top Esports | Top Esports | Top Esports | yes |
| Invictus Gaming | Invictus Gaming | Invictus Gaming | Invictus Gaming | yes |
| IG | Invictus Gaming | Invictus Gaming | Invictus Gaming | yes |
| G2 Esports | G2 Esports | G2 Esports | G2 Esports | yes |
| G2 | G2 Esports | G2 Esports | G2 Esports | yes |
| Movistar KOI | Movistar KOI | Movistar KOI | Movistar KOI | yes |
| MKOI | Movistar KOI | Movistar KOI | Movistar KOI | yes |
| Karmine Corp | Karmine Corp | Karmine Corp | Karmine Corp | yes |
| KC | Karmine Corp | Karmine Corp | Karmine Corp | yes |
| Team Liquid Alienware | Team Liquid | Team Liquid | Team Liquid | yes |
| TL Alienware | Team Liquid | Team Liquid | Team Liquid | yes |
| TLA | Team Liquid | Team Liquid | Team Liquid | yes |
| TL | Team Liquid | Team Liquid | Team Liquid | yes |
| LYON | LYON | LYON | LYON | yes |
| Lyon Gaming | LYON | LYON | LYON | yes |
| Team Secret Whales | Team Secret Whales | Team Secret Whales | Team Secret Whales | yes |
| TSW | Team Secret Whales | Team Secret Whales | Team Secret Whales | yes |
| CTBC Flying Oyster | CTBC Flying Oyster | CTBC Flying Oyster | CTBC Flying Oyster | yes |
| CFO | CTBC Flying Oyster | CTBC Flying Oyster | CTBC Flying Oyster | yes |
| LOS | LØS | LØS | LØS | yes |
| LØS | LØS | LØS | LØS | yes |
| Cloud9 Kia | Cloud9 | Cloud9 | Cloud9 | yes |
| C9 Kia | Cloud9 | Cloud9 | Cloud9 | yes |
| C9 | Cloud9 | Cloud9 | Cloud9 | yes |
| FURIA | FURIA | FURIA | FURIA | yes |
| MVK | MVK Esports | MVK Esports | MVK Esports | yes |
| MVK Esports | MVK Esports | MVK Esports | MVK Esports | yes |

## Kill environment of the Round 1 matchups

`target` = blend of the measured WLDs average (plain mean of the last 120 WLDs games, any year, weight min(1, n/40)) and mean(home-league envs) x event prior (Worlds 1.00). Each team's pool values are divided by the env of the league they were played in and multiplied by `target` (cross-league scaling), then the outcome-conditional opponent adjustment applies.

Measured WLDs: 14.58 team kills / map over the last 120 games (weight 1.00). Home envs: LCK 14.18, LPL 14.92, LEC 13.83, LCS 13.10, LCP 15.20, CBLOL 13.68; global 15.43.

Mean team kills per map in the pools (win / loss):

| matchup | home | home_env | target | a_win_loss | b_win_loss |
|---|---|---|---|---|---|
| Gen.G vs Invictus Gaming | LCK / LPL | 14.5500 | 14.5800 | 21.1 / 10.3 | 16.9 / 10.3 |
| Team Liquid vs CTBC Flying Oyster | LCS / LCP | 14.1500 | 14.5800 | 20.4 / 9.8 | 21.6 / 7.7 |
| Anyone's Legend vs Karmine Corp | LPL / LEC | 14.3700 | 14.5800 | 19.8 / 9.2 | 16.9 / 7.2 |
| G2 Esports vs Dplus Kia | LEC / LCK | 14.0000 | 14.5800 | 19.2 / 8.5 | 17.7 / 11.7 |
| Hanwha Life Esports vs Top Esports | LCK / LPL | 14.5500 | 14.5800 | 19.7 / 12.4 | 24.5 / 10.0 |
| LØS vs T1 | CBLOL / LCK | 13.9300 | 14.5800 | 19.3 / 11.9 | 21.8 / 10.6 |
| Team Secret Whales vs LYON | LCP / LCS | 14.1500 | 14.5800 | 18.3 / 9.5 | 19.9 / 8.7 |
| Bilibili Gaming vs Movistar KOI | LPL / LEC | 14.3700 | 14.5800 | 20.9 / 8.4 | 20.8 / 9.7 |

## Swiss starters with fewer than 15 effective games

Effective games = Kish n_eff of the patch-decay weights (0.65 per patch, 12 patches) on the player's games: what the kill share actually rests on. `games` = raw count in the window (the share shrink uses min(1, games/30)). Starters = latest lineup in the data.

None.

All 80 starters: median 40.0 effective games, min 31.3.

## Format -> market

| format | app_market | backtest_market |
|---|---|---|
| BO1 | Map 1 | Map 1 |
| BO3 | Maps 1-2 | Maps 1-2 |
| BO5 | Maps 1-3 | Maps 1-3 |
