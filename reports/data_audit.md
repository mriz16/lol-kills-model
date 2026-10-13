# Data audit

Sources: oe_fixture.csv  
Raw rows: 1,716; raw games: 143  
Date range: 2025-01-12 08:50:00 -> 2025-04-19 12:40:00  
Patch range: 25.01 -> 25.08 (8 patches)  
Kept (complete, in-scope): 127 games, 61 series, 1,270 player rows  
Incomplete in-scope games dropped: 1  
Series with a missing map (excluded from backtest): 1  
Best-of inferred: {1: 23, 3: 30, 5: 8}

## Expected league codes

MISSING: LPL, LCS, LCKC, EM, LFL, WLDs, MSI

## League codes found in the file

FST, LCK, LDL, LEC

## Per league

| league | raw_games | incomplete_games | raw_rows | kept_games | kept_series | in_scope |
|---|---|---|---|---|---|---|
| LCK | 66 | 0 | 792 | 66 | 26 | True |
| LEC | 37 | 0 | 444 | 37 | 26 | True |
| FST | 25 | 1 | 300 | 24 | 9 | True |
| LDL | 15 | 0 | 180 | 0 | 0 | False |
