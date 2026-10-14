# Softer O fade (--quick, real OE data, latest 6 months, 20,000 sims)

O's fade is `min(1, nDef/10)`, where nDef = opponent games with the relevant result. This run
tests `/15` and `/20`. The bar to lock a softer fade: CRPS better than O by more than 2 se,
with ECE no worse. All deltas are paired against L+O on the same 8,816 player-series.

| variant | CRPS | ΔCRPS vs O (se) | Δlog vs O (se) | ECE median / half-pt | bias |
|---|---|---|---|---|---|
| **L+O (fade /10)** | 1.8898 | – | – | 0.0107 / 0.0128 | -0.016 |
| L+O15 | 1.8904 | +0.0006 (0.0004) | +0.0002 (0.0003) | 0.0112 / 0.0125 | -0.013 |
| L+O20 | 1.8906 | +0.0008 (0.0007) | +0.0004 (0.0005) | 0.0101 / 0.0118 | -0.010 |

**Result: no softer fade is locked.** Both are slightly worse on CRPS and log score. The
differences are within about 1.5 se, nowhere near a 2-se gain. O stays at `min(1, nDef/10)`.

Reproduce: `python scripts/run_backtest.py --quick --rows-only --tag ofade --variants L+O L+O15 L+O20`
