"""Worlds modeling check -> reports/worlds_check.md.

- every Worlds team label (DFS / sportsbook spelling) resolves to the right OE team
- kill environment of each Swiss Round 1 matchup: measured event average, the two home
  leagues, the blend, and how pool values are rescaled
- Worlds starters whose kill share rests on fewer than MIN_EFF effective games
- format -> market (BO1 Map 1, BO3 Maps 1-2, BO5 Maps 1-3)

    python scripts/worlds_check.py [--data data/]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_backtest import _md  # noqa: E402
from src import config as C  # noqa: E402
from src.app.names import search_teams  # noqa: E402
from src.app.slate import MARKET_BY_FORMAT, DataStore  # noqa: E402
from src.backtest.walkforward import primary_market  # noqa: E402
from src.data.teams import resolve  # noqa: E402
from src.engine.features import build_matchup, kish_neff, locked_config  # noqa: E402

MIN_EFF = 15
# label as typed on a board -> expected OE team (Swiss stage + eliminated play-in teams)
LABELS = {
    "Gen.G": "Gen.G", "GEN": "Gen.G", "Hanwha Life": "Hanwha Life Esports",
    "HLE": "Hanwha Life Esports", "T1": "T1", "Dplus KIA": "Dplus Kia", "DK": "Dplus Kia",
    "Anyone's Legend": "Anyone's Legend", "AL": "Anyone's Legend",
    "Bilibili Gaming": "Bilibili Gaming", "BLG": "Bilibili Gaming",
    "Top Esports": "Top Esports", "TES": "Top Esports",
    "Invictus Gaming": "Invictus Gaming", "IG": "Invictus Gaming",
    "G2 Esports": "G2 Esports", "G2": "G2 Esports", "Movistar KOI": "Movistar KOI",
    "MKOI": "Movistar KOI", "Karmine Corp": "Karmine Corp", "KC": "Karmine Corp",
    "Team Liquid Alienware": "Team Liquid", "TL Alienware": "Team Liquid",
    "TLA": "Team Liquid", "TL": "Team Liquid", "LYON": "LYON", "Lyon Gaming": "LYON",
    "Team Secret Whales": "Team Secret Whales", "TSW": "Team Secret Whales",
    "CTBC Flying Oyster": "CTBC Flying Oyster", "CFO": "CTBC Flying Oyster",
    "LOS": "LØS", "LØS": "LØS",
    "Cloud9 Kia": "Cloud9", "C9 Kia": "Cloud9", "C9": "Cloud9", "FURIA": "FURIA",
    "MVK": "MVK Esports", "MVK Esports": "MVK Esports",
}
SWISS = ["Gen.G", "Hanwha Life Esports", "T1", "Dplus Kia", "Anyone's Legend",
         "Bilibili Gaming", "Top Esports", "Invictus Gaming", "G2 Esports", "Movistar KOI",
         "Karmine Corp", "Team Liquid", "LYON", "Team Secret Whales", "CTBC Flying Oyster",
         "LØS"]
ROUND1 = [("Gen.G", "Invictus Gaming"), ("Team Liquid", "CTBC Flying Oyster"),
          ("Anyone's Legend", "Karmine Corp"), ("G2 Esports", "Dplus Kia"),
          ("Hanwha Life Esports", "Top Esports"), ("LØS", "T1"),
          ("Team Secret Whales", "LYON"), ("Bilibili Gaming", "Movistar KOI")]


def player_games(store, player, cfg):
    h = store.hist
    idx = h.player_idx.get(player)
    if idx is None:
        return 0, 0.0, ""
    idx = h._prior(idx, h.p_time, store.cutoff)
    keep, w = h._weights(h.p_patch[idx], store.patch, cfg)
    idx, w = idx[keep], w[keep]
    lg = pd.Series(h.p_league[idx]).value_counts()
    return len(idx), kish_neff(w) if len(w) else 0.0, ", ".join(f"{k} {v}" for k, v in
                                                               lg.items())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=C.DATA_DIR)
    ap.add_argument("--out", type=Path, default=C.REPORTS_DIR / "worlds_check.md")
    args = ap.parse_args(argv)
    players = pd.read_parquet(args.data / "players.parquet")
    teams = pd.read_parquet(args.data / "teams.parquet")
    store = DataStore(players, teams)
    cfg = store.config()
    known = store.team_names
    lines = [f"# Worlds 2026 modeling check (data through {store.last_date:%Y-%m-%d}, "
             f"patch {store.patch_name})", ""]

    # ---- aliases
    rows = []
    for label, want in LABELS.items():
        got, top = resolve(label, known), (search_teams(label, known) or [None])[0]
        rows.append(dict(label=label, expected=want, resolve=got, search_top=top,
                         ok="yes" if top == want else "NO"))
    al = pd.DataFrame(rows)
    lines += ["## Team labels", "",
              f"{(al.ok == 'yes').sum()} / {len(al)} labels put the right team first in the "
              "slate's team search (`names.search_teams`).", "", _md(al), ""]

    # ---- environment
    lines += ["## Kill environment of the Round 1 matchups", "",
              "`target` = blend of the measured WLDs average (plain mean of the last 120 WLDs "
              "games, any year, weight min(1, n/40)) and mean(home-league envs) x event prior "
              "(Worlds 1.00). Each team's pool values are divided by the env of the league "
              "they were played in and multiplied by `target` (cross-league scaling), then "
              "the outcome-conditional opponent adjustment applies.", ""]
    envs = store.hist.env_table(store.cutoff, store.patch, cfg)
    m, n = store.hist.measured_env("WLDs", store.cutoff, store.patch, cfg)
    erows = []
    for a, b in ROUND1:
        ha, hb = store.home_league(a), store.home_league(b)
        p = build_matchup(store.hist, a, b, "WLDs", 1, 0.5,
                          (store.default_lineup(a), store.default_lineup(b)), store.cutoff,
                          store.patch, cfg)
        mean_k = [[float(np.dot(*p.pools[s][o])) for o in (0, 1)] for s in (0, 1)]
        erows.append(dict(matchup=f"{a} vs {b}", home=f"{ha} / {hb}",
                          home_env=round((envs.get(ha, envs['_global'])
                                          + envs.get(hb, envs['_global'])) / 2, 2),
                          target=round(p.info["target_env"], 2),
                          a_win_loss=f"{mean_k[0][1]:.1f} / {mean_k[0][0]:.1f}",
                          b_win_loss=f"{mean_k[1][1]:.1f} / {mean_k[1][0]:.1f}"))
    lines += [f"Measured WLDs: {m:.2f} team kills / map over the last {n} games "
              f"(weight {min(1, n / C.EVENT_ENV_FULL_WEIGHT_GAMES):.2f}). Home envs: "
              + ", ".join(f"{lg} {envs[lg]:.2f}" for lg in
                          ["LCK", "LPL", "LEC", "LCS", "LCP", "CBLOL"] if lg in envs)
              + f"; global {envs['_global']:.2f}.", "",
              "Mean team kills per map in the pools (win / loss):", "",
              _md(pd.DataFrame(erows)), ""]

    # ---- thin samples
    prow = []
    for tm in SWISS:
        for pos, pl in zip(C.POSITIONS, store.default_lineup(tm)):
            if pl is None:
                prow.append(dict(team=tm, role=pos, player="(none)", games=0, eff_games=0.0,
                                 leagues=""))
                continue
            n_g, eff, lgs = player_games(store, pl, cfg)
            prow.append(dict(team=tm, role=pos, player=pl, games=n_g,
                             eff_games=round(eff, 1), leagues=lgs))
    pr = pd.DataFrame(prow)
    thin = pr[pr["eff_games"] < MIN_EFF].sort_values("eff_games")
    lines += [f"## Swiss starters with fewer than {MIN_EFF} effective games", "",
              "Effective games = Kish n_eff of the patch-decay weights (0.65 per patch, 12 "
              "patches) on the player's games: what the kill share actually rests on. "
              "`games` = raw count in the window (the share shrink uses min(1, games/30)). "
              "Starters = latest lineup in the data.", "",
              _md(thin) if len(thin) else "None.", "",
              f"All {len(pr)} starters: median {pr['eff_games'].median():.1f} effective "
              f"games, min {pr['eff_games'].min():.1f}.", ""]

    # ---- markets
    mk = pd.DataFrame([dict(format=f"BO{bo}", app_market=MARKET_BY_FORMAT[bo],
                            backtest_market=primary_market(bo)) for bo in (1, 3, 5)])
    lines += ["## Format -> market", "", _md(mk), ""]
    text = "\n".join(lines)
    args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
