"""Process a local Oracle's Elixir CSV (or .zip) into data/*.parquet: the same pipeline
the app's uploader runs (src.data.upload), for dev and one-off rebuilds.

Usage:
    python scripts/refresh_data.py --csv 2026_LoL_esports_match_data_from_OraclesElixir.csv
    python scripts/refresh_data.py --csv 2026.zip --out data/      # merge into data/
    python scripts/refresh_data.py --csv 2025.csv 2026.csv --replace

By default each file is merged into the parquet already in --out by gameid (uploaded games
replace stored ones, nothing else is dropped); --replace starts from empty. Data is no
longer downloaded from Google Drive: the user uploads the OE file in the app, which saves
the parquet to the `data` branch (src.data.databranch).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA_DIR, REPORTS_DIR  # noqa: E402
from src.data.upload import UploadError, process_upload  # noqa: E402

MAX_PARQUET_MB = 50


def write(players: pd.DataFrame, teams: pd.DataFrame, out_dir: Path) -> float:
    out_dir.mkdir(parents=True, exist_ok=True)
    players.to_parquet(out_dir / "players.parquet", index=False, compression="zstd")
    teams.to_parquet(out_dir / "teams.parquet", index=False, compression="zstd")
    mb = sum((out_dir / f).stat().st_size for f in ("players.parquet", "teams.parquet")) / 2**20
    if mb > MAX_PARQUET_MB:
        raise UploadError(f"parquet is {mb:.1f} MB (> {MAX_PARQUET_MB} MB limit)")
    return mb


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, nargs="+", required=True,
                    help="OE match CSV(s) or .zip(s)")
    ap.add_argument("--out", type=Path, default=DATA_DIR)
    ap.add_argument("--audit", type=Path, default=REPORTS_DIR / "data_audit.md")
    ap.add_argument("--replace", action="store_true",
                    help="ignore the parquet already in --out")
    args = ap.parse_args(argv)
    players = teams = None
    if not args.replace and (args.out / "teams.parquet").exists():
        players = pd.read_parquet(args.out / "players.parquet")
        teams = pd.read_parquet(args.out / "teams.parquet")
    audits = []
    try:
        for path in args.csv:
            with open(path, "rb") as f:
                res = process_upload(f, path.name, players, teams)
            players, teams = res.players, res.teams
            audits.append(res.audit_md)
            print(f"{path.name}: {res.games_added} games added, {res.games_updated} replaced")
            for w in res.warnings:
                print(f"::warning::{w}")
        mb = write(players, teams, args.out)
    except (UploadError, OSError) as exc:
        print(f"::error::{exc}")
        print(f"error: {exc}", file=sys.stderr)
        return 1
    text = "\n\n".join(audits)
    print(text)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(text)
    print(f"wrote {len(players)} player rows, {len(teams)} team rows ({mb:.2f} MB) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
