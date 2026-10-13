"""Download Oracle's Elixir yearly match CSV(s) and write compact parquet to data/.

Usage:
    python scripts/refresh_data.py                 # current + previous year from Drive
    python scripts/refresh_data.py --years 2025 2026
    python scripts/refresh_data.py --csv path/to/2026_LoL_esports_match_data_from_OraclesElixir.csv

OE publishes one CSV per year in a public Google Drive folder. The folder id can be
overridden with OE_DRIVE_FOLDER_ID; individual file ids with OE_FILE_ID_<YEAR>.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA_DIR, LEAGUES  # noqa: E402
from src.data.oe import process  # noqa: E402

DEFAULT_FOLDER_ID = "1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH"
FILE_TEMPLATE = "{year}_LoL_esports_match_data_from_OraclesElixir.csv"


def download_year(year: int, out_dir: Path) -> Path:
    import gdown

    target = out_dir / FILE_TEMPLATE.format(year=year)
    file_id = os.environ.get(f"OE_FILE_ID_{year}")
    if file_id:
        gdown.download(id=file_id, output=str(target), quiet=False)
        return target
    folder = os.environ.get("OE_DRIVE_FOLDER_ID", DEFAULT_FOLDER_ID)
    files = gdown.download_folder(id=folder, skip_download=True, quiet=True)
    match = [f for f in files if Path(f.path).name == target.name]
    if not match:
        raise FileNotFoundError(f"{target.name} not found in Drive folder {folder}")
    gdown.download(id=match[0].id, output=str(target), quiet=False)
    return target


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def write(players: pd.DataFrame, teams: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    players.to_parquet(out_dir / "players.parquet", index=False, compression="zstd")
    teams.to_parquet(out_dir / "teams.parquet", index=False, compression="zstd")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="*")
    ap.add_argument("--csv", type=Path, nargs="*", help="use local CSV(s) instead of Drive")
    ap.add_argument("--out", type=Path, default=DATA_DIR)
    args = ap.parse_args(argv)

    if args.csv:
        paths = list(args.csv)
    else:
        this_year = dt.date.today().year
        years = args.years or [this_year - 1, this_year]
        tmp = Path(tempfile.mkdtemp(prefix="oe_"))
        paths = []
        for y in years:
            try:
                paths.append(download_year(y, tmp))
            except Exception as exc:  # a missing future/old year should not kill refresh
                print(f"warning: could not download {y}: {exc}", file=sys.stderr)
        if not paths:
            print("error: nothing downloaded", file=sys.stderr)
            return 1

    raw = pd.concat([read_csv(p) for p in paths], ignore_index=True)
    players, teams = process(raw, LEAGUES)
    write(players, teams, args.out)
    print(f"wrote {len(players)} player rows, {len(teams)} team rows, "
          f"{teams['gameid'].nunique()} games, {teams['series_id'].nunique()} series "
          f"-> {args.out}")
    print(teams.groupby("league", observed=True)["gameid"].nunique().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
