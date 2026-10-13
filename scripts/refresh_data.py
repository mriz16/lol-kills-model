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
from src.config import DATA_DIR, LEAGUES, REPORTS_DIR  # noqa: E402
from src.data.audit import audit  # noqa: E402
from src.data.oe import REQUIRED_COLUMNS, process  # noqa: E402

MAX_PARQUET_MB = 50


class DataError(RuntimeError):
    """Download/parse produced unusable data: fail the job loudly."""


DEFAULT_FOLDER_ID = "1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH"
FILE_TEMPLATE = "{year}_LoL_esports_match_data_from_OraclesElixir.csv"


def download_year(year: int, out_dir: Path) -> Path:
    import gdown

    target = out_dir / FILE_TEMPLATE.format(year=year)
    file_id = os.environ.get(f"OE_FILE_ID_{year}")
    if file_id:
        gdown.download(id=file_id, output=str(target), quiet=False)
        return target
    folder = os.environ.get("OE_DRIVE_FOLDER_ID") or DEFAULT_FOLDER_ID
    files = gdown.download_folder(id=folder, skip_download=True, quiet=True) or []
    match = [f for f in files if Path(f.path).name == target.name]
    if not match:
        raise FileNotFoundError(f"{target.name} not found in Drive folder {folder}")
    gdown.download(id=match[0].id, output=str(target), quiet=False)
    return target


def check_not_html(path: Path) -> None:
    """Drive quota/permission errors come back as an HTML page saved under the CSV name."""
    if not path.exists() or path.stat().st_size == 0:
        raise DataError(f"{path.name}: download is missing or empty")
    head = path.read_bytes()[:2048].lstrip().lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html") or b"<html" in head:
        raise DataError(f"{path.name}: download returned HTML, not CSV "
                        f"(Drive quota / permissions / wrong file id?)")


def read_csv(path: Path) -> pd.DataFrame:
    check_not_html(path)
    df = pd.read_csv(path, low_memory=False)
    if len(df) == 0:
        raise DataError(f"{path.name}: 0 rows")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"{path.name}: not an OE match file, missing columns {missing}")
    return df


def write(players: pd.DataFrame, teams: pd.DataFrame, out_dir: Path) -> float:
    out_dir.mkdir(parents=True, exist_ok=True)
    players.to_parquet(out_dir / "players.parquet", index=False, compression="zstd")
    teams.to_parquet(out_dir / "teams.parquet", index=False, compression="zstd")
    mb = sum((out_dir / f).stat().st_size for f in ("players.parquet", "teams.parquet")) / 2**20
    if mb > MAX_PARQUET_MB:
        raise DataError(f"parquet is {mb:.1f} MB (> {MAX_PARQUET_MB} MB limit)")
    return mb


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="*")
    ap.add_argument("--csv", type=Path, nargs="*", help="use local CSV(s) instead of Drive")
    ap.add_argument("--out", type=Path, default=DATA_DIR)
    ap.add_argument("--audit", type=Path, default=REPORTS_DIR / "data_audit.md")
    args = ap.parse_args(argv)
    try:
        return run(args)
    except DataError as exc:
        print(f"::error::{exc}")
        print(f"error: {exc}", file=sys.stderr)
        return 1


def run(args) -> int:
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
            except FileNotFoundError as exc:  # e.g. next year's file not published yet
                print(f"::warning::{exc}")
        if not paths:
            raise DataError("nothing downloaded")

    raw = pd.concat([read_csv(p) for p in paths], ignore_index=True)
    players, teams = process(raw, LEAGUES)
    if len(teams) == 0:
        raise DataError("0 complete in-scope games after filtering (league codes changed?)")
    text, missing = audit(raw, players, teams, [p.name for p in paths])
    print(text)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(text)
    for lg in missing:
        print(f"::warning::expected league code {lg} not found in OE data")
    mb = write(players, teams, args.out)
    print(f"wrote {len(players)} player rows, {len(teams)} team rows ({mb:.2f} MB) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
