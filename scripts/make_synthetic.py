"""Generate synthetic OE-schema data.

    python scripts/make_synthetic.py --fixture     # tests/fixtures/oe_fixture.csv (small)
    python scripts/make_synthetic.py               # data/synthetic/{players,teams}.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import DATA_DIR  # noqa: E402
from src.data.oe import process  # noqa: E402
from src.data.synthetic import generate, generate_fixture  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    if args.fixture:
        out = ROOT / "tests" / "fixtures" / "oe_fixture.csv"
        df = generate_fixture()
        df.to_csv(out, index=False)
        print(f"wrote {len(df)} rows -> {out}")
        return 0
    raw = generate(seed=args.seed)
    players, teams = process(raw)
    out = DATA_DIR / "synthetic"
    out.mkdir(parents=True, exist_ok=True)
    players.to_parquet(out / "players.parquet", index=False, compression="zstd")
    teams.to_parquet(out / "teams.parquet", index=False, compression="zstd")
    print(f"wrote {teams['gameid'].nunique()} games / {teams['series_id'].nunique()} series "
          f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
