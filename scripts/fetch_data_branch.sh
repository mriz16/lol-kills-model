#!/usr/bin/env bash
# Copy the processed OE parquet from the `data` branch (written by the app's uploader)
# into $1 (default: data_branch/). Falls back to the parquet bundled in main when the
# branch does not exist yet. Prints the directory to use as --data.
set -euo pipefail
out="${1:-data_branch}"
branch="${DATA_BRANCH:-data}"
mkdir -p "$out"
if git ls-remote --exit-code --heads origin "$branch" >/dev/null; then
  git fetch --depth 1 origin "$branch" >&2
  for f in players teams; do
    git show "FETCH_HEAD:data/$f.parquet" > "$out/$f.parquet"
  done
  git show "FETCH_HEAD:data/meta.json" >&2 2>/dev/null || true
  echo "using the $branch branch parquet ($(git rev-parse --short FETCH_HEAD))" >&2
else
  echo "::warning::no $branch branch yet (nothing uploaded): using the parquet in main" >&2
  cp data/players.parquet data/teams.parquet "$out/"
fi
echo "$out"
