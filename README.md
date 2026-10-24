# LoL Player-Kills Prop Model

This app prices League of Legends player **kills** props (PrizePicks / Underdog). It uses a
Monte Carlo series simulator (20,000 sims per series) built on Oracle's Elixir match data.
The model is locked to shrink30 + A + B50 + O, with game-script, C, M and P off. See
`CLAUDE.md` for the full spec and `reports/` for the walk-forward backtests.

- `streamlit_app.py`: the phone-first app (Slate, Screenshot, Results, Slip, Record tabs)
- `src/engine/`: the simulator. `src/app/`: app logic (names, odds input, slate, OCR, pick
  log, grading).
- Match data: you upload the Oracle's Elixir yearly CSV (or a .zip of it) at the top of
  the app. It is merged by `gameid` into the stored data and saved as parquet to the
  **`data`** branch through the GitHub API (never `main`, so an upload does not redeploy
  the app mid-slate). The app loads the latest `data` commit on startup. See "Updating the
  match data" below.
- `.github/workflows/grade.yml`: runs three times a day. It grades logged picks and the
  slate log on the `picks-log` branch against the `data` branch parquet.
- The sims carry the locked spread inflation S. Probabilities shown in the app are
  calibrated (I, beta) and then shrunk toward the line (W: `p_line + λ (p_cal − p_line)`,
  λ per market). Calibrated-only and raw numbers are one tap away. A PLAY needs a shown
  edge of at least 7 points. See `reports/backtest_spread.md` and
  `reports/calibration_fit.md`.
- `.github/workflows/calibrate.yml`: runs weekly (Mondays 11:07 UTC) and on demand. It
  re-runs the walk-forward backtest and refits S / I / W (`scripts/refit_calibration.py`),
  then commits `data/calib_pairs.parquet`, `data/spread_rows.parquet`,
  `data/w_rows.parquet`, `data/calib_params.json` and `reports/calibration_fit.md`. If any
  parameter moves more than the drift limits (S k ±0.10, I map ±0.05, W λ ±0.25), the job
  fails and commits nothing. The report in the job summary shows what moved. To accept
  the move, re-run with **force**.

## Deploy on Streamlit Community Cloud

### 1. Put the app on `main`
Merge the Phase 2 branch into `main`. The app deploys from `main`. Each daily data commit
to `main` redeploys it; data uploads go to the `data` branch and do not.

### 2. Create a fine-grained GitHub token (for the data and the pick log)
Streamlit Cloud's disk is wiped on every restart. So the app stores uploaded match data on
a **`data`** branch and your pick log as `logs/picks.csv` on a separate **`picks-log`**
branch, through the GitHub API. Using separate branches means an upload or a logged pick
never redeploys the app.

1. GitHub → your avatar → **Settings** → **Developer settings** → **Personal access tokens**
   → **Fine-grained tokens** → **Generate new token**.
2. Fill in the token settings:
   - Name: `lol-kills-model app`. Expiration: your choice (you will re-paste the token when it expires).
   - Resource owner: `mriz16`.
   - Repository access: **Only select repositories** → `mriz16/lol-kills-model`.
   - Repository permissions → **Contents: Read and write**. Metadata: read-only is added
     automatically. Leave everything else at "No access".
3. Generate the token and copy it (`github_pat_...`).

### 3. Create the app
1. Go to <https://share.streamlit.io>, sign in with GitHub and authorize Streamlit.
2. Click **Create app** → **Deploy a public app from GitHub** (or "from repo").
3. Set Repository `mriz16/lol-kills-model`, Branch `main`, Main file path `streamlit_app.py`.
4. Optionally pick a custom subdomain, e.g. `lol-kills`.
5. Open **Advanced settings**:
   - **Python version: 3.11**
   - **Secrets**: paste the block below with your token (format in
     `.streamlit/secrets.toml.example`):
     ```toml
     [github]
     token = "github_pat_..."
     repo = "mriz16/lol-kills-model"
     branch = "picks-log"
     data_branch = "data"
     ```
6. Click **Deploy**. The first build installs `requirements.txt` plus `tesseract-ocr` from
   `packages.txt`, which takes a few minutes.

### 4. Lock it down
Anyone who can open the app can write to your pick log. After the deploy, open the app →
**Share** (top right) and limit viewers to your own email. If the repo is private, the app
is private by default.

### 5. Use it on your phone
Open the app URL and use the browser's **Add to Home Screen**.

### Changing secrets later
On share.streamlit.io, open the app's **⋮** menu → **Settings** → **Secrets**, edit the
values and save. The app restarts with the new values.

### Updating the match data
1. Download the current year's file from Oracle's Elixir
   (`2026_LoL_esports_match_data_from_OraclesElixir.csv`; zipping it first makes the upload
   much faster on a phone).
2. In the app, the banner at the top shows "Data through" per league. Use the uploader
   under it, then press **Merge into data**.
3. The app parses only the columns the model needs, applies the same rules as always
   (complete games + partial games that pass the kill checks, league aliases, series
   inference, audit), and merges by `gameid`: games in the file replace the stored ones,
   every other stored game (all of 2025 included) is kept. It shows the games added, the
   new date range per league, and the audit's warnings.
4. The merged parquet is committed to the `data` branch (`data/players.parquet`,
   `data/teams.parquet`, `data/meta.json`, `reports/data_audit.md`). The team pools are
   rebuilt right away, without a restart. Other open sessions pick up the new data within
   5 minutes.

Without GitHub secrets, or if the save fails, the merged data stays on the server's disk
until the next restart, and the app says so. Uploads up to 600 MB are accepted
(`.streamlit/config.toml`). Locally, `python scripts/refresh_data.py --csv FILE` runs the
same pipeline into `data/`.

### Pick-log grading
The `grade-picks` workflow runs three times a day (03:17, 11:17, 19:17 UTC), or on demand
from Actions → grade-picks → Run workflow. It copies the parquet from the `data` branch
(falling back to the parquet in `main` if nothing was uploaded yet), checks out
`picks-log`, grades every pending pick, and commits `logs/graded.csv`. It needs no extra secret; it
uses the workflow's own `GITHUB_TOKEN`. The Record tab also grades pending picks live from
the app's data, so results show up as soon as the data has them.

### Slate log and backfill
Every **Run slate** writes every entered line (played or not) to `logs/slates.csv` on the
same branch, with all model outputs (calibrated and raw) and a `played` flag. Logging a
slip, or "Mark lines as played" on the Results tab, sets the flag. The workflow grades
every row into `logs/slates_graded.csv`; the Record tab shows the result. This is the
real-world evaluation set.

To enter a past slate, open **Mode: live or backfill** at the top, turn on backfill and
pick the slate date. Everything (pools, rosters, lineups, calibration) is then built from
games before 00:00 UTC of that date only, and the rows are logged with `backfill = 1`.

### Refitting W on real PrizePicks lines
W's λ is fitted on stand-in lines (each player's rolling median + 0.5) until the slate log
has real data. The Record tab shows how many graded PrizePicks lines exist, out of the 300
needed. Once it reaches 300:

```bash
git fetch origin picks-log && git worktree add ../picks-log origin/picks-log
python scripts/refit_w_real.py --log-root ../picks-log            # dry run: prints λ
python scripts/refit_w_real.py --log-root ../picks-log --write    # data/w_lambda_real.json
```

Commit `data/w_lambda_real.json` to `main`. The app then uses the real-line λ for every
cutoff after the last slate it was fitted on; backfills of earlier dates keep the
proxy-line λ. The fit uses the same rule: the top 5 edges per logged slate should hit at
their shown probability. Re-run it as the log grows.

Grading rules:
- A pick is matched to the first series of team vs opponent that starts between 1 day
  before and 5 days after the time it was logged (slate-log rows: after the slate date).
- Kills count only on maps actually played in the market range.
- A player who missed a map in the range → void.
- A pick with no series found within 7 days → void.
- Slips: pushed and void legs drop out, and the slip pays the table multiplier for the
  legs that remain. Fewer than 2 legs left → refund. This is an approximation of the
  platforms' reduced-slip rules.

## Run locally

```bash
pip install -r requirements.txt            # plus: apt install tesseract-ocr (for OCR)
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # optional; edit the token
streamlit run streamlit_app.py
pytest
```

Without secrets, the pick log is written to `logs/` on local disk (git-ignored).

## Data and backtest

```bash
python scripts/refresh_data.py --csv 2026_LoL_esports_match_data_from_OraclesElixir.csv
                                           # local OE CSV / .zip -> merged into data/*.parquet
scripts/fetch_data_branch.sh data_branch   # the app's uploaded parquet -> data_branch/
python scripts/worlds_check.py             # Worlds labels / env / thin samples -> reports/
python scripts/intl_report.py              # international-event backtest -> reports/
python scripts/run_backtest.py --quick     # walk-forward backtest -> reports/
python scripts/refit_calibration.py        # backtest + S/I/W refit -> data/calib_*, w_rows
# spread / calibration study (report only)
python scripts/run_backtest.py --rows-only --keep-pmf --tag spread --variants L+O
python scripts/spread_report.py --rows reports/backtest_rows_spread_L+O.parquet
```
