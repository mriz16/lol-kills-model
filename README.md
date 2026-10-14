# LoL Player-Kills Prop Model

This app prices League of Legends player **kills** props (PrizePicks / Underdog). It uses a
Monte Carlo series simulator (20,000 sims per series) built on Oracle's Elixir match data.
The model is locked to shrink30 + A + B50 + O, with game-script, C, M and P off. See
`CLAUDE.md` for the full spec and `reports/` for the walk-forward backtests.

- `streamlit_app.py`: the phone-first app (Slate, Screenshot, Results, Slip, Record tabs)
- `src/engine/`: the simulator. `src/app/`: app logic (names, odds input, slate, OCR, pick
  log, grading).
- `.github/workflows/refresh.yml`: runs daily. It refreshes `data/*.parquet` on `main`,
  then grades logged picks on the `picks-log` branch.

## Deploy on Streamlit Community Cloud

### 1. Put the app on `main`
Merge the Phase 2 branch into `main`. The app deploys from `main`. Each daily data commit
to `main` redeploys it with fresh data.

### 2. Create a fine-grained GitHub token (for the pick log)
Streamlit Cloud's disk is wiped on every restart. So the app stores your pick log as
`logs/picks.csv` on a separate **`picks-log`** branch, through the GitHub API. Using a
separate branch means a logged pick never redeploys the app.

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

### Pick-log grading
The `refresh-data` workflow runs daily at 09:17 UTC, or on demand from Actions → refresh-data →
Run workflow. After refreshing the data, it checks out `picks-log`, grades every pending
pick against the new OE data, and commits `logs/graded.csv`. It needs no extra secret; it
uses the workflow's own `GITHUB_TOKEN`. The Record tab also grades pending picks live from
the app's data, so results show up as soon as the data has them.

Grading rules:
- A pick is matched to the first series of team vs opponent that starts between 1 day
  before and 5 days after the time it was logged.
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
python scripts/refresh_data.py             # OE CSVs from Google Drive -> data/*.parquet
python scripts/run_backtest.py --quick     # walk-forward backtest -> reports/
```
