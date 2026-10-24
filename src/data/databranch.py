"""Processed parquet on a dedicated `data` branch of the repo (not `main`).

The app persists uploaded OE data here through the GitHub API (same token as the pick log)
so it survives restarts without redeploying the app, and loads the latest commit of the
branch on startup. The grading and calibration workflows check the same files out of the
branch. The branch is an orphan that mirrors the repo layout:

    data/players.parquet   data/teams.parquet   data/meta.json   reports/data_audit.md

`GitHubDataBranch` uses the Git Data API, so all files of one upload land in one commit.
`LocalDataBranch` is the same interface on disk (dev without secrets, tests).
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DATA_BRANCH = "data"
PLAYERS_PATH = "data/players.parquet"
TEAMS_PATH = "data/teams.parquet"
META_PATH = "data/meta.json"
AUDIT_PATH = "reports/data_audit.md"


class ConflictError(RuntimeError):
    """The branch moved between reading its head and writing (another upload won)."""


class GitHubDataBranch:
    API = "https://api.github.com"

    def __init__(self, token: str, repo: str, branch: str = DATA_BRANCH, session=None):
        import requests
        self.repo, self.branch = repo, branch
        self.s = session or requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}",
                               "Accept": "application/vnd.github+json",
                               "X-GitHub-Api-Version": "2022-11-28"})

    def _api(self, path: str) -> str:
        return f"{self.API}/repos/{self.repo}/{path}"

    def head(self) -> str | None:
        """Commit sha of the branch tip, None when the branch does not exist yet."""
        r = self.s.get(self._api(f"git/ref/heads/{self.branch}"), timeout=20)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()["object"]["sha"]

    def read_bytes(self, path: str, ref: str | None = None) -> bytes | None:
        r = self.s.get(self._api(f"contents/{path}"), params={"ref": ref or self.branch},
                       headers={"Accept": "application/vnd.github.raw+json"}, timeout=120)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.content

    def commit(self, files: dict[str, bytes], message: str,
               expected_head: str | None = "unchecked") -> str:
        """Write files (path -> bytes) in one commit on top of the branch tip; returns the
        new commit sha. expected_head (a sha, or None for "branch must not exist") makes a
        moved branch a ConflictError instead of silently stacking on it."""
        head = self.head()
        if expected_head != "unchecked" and head != expected_head:
            raise ConflictError(f"{self.branch} moved: {expected_head} -> {head}")
        tree = []
        for path, data in files.items():
            b = self.s.post(self._api("git/blobs"), timeout=300, json={
                "content": base64.b64encode(data).decode("ascii"), "encoding": "base64"})
            b.raise_for_status()
            tree.append({"path": path, "mode": "100644", "type": "blob",
                         "sha": b.json()["sha"]})
        body = {"tree": tree}
        if head:
            c = self.s.get(self._api(f"git/commits/{head}"), timeout=20)
            c.raise_for_status()
            body["base_tree"] = c.json()["tree"]["sha"]
        t = self.s.post(self._api("git/trees"), json=body, timeout=60)
        t.raise_for_status()
        c = self.s.post(self._api("git/commits"), timeout=30, json={
            "message": message, "tree": t.json()["sha"], "parents": [head] if head else []})
        c.raise_for_status()
        sha = c.json()["sha"]
        if head:  # fast-forward only: a concurrent upload makes this a 422
            r = self.s.patch(self._api(f"git/refs/heads/{self.branch}"), timeout=20,
                             json={"sha": sha, "force": False})
        else:     # first upload: the branch is an orphan holding only the data files
            r = self.s.post(self._api("git/refs"), timeout=20,
                            json={"ref": f"refs/heads/{self.branch}", "sha": sha})
        if r.status_code in (409, 422):
            raise ConflictError(f"{self.branch}: {r.text[:200]}")
        r.raise_for_status()
        return sha


class LocalDataBranch:
    """Files under root; the 'head' is a hash of the stored parquet."""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def head(self) -> str | None:
        h = hashlib.sha1()
        found = False
        for p in (PLAYERS_PATH, TEAMS_PATH):
            f = self.root / p
            if f.exists():
                found = True
                h.update(f.read_bytes())
        return h.hexdigest() if found else None

    def read_bytes(self, path: str, ref: str | None = None) -> bytes | None:
        f = self.root / path
        return f.read_bytes() if f.exists() else None

    def commit(self, files: dict[str, bytes], message: str,
               expected_head: str | None = "unchecked") -> str:
        if expected_head != "unchecked" and self.head() != expected_head:
            raise ConflictError(str(self.root))
        for path, data in files.items():
            f = self.root / path
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(data)
        return self.head()


# ---------------------------------------------------------------- load / save --
def to_parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, compression="zstd")
    return buf.getvalue()


def load_frames(branch, ref: str | None = None
                ) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """(players, teams) from the branch (at commit `ref`), None when it has no data."""
    pb = branch.read_bytes(PLAYERS_PATH, ref)
    tb = branch.read_bytes(TEAMS_PATH, ref)
    if pb is None or tb is None:
        return None
    return pd.read_parquet(io.BytesIO(pb)), pd.read_parquet(io.BytesIO(tb))


def save_frames(branch, players: pd.DataFrame, teams: pd.DataFrame, audit_md: str,
                source: str, expected_head: str | None = "unchecked",
                now: datetime | None = None) -> str:
    """One commit: both parquet files, the audit and a small meta.json. Returns its sha."""
    now = now or datetime.now(timezone.utc)
    t = teams.astype({"league": str})
    meta = {
        "saved_at": now.isoformat(timespec="seconds"), "source": source,
        "games": int(teams["gameid"].nunique()), "player_rows": int(len(players)),
        "last_game": str(pd.Timestamp(teams["date"].max())),
        "last_by_league": {lg: str(d.date()) for lg, d in
                           t.groupby("league")["date"].max().items()},
    }
    files = {PLAYERS_PATH: to_parquet_bytes(players), TEAMS_PATH: to_parquet_bytes(teams),
             META_PATH: json.dumps(meta, indent=1).encode(),
             AUDIT_PATH: audit_md.encode()}
    msg = (f"data: upload {source} ({meta['games']} games, through "
           f"{meta['last_game'][:10]})")
    return branch.commit(files, msg, expected_head=expected_head)
