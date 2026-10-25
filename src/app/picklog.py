"""Pick log: CSV rows of the picks actually played, stored in the repo.

Streamlit Community Cloud's disk is temporary, so the app writes the log through the
GitHub contents API to a dedicated branch (default `picks-log`, never `main`, so a logged
pick does not redeploy the app). The refresh workflow grades the log into `graded.csv`
on the same branch. `LocalStore` is the same interface on disk (dev / tests / workflow).
"""
from __future__ import annotations

import base64
import hashlib
import io
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PICKS_PATH = "logs/picks.csv"
GRADED_PATH = "logs/graded.csv"
LOG_BRANCH = "picks-log"

PICK_COLUMNS = [
    "pick_id", "slip_id", "logged_at", "date", "platform", "n_legs", "stake", "payout",
    "slip_p_joint", "slip_ev", "league", "best_of", "matchup", "team", "opponent",
    "player", "role", "market", "line", "side", "p_more", "p_less", "p_push", "p_side",
    "model_mean", "fair_line", "mode", "kind",
]
GRADED_COLUMNS = ["pick_id", "status", "actual", "series_id", "graded_at"]


class ConflictError(RuntimeError):
    """The file changed between read and write (someone else wrote first)."""


class LocalStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    def read(self, path: str) -> tuple[str | None, str | None]:
        f = self.root / path
        if not f.exists():
            return None, None
        text = f.read_text()
        return text, hashlib.sha1(text.encode()).hexdigest()

    def write(self, path: str, text: str, message: str, sha: str | None) -> None:
        f = self.root / path
        cur, cur_sha = self.read(path)
        if cur_sha != sha:
            raise ConflictError(path)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)


class GitHubStore:
    """Files on one branch of a repo via the GitHub contents API.

    Needs a fine-grained token with Contents: read and write on the repo. The branch is
    created from the default branch on the first write if it does not exist.
    """
    API = "https://api.github.com"

    def __init__(self, token: str, repo: str, branch: str = LOG_BRANCH, session=None):
        import requests
        self.repo, self.branch = repo, branch
        self.s = session or requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}",
                               "Accept": "application/vnd.github+json",
                               "X-GitHub-Api-Version": "2022-11-28"})

    def _url(self, path: str) -> str:
        return f"{self.API}/repos/{self.repo}/contents/{path}"

    def read(self, path: str) -> tuple[str | None, str | None]:
        r = self.s.get(self._url(path), params={"ref": self.branch}, timeout=20)
        if r.status_code == 404:
            return None, None
        r.raise_for_status()
        j = r.json()
        if j.get("encoding") == "base64" and j.get("content") is not None:
            text = base64.b64decode(j["content"]).decode("utf-8")
        else:  # > 1 MB files come back without content: fetch the raw blob
            raw = self.s.get(self._url(path), params={"ref": self.branch}, timeout=30,
                             headers={"Accept": "application/vnd.github.raw+json"})
            raw.raise_for_status()
            text = raw.text
        return text, j["sha"]

    def _ensure_branch(self) -> None:
        r = self.s.get(f"{self.API}/repos/{self.repo}/branches/{self.branch}", timeout=20)
        if r.status_code != 404:
            r.raise_for_status()
            return
        repo = self.s.get(f"{self.API}/repos/{self.repo}", timeout=20)
        repo.raise_for_status()
        base = repo.json()["default_branch"]
        ref = self.s.get(f"{self.API}/repos/{self.repo}/git/ref/heads/{base}", timeout=20)
        ref.raise_for_status()
        c = self.s.post(f"{self.API}/repos/{self.repo}/git/refs", timeout=20,
                        json={"ref": f"refs/heads/{self.branch}",
                              "sha": ref.json()["object"]["sha"]})
        if c.status_code not in (201, 422):  # 422: created concurrently
            c.raise_for_status()

    def write(self, path: str, text: str, message: str, sha: str | None) -> None:
        if sha is None:
            self._ensure_branch()
        body = {"message": message, "branch": self.branch,
                "content": base64.b64encode(text.encode("utf-8")).decode("ascii")}
        if sha:
            body["sha"] = sha
        r = self.s.put(self._url(path), json=body, timeout=30)
        if r.status_code in (409, 422) and "sha" in r.text.lower():
            raise ConflictError(path)
        if r.status_code == 409:
            raise ConflictError(path)
        r.raise_for_status()


def check_store(store) -> tuple[bool, str]:
    """Can the app write the log (GitHubStore)? Creates the log branch if missing (that is
    the write test: it needs the same Contents: write permission as saving a pick). A
    function, not a method, so it also works on a store that st.cache_resource kept from
    an older deploy of this module."""
    try:
        r = store.s.get(f"{store.API}/repos/{store.repo}", timeout=20)
        if r.status_code == 401:
            return False, "GitHub rejected the token (expired or mistyped): make a new one."
        if r.status_code == 404:
            return False, (f"The token cannot see {store.repo}: check `repo` in the "
                           "secrets and the token's repository access.")
        r.raise_for_status()
        b = store.s.get(f"{store.API}/repos/{store.repo}/branches/{store.branch}", timeout=20)
        existed = b.status_code != 404
        store._ensure_branch()
    except Exception as exc:
        code = getattr(getattr(exc, "response", None), "status_code", None)
        if code == 403:
            return False, ("The token can read the repo but not write: give it "
                           "Contents: Read and write.")
        return False, f"GitHub error: {exc}"
    return True, (f"OK: writing to {store.repo}, branch `{store.branch}`"
                  + ("." if existed else " (created now)."))


# --------------------------------------------------------------------- read / write --
def _frame(text: str | None, columns: list[str]) -> pd.DataFrame:
    if not text:
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(io.StringIO(text), dtype={"pick_id": str, "slip_id": str})
    for c in columns:
        if c not in df.columns:
            df[c] = pd.NA
    return df[columns + [c for c in df.columns if c not in columns]]


def read_picks(store) -> pd.DataFrame:
    return _frame(store.read(PICKS_PATH)[0], PICK_COLUMNS)


def read_graded(store) -> pd.DataFrame:
    return _frame(store.read(GRADED_PATH)[0], GRADED_COLUMNS)


def new_slip_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d") + "-" + uuid.uuid4().hex[:6]


def make_pick_rows(legs: list[dict], slip: dict, platform: str, stake: float,
                   slip_id: str | None = None, now: datetime | None = None,
                   mode: str = "power") -> list[dict]:
    """Log rows for one played slip. legs are rows of the results table (goblin legs:
    slate.goblin_row). mode "power" or "flex"."""
    now = now or datetime.now(timezone.utc)
    slip_id = slip_id or new_slip_id()
    rows = []
    for i, r in enumerate(legs):
        rows.append(dict(
            pick_id=f"{slip_id}-{i + 1}", slip_id=slip_id,
            logged_at=now.isoformat(timespec="seconds"), date=now.date().isoformat(),
            platform=platform, n_legs=len(legs), stake=float(stake),
            payout=slip.get("payout"), slip_p_joint=round(float(slip["p_joint"]), 4),
            slip_ev=None if slip.get("ev") is None else round(float(slip["ev"]), 4),
            league=r["league"], best_of=int(r["best_of"]), matchup=r["matchup"],
            team=r["team"], opponent=r["opponent"], player=r["player"], role=r["role"],
            market=r["market"], line=float(r["line"]), side=r["side"],
            p_more=round(float(r["p_more"]), 4), p_less=round(float(r["p_less"]), 4),
            p_push=round(float(r["p_push"]), 4),
            p_side=round(float(r["p_more"] if r["side"] == "more" else r["p_less"]), 4),
            model_mean=round(float(r["mean"]), 3), fair_line=float(r["fair_line"]),
            mode=mode, kind=r.get("kind", "standard"),
        ))
    return rows


def missing_picks(store, pick_ids) -> list[str]:
    """Pick ids not in the stored log, read back after a write."""
    have = set(read_picks(store)["pick_id"].astype(str))
    return [str(x) for x in pick_ids if str(x) not in have]


def append_picks(store, rows: list[dict], retries: int = 3) -> pd.DataFrame:
    """Append rows to the pick log (read-modify-write, retried on a write conflict)."""
    for attempt in range(retries + 1):
        text, sha = store.read(PICKS_PATH)
        df = _frame(text, PICK_COLUMNS)
        new = pd.DataFrame(rows, columns=PICK_COLUMNS)
        new = new[~new["pick_id"].isin(set(df["pick_id"].astype(str)))]
        out = pd.concat([df, new], ignore_index=True) if len(df) else new
        msg = f"log: {len(new)} pick(s), slip {rows[0]['slip_id'] if rows else ''}"
        try:
            store.write(PICKS_PATH, out.to_csv(index=False), msg, sha)
            return out
        except ConflictError:
            if attempt == retries:
                raise
            time.sleep(0.5 * (attempt + 1))
    raise AssertionError("unreachable")
