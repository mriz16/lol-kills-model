"""In-app OE upload: parser (CSV / zip), merge by gameid, and the `data` branch store."""
import base64
import hashlib
import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from src.data import databranch
from src.data.oe import REQUIRED_COLUMNS, process
from src.data.upload import UploadError, merge_frames, process_upload, read_oe_csv

FIXTURE = Path(__file__).parent / "fixtures" / "oe_fixture.csv"


@pytest.fixture(scope="module")
def csv_bytes():
    return FIXTURE.read_bytes()


def _zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def _same(a: pd.DataFrame, b: pd.DataFrame):
    assert list(a.columns) == list(b.columns)
    assert len(a) == len(b)
    for c in a.columns:
        assert a[c].astype(str).reset_index(drop=True).equals(
            b[c].astype(str).reset_index(drop=True)), c


# ------------------------------------------------------------------------- parser --
def test_parse_csv_keeps_only_needed_columns_with_dtypes(csv_bytes, raw):
    df = read_oe_csv(csv_bytes, "2025.csv")
    assert list(df.columns) == [c for c in raw.columns if c in REQUIRED_COLUMNS]
    assert set(df.columns) == set(REQUIRED_COLUMNS)    # url, champion, ... not loaded
    assert len(df) == len(raw)
    assert isinstance(df["league"].dtype, pd.CategoricalDtype)
    assert str(df["kills"].dtype) == "float32"
    # same processed result as the old full read
    for a, b in zip(process(df), process(raw)):
        _same(a, b)


def test_parse_zip_and_file_object(csv_bytes, raw):
    z = _zip({"__MACOSX/._x.csv": b"junk", "notes.txt": b"hi",
              "2025_LoL_esports_match_data_from_OraclesElixir.csv": csv_bytes})
    assert len(read_oe_csv(z, "oe.zip")) == len(raw)
    assert len(read_oe_csv(io.BytesIO(z), "upload")) == len(raw)   # sniffed by magic
    assert len(read_oe_csv(io.BytesIO(csv_bytes), "x.csv")) == len(raw)


def test_parse_errors(csv_bytes, raw):
    with pytest.raises(UploadError, match="no .csv"):
        read_oe_csv(_zip({"a.txt": b"x"}), "a.zip")
    with pytest.raises(UploadError, match="not a valid .zip"):
        read_oe_csv(b"definitely not a zip", "a.zip")
    with pytest.raises(UploadError, match="web page"):
        read_oe_csv(b"<!DOCTYPE html><html>quota</html>", "x.csv")
    with pytest.raises(UploadError, match="missing columns.*kills"):
        read_oe_csv(_csv(raw.drop(columns=["kills"])), "x.csv")
    with pytest.raises(UploadError, match="0 rows"):
        read_oe_csv(_csv(raw.head(0)), "x.csv")


def test_process_upload_rejects_out_of_scope_file(raw):
    other = raw.assign(league="KeSPA Cup")
    with pytest.raises(UploadError, match="0 complete in-scope games"):
        process_upload(_csv(other), "x.csv", None, None)


# -------------------------------------------------------------------------- merge --
def test_merge_split_upload_equals_full_processing(raw, processed):
    dates = pd.to_datetime(raw["date"])
    cut = dates.sort_values().iloc[len(dates) // 2]
    old = process(raw[dates < cut])
    # the second file overlaps the first by a few days, like a re-downloaded yearly file
    new = process(raw[dates >= cut - pd.Timedelta(days=5)])
    players, teams = merge_frames(*old, *new)
    _same(players, processed[0])
    _same(teams, processed[1])


def test_merge_replaces_by_gameid_and_never_drops_old_years(raw, processed):
    old_p, old_t = processed
    # "2026 file": a few games already stored (one with a corrected kill count) + new games
    g_old = sorted(old_t["gameid"].astype(str).unique())[:3]
    upd = raw[raw["gameid"].isin(g_old)].copy()
    fix = upd["gameid"] == g_old[0]
    pl0 = fix & (upd["position"] == "top") & (upd["side"] == "Blue")
    upd.loc[pl0, "kills"] += 1
    upd.loc[fix & (upd["side"] == "Blue"), "teamkills"] += 1     # team + player rows
    g_new = sorted(old_t["gameid"].astype(str).unique())[-4:]
    new26 = raw[raw["gameid"].isin(g_new)].copy()
    new26["gameid"] = new26["gameid"] + "_26"
    new26["date"] = (pd.to_datetime(new26["date"]) + pd.DateOffset(years=1)).astype(str)
    new26["year"] = 2026
    up = pd.concat([upd, new26])
    res = process_upload(_csv(up), "2026.csv", old_p, old_t)
    p, t = res.players, res.teams

    assert res.games_added == 4 and res.games_updated == 3
    assert res.player_rows_added == 40
    assert set(old_t["gameid"]) <= set(t["gameid"])               # nothing dropped
    assert t["gameid"].nunique() == old_t["gameid"].nunique() + 4
    years = pd.to_datetime(t["date"]).dt.year.value_counts()
    assert years[2025] == len(old_t) and years[2026] == 8          # 2025 intact
    # the corrected game replaced the stored one
    top = p[(p["gameid"] == g_old[0]) & (p["position"] == "top") & (p["side"] == "Blue")]
    was = old_p[(old_p["gameid"] == g_old[0]) & (old_p["position"] == "top")
                & (old_p["side"] == "Blue")]
    assert int(top["kills"].iloc[0]) == int(was["kills"].iloc[0]) + 1
    assert len(p[p["gameid"] == g_old[0]]) == 10                    # replaced, not doubled
    # untouched games carry the same stats; series / patch ordinals rebuilt consistently
    keep = ~old_p["gameid"].isin(g_old)
    m = old_p[keep].merge(p, on=["gameid", "side", "position"], suffixes=("", "_n"))
    assert (m["kills"] == m["kills_n"]).all() and len(m) == keep.sum()
    assert t.groupby("series_id", observed=True)["date"].apply(
        lambda d: d.max() - d.min()).max() <= pd.Timedelta(days=1)
    # summary: per-league ranges and the new last date
    r = res.ranges.set_index("league")
    lg = str(new26["league"].iloc[0])
    assert r.loc[lg, "added"] >= 1
    assert r.loc[lg, "last"] > r.loc[lg, "before_last"]


def test_upload_of_stored_data_adds_nothing(csv_bytes, processed):
    res = process_upload(csv_bytes, "again.csv", *processed)
    assert res.games_added == 0 and res.games_updated == processed[1]["gameid"].nunique()
    assert any("No new games" in w for w in res.warnings)
    _same(res.teams, processed[1])


# ---------------------------------------------------------------------- data branch --
class _Resp:
    def __init__(self, code, payload=None, content=b""):
        self.status_code, self._p, self.content = code, payload or {}, content
        self.text = str(self._p)

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeGitData:
    """In-memory Git Data + contents API for one repo (o/r)."""
    base = "https://api.github.com/repos/o/r/"

    def __init__(self):
        self.headers, self.blobs, self.trees, self.commits, self.refs = {}, {}, {}, {}, {}
        self.race = False

    @staticmethod
    def _sha(*parts) -> str:
        return hashlib.sha1(repr(parts).encode()).hexdigest()

    def _files(self, commit: str) -> dict:
        return self.trees[self.commits[commit]["tree"]]

    def get(self, url, params=None, timeout=None, headers=None):
        path = url.removeprefix(self.base)
        if path.startswith("git/ref/heads/"):
            b = path.removeprefix("git/ref/heads/")
            return _Resp(200, {"object": {"sha": self.refs[b]}}) if b in self.refs \
                else _Resp(404)
        if path.startswith("git/commits/"):
            sha = path.removeprefix("git/commits/")
            return _Resp(200, {"sha": sha, "tree": {"sha": self.commits[sha]["tree"]}})
        if path.startswith("contents/"):
            ref = params["ref"]
            commit = self.refs.get(ref, ref)
            if commit not in self.commits:
                return _Resp(404)
            f = path.removeprefix("contents/")
            sha = self._files(commit).get(f)
            assert headers["Accept"] == "application/vnd.github.raw+json"
            return _Resp(200, content=self.blobs[sha]) if sha else _Resp(404)
        return _Resp(404)

    def post(self, url, json=None, timeout=None):
        path = url.removeprefix(self.base)
        if path == "git/blobs":
            data = base64.b64decode(json["content"])
            sha = self._sha("blob", data)
            self.blobs[sha] = data
            return _Resp(201, {"sha": sha})
        if path == "git/trees":
            files = dict(self.trees[json["base_tree"]]) if "base_tree" in json else {}
            files.update({e["path"]: e["sha"] for e in json["tree"]})
            sha = self._sha("tree", sorted(files.items()))
            self.trees[sha] = files
            return _Resp(201, {"sha": sha})
        if path == "git/commits":
            sha = self._sha("commit", json["tree"], json["parents"], json["message"])
            self.commits[sha] = json
            return _Resp(201, {"sha": sha})
        if path == "git/refs":
            b = json["ref"].removeprefix("refs/heads/")
            if b in self.refs:
                return _Resp(422, {"message": "Reference already exists"})
            self.refs[b] = json["sha"]
            return _Resp(201)
        return _Resp(404)

    def patch(self, url, json=None, timeout=None):
        b = url.removeprefix(self.base + "git/refs/heads/")
        if self.race or self.refs[b] not in self.commits[json["sha"]]["parents"]:
            return _Resp(422, {"message": "Update is not a fast forward"})
        self.refs[b] = json["sha"]
        return _Resp(200)


def test_data_branch_save_and_load_github(processed):
    fake = FakeGitData()
    br = databranch.GitHubDataBranch("tok", "o/r", session=fake)
    assert fake.headers["Authorization"] == "Bearer tok"
    assert br.head() is None and databranch.load_frames(br) is None
    p, t = processed
    sha1 = databranch.save_frames(br, p, t, "# audit", "2025.csv", expected_head=None)
    assert br.head() == sha1 and fake.commits[sha1]["parents"] == []   # orphan branch
    assert set(fake._files(sha1)) == {databranch.PLAYERS_PATH, databranch.TEAMS_PATH,
                                      databranch.META_PATH, databranch.AUDIT_PATH}
    lp, lt = databranch.load_frames(br)
    _same(lp, p)
    _same(lt, t)
    assert lt["league"].dtype == t["league"].dtype and lp["kills"].dtype == p["kills"].dtype

    # second upload: stacked on the first commit, readable at either commit
    t2 = t[t["date"] > t["date"].min()]
    sha2 = databranch.save_frames(br, p, t2, "# audit 2", "2026.csv", expected_head=sha1)
    assert fake.commits[sha2]["parents"] == [sha1] and br.head() == sha2
    assert len(databranch.load_frames(br)[1]) == len(t2)
    assert len(databranch.load_frames(br, ref=sha1)[1]) == len(t)
    assert br.read_bytes(databranch.AUDIT_PATH) == b"# audit 2"

    # the branch moved since this session read it -> conflict, nothing written
    with pytest.raises(databranch.ConflictError):
        databranch.save_frames(br, p, t, "x", "x.csv", expected_head=sha1)
    fake.race = True               # moved between head() and the ref update
    with pytest.raises(databranch.ConflictError):
        databranch.save_frames(br, p, t, "x", "x.csv", expected_head=sha2)
    assert br.head() == sha2


def test_data_branch_local(tmp_path, processed):
    br = databranch.LocalDataBranch(tmp_path)
    assert br.head() is None and databranch.load_frames(br) is None
    h = databranch.save_frames(br, *processed, "# audit", "2025.csv")
    assert br.head() == h and (tmp_path / "data" / "meta.json").exists()
    lp, lt = databranch.load_frames(br)
    _same(lt, processed[1])


def test_refresh_script_merges_zip(tmp_path, csv_bytes):
    from scripts.refresh_data import main
    z = tmp_path / "oe.zip"
    z.write_bytes(_zip({"oe.csv": csv_bytes}))
    assert main(["--csv", str(z), "--out", str(tmp_path), "--audit",
                 str(tmp_path / "a.md")]) == 0
    n = len(pd.read_parquet(tmp_path / "teams.parquet"))
    assert main(["--csv", str(FIXTURE), "--out", str(tmp_path), "--audit",
                 str(tmp_path / "a.md")]) == 0                     # merge: same games
    assert len(pd.read_parquet(tmp_path / "teams.parquet")) == n
