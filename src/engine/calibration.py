"""Post-sim spread and probability calibration (toggles S and I; see
reports/backtest_spread.md).

S) spread inflation: x -> mean + k (x - mean) per sim, stochastically rounded, clipped at
   0. k = sqrt(sum (y - mean)^2 / sum predicted var) per position x market, fitted on prior
   backtest rows and shrunk toward the pooled k.
I) probability calibration: a monotone map of P(over) per position x market, fitted on
   (P(over), hit) pairs at half-point lines rounded median -2.5 .. +2.5 of prior backtest
   rows. Isotonic (PAV on binned p) or beta calibration; shrunk toward the pooled map.

Both act on the player's market total only, so slips still use the shared sims (S keeps
each sim's rank; I is applied to the joint probability's legs only through P(side)).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

OFFSETS = (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)
CAL_MIN_PAIRS = 3000        # pairs before a calibration map departs from identity
CAL_SHRINK_PAIRS = 3000     # group map weight n / (n + this), rest = pooled map
SPREAD_MIN_ROWS = 500       # rows before k departs from 1
SPREAD_SHRINK_ROWS = 300    # group k^2 weight n / (n + this), rest = pooled k^2
N_BINS = 200


# ------------------------------------------------------------------ S: inflation ----
def inflate_samples(samples: np.ndarray, k: float, rng=0) -> np.ndarray:
    """mean + k (x - mean), clipped at 0, stochastically rounded (keeps sim order)."""
    if k == 1.0:
        return samples
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    mu = samples.mean()
    x = np.maximum(mu + k * (samples - mu), 0.0)
    base = np.floor(x)
    return (base + (rng.random(len(x)) < (x - base))).astype(np.int64)


def inflate_pmf(pmf: np.ndarray, k: float) -> np.ndarray:
    """Exact distribution of inflate_samples for a pmf on 0..len-1 (last bin absorbs)."""
    if k == 1.0:
        return pmf
    n = len(pmf)
    support = np.arange(n)
    mu = float(np.sum(support * pmf))
    x = np.clip(mu + k * (support - mu), 0.0, n - 1)
    lo = np.floor(x).astype(int)
    frac = x - lo
    out = np.zeros(n)
    np.add.at(out, lo, pmf * (1 - frac))
    np.add.at(out, np.minimum(lo + 1, n - 1), pmf * frac)
    return out


def spread_k(sse: float, var: float, n: int, k2_pool: float | None = None,
             shrink: float = SPREAD_SHRINK_ROWS) -> float:
    """sqrt(sum (y - mean)^2 / sum var), shrunk toward the pooled k^2 on the ratio scale."""
    if var <= 0 or n == 0:
        return 1.0 if k2_pool is None else float(np.sqrt(k2_pool))
    r = sse / var
    if k2_pool is not None:
        w = n / (n + shrink)
        r = w * r + (1 - w) * k2_pool
    return float(np.sqrt(r))


def fit_spread(rows: pd.DataFrame) -> dict:
    """{(position, market): k, "_pool": k} from rows with mean, var, y, position, market."""
    if len(rows) < SPREAD_MIN_ROWS:
        return {"_pool": 1.0}
    sse = (rows["y"] - rows["mean"]) ** 2
    k2 = float(sse.sum() / rows["var"].sum())
    out = {"_pool": float(np.sqrt(k2))}
    for (pos, mkt), g in rows.assign(sse=sse).groupby(["position", "market"]):
        out[(pos, mkt)] = spread_k(g["sse"].sum(), g["var"].sum(), len(g), k2)
    return out


# ------------------------------------------------------------------ I: calibration ----
def pav(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted pool-adjacent-violators: non-decreasing fit of y (already x-sorted)."""
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi)
        wts.append(wi)
        cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnt.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), cnt.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2))
            wts.append(w1 + w2)
            cnt.append(c1 + c2)
    return np.repeat(vals, cnt)


def _bin(p: np.ndarray, y: np.ndarray, n_bins: int = N_BINS):
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    w = np.bincount(b, minlength=n_bins).astype(float)
    sp = np.bincount(b, weights=p, minlength=n_bins)
    sy = np.bincount(b, weights=y, minlength=n_bins)
    m = w > 0
    return sp[m] / w[m], sy[m] / w[m], w[m]


@dataclass
class IsoMap:
    x: np.ndarray
    y: np.ndarray

    @classmethod
    def fit(cls, p: np.ndarray, hit: np.ndarray) -> "IsoMap":
        xb, yb, wb = _bin(p, hit)
        fy = pav(yb, wb)
        # anchors: P = 0 and P = 1 stay put (no sims beyond the support)
        return cls(np.concatenate([[0.0], xb, [1.0]]), np.concatenate([[0.0], fy, [1.0]]))

    def __call__(self, p):
        return np.interp(p, self.x, self.y)


@dataclass
class BetaMap:
    """Beta calibration (Kull et al. 2017): sigmoid(a ln p - b ln(1-p) + c), a, b >= 0."""
    a: float = 1.0
    b: float = 1.0
    c: float = 0.0

    @classmethod
    def fit(cls, p: np.ndarray, hit: np.ndarray, iters: int = 50) -> "BetaMap":
        xb, yb, wb = _bin(p, hit)
        xb = np.clip(xb, 1e-4, 1 - 1e-4)
        X = np.column_stack([np.log(xb), -np.log1p(-xb), np.ones_like(xb)])
        th = np.array([1.0, 1.0, 0.0])
        for _ in range(iters):
            q = 1 / (1 + np.exp(-X @ th))
            g = X.T @ (wb * (yb - q))
            H = (X * (wb * q * (1 - q))[:, None]).T @ X + 1e-6 * np.eye(3)
            step = np.linalg.solve(H, g)
            th = th + step
            if np.abs(step).max() < 1e-8:
                break
        th[:2] = np.maximum(th[:2], 0.0)
        return cls(*map(float, th))

    def __call__(self, p):
        p = np.asarray(p, dtype=float)
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        z = self.a * np.log(pc) - self.b * np.log1p(-pc) + self.c
        out = 1 / (1 + np.exp(-z))
        return np.where(p <= 0, 0.0, np.where(p >= 1, 1.0, out))


METHODS = {"iso": IsoMap, "beta": BetaMap}


@dataclass
class Calibrator:
    """P(over) -> calibrated P(over), per (position, market) shrunk toward the pool."""
    method: str = "iso"
    pool: object | None = None
    groups: dict = field(default_factory=dict)      # (pos, mkt) -> (map, n pairs)
    n_pairs: int = 0

    @classmethod
    def fit(cls, pairs: pd.DataFrame, method: str = "iso") -> "Calibrator":
        """pairs: columns p, y (0/1), position, market."""
        cal = cls(method=method, n_pairs=len(pairs))
        if len(pairs) < CAL_MIN_PAIRS:
            return cal
        M = METHODS[method]
        cal.pool = M.fit(pairs["p"].to_numpy(float), pairs["y"].to_numpy(float))
        for key, g in pairs.groupby(["position", "market"]):
            if len(g) >= 200:
                cal.groups[key] = (M.fit(g["p"].to_numpy(float), g["y"].to_numpy(float)),
                                   len(g))
        return cal

    def __call__(self, p, position: str, market: str):
        p = np.asarray(p, dtype=float)
        if self.pool is None:
            return p
        out = self.pool(p)
        if (position, market) in self.groups:
            m, n = self.groups[(position, market)]
            w = n / (n + CAL_SHRINK_PAIRS)
            out = w * m(p) + (1 - w) * out
        return np.clip(out, 0.0, 1.0)


def calibrate_pmf(pmf: np.ndarray, cal: Calibrator, position: str, market: str,
                  center: float) -> np.ndarray:
    """Calibrated pmf: S'(j) = cal(P(X > j)) at the half-point thresholds j + 0.5 within
    `center` (the rounded median line) +/- 2.5, the range the map is fitted on. Outside it
    the raw survival is kept, clipped so the survival stays monotone."""
    n = len(pmf)
    surv = np.clip(1.0 - np.cumsum(pmf), 0.0, 1.0)         # P(X > j), j = 0..n-1
    lo = int(max(0, np.floor(center + OFFSETS[0])))
    hi = int(min(n - 1, np.floor(center + OFFSETS[-1])))
    s2 = surv.copy()
    if cal is not None and cal.pool is not None and hi >= lo:
        s2[lo:hi + 1] = cal(surv[lo:hi + 1], position, market)
        s2[lo:hi + 1] = np.minimum.accumulate(s2[lo:hi + 1])
        s2[:lo] = np.maximum(surv[:lo], s2[lo])
        s2[hi + 1:] = np.minimum(surv[hi + 1:], s2[hi])
    s2 = np.minimum.accumulate(s2)
    cdf = 1.0 - s2
    out = np.diff(np.concatenate([[0.0], cdf]))
    out[-1] += 1.0 - cdf[-1]
    return np.clip(out, 0.0, None)


def median_line(kills: np.ndarray) -> float:
    """Rounded median of the samples (the backtest's 'median line')."""
    return float(np.floor(np.median(kills) + 0.5))


def calibrated_prop(kills: np.ndarray, line: float, cal: Calibrator | None, position: str,
                    market: str) -> dict:
    """Calibrated P(over), P(under), P(push) at `line` from sim samples, through the same
    calibrated pmf the backtest scores (calibrate_pmf around the rounded median line)."""
    n = len(kills)
    if cal is None or cal.pool is None:
        return {"p_over": float(np.count_nonzero(kills > line) / n),
                "p_under": float(np.count_nonzero(kills < line) / n),
                "p_push": float(np.count_nonzero(kills == line) / n)}
    pmf = np.bincount(np.asarray(kills, dtype=np.int64)) / n
    pmf = np.concatenate([pmf, [0.0] * 4])
    P = calibrate_pmf(pmf, cal, position, market, median_line(kills))
    ks = np.arange(len(P))
    return {"p_over": float(P[ks > line].sum()), "p_under": float(P[ks < line].sum()),
            "p_push": float(P[ks == line].sum())}


# ------------------------------------------------------------------ pairs from rows ----
def pairs_from_rows(rows: pd.DataFrame, pmf_col: str = "pmf") -> pd.DataFrame:
    """(date, position, market, offset, p, y) at rounded median + OFFSETS (lines > 0)."""
    P = np.stack(rows[pmf_col].to_numpy())
    surv = 1.0 - np.cumsum(P, axis=1)
    line = rows["line"].to_numpy(float)
    y = rows["y"].to_numpy(float)
    parts = []
    for off in OFFSETS:
        hl = line + off
        ok = hl > 0
        j = np.clip(np.floor(hl).astype(int), 0, P.shape[1] - 1)
        p = surv[np.arange(len(P)), j]
        parts.append(pd.DataFrame({
            "date": rows["date"].to_numpy()[ok], "position": rows["position"].to_numpy()[ok],
            "market": rows["market"].to_numpy()[ok], "offset": off, "p": p[ok],
            "y": (y > hl).astype(int)[ok]}))
    return pd.concat(parts, ignore_index=True)


def fit_calibrator_asof(pairs: pd.DataFrame, cutoff, method: str = "iso",
                        lag: pd.Timedelta = pd.Timedelta(days=1)) -> Calibrator:
    """Calibrator from pairs whose series started at least `lag` before cutoff."""
    d = pd.to_datetime(pairs["date"])
    return Calibrator.fit(pairs[d <= pd.Timestamp(cutoff) - lag], method)


def fit_spread_asof(rows: pd.DataFrame, cutoff,
                    lag: pd.Timedelta = pd.Timedelta(days=1)) -> dict:
    d = pd.to_datetime(rows["date"])
    return fit_spread(rows[d <= pd.Timestamp(cutoff) - lag])


# ------------------------------------------------------------------ W: line shrinkage ----
# Displayed P = p_line + lambda (p_cal - p_line). p_line is the line's push-adjusted implied
# probability of a side: (1 - P(push)) / 2 (both sides priced alike). lambda per market is
# fitted so that the top-W_TOP_N edges per slate are calibrated (predicted ~ hit).
W_TOP_N = 5
W_MIN_PICKS = 300           # selected picks before lambda departs from 1
W_SHRINK_PICKS = 150        # market lambda weight n / (n + this), rest = pooled lambda
W_CLIP = (0.2, 1.2)
MARKETS = ("Map 1", "Maps 1-2", "Maps 1-3")


def w_adjust(p_more, p_less, p_push, lam):
    """(more, less) after W. Keeps P(more) + P(less) = 1 - P(push) and the side order."""
    p_line = (1.0 - np.asarray(p_push, dtype=float)) / 2.0
    return (p_line + lam * (np.asarray(p_more, dtype=float) - p_line),
            p_line + lam * (np.asarray(p_less, dtype=float) - p_line))


def lambda_for(lam: dict | None, market: str) -> float:
    if not lam:
        return 1.0
    return float(lam.get(market, lam.get("_pool", 1.0)))


def select_top(rows: pd.DataFrame, lam: dict | None, breakeven: float,
               top_n: int | None = W_TOP_N) -> pd.DataFrame:
    """The top_n W-adjusted edges (edge > 0) per slate. rows: slate, market, p_more, p_less,
    p_push, y, line. Adds side, p_cal (calibrated P of the side), p_line, p_w, edge, hit."""
    lams = rows["market"].map(lambda m: lambda_for(lam, m)).to_numpy(float)
    more, less = w_adjust(rows["p_more"], rows["p_less"], rows["p_push"], lams)
    d = rows.copy()
    d["side"] = np.where(d["p_more"].to_numpy() >= d["p_less"].to_numpy(), "more", "less")
    is_more = d["side"].to_numpy() == "more"
    d["p_cal"] = np.where(is_more, d["p_more"], d["p_less"])
    d["p_line"] = (1.0 - d["p_push"].to_numpy(float)) / 2.0
    d["p_w"] = np.where(is_more, more, less)
    d["edge"] = d["p_w"] - breakeven
    y, line = d["y"].to_numpy(float), d["line"].to_numpy(float)
    d["hit"] = np.where(is_more, y > line, y < line).astype(float)
    d = d[d["edge"] > 0].sort_values("edge", ascending=False)
    if top_n is not None:
        d = d.groupby("slate", sort=False).head(top_n)
    return d


def fit_w(rows: pd.DataFrame, breakeven: float, top_n: int = W_TOP_N,
          iters: int = 20, min_picks: int = W_MIN_PICKS) -> dict:
    """{market: lambda, "_pool": lambda, "_n": picks}: lambda = sum(hit - p_line) /
    sum(p_cal - p_line) over the selected picks, re-selected with the new lambda until it
    settles (selection depends on lambda). Shrunk to the pooled lambda, clipped."""
    lam = {"_pool": 1.0}
    if len(rows) == 0:
        return {"_pool": 1.0, "_n": 0}
    for _ in range(iters):
        sel = select_top(rows, lam, breakeven, top_n)
        if len(sel) < min_picks:
            return {"_pool": 1.0, "_n": len(sel)}

        def ratio(g):
            den = float((g["p_cal"] - g["p_line"]).sum())
            return float((g["hit"] - g["p_line"]).sum()) / den if den > 0 else 1.0

        pool = float(np.clip(ratio(sel), *W_CLIP))
        new = {"_pool": pool}
        for m, g in sel.groupby("market"):
            w = len(g) / (len(g) + W_SHRINK_PICKS)
            new[m] = float(np.clip(w * ratio(g) + (1 - w) * pool, *W_CLIP))
        done = all(abs(new.get(k, pool) - lam.get(k, lam["_pool"])) < 1e-4
                   for k in set(new) | set(lam))
        lam = new
        if done:
            break
    lam["_n"] = len(select_top(rows, lam, breakeven, top_n))
    return lam


def fit_w_asof(rows: pd.DataFrame, cutoff, breakeven: float,
               lag: pd.Timedelta = pd.Timedelta(days=1)) -> dict:
    d = pd.to_datetime(rows["date"])
    return fit_w(rows[d <= pd.Timestamp(cutoff) - lag], breakeven)


# ------------------------------------------------------------------ G: goblin lines ----
# P(more) at lines 1-4 kills under a PP-like line (reports/goblin_fit.md). Walk-forward
# since 2025-06, the S pmf is already sharp there for top / jng / mid / bot: picks at the
# highest goblin line with P >= 73.3% predicted 79.1% and hit 80.0% (n=16,032). Supports
# run hot (81.1% predicted, 79.2% hit; 92% -> 84%), so supports get their own beta map.
GOBLIN_DEPTHS = (1, 2, 3, 4)
GOBLIN_MAP_POSITIONS = ("sup",)
GOBLIN_MIN_PAIRS = 2000


def goblin_depth(std_line: float, line: float) -> int:
    """Whole kills a goblin line sits under the standard line (4.0 -> 3.5 is depth 1)."""
    return int(min(max(np.ceil(float(std_line) - float(line) - 1e-9), 1), GOBLIN_DEPTHS[-1]))


def goblin_pairs(base: pd.DataFrame, P: np.ndarray, line_col: str = "proxy_mean",
                 extra: tuple = ()) -> pd.DataFrame:
    """(date, position, market, depth, p, y): P(X > line - d) of the pmfs and the outcome,
    d = 1..4 under each row's PP-like line (rows without one, or lines <= 0, dropped).
    `extra`: more columns of base to carry (reports)."""
    surv = 1.0 - np.cumsum(P, axis=1)
    std = base[line_col].to_numpy(float)
    y = base["y"].to_numpy(float)
    parts = []
    for d in GOBLIN_DEPTHS:
        line = std - d
        ok = np.isfinite(line) & (line > 0)
        j = np.clip(np.floor(np.nan_to_num(line)).astype(int), 0, P.shape[1] - 1)
        parts.append(pd.DataFrame({
            "date": base["date"].to_numpy()[ok], "position": base["position"].to_numpy()[ok],
            "market": base["market"].to_numpy()[ok], "depth": np.int8(d),
            "p": surv[np.arange(len(P)), j][ok].astype("float32"),
            "y": (y > line)[ok].astype("int8"),
            **{c: base[c].to_numpy()[ok] for c in extra}}))
    return pd.concat(parts, ignore_index=True)


@dataclass
class GoblinCal:
    """position -> BetaMap for the positions whose goblin P(more) is miscalibrated; every
    other position keeps the model's P."""
    maps: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, pairs: pd.DataFrame) -> "GoblinCal":
        out = cls()
        for pos in GOBLIN_MAP_POSITIONS:
            g = pairs[pairs["position"] == pos]
            if len(g) >= GOBLIN_MIN_PAIRS:
                out.maps[pos] = BetaMap.fit(g["p"].to_numpy(float), g["y"].to_numpy(float))
        return out

    def __call__(self, p, position: str):
        p = np.asarray(p, dtype=float)
        m = self.maps.get(position)
        return np.clip(m(p), 0.0, 1.0) if m is not None else p


def fit_goblin_asof(pairs: pd.DataFrame, cutoff,
                    lag: pd.Timedelta = pd.Timedelta(days=1)) -> GoblinCal:
    d = pd.to_datetime(pairs["date"])
    return GoblinCal.fit(pairs[d <= pd.Timestamp(cutoff) - lag])
