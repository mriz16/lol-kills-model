"""Scoring rules and calibration for discrete (count) predictive samples."""
from __future__ import annotations

import numpy as np
import pandas as pd


def pmf_from_samples(samples: np.ndarray, upto: int) -> np.ndarray:
    counts = np.bincount(samples.astype(np.int64), minlength=upto + 1)[: upto + 1]
    return counts / len(samples)


def crps_discrete(samples: np.ndarray, y: int) -> float:
    """CRPS for integer outcomes: sum_k (F(k) - 1{y <= k})^2."""
    upto = int(max(samples.max(), y)) + 1
    F = np.cumsum(pmf_from_samples(samples, upto))
    H = (np.arange(upto + 1) >= y).astype(float)
    return float(np.sum((F - H) ** 2))


def log_score(samples: np.ndarray, y: int) -> float:
    """Negative log predictive probability of y (add-half smoothing). Lower is better."""
    upto = int(max(samples.max(), y)) + 1
    counts = np.bincount(samples.astype(np.int64), minlength=upto + 1)
    p = (counts[y] + 0.5) / (len(samples) + 0.5 * (upto + 1))
    return float(-np.log(p))


def poisson_scores(mu: float, y: int) -> tuple[float, float]:
    """(CRPS, log score) for a Poisson(mu) forecast (naive baseline)."""
    from scipy.stats import poisson
    mu = max(mu, 1e-6)
    upto = int(max(y, poisson.ppf(0.99999, mu))) + 1
    ks = np.arange(upto + 1)
    F = poisson.cdf(ks, mu)
    crps = float(np.sum((F - (ks >= y)) ** 2))
    return crps, float(-poisson.logpmf(y, mu))


def calibration_table(p: pd.Series, outcome: pd.Series, width: float = 0.05) -> pd.DataFrame:
    edges = np.round(np.arange(0, 1 + width, width), 4)
    b = pd.cut(p, edges, include_lowest=True, right=False)
    df = pd.DataFrame({"bucket": b, "p": p, "y": outcome.astype(float)})
    g = df.groupby("bucket", observed=True).agg(n=("y", "size"), pred=("p", "mean"),
                                                obs=("y", "mean"))
    g["diff"] = g["obs"] - g["pred"]
    g["se"] = np.sqrt(g["obs"] * (1 - g["obs"]) / g["n"])
    return g.reset_index()


def ece(p: pd.Series, outcome: pd.Series, width: float = 0.05) -> float:
    t = calibration_table(p, outcome, width)
    return float(np.sum(t["n"] * (t["obs"] - t["pred"]).abs()) / t["n"].sum())
