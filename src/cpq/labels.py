from dataclasses import dataclass

import numpy as np

from .conformal import pac_missed

VIOLATES, POSSIBLE, CERTAIN = 0, 1, 2


def measured_labels(y: np.ndarray, t: float, eps: float = 0.0) -> np.ndarray:
    """Noise-band labels of measured values: satisfies, undecided or violates."""
    band = eps * np.abs(y)
    return np.where(y + band <= t, CERTAIN, np.where(y - band > t, VIOLATES, POSSIBLE))


def label(yhat, y, meas, sel, qh, t, eps=0.0) -> np.ndarray:
    """Provenance label of every configuration; y is only read where meas is set."""
    lab = np.where(sel | (yhat - qh <= t), POSSIBLE, VIOLATES)
    lab = np.where(sel, CERTAIN, lab)
    lab[meas] = measured_labels(y[meas], t, eps)
    return lab


def filter_(lab: np.ndarray, level: int = CERTAIN) -> np.ndarray:
    return np.flatnonzero(lab >= level)


@dataclass
class Count:
    certain: int
    possible: int
    low: int
    high: int


def count(yhat, y, meas, sel, qh, t, m, alpha, delta, eps=0.0) -> Count:
    """Certain and possible counts and the P4 interval, valid with probability 1 - delta."""
    lab = label(yhat, y, meas, sel, qh, t, eps)
    u = ~meas
    n_u = int(u.sum())
    xmax = pac_missed(m, n_u, alpha, delta)
    # the P4 bounds on U, intersected with the trivial range [0, |U|]
    lo = int((lab[meas] == CERTAIN).sum()) + max(int((yhat[u] + qh <= t).sum()) - xmax, 0)
    hi = int((lab[meas] >= POSSIBLE).sum()) + min(int((yhat[u] - qh <= t).sum()) + xmax, n_u)
    return Count(int((lab == CERTAIN).sum()), int((lab >= POSSIBLE).sum()), lo, hi)


def hybrid(total, pred_m, meas_m):
    """Exact correction over M: measured contributions replace the predicted ones."""
    return total - np.sum(pred_m, axis=0) + np.sum(meas_m, axis=0)


def optimise(yhat, y, meas, qh) -> tuple[float, int]:
    """Screen then measure: best measured value after measuring S = {v : lo(v) <= min hi}."""
    u = ~meas
    best = y[meas].min()
    bound = min(best, (yhat[u] + qh).min()) if u.any() else best
    cand = u & (yhat - qh <= bound)
    found = min(best, y[cand].min()) if cand.any() else best
    return float(found), int(cand.sum())
