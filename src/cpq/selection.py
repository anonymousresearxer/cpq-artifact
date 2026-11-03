import numpy as np


def top(cal_pred, test_pred, t: float, gamma: float, mirror: bool = False) -> float:
    """Clipping value M: above every finite score of Cal and U, from predictions only."""
    pred = np.concatenate([cal_pred, test_pred])
    return float(((t - pred) if mirror else (pred - t)).max() + gamma)


def scores(cal_pred, cal_y, test_pred, t: float, big: float, mirror: bool = False):
    """Clipped scores oriented to y <= t, or to y > t for the mirror selection."""
    if mirror:
        return np.where(cal_y > t, big, t - cal_pred), t - test_pred
    return np.where(cal_y <= t, big, cal_pred - t), test_pred - t


def pvalues(s_cal: np.ndarray, s_test: np.ndarray) -> np.ndarray:
    """Deterministic conformal p-values (Jin and Candes 2023, Theorem 6)."""
    return (1 + np.searchsorted(np.sort(s_cal), s_test, side="left")) / (len(s_cal) + 1)


def bh(p: np.ndarray, q: float) -> np.ndarray:
    """Boolean mask of the Benjamini-Hochberg rejections at level q."""
    n = len(p)
    order = np.argsort(p)
    ok = np.flatnonzero(p[order] <= q * np.arange(1, n + 1) / n)
    sel = np.zeros(n, dtype=bool)
    if len(ok):
        sel[order[: ok[-1] + 1]] = True
    return sel


def select(cal_pred, cal_y, test_pred, t, q, xi_cal, xi_test, gamma, mirror: bool = False):
    """BH over U; xi < gamma/2 is per-configuration noise that only breaks ties."""
    big = top(cal_pred, test_pred, t, gamma, mirror)
    s_cal, s_test = scores(cal_pred, cal_y, test_pred, t, big, mirror)
    return bh(pvalues(s_cal + xi_cal, s_test + xi_test), q)
