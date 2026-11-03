import numpy as np

Z = 1.959964


def wilson(k: int, n: int) -> tuple[float, float]:
    """Wilson score interval of a binomial proportion, 95 %."""
    p = k / n
    d = 1 + Z**2 / n
    c = (p + Z**2 / (2 * n)) / d
    h = Z * np.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / d
    return float(max(c - h, 0.0)), float(min(c + h, 1.0))


def ppi_mean(f_unl: np.ndarray, f_lab: np.ndarray, y_lab: np.ndarray) -> tuple[float, float, float]:
    """Prediction-powered mean and 95 % CLT interval (Angelopoulos et al. 2023)."""
    rect = y_lab - f_lab
    est = f_unl.mean() + rect.mean()
    se = np.sqrt(f_unl.var(ddof=1) / len(f_unl) + rect.var(ddof=1) / len(rect))
    return float(est), float(max(est - Z * se, 0.0)), float(min(est + Z * se, 1.0))


def returned(ret: np.ndarray, truth: np.ndarray) -> tuple[int, float, float]:
    """Size, precision and recall of a returned set against the configurations that comply."""
    tp = int((ret & truth).sum())
    n = int(ret.sum())
    return n, (tp / n if n else np.nan), (tp / truth.sum() if truth.any() else np.nan)


def interval_error(truth: float, lo: float, hi: float) -> float:
    return float(max(lo - truth, truth - hi, 0.0))
