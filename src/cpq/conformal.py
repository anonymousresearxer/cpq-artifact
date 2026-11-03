from fractions import Fraction

import numpy as np
from scipy.stats import betabinom


def qhat(res: np.ndarray, alpha: float) -> float:
    """Split-conformal quantile of absolute residuals."""
    m = len(res)
    k = m + 1 - n_missed(m, alpha)
    if k < 1:
        return -np.inf
    return np.inf if k > m else float(np.sort(res)[k - 1])


def n_missed(m: int, alpha: float) -> int:
    # l = floor((m + 1) alpha) on the decimal alpha, so that ceil((m + 1)(1 - alpha)) = m + 1 - l
    return int((m + 1) * Fraction(str(alpha)) // 1)


def pac_missed(m: int, n_u: int, alpha: float, delta: float) -> int:
    """Smallest x with P(X <= x) >= 1 - delta, X ~ BetaBinomial(n_u, l, m + 1 - l)."""
    k = n_missed(m, alpha)
    # degenerate laws: l = 0 gives qhat = inf and X = 0, l > m gives qhat = -inf and X = n_u
    if k == 0 or n_u == 0:
        return 0
    if k > m:
        return n_u
    return max(int(betabinom.ppf(1 - delta, n_u, k, m + 1 - k)), 0)
