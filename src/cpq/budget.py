import math
from fractions import Fraction

import numpy as np


def rank(yhat: np.ndarray, xi: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Indices u by increasing yhat, ties broken by increasing xi."""
    return u[np.lexsort((xi[u], yhat[u]))]


def n_min(total: int, count: int, q: float, alpha: float, fdp: bool = False) -> int | None:
    """Smallest even n with n / 2 + 1 >= |V| / (q T), times 1 - q if fdp; None above |V| / 2."""
    q, alpha = Fraction(str(q)), Fraction(str(alpha))
    x = Fraction(total) / (q * count)
    if fdp:
        x *= 1 - q
    n = 2 * max(math.ceil(x - 1), math.ceil(1 / alpha) - 1)
    return n if 2 * n <= total else None
