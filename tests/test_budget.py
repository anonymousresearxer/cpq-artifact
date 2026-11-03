from fractions import Fraction

import numpy as np
import pytest

from cpq.budget import n_min, rank


def test_rank_ties_by_noise():
    yhat = np.array([1.0, 1.0, 0.0, 1.0])
    xi = np.array([0.3, 0.1, 0.9, 0.2])
    assert rank(yhat, xi, np.arange(4)).tolist() == [2, 1, 3, 0]
    assert rank(yhat, xi, np.array([3, 1, 0])).tolist() == [1, 3, 0]


def scan(total, count, q, fdp):
    x = Fraction(total) / (Fraction(str(q)) * count)
    x = x * (1 - Fraction(str(q))) if fdp else x
    return next((n for n in range(18, total // 2 + 1, 2) if Fraction(n, 2) + 1 >= x), None)


@pytest.mark.parametrize("fdp", [False, True])
@pytest.mark.parametrize("total", [432, 1000, 2560, 13485])
@pytest.mark.parametrize("share", [0.05, 0.25, 0.5, 0.75])
def test_n_min_matches_scan(total, share, fdp):
    count = max(int(share * total), 1)
    assert n_min(total, count, 0.1, 0.1, fdp) == scan(total, count, 0.1, fdp)


def test_n_min_exact_equality():
    assert n_min(1000, 250, 0.1, 0.1) == 78
    assert n_min(100, 100, 0.1, 0.1) == 18
    assert n_min(100, 5, 0.1, 0.1) is None
