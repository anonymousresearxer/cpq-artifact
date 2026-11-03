import numpy as np
import pytest
from mapie.regression import SplitConformalRegressor
from sklearn.linear_model import LinearRegression

from cpq.conformal import n_missed, pac_missed, qhat


def test_matches_mapie():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 3))
    y = X @ [1.0, -2.0, 0.5] + rng.normal(size=300)
    est = LinearRegression().fit(X[:100], y[:100])
    cal, new = slice(100, 160), X[200:]
    scr = SplitConformalRegressor(est, confidence_level=0.9, prefit=True)
    scr.conformalize(X[cal], y[cal])
    _, ints = scr.predict_interval(new)
    q = qhat(np.abs(y[cal] - est.predict(X[cal])), 0.1)
    pred = est.predict(new)
    assert np.allclose(ints[:, 0, 0], pred - q)
    assert np.allclose(ints[:, 1, 0], pred + q)


def test_qhat_small_m_is_infinite():
    assert qhat(np.arange(5.0), 0.1) == np.inf
    assert qhat(np.arange(9.0), 0.1) == 8.0


@pytest.mark.parametrize("m,want", [(25, 0.18), (50, 0.17), (100, 0.15)])
def test_pac_bound_values(m, want):
    n_u = 200_000
    assert round(pac_missed(m, n_u, 0.1, 0.05) / n_u, 2) == want


def test_pac_bound_holds_by_simulation():
    rng = np.random.default_rng(7)
    m, n_u, alpha, delta = 50, 400, 0.1, 0.05
    bound = pac_missed(m, n_u, alpha, delta)
    reps, bad = 2000, 0
    for _ in range(reps):
        r = rng.normal(size=m + n_u)
        q = qhat(np.abs(r[:m]), alpha)
        bad += (np.abs(r[m:]) > q).sum() > bound
    assert bad / reps <= delta + 3 * np.sqrt(delta * (1 - delta) / reps)
    assert n_missed(m, alpha) == 5
