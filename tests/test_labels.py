import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from cpq.labels import CERTAIN, POSSIBLE, VIOLATES, count, hybrid, label, measured_labels, optimise


def test_noise_band_labels():
    y = np.array([9.0, 10.0, 11.0, 12.0])
    assert measured_labels(y, 10.0, 0.1).tolist() == [CERTAIN, POSSIBLE, POSSIBLE, VIOLATES]


def test_certain_subset_of_possible():
    rng = np.random.default_rng(0)
    yhat = rng.normal(10, 2, 500)
    y = yhat + rng.normal(0, 1, 500)
    meas = rng.random(500) < 0.2
    sel = ~meas & (yhat < 8)
    lab = label(yhat, y, meas, sel, 1.0, 10.0)
    c = count(yhat, y, meas, sel, 1.0, 10.0, m=50, alpha=0.1, delta=0.05)
    assert c.certain == (lab == CERTAIN).sum() <= c.possible == (lab >= POSSIBLE).sum()
    assert c.low <= c.certain + c.possible


@given(st.lists(st.tuples(st.integers(0, 12), st.integers(0, 12)), min_size=1, max_size=30))
def test_hybrid_equals_recompute(rows):
    D = 10
    pred = np.array([min(a, D + 1) for a, _ in rows])
    meas = np.array([min(b, D + 1) for _, b in rows])
    hist = lambda v: np.bincount(v, minlength=D + 2)  # noqa: E731
    onehot = lambda v: np.eye(D + 2, dtype=np.int64)[v]  # noqa: E731
    in_m = np.arange(len(rows)) % 2 == 0
    total = hist(pred)
    got = hybrid(total, onehot(pred[in_m]), onehot(meas[in_m]))
    swapped = np.where(in_m, meas, pred)
    assert got.tolist() == hist(swapped).tolist()
    ind = lambda v: (v <= 5).astype(np.int64)  # noqa: E731
    assert hybrid(ind(pred).sum(), ind(pred[in_m]), ind(meas[in_m])) == ind(swapped).sum()


@given(st.integers(0, 10_000))
def test_optimum_contained_under_coverage(seed):
    rng = np.random.default_rng(seed)
    n = 200
    y = rng.gamma(2.0, 3.0, n)
    q = 1.5
    yhat = y + rng.uniform(-q, q, n)
    meas = rng.random(n) < 0.1
    meas[0] = True
    found, extra = optimise(yhat, y, meas, q)
    assert found == y.min()
    assert extra <= (~meas).sum()
