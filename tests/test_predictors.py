import numpy as np

from cpq.predictors import MAX_PAIRS, fit_grid


def toy(n=120, f=12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n, f)).astype(np.uint8)
    y = 5 + X @ rng.uniform(-1, 3, f) + 2.5 * X[:, 0] * X[:, 1] + rng.normal(0, 0.05, n)
    return X, y


def test_grid_weights_are_non_negative_integers():
    X, y = toy()
    g = fit_grid(X[:60], y[:60])
    c0 = g.floor(y.min())
    allx = np.array(np.meshgrid(*[[0, 1]] * 12)).reshape(12, -1).T.astype(np.uint8)
    w = g.weights(allx, c0)
    assert w.min() >= 0
    assert np.allclose(g.predict(allx), g.delta * (w + c0))
    assert c0 * g.delta <= y.min()


def test_at_most_15_pairs_and_finds_interaction():
    X, y = toy(f=30)
    g = fit_grid(X[:80], y[:80])
    assert len(g.pairs) <= MAX_PAIRS
    assert (0, 1) in {tuple(p) for p in g.pairs}


def test_delta_depends_on_train_only():
    X, y = toy()
    a = fit_grid(X[:60], y[:60])
    y2 = y.copy()
    y2[60:] += 100
    b = fit_grid(X[:60], y2[:60])
    assert a.delta == b.delta and np.array_equal(a.kp, b.kp)


def test_lazy_pairs_match_dense():
    X, y = toy()
    g = fit_grid(X[:60], y[:60])
    dense = g.delta * (
        g.k0
        + X @ g.ko
        + sum(k * X[:, i] * X[:, j] for (i, j), k in zip(g.pairs, g.kp, strict=True))
    )
    assert np.allclose(g.predict(X), dense)
