import math

import numpy as np

from cpq.conformal import qhat
from cpq.selection import bh, pvalues, scores, select, top


def world(rng, n=6000):
    X = rng.integers(0, 2, size=(n, 6))
    yhat = 10 + X @ [1.0, 2.0, 0.5, 1.5, 0.0, 1.0]
    # the interaction hits yhat in {11, 11.5}; yhat in {10, 10.5} stays clearly below t
    hidden = 6.0 * ((X[:, 0] == 1) & (X[:, 1] == 0) & (X[:, 3] == 0) & (X[:, 5] == 0))
    return yhat + hidden + rng.normal(0, 0.5, n), yhat


def run(seed, q=0.1, m=200, t=13.0):
    rng = np.random.default_rng(seed)
    y, yhat = world(rng)
    perm = rng.permutation(len(y))
    cal, u = perm[:m], perm[m:]
    xi = 0.25 * rng.random(len(y))
    sel = select(yhat[cal], y[cal], yhat[u], t, q, xi[cal], xi[u], 0.5)
    qh = qhat(np.abs(y[cal] - yhat[cal]), 0.1)
    return sel, yhat[u] + qh <= t, y[u] <= t


def fdp(sel, truth):
    return 0.0 if not sel.any() else (~truth[sel]).mean()


def test_fdr_below_q():
    res = [run(s) for s in range(150)]
    fdr = np.mean([fdp(s, t) for s, _, t in res])
    assert fdr <= 0.1 + 0.02
    assert np.mean([s.sum() for s, _, _ in res]) > 0


def test_unselected_intervals_miss_precision():
    res = [run(s) for s in range(60)]
    assert np.mean([fdp(n, t) for _, n, t in res]) > 0.1 + 0.05


def test_pvalues_super_uniform_on_nulls():
    rng = np.random.default_rng(3)
    t, hits = 0.0, {0.1: 0, 0.25: 0, 0.5: 0}
    reps, k = 600, 30
    for _ in range(reps):
        yhat = rng.normal(size=60 + k)
        y = yhat + rng.normal(size=60 + k)
        big = top(yhat[:60], yhat[60:], t, 0.1)
        s_cal, s_test = scores(yhat[:60], y[:60], yhat[60:], t, big)
        p = pvalues(s_cal, s_test)
        for a in hits:
            hits[a] += ((p <= a) & (y[60:] > t)).sum()
    for a, h in hits.items():
        assert h / (reps * k) <= a + 0.01


def test_bh_known_case():
    p = np.array([0.001, 0.02, 0.03, 0.5, 0.9])
    assert bh(p, 0.1).tolist() == [True, True, True, False, False]


def test_mirror_scores_flip_orientation():
    yhat, y = np.array([1.0, 5.0]), np.array([1.0, 5.0])
    s_cal, s_test = scores(yhat, y, np.array([2.0]), 3.0, 9.0, mirror=True)
    assert s_cal.tolist() == [2.0, 9.0] and s_test.tolist() == [1.0]


def grid_world(rng, m, k, delta=0.5):
    pred = delta * rng.integers(0, 12, m + k)
    y = pred + rng.normal(0, 1, m + k)
    return pred[:m], y[:m], pred[m:], delta * rng.random(m + k) / 2


def test_no_ties_after_jitter():
    rng = np.random.default_rng(5)
    for mirror in (False, True):
        cp, cy, tp, xi = grid_world(rng, 80, 400)
        big = top(cp, tp, 3.0, 0.5, mirror)
        s = np.concatenate(scores(cp, cy, tp, 3.0, big, mirror))
        z = s + xi
        assert len(np.unique(s)) < len(s) and len(np.unique(z)) == len(z)
        i, j = np.triu_indices(len(s), 1)
        d = s[i] != s[j]
        assert ((s[i][d] < s[j][d]) == (z[i][d] < z[j][d])).all()


def test_min_selection_size():
    rng = np.random.default_rng(11)
    seen = 0
    for _ in range(300):
        m, k = int(rng.integers(10, 120)), int(rng.integers(50, 3000))
        q, mirror = float(rng.choice([0.05, 0.1, 0.2])), bool(rng.integers(2))
        cp, cy, tp, xi = grid_world(rng, m, k)
        sel = select(cp, cy, tp, 3.0, q, xi[:m], xi[m:], 0.5, mirror)
        if sel.any():
            seen += 1
            assert sel.sum() >= math.ceil(k / (q * (m + 1)))
    assert seen > 20


def test_top_ignores_labels():
    rng = np.random.default_rng(2)
    for mirror in (False, True):
        cp, cy, tp, xi = grid_world(rng, 60, 300)
        big = top(cp, tp, 3.0, 0.5, mirror)
        s_cal, s_test = scores(cp, cy, tp, 3.0, big, mirror)
        assert scores(cp, rng.permutation(cy), tp, 3.0, big, mirror)[0].max() == big
        fin = np.concatenate([s_cal[s_cal < big] + xi[:60][s_cal < big], s_test + xi[60:]])
        assert (big + xi[:60] > fin.max()).all()
