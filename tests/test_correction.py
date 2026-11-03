import numpy as np
import pytest

from cpq.labels import hybrid
from cpq.predictors import fit_grid
from cpq.tn import contract


@pytest.mark.parametrize("name", ["x264", "berkeleydb", "llvm"])
def test_corrected_prefix_and_mindeg(pops, name):
    p = pops(name)
    perm = np.random.default_rng(1).permutation(len(p))
    m = perm[:60]
    g = fit_grid(p.X[perm[:30]], p.y[perm[:30]]).top_pairs(5)
    pl = contract.plan(p.vm, g)
    c0 = min(g.low() + contract.minimum(pl, None, 0), int(np.floor(p.y[m].min() / g.delta)))
    w = np.rint(g.predict(p.X) / g.delta).astype(np.int64) - c0
    wm = np.rint(p.y / g.delta).astype(np.int64) - c0
    D = int(np.percentile(w, 75))
    h = contract.histogram(pl, D, len(p.vm.names), g.low() - c0)
    eye = np.eye(D + 2, dtype=np.int64)
    pred_m, meas_m = eye[np.minimum(w[m], D + 1)], eye[np.minimum(wm[m], D + 1)]
    hc = hybrid(h, pred_m, meas_m)
    u = np.ones(len(p), dtype=bool)
    u[m] = False
    hu = hc - meas_m.sum(axis=0)
    for k in (D // 2, D):
        assert contract.prefix(hu, k) == int((w[u] <= k).sum())
    top = np.minimum(np.concatenate([wm[m], w[u]]), D + 1)
    assert contract.mindeg(hc) == top.min()
