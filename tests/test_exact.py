import numpy as np
import pytest

from cpq.data import SYSTEMS
from cpq.predictors import fit_grid
from cpq.tn import contract


@pytest.mark.parametrize("name", list(SYSTEMS))
def test_float64_and_int64_agree_on_counts_and_histograms(pops, name, monkeypatch):
    p = pops(name)
    tr = np.random.default_rng(0).permutation(len(p))[:200]
    g = fit_grid(p.X[tr], p.y[tr])
    c0 = g.floor(p.y.min())
    base = g.low() - c0
    w = np.rint(g.predict(p.X) / g.delta).astype(np.int64) - c0
    D = int(np.percentile(w, 75))
    pl = contract.plan(p.vm, g)
    nf = len(p.vm.names)
    f = contract.histogram(pl, D, nf, base)
    assert f.dtype == (np.float64 if nf <= 52 else np.int64)
    monkeypatch.setattr(contract, "exact_dtype", lambda n: np.int64)
    i = contract.histogram(pl, D, nf, base)
    assert i.dtype == np.int64
    assert f.astype(np.int64).tolist() == i.tolist()
    assert contract.count(p.vm, nf) == len(p)
    assert contract.count_from_hist(i) == len(p)
