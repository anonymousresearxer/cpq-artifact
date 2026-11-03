import numpy as np
import pytest

from cpq.predictors import fit_grid
from cpq.tn import contract

SMALL = ["x264", "berkeleydb", "lrzip", "llvm", "dune"]


def setup(pops, name, npairs=15, k=100):
    p = pops(name)
    tr = np.random.default_rng(0).permutation(len(p))[:k]
    g = fit_grid(p.X[tr], p.y[tr]).top_pairs(npairs)
    c0 = g.floor(p.y.min())
    w = np.rint(g.predict(p.X) / g.delta).astype(np.int64) - c0
    return p, g, w, g.low() - c0


@pytest.mark.parametrize("name", SMALL)
def test_histogram_matches_enumeration(pops, name):
    p, g, w, base = setup(pops, name)
    pl = contract.plan(p.vm, g)
    for D in (3, int(np.percentile(w, 50)), int(w.max()) + 4):
        want = np.bincount(np.minimum(w, D + 1), minlength=D + 2)
        assert contract.histogram(pl, D, len(p.vm.names), base).tolist() == want.tolist()


@pytest.mark.parametrize("name", SMALL)
def test_tropical_matches_enumeration(pops, name):
    p, g, w, base = setup(pops, name)
    pl = contract.plan(p.vm, g)
    assert contract.minimum(pl, None, base) == w.min()
    assert contract.minimum(pl, int(w.min()) - 1, base) == int(w.min())


@pytest.mark.parametrize("name", SMALL)
def test_prefix_sums_are_threshold_counts(pops, name):
    p, g, w, base = setup(pops, name)
    D = int(np.percentile(w, 75))
    h = contract.histogram(contract.plan(p.vm, g), D, len(p.vm.names), base)
    for k in (int(w.min()), D // 2 + int(w.min()) // 2, D):
        assert contract.prefix(h, k) == int((w <= k).sum())


@pytest.mark.parametrize("name", SMALL)
def test_dropping_pairs_changes_predictor_not_exactness(pops, name):
    p, g, w, base = setup(pops, name, npairs=3)
    D = int(w.max()) + 1
    h = contract.histogram(contract.plan(p.vm, g), D, len(p.vm.names), base)
    assert h.tolist() == np.bincount(w, minlength=D + 2).tolist()
    assert len(g.pairs) <= 3


@pytest.mark.parametrize("name", SMALL)
def test_tight_offset_from_the_tropical_minimum(pops, name):
    p, g, _, _ = setup(pops, name)
    pl = contract.plan(p.vm, g)
    c0 = g.low() + contract.minimum(pl, None, 0)
    base = g.low() - c0
    w = np.rint(g.predict(p.X) / g.delta).astype(np.int64) - c0
    assert w.min() == 0 and base <= 0
    D = int(np.percentile(w, 60))
    want = np.bincount(np.minimum(w, D + 1), minlength=D + 2)
    assert contract.histogram(pl, D, len(p.vm.names), base).tolist() == want.tolist()


@pytest.mark.parametrize("name", SMALL)
def test_saturated_tropical_with_offsets(pops, name):
    p, g, _, base = setup(pops, name)
    pl = contract.plan(p.vm, g)
    tight = g.low() + contract.minimum(pl, None, 0)
    for b in (base, g.low() - tight):
        wv = np.rint(g.predict(p.X) / g.delta).astype(np.int64) - (g.low() - b)
        for D in (0, 2, int(wv.min()) - 2, int(wv.min()) - 1, int(wv.min()), int(wv.min()) + 3):
            if D >= 0:
                assert contract.minimum(pl, D, b) == min(int(wv.min()), D + 1)
