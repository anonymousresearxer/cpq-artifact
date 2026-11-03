import itertools

import numpy as np
import pytest

from cpq.predictors import GridPredictor
from cpq.tn import contract
from cpq.vm import VM


def conv_ref(A, ia, B, ib, keep, L):
    lab = {x: i for i, x in enumerate(dict.fromkeys(ia + ib))}
    sa, sb, so = [lab[x] for x in ia], [lab[x] for x in ib], [lab[x] for x in keep]
    size = dict(zip(ia, A.shape, strict=False)) | dict(zip(ib, B.shape, strict=False))
    out = np.zeros(tuple(size[x] for x in keep) + (L,), dtype=object)
    for d, e in itertools.product(range(L), repeat=2):
        out[..., min(d + e, L - 1)] += np.einsum(A[..., d], sa, B[..., e], sb, so)
    return out


def factors(rng, L, dt):
    ia = tuple(rng.choice(list("abcdef"), rng.integers(0, 4), replace=False))
    ib = tuple(rng.choice(list("abcdef"), rng.integers(0, 4), replace=False))
    keep = [x for x in dict.fromkeys(ia + ib) if rng.random() < 0.5]
    A = rng.integers(0, 5, (2,) * len(ia) + (L,))
    B = rng.integers(0, 5, (2,) * len(ib) + (L,))
    # clause lifts and clause-only intermediates carry degree 0 only
    for T in [A, B][: rng.integers(0, 3)]:
        T[..., 1:] = 0
    return A.astype(dt), ia, B.astype(dt), ib, keep


def ints(a):
    return [int(x) for x in np.asarray(a).ravel()]


@pytest.mark.parametrize("dt", [np.float64, np.int64, object])
@pytest.mark.parametrize("L", [2, 3, 9, 40])
def test_conv_matches_definition(dt, L):
    rng = np.random.default_rng(L)
    for _ in range(30):
        A, ia, B, ib, keep = factors(rng, L, dt)
        ref = conv_ref(A.astype(object), ia, B.astype(object), ib, keep, L)
        got = contract._conv(A, ia, B, ib, keep, L)
        assert got.shape == ref.shape and ints(got) == ints(ref)


def instance(seed, nv=9):
    rng = np.random.default_rng(seed)
    clauses = []
    for _ in range(rng.integers(4, 9)):
        vs = rng.choice(np.arange(1, nv + 1), rng.integers(2, 7), replace=False)
        clauses.append([int(v) * int(rng.choice([-1, 1])) for v in vs])
    vm = VM("r", [f"x{i}" for i in range(1, nv + 1)], nv, clauses)
    ko = rng.integers(-3, 6, nv).astype(float)
    pairs = np.unique(np.sort(rng.choice(nv, (3, 2), replace=True), axis=1), axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    kp = rng.integers(-3, 4, len(pairs)).astype(float)
    return vm, GridPredictor(1.0, 0, ko, pairs, kp, (0.0, ko, kp))


def enumerate_w(vm, g):
    X = np.array(list(itertools.product((0, 1), repeat=vm.nvars)), dtype=np.int64)
    X = X[vm.valid(X)]
    w = X @ g.ko.astype(np.int64)
    for (i, j), k in zip(g.pairs, g.kp.astype(np.int64), strict=True):
        w = w + k * X[:, i] * X[:, j]
    return w - int(np.minimum(g.ko, 0).sum() + np.minimum(g.kp, 0).sum())


@pytest.mark.parametrize("elim", [2000, 0])
@pytest.mark.parametrize("seed", range(12))
def test_semirings_match_enumeration(seed, elim, monkeypatch):
    monkeypatch.setattr(contract, "ELIM", elim)
    vm, g = instance(seed)
    w = enumerate_w(vm, g)
    pl = contract.plan(vm, g)
    assert not pl.idx or len(contract._elimination(pl.idx)) == len(pl.idx) - 1
    t = contract.minimum(pl)
    assert t == int(w.min()) if len(w) else t >= contract.BIG
    for D in (0, 1, 3, 8, 30):
        ref = np.bincount(np.minimum(w, D + 1), minlength=D + 2).tolist()
        hs = [contract.histogram(pl, D, nf).tolist() for nf in (vm.nvars, 60, 70)]
        assert all([int(x) for x in h] == ref for h in hs)
        assert contract.minimum(pl, D) == min(int(w.min()) if len(w) else D + 1, D + 1)


def test_tropical_zero_absorbs():
    vm = VM("z", ["a", "b", "c"], 3, [[1, 2], [1, -2], [-1, 2], [-1, -2], [2, 3]])
    pl = contract.plan(vm)
    assert not pl.unsat
    assert contract.count(vm, 3, pl) == 0
    assert contract.minimum(pl) == contract.BIG
    for D in (0, 1, 5):
        assert contract.minimum(pl, D) == D + 1
        assert not contract.histogram(pl, D, 3).any()
