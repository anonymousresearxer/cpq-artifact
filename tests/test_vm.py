import itertools

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cpq.data import SYSTEMS
from cpq.tn import contract
from cpq.vm import VM, _Encoder, _nnf, _parse


@pytest.mark.parametrize("name", list(SYSTEMS))
def test_count_matches_rows(pops, name):
    p = pops(name)
    assert p.vm.valid(p.X).all()
    assert len(np.unique(p.X, axis=0)) == len(p)
    assert contract.count(p.vm) == len(p)


def brute(vm: VM) -> int:
    n = len(vm.names)
    X = np.array(list(itertools.product((0, 1), repeat=n)), dtype=np.uint8)
    return int(vm.valid(X).sum())


lits = st.sampled_from(["a", "b", "c", "d", "!a", "!b", "!c", "!d"])
exprs = st.recursive(
    lits,
    lambda sub: st.one_of(
        st.tuples(sub, sub).map(lambda t: f"({t[0]} & {t[1]})"),
        st.tuples(sub, sub).map(lambda t: f"({t[0]} | {t[1]})"),
        sub.map(lambda e: f"!({e})"),
    ),
    max_leaves=8,
)


@given(st.lists(exprs, min_size=1, max_size=3))
@settings(max_examples=60, deadline=None)
def test_tseitin_preserves_count(cs):
    idx = {n: i + 1 for i, n in enumerate("abcd")}
    enc = _Encoder(4)
    for c in cs:
        enc.require(_nnf(_parse(c, idx)))
    vm = VM("t", list("abcd"), enc.nvars, enc.clauses, enc.defs)
    X = np.array(list(itertools.product((0, 1), repeat=4)), dtype=np.uint8)
    names = dict(zip("abcd", X.T.astype(bool), strict=True))
    want = np.ones(len(X), dtype=bool)
    for c in cs:
        want &= eval(c.replace("!", "~"), {}, names)
    assert contract.count(vm) == int(want.sum())
    assert (vm.valid(X) == want).all()


def test_long_clause_chain():
    n = 11
    vm = VM("t", [f"x{i}" for i in range(n)], n, [list(range(1, n + 1)), [-1, -2]])
    assert contract.count(vm) == brute(vm)
