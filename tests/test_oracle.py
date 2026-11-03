import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cpq import spec
from cpq.predictors import GridPredictor
from cpq.tn import oracle
from cpq.vm import VM


@st.composite
def instance(draw):
    n = draw(st.integers(3, 9))
    lit = st.integers(1, n).flatmap(lambda v: st.sampled_from([v, -v]))
    clauses = draw(st.lists(st.lists(lit, min_size=1, max_size=6), min_size=0, max_size=8))
    ko = np.array(draw(st.lists(st.integers(-3, 4), min_size=n, max_size=n)), dtype=float)
    pair = st.tuples(st.integers(0, n - 1), st.integers(0, n - 1)).filter(lambda t: t[0] < t[1])
    pairs = draw(st.lists(pair, max_size=4, unique=True))
    kp = np.array(
        draw(st.lists(st.integers(-3, 4), min_size=len(pairs), max_size=len(pairs))), dtype=float
    )
    k0 = draw(st.integers(-5, 5))
    pred = GridPredictor(
        1.0, k0, ko, np.array(pairs, dtype=np.int64).reshape(-1, 2), kp, (0, ko, kp)
    )
    vm = VM("rnd", [f"x{i}" for i in range(n)], n, clauses)
    return vm, pred, draw(st.integers(1, 25)), draw(st.integers(0, 6))


@given(instance())
@settings(max_examples=60, deadline=None)
def test_homomorphism_identities_hold(inst):
    vm, pred, D, base = inst
    assert oracle.violations(vm, pred, D, base) == []


@pytest.mark.parametrize("fx", ["p1", "p2", "p3"])
@pytest.mark.parametrize("q", ["cfg", "measured", "hist", "pairs", "pairs_vc", "pred", "noise"])
def test_compiled_queries_satisfy_the_identities(fx, q):
    bridge = pytest.importorskip("cpq.bridge")
    tabs = bridge.tables(bridge.fixture(fx))
    yh = tabs["Pred"]["cols"]["yhat"]
    ann = {"Pred": np.rint((yh - yh.min()) / 0.5).astype(int)} if len(yh) else {}
    res = spec.evaluate(bridge.query_spec(q), tabs, ann, 12)
    assert spec.identities(res, 12) == []
