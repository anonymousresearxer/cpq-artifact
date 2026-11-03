import itertools

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from cpq import enum, instance
from cpq.baselines import methods
from cpq.vm import VM


def test_wilson_contains_the_proportion():
    lo, hi = methods.wilson(30, 100)
    assert lo < 0.3 < hi and 0 <= lo and hi <= 1
    assert methods.wilson(0, 50)[0] == 0.0


def test_ppi_is_exact_for_a_perfect_predictor():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.3).astype(float)
    est, lo, hi = methods.ppi_mean(y[100:], y[:100], y[:100])
    assert est == y[100:].mean() and lo <= est <= hi


def test_ppi_covers_the_truth_often():
    rng = np.random.default_rng(1)
    hits = 0
    for _ in range(300):
        x = rng.normal(size=2000)
        y = (x + rng.normal(0, 0.7, 2000) <= 0.2).astype(float)
        f = (x <= 0.2).astype(float)
        _, lo, hi = methods.ppi_mean(f[200:], f[:200], y[:200])
        hits += lo <= y[200:].mean() <= hi
    assert hits / 300 > 0.9


def test_returned_set_metrics():
    ret = np.array([1, 1, 0, 0], dtype=bool)
    truth = np.array([1, 0, 1, 0], dtype=bool)
    assert methods.returned(ret, truth) == (2, 0.5, 0.5)
    assert methods.interval_error(5, 2, 4) == 1 and methods.interval_error(3, 2, 4) == 0


@given(
    st.lists(st.lists(st.integers(1, 6).flatmap(lambda v: st.sampled_from([v, -v])), min_size=1,
                      max_size=4), max_size=7)
)  # fmt: skip
@settings(max_examples=80, deadline=None)
def test_enumeration_matches_brute_force(clauses):
    vm = VM("t", [f"x{i}" for i in range(6)], 6, clauses)
    allx = np.array(list(itertools.product((0, 1), repeat=6)), dtype=np.uint8)
    want = {tuple(map(int, r)) for r in allx[vm.valid(allx)]}
    got = enum.models(vm)
    assert len(got) == len(set(got)) and set(got) == want


def test_instance_tables_partition_the_configurations():
    tabs, g = instance.qvm_tables("lrzip", n=100, seed=1)
    n, total = tabs["CCs"]["n"], tabs["Cfg"]["n"]
    assert n + tabs["VCs"]["n"] == total == tabs["Pred"]["n"] == 432
    cfg = np.concatenate([tabs["CCs"]["cols"]["cc_cfg"], tabs["VCs"]["cols"]["vc_cfg"]])
    assert sorted(cfg) == list(range(1, total + 1))
    assert tabs["CClinks"]["n"] == int(tabs["CClinks"]["cols"]["cc"].size)
