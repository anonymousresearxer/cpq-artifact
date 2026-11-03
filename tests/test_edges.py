import importlib.util
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

from cpq import conformal, hw, labels, selection, spec


def test_small_calibration():
    for m in range(9):
        assert conformal.qhat(np.arange(m, dtype=float), 0.1) == np.inf
        assert conformal.pac_missed(m, 1000, 0.1, 0.05) == 0


def test_rank_matches_l():
    for alpha in (0.05, 0.1, 0.2):
        for m in range(1, 600):
            k = m + 1 - conformal.n_missed(m, alpha)
            assert k == math.ceil((m + 1) * (1 - Fraction(str(alpha))))


def test_pac_extremes():
    assert conformal.pac_missed(100, 0, 0.1, 0.05) == 0
    assert conformal.pac_missed(100, 500, 0.1, 1.0) == 0
    assert conformal.pac_missed(100, 500, 0.1, 0.0) == 500
    assert conformal.qhat(np.arange(20.0), 1.0) == -np.inf
    assert conformal.pac_missed(20, 300, 1.0, 0.05) == 300


def test_pvalues_follow_theorem6():
    rng = np.random.default_rng(3)
    s_cal = np.where(rng.random(40) < 0.3, np.inf, rng.integers(-3, 4, 40).astype(float))
    s_test = rng.integers(-4, 5, 25).astype(float)
    p = selection.pvalues(s_cal, s_test)
    assert np.array_equal(p, [(1 + (s_cal < s).sum()) / 41 for s in s_test])


def test_count_extreme_thresholds():
    rng = np.random.default_rng(1)
    y = rng.gamma(2.0, 1.0, 400)
    yhat = y + rng.normal(0, 0.3, 400)
    meas = np.zeros(400, dtype=bool)
    meas[:60] = True
    sel = np.zeros(400, dtype=bool)
    for t in (-1e9, y.min(), np.median(y), y.max(), 1e9):
        c = labels.count(yhat, y, meas, sel, 0.4, t, 30, 0.1, 0.05)
        assert 0 <= c.low <= c.high <= len(y)
    assert labels.count(yhat, y, meas, sel, 0.4, 1e9, 30, 0.1, 0.05).high == len(y)
    assert labels.count(yhat, y, meas, sel, 0.4, -1e9, 30, 0.1, 0.05).low == 0


def test_boolean_semiring_evaluated():
    tabs = {"R": {"n": 3, "cols": {"a": np.array([1, 1, 2])}},
            "S": {"n": 2, "cols": {"a": np.array([1, 3])}}}  # fmt: skip
    axes = [{"index": "x", "column": "a"}]
    q = {"indices": [{"name": "x", "kind": "attr"}], "output": ["x"], "einsum": "a,a->a",
         "tensors": [{"table": "R", "axes": axes}, {"table": "S", "axes": axes}]}  # fmt: skip
    res = spec.evaluate(q, tabs, D=2)
    assert res["N"] == {(1,): 2} and res["B"] == {(1,): True}
    assert spec.identities(res, 2) == []


def test_cpu_lists():
    assert hw._cpus("0-3,8,10-11\n") == {0, 1, 2, 3, 8, 10, 11}
    assert hw._size("98304K\n") == 96 * 2**20


def fake_sysfs(root, sizes):
    for cpu, (size, shared) in enumerate(sizes):
        d = root / f"cpu{cpu}/cache/index3"
        d.mkdir(parents=True)
        (d / "level").write_text("3\n")
        (d / "size").write_text(size)
        (d / "shared_cpu_list").write_text(shared)


def test_vcache_ccd(tmp_path, monkeypatch):
    fake_sysfs(tmp_path / "x3d", [("98304K", "0-1")] * 2 + [("32768K", "2-3")] * 2)
    fake_sysfs(tmp_path / "flat", [("32768K", "0-3")] * 4)
    monkeypatch.setattr(hw, "CPU", tmp_path / "x3d")
    assert hw.vcache_cpus() == {0, 1}
    monkeypatch.setattr(hw, "CPU", tmp_path / "flat")
    assert hw.vcache_cpus() is None
    monkeypatch.setattr(hw, "CPU", tmp_path / "missing")
    assert hw.vcache_cpus() is None and hw.pin_vcache() is None
    assert hw.physical_cores() >= 1


def rq4_module():
    path = Path(__file__).resolve().parents[1] / "experiments/rq4.py"
    spec_ = importlib.util.spec_from_file_location("rq4", path)
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    return mod


def test_fastest_exact_layout(monkeypatch):
    rq4 = rq4_module()
    full = dict(layout="full", s_count=1.0, s_hist=2.0, s_min=0.5, count_ok=True)
    vc = dict(layout="vcache", s_count=0.5, s_hist=1.0, s_min=0.5, count_ok=True)
    wrong = dict(layout="vcache", s_count=0.1, s_hist=0.1, s_min=0.1, count_ok=False)
    assert rq4.fastest([full, vc])["layout"] == "vcache"
    best = rq4.fastest([full, wrong])
    assert best["layout"] == "full" and [r["layout"] for r in best["runs"]] == ["full", "vcache"]
    monkeypatch.setattr(rq4.hw, "vcache_cpus", lambda: {0, 1})
    assert rq4.layouts("study") == ("full", "vcache") and rq4.layouts("limits") == ("full",)
