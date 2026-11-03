import importlib.util
import inspect
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "run", Path(__file__).parents[1] / "experiments/run.py"
)
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)

CFG = dict(alpha=0.1, q=[0.1], pac_delta=0.05, percentiles=[25, 50, 75], eps=[0.0, 0.1])


def test_rows_are_consistent_and_reproducible():
    a = run.one(("lrzip", 100, 3, CFG))
    b = run.one(("lrzip", 100, 3, CFG))
    assert [r["n_sel"] for r in a["sel"]] == [r["n_sel"] for r in b["sel"]]
    for r in a["sel"]:
        assert 0 <= r["fdp"] <= 1
        assert r["certain"] <= r["possible"]
        assert r["m"] == 50 and r["n_u"] == r["n_valid"] - 100
    assert np.isfinite([r["qhat"] for r in a["sel"]]).all()
    methods = {r["method"] for r in a["cmp"]}
    assert methods == {"ours", "measured_only", "predictor_only", "intervals", "ppi"}
    assert len(a["cmp"]) == 3 * 2 * 5
    ours = [r for r in a["opt"] if r["method"] == "ours"][0]
    uni = [r for r in a["opt"] if r["method"] == "uniform_total"][0]
    assert ours["total"] == 100 + ours["extra"] and ours["regret"] >= 0 and uni["regret"] >= 0


def test_noise_independent_of_y_and_split():
    # xi takes neither y nor the permutation that splits Train, Cal and U
    assert list(inspect.signature(run.noise).parameters) == ["sid", "n", "rep", "N"]
    xi = run.noise(2, 100, 3, 432)
    assert xi.shape == (432,) and ((0 <= xi) & (xi < 1)).all()
    assert np.array_equal(xi, run.noise(2, 100, 3, 432))
    assert not np.array_equal(xi, run.noise(2, 100, 4, 432))
