import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "results", Path(__file__).resolve().parents[1] / "experiments/results.py"
)
results = importlib.util.module_from_spec(spec)
spec.loader.exec_module(results)

pytestmark = pytest.mark.skipif(
    not any((results.RAW / "grid/sel").glob("*.parquet")), reason="grid not run"
)


def test_grid_fdr_within_mc_error():
    sel = results.grid("sel")
    cells = sel.groupby(["system", "n", "pct", "q"])
    z = [
        (d.fdp.mean() - q) / results.se(d.fdp) if d.fdp.std() > 0 else -np.inf
        for (_, _, _, q), d in cells
    ]
    # Bonferroni over the cells at 0.05, one-sided
    assert max(z) <= 3.5
    for q, d in sel.groupby("q"):
        assert d.fdp.mean() <= q + 2 * results.se_pooled(d, "fdp")
