import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location(
    "results", Path(__file__).resolve().parents[1] / "experiments/results.py"
)
results = importlib.util.module_from_spec(spec)
spec.loader.exec_module(results)


def test_se_of_known_sample():
    assert np.isclose(results.se([1.0, 2.0, 3.0, 4.0]), np.std([1, 2, 3, 4], ddof=1) / 2)


def test_se_pooled_combines_systems():
    d = pd.DataFrame({
        "system": ["a"] * 4 + ["b"] * 4,
        "rep": [0, 1, 2, 3] * 2,
        "x": [0.0, 1.0, 0.0, 1.0, 2.0, 2.0, 4.0, 4.0],
    })  # fmt: skip
    va, vb = np.var([0, 1, 0, 1], ddof=1) / 4, np.var([2, 2, 4, 4], ddof=1) / 4
    assert np.isclose(results.se_pooled(d, "x"), np.sqrt(va + vb) / 2)


def test_md_aligns_numbers_right_and_escapes_pipes():
    out = results.md(["system", "|V|", "ok"], [["x", 3, "True"], ["y", "0.5 +- 0.1", "False"]])
    assert out.splitlines()[1] == "|---|---:|---|"
    assert "\\|V\\|" in out.splitlines()[0]
