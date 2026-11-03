import numpy as np

from cpq.predictors import GridPredictor
from cpq.tn import oracle
from cpq.vm import VM


def test_identities_with_zero_box():
    vm = VM("o", [f"x{i}" for i in range(1, 7)], 6, [[1, 2], [-2, 3, 4], [-1, -5], [4, 5, 6]])
    ko = np.array([2.0, -1.0, 3.0, 1.0, 2.0, 1.0])
    kp = np.array([-2.0, 1.0])
    g = GridPredictor(1.0, 0, ko, np.array([[0, 2], [3, 5]]), kp, (0.0, ko, kp))
    for D in (1, 4, 9):
        assert oracle.violations(vm, g, D) == []
