import numpy as np

from ..predictors import GridPredictor
from ..vm import VM
from . import contract


def saturate(h: np.ndarray, D: int) -> list[int]:
    """The homomorphism H_D' to H_D for D < D'."""
    h = [int(x) for x in h]
    return h[: D + 1] + [sum(h[D + 1 :])]


def zero(vm: VM) -> VM:
    """vm times a box that no assignment satisfies and that unit propagation cannot see."""
    a, b = vm.nvars + 1, vm.nvars + 2
    return VM(vm.name, vm.names, vm.nvars + 2, vm.clauses + [[a, b], [a, -b], [-a, b], [-a, -b]])


def violations(vm: VM, pred: GridPredictor, D: int, base: int = 0) -> list[str]:
    """Homomorphism identities between the evaluations of one diagram; empty if all hold."""
    nf = len(vm.names)
    n = contract.count(vm, nf)
    pl = contract.plan(vm, pred)
    h = contract.histogram(pl, D, nf, base)
    t = contract.minimum(pl, D, base)
    t_free = contract.minimum(pl, None, base)
    bad = []
    if contract.count_from_hist(h) != n:
        bad.append("ev1")
    if contract.mindeg(h) != min(t_free, D + 1):
        bad.append("mindeg")
    if t != min(t_free, D + 1):
        bad.append("tropical")
    if (n > 0) != (t_free < contract.BIG):
        bad.append("supp")
    # t_free - 1 is the last degree at which every evaluation must saturate
    below = {t_free - 1} if 0 < t_free <= D else set()
    for D2 in sorted({D + 7, D // 2 + 1} | below):
        h2 = contract.histogram(pl, D2, nf, base)
        small, big = (h, h2) if D2 > D else (h2, h)
        if saturate(big, min(D, D2)) != [int(x) for x in small]:
            bad.append(f"saturation {D2}")
        e = min(t_free, D2 + 1)
        if contract.minimum(pl, D2, base) != e or contract.mindeg(h2) != e:
            bad.append(f"tropical {D2}")
    shifted = contract.histogram(pl, D + 1, nf, base + 1)
    if [int(x) for x in shifted[1:]] != [int(x) for x in contract.histogram(pl, D, nf, base)]:
        bad.append("shift")
    # the semiring zeros are absorbing and every homomorphism preserves them
    z = zero(vm)
    pz = contract.plan(z)
    if contract.count(z, nf, pz) != 0 or any(int(x) for x in contract.histogram(pz, D, nf)):
        bad.append("zero")
    if contract.minimum(pz, D) != D + 1 or contract.minimum(pz) != contract.BIG:
        bad.append("tropical zero")
    return bad
