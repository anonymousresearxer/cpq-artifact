import itertools
from dataclasses import dataclass

import numpy as np

from ..predictors import GridPredictor
from ..vm import VM

DENSE = 4
CHUNK = 3


def _dense(clause: list[int]) -> tuple[list[int], np.ndarray]:
    vs = sorted({abs(lt) for lt in clause})
    pos = {v: i for i, v in enumerate(vs)}
    t = np.zeros((2,) * len(vs), dtype=np.int8)
    for a in itertools.product((0, 1), repeat=len(vs)):
        if any(a[pos[abs(lt)]] == (lt > 0) for lt in clause):
            t[a] = 1
    return vs, t


def _chain(clause: list[int], fresh) -> list[tuple[list[int], np.ndarray]]:
    """Long clause as a chain of tensors; the bond carries 'satisfied so far' and is determined."""
    parts = [clause[i : i + CHUNK] for i in range(0, len(clause), CHUNK)]
    out = []
    prev = None
    for k, part in enumerate(parts):
        last = k == len(parts) - 1
        nxt = None if last else fresh()
        vs = ([prev] if prev else []) + [abs(lt) for lt in part] + ([nxt] if nxt else [])
        t = np.zeros((2,) * len(vs), dtype=np.int8)
        for a in itertools.product((0, 1), repeat=len(vs)):
            s_prev = a[0] if prev else 0
            xs = a[1 if prev else 0 : len(a) - (0 if last else 1)]
            hit = s_prev or any(x == (lt > 0) for x, lt in zip(xs, part, strict=True))
            if last:
                t[a] = int(bool(hit))
            else:
                t[a] = int(a[-1] == int(bool(hit)))
        out.append((vs, t))
        prev = nxt
    return out


class Unsat(Exception):
    pass


def propagate(clauses: list[list[int]]) -> tuple[list[list[int]], dict[int, bool]]:
    """Unit propagation: simplified clauses and the values it fixes."""
    fixed: dict[int, bool] = {}
    cl = [list(dict.fromkeys(c)) for c in clauses]
    while True:
        units = [c[0] for c in cl if len(c) == 1]
        if not units:
            return cl, fixed
        for u in units:
            fixed[abs(u)] = u > 0
        nxt = []
        for c in cl:
            if any(fixed.get(abs(x)) == (x > 0) for x in c):
                continue
            nxt.append([x for x in c if abs(x) not in fixed])
        if any(not c for c in nxt):
            raise Unsat
        cl = nxt


@dataclass
class Net:
    items: list[tuple[list[int], np.ndarray]]
    fixed: dict[int, bool]


def build(vm: VM) -> Net:
    """Clause tensors over the variables left after unit propagation."""
    nxt = vm.nvars
    clauses, fixed = propagate(vm.clauses)
    merged: dict[tuple[int, ...], np.ndarray] = {}
    extra: list[tuple[list[int], np.ndarray]] = []

    def fresh() -> int:
        nonlocal nxt
        nxt += 1
        return nxt

    for c in clauses:
        if any(-lt in c for lt in c):
            continue
        if len(c) <= DENSE:
            vs, t = _dense(c)
            key = tuple(vs)
            merged[key] = merged[key] * t if key in merged else t
        else:
            extra += _chain(c, fresh)
    return Net([(list(k), t) for k, t in merged.items()] + extra, fixed)


def weight_items(pred: GridPredictor, fixed: dict[int, bool]):
    """Exponent tensors of the grid predictor, shifted to be non-negative, and their constant.

    The exponent of a configuration is E(v) = const + sum of the tensors, and its grid index
    is E(v) + base with base = pred.low() - c0.
    """
    const = 0
    unary: dict[int, np.ndarray] = {}
    items: list[tuple[list[int], np.ndarray]] = []

    def add_unary(v: int, e: np.ndarray) -> None:
        nonlocal const
        if v in fixed:
            const += int(e[int(fixed[v])])
        else:
            unary[v] = unary.get(v, 0) + e

    for j, k in enumerate(pred.ko.astype(np.int64)):
        add_unary(j + 1, np.array([-min(0, k), k - min(0, k)], dtype=np.int64))
    for (i, j), k in zip(pred.pairs, pred.kp.astype(np.int64), strict=True):
        vi, vj = int(i) + 1, int(j) + 1
        e = np.array([[0, 0], [0, k]], dtype=np.int64) - min(0, k)
        if vi in fixed and vj in fixed:
            const += int(e[int(fixed[vi]), int(fixed[vj])])
        elif vi in fixed:
            add_unary(vj, e[int(fixed[vi])])
        elif vj in fixed:
            add_unary(vi, e[:, int(fixed[vj])])
        else:
            items.append(([vi, vj], e))
    items += [([v], e) for v, e in unary.items() if e.any()]
    return items, const
