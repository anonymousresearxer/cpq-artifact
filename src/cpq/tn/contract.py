import heapq
import os
from collections import Counter
from dataclasses import dataclass, field

import cotengra as ctg
import numpy as np

from ..predictors import GridPredictor
from ..vm import VM
from .network import Unsat, build, weight_items

CAP_BYTES = int(float(os.environ.get("CPQ_MEM_GB", 32)) * 2**30)
BIG = 2**40
PLAN_S = float(os.environ.get("CPQ_PLAN_S", "600"))
ELIM = 2000


class TooBig(Exception):
    pass


@dataclass
class Plan:
    tree: ctg.ContractionTree | None
    arrays: list[np.ndarray]
    idx: list[tuple[str, ...]]
    nclause: int
    free: int
    const: int
    width: float
    peak: float
    fixed: dict[int, bool] = field(default_factory=dict)
    unsat: bool = False

    @property
    def weighted(self) -> bool:
        return len(self.arrays) > self.nclause


def exact_dtype(nfeat: int):
    if nfeat <= 52:
        return np.float64
    return np.int64 if nfeat <= 62 else object


def _bytes(dt, nfeat: int) -> int:
    # an object entry is a pointer and a Python int below 2^nfeat, stored in 30-bit digits
    return 32 + 4 * -(-nfeat // 30) if dt is object else np.dtype(dt).itemsize


def _elimination(idx) -> list[tuple[int, int]]:
    """SSA path of bucket elimination in greedy min-degree order on the primal graph."""
    nb: dict[str, set[str]] = {}
    holders: dict[str, set[int]] = {}
    for i, ix in enumerate(idx):
        for x in ix:
            nb.setdefault(x, set()).update(ix)
            holders.setdefault(x, set()).add(i)
    for x, n in nb.items():
        n.discard(x)
    heap = [(len(n), x) for x, n in nb.items()]
    heapq.heapify(heap)
    tix = {i: set(ix) for i, ix in enumerate(idx)}
    path, nxt = [], len(idx)
    while heap:
        d, x = heapq.heappop(heap)
        if x not in nb or d != len(nb[x]):
            continue
        n = nb.pop(x)
        for y in n:
            nb[y] |= n
            nb[y] -= {x, y}
            heapq.heappush(heap, (len(nb[y]), y))
        group = sorted(holders.pop(x))
        cur = group[0]
        for j in group[1:]:
            s = tix.pop(cur) | tix.pop(j)
            for y in s - {x}:
                holders[y] -= {cur, j}
                holders[y].add(nxt)
            path.append((cur, j))
            tix[nxt], cur, nxt = s, nxt, nxt + 1
        tix[cur].discard(x)
    rest = sorted(tix)
    while len(rest) > 1:
        path.append((rest[0], rest[1]))
        rest, nxt = rest[2:] + [nxt], nxt + 1
    return path


def plan(vm: VM, pred: GridPredictor | None = None, repeats: int = 16, seed: int = 0) -> Plan:
    """Contraction plan of the clause network, with the predictor's weight tensors if given."""
    try:
        net = build(vm)
    except Unsat:
        return Plan(None, [], [], 0, 0, 0, 0.0, 1.0, unsat=True)
    items, const = list(net.items), 0
    if pred is not None:
        w, const = weight_items(pred, net.fixed)
        items += w
    used = {v for vs, _ in items for v in vs}
    free = sum(1 for v in range(1, vm.nvars + 1) if v not in used and v not in net.fixed)
    if not items:
        return Plan(None, [], [], 0, free, const, 0.0, 1.0, net.fixed)
    syms: dict[int, str] = {}
    for vs, _ in items:
        for v in vs:
            syms.setdefault(v, ctg.get_symbol(len(syms)))
    idx = [tuple(syms[v] for v in vs) for vs, _ in items]
    size = {s: 2 for s in syms.values()}
    # min-degree bucket elimination is deterministic and scales; the search may beat it when small
    tree = ctg.ContractionTree.from_path(idx, (), size, ssa_path=_elimination(idx))
    if len(idx) <= ELIM:
        # subtree reconfiguration after every trial dominates planning in pure Python; run it once
        opt = ctg.HyperOptimizer(
            methods=["greedy", "labels"], max_repeats=repeats, max_time=PLAN_S, minimize="size",
            reconf_opts=None, parallel=False, progbar=False, seed=seed,
        )  # fmt: skip
        found = ctg.array_contract_tree(idx, output=(), size_dict=size, optimize=opt)
        found.subtree_reconfigure_(minimize="size")
        if (found.max_size(), found.total_flops()) < (tree.max_size(), tree.total_flops()):
            tree = found
    return Plan(tree, [t for _, t in items], idx, len(net.items), free, const,
                tree.contraction_width(), tree.max_size(), net.fixed)  # fmt: skip


def _guard(pl: Plan, per_elem: int, cap: int) -> None:
    if pl.peak * per_elem > cap:
        raise TooBig(f"width {pl.width:.1f}, {pl.peak:.3g} elements, cap {cap / 2**30:.0f} GiB")


def count(vm: VM, nfeat: int | None = None, pl: Plan | None = None, cap: int = CAP_BYTES) -> int:
    """Number of satisfying assignments by exact contraction (semiring N)."""
    pl = pl or plan(vm)
    assert not pl.weighted
    if pl.unsat:
        return 0
    nf = nfeat or len(vm.names)
    dt = exact_dtype(nf)
    if pl.tree is None:
        return 1 << pl.free
    _guard(pl, _bytes(dt, nf), cap)
    # astype(object) already yields Python ints
    val = pl.tree.contract([a.astype(dt) for a in pl.arrays], backend="numpy")
    return int(val) << pl.free


def _drive(pl: Plan, arrays, pair, reduce_all, join, output=()):
    """Run the plan's pairwise contractions with the semiring operations given."""
    ten, ixs = list(arrays), [tuple(i) for i in pl.idx]
    counts = Counter(i for ix in ixs for i in ix)
    counts.update(output)
    for a, b in pl.tree.get_ssa_path():
        ia, ib = ixs[a], ixs[b]
        keep = []
        for x in dict.fromkeys(ia + ib):
            n_in = (x in ia) + (x in ib)
            if counts[x] > n_in:
                keep.append(x)
            counts[x] += (x in keep) - n_in
        ten.append(pair(ten[a], ia, ten[b], ib, keep))
        ixs.append(tuple(keep))
        ten[a] = ten[b] = None
    out = None
    for t, ix in zip(ten, ixs, strict=True):
        if t is None:
            continue
        t = reduce_all(t, ix, tuple(output))
        out = t if out is None else join(out, t)
    return out


def _conv(A, ia, B, ib, keep, L):
    """Product in H_D: saturated convolution on the last axis, overflow in slot L - 1."""
    lab = {x: i for i, x in enumerate(dict.fromkeys(ia + ib))}
    sa, sb, so = [lab[x] for x in ia], [lab[x] for x in ib], [lab[x] for x in keep]
    k = len(lab)
    # x^0 is the unit of H_D, so a factor of degree 0 only scales the other one
    if not A[..., 1:].any():
        return np.einsum(A[..., 0], sa, B, [*sb, k], [*so, k])
    if not B[..., 1:].any():
        return np.einsum(A, [*sa, k], B[..., 0], sb, [*so, k])
    size = dict(zip(ia, A.shape, strict=False)) | dict(zip(ib, B.shape, strict=False))
    out = np.zeros(tuple(size[x] for x in keep) + (L,), dtype=A.dtype)
    suf = np.cumsum(B[..., ::-1], axis=-1)[..., ::-1]
    for d in range(L):
        Ad = A[..., d]
        if not Ad.any():
            continue
        if d < L - 1:
            out[..., d : L - 1] += np.einsum(Ad, sa, B[..., : L - 1 - d], [*sb, k], [*so, k])
        out[..., L - 1] += np.einsum(Ad, sa, suf[..., L - 1 - d], sb, so)
    return out


def _lift_h(pl: Plan, L: int, dt):
    out = []
    for i, a in enumerate(pl.arrays):
        t = np.zeros(a.shape + (L,), dtype=dt)
        if i < pl.nclause:
            t[..., 0] = a
        else:
            t[...] = np.eye(L, dtype=dt)[np.minimum(a, L - 1)]
        out.append(t)
    return out


def _hist(pl: Plan, D: int, nfeat: int, cap: int) -> np.ndarray:
    """Histogram of E(v) saturated at D (semiring H_D), length D + 2."""
    L = D + 2
    dt = exact_dtype(nfeat)
    _guard(pl, _bytes(dt, nfeat) * L, cap)
    if pl.tree is None:
        h = np.zeros(L, dtype=dt)
        h[0] = 0 if pl.unsat else 1 << pl.free
        return h
    arrays = _lift_h(pl, L, dt)
    h = _drive(
        pl, arrays,
        lambda A, ia, B, ib, keep: _conv(A, ia, B, ib, keep, L),
        lambda t, ix, keep: _reduce(t, ix, keep, np.sum),
        lambda u, v: _conv(u, (), v, (), [], L),
    )  # fmt: skip
    return h * (1 << pl.free)


def histogram(pl: Plan, D: int, nfeat: int, base: int = 0, cap: int = CAP_BYTES) -> np.ndarray:
    """Histogram of grid indices E(v) + const + base saturated at D, length D + 2 (H_D)."""
    shift = pl.const + base
    if shift > D:
        h = np.zeros(D + 2, dtype=exact_dtype(nfeat))
        h[-1] = count_from_hist(_hist(pl, 0, nfeat, cap))
        return h
    inner = _hist(pl, D - shift, nfeat, cap)
    if shift < 0:
        return inner[-shift:]
    return np.concatenate([np.zeros(shift, dtype=inner.dtype), inner])


def count_from_hist(h: np.ndarray) -> int:
    """The homomorphism ev1: H_D to N."""
    return int(sum(int(x) for x in h))


def _reduce(t, ix, keep, op, **kw):
    """Reduce the axes of t outside keep with op, leaving the rest in the order of keep."""
    axes = tuple(i for i, x in enumerate(ix) if x not in keep)
    r = op(t, axis=axes, **kw) if axes else t
    rem = [x for x in ix if x in keep]
    return np.transpose(r, [rem.index(x) for x in keep] + list(range(len(rem), r.ndim)))


def _pair_t(A, ia, B, ib, keep, top):
    union = list(dict.fromkeys(ia + ib))
    size = dict(zip(ia, A.shape, strict=True)) | dict(zip(ib, B.shape, strict=True))

    def align(T, ix):
        order = np.argsort([union.index(x) for x in ix])
        T = np.transpose(T, order) if len(ix) else T
        return T.reshape([size[x] if x in ix else 1 for x in union])

    S = np.minimum(align(A, ia) + align(B, ib), top)
    axes = tuple(i for i, x in enumerate(union) if x not in keep)
    return S.min(axis=axes, initial=top) if axes else S


def minimum(pl: Plan, D: int | None = None, base: int = 0, cap: int = CAP_BYTES) -> int:
    """Smallest grid index E(v) + const + base over valid configurations (semiring T_D)."""
    shift = pl.const + base
    top = BIG if D is None else D + 1 - shift
    if pl.unsat or top <= 0:
        return BIG if D is None else D + 1
    if pl.tree is None:
        m = 0
    else:
        # entries never exceed top, so a sum of two fits in int32 when 2 top < 2^31
        dt = np.int32 if top < 2**30 else np.int64
        _guard(pl, np.dtype(dt).itemsize, cap)
        arrays = []
        for i, a in enumerate(pl.arrays):
            if i < pl.nclause:
                arrays.append(np.where(a == 1, 0, top).astype(dt))
            else:
                arrays.append(np.minimum(a, top).astype(dt))
        m = int(
            _drive(
                pl, arrays,
                lambda A, ia, B, ib, keep: _pair_t(A, ia, B, ib, keep, top),
                lambda t, ix, keep: _reduce(t, ix, keep, np.min, initial=top),
                lambda u, v: np.minimum(u + v, top),
            )
        )  # fmt: skip
    if D is None:
        # a saturated minimum means no valid configuration, whatever the shift
        return BIG if m >= top else m + shift
    return min(m + shift, D + 1)


def mindeg(h: np.ndarray) -> int:
    """The homomorphism mindeg: H_D to T_D, exact when the optimum is at most D."""
    nz = np.flatnonzero(np.array([int(x) != 0 for x in h]))
    return int(nz[0]) if len(nz) else len(h) - 1


def prefix(h: np.ndarray, k: int) -> int:
    """Number of configurations with grid index at most k (k <= D)."""
    return int(sum(int(x) for x in h[: k + 1]))
