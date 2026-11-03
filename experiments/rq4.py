import json
import multiprocessing as mp
import os
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from cpq import conformal, enum, ganak, hw, runinfo
from cpq.data import DATA, SYSTEMS, load
from cpq.predictors import GridPredictor, fit_grid
from cpq.tn import contract
from cpq.vm import read_dimacs

OUT = Path(os.environ.get("CPQ_OUT", Path(__file__).resolve().parents[1] / "results/raw")) / "rq4"
MAX_WIDTH = 16
BLAS = int(os.environ.get("CPQ_BLAS_THREADS", "16"))
DIMACS = ["berkeleydb", "busybox-1.18.0"]
LIMITS = ["financial01", "linux-2.6.33.3"]


def traced(f, *a, **kw):
    tracemalloc.start()
    t = time.perf_counter()
    r = f(*a, **kw)
    dt = time.perf_counter() - t
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return r, dt, peak / 2**20


def study_case(system):
    p = load(system)
    N, nf = len(p), len(p.vm.names)
    rng = np.random.default_rng(np.random.SeedSequence([list(SYSTEMS).index(system), 200, 0]))
    perm = rng.permutation(N)
    g = fit_grid(p.X[perm[:100]], p.y[perm[:100]], seed=0)
    full = g
    while True:
        pl = contract.plan(p.vm, g)
        if pl.width <= MAX_WIDTH or len(g.pairs) == 0:
            break
        g = g.top_pairs(len(g.pairs) - 1)
    ymin = float(p.y[perm[:200]].min())
    e_min = contract.minimum(pl, None, 0)
    c0 = min(g.low() + e_min, int(np.floor(ymin / g.delta)))
    base = g.low() - c0
    cal = perm[100:200]
    qh = conformal.qhat(np.abs(p.y[cal] - g.predict(p.X[cal])), 0.1)
    t = float(np.percentile(p.y, 75))
    D = max(int(np.floor((t + qh) / g.delta)) - c0, 1)
    w = np.rint(g.predict(p.X) / g.delta).astype(np.int64) - c0
    pc = contract.plan(p.vm)
    n_tn, t_n, m_n = traced(contract.count, p.vm, nf, pc)
    h, t_h, m_h = traced(contract.histogram, pl, D, nf, base)
    mn, t_t, m_t = traced(contract.minimum, pl, D, base)
    ref = np.bincount(np.minimum(w, D + 1), minlength=D + 2)
    row = dict(system=system, kind="study", n_valid=N, features=nf, pairs=len(g.pairs))
    row |= dict(pairs_dropped=len(full.pairs) - len(g.pairs), width_n=pc.width)
    row |= dict(width_weighted=pl.width, D=D, delta=g.delta)
    row |= dict(dtype=str(np.dtype(contract.exact_dtype(nf))), count_ok=n_tn == N)
    row |= dict(hist_ok=h.tolist() == ref.tolist(), min_ok=mn == min(int(w.min()), D + 1))
    row |= dict(ev1_ok=contract.count_from_hist(h) == N)
    row |= dict(mindeg_ok=contract.mindeg(h) == min(mn, D + 1))
    row |= dict(prefix_ok=contract.prefix(h, D) + int(h[-1]) == N)
    row |= dict(s_count=t_n, s_hist=t_h, s_min=t_t, mb_count=m_n, mb_hist=m_h, mb_min=m_t)
    row |= dict(mae_raw15=float(np.abs(p.y - full.predict_raw(p.X)).mean()))
    row |= dict(mae_grid15=float(np.abs(p.y - full.predict(p.X)).mean()))
    row |= dict(mae_used_grid=float(np.abs(p.y - g.predict(p.X)).mean()))
    row |= dict(mae_used_raw=float(np.abs(p.y - g.predict_raw(p.X)).mean()))
    return row


def synthetic(vm, seed=0, density=0.1, npairs=5):
    rng = np.random.default_rng(seed)
    nf = len(vm.names)
    on = rng.choice(nf, max(int(density * nf), 2), replace=False)
    ko = np.zeros(nf)
    ko[on] = rng.integers(1, 6, len(on))
    pairs = np.array([rng.choice(on, 2, replace=False) for _ in range(npairs)])
    pairs = np.sort(pairs, axis=1)
    pairs = np.unique(pairs, axis=0)
    kp = rng.integers(-3, 4, len(pairs)).astype(float)
    return GridPredictor(1.0, 0, ko, pairs, kp, (0, ko, kp))


def dimacs_case(name):
    vm = read_dimacs(DATA / "dimacs" / f"{name}.dimacs")
    g = synthetic(vm)
    D = 64
    pc = contract.plan(vm)
    pl = contract.plan(vm, g)
    n_tn, t_n, m_n = traced(contract.count, vm, vm.nvars, pc)
    h, t_h, m_h = traced(contract.histogram, pl, D, vm.nvars, 0)
    mn, t_t, m_t = traced(contract.minimum, pl, D, 0)
    n_g, t_g = None, None
    t0 = time.perf_counter()
    n_g = ganak.count(vm, timeout=1800)
    t_g = time.perf_counter() - t0
    row = dict(system=name, kind="dimacs_synthetic", features=vm.nvars, clauses=len(vm.clauses),
               pairs=len(g.pairs), width_n=pc.width, width_weighted=pl.width, D=D,
               count=str(n_tn), ganak=str(n_g), count_ok=n_tn == n_g,
               ev1_ok=contract.count_from_hist(h) == n_tn,
               mindeg_ok=contract.mindeg(h) == min(mn, D + 1), s_count=t_n, s_hist=t_h, s_min=t_t,
               s_ganak=t_g, mb_count=m_n, mb_hist=m_h, mb_min=m_t)  # fmt: skip
    if n_g < 10**5:
        ms = np.array(enum.models(vm), dtype=np.int64)[:, : vm.nvars]
        w = ms[:, : len(g.ko)] @ g.ko.astype(np.int64)
        for (i, j), k in zip(g.pairs, g.kp.astype(np.int64), strict=True):
            w = w + k * ms[:, i] * ms[:, j]
        w = w - (np.minimum(g.ko, 0).sum() + np.minimum(g.kp, 0).sum()).astype(np.int64)
        row["enum_hist_ok"] = (
            h.tolist() == np.bincount(np.minimum(w, D + 1), minlength=D + 2).tolist()
        )
        row["enum_min_ok"] = mn == min(int(w.min()), D + 1)
    return row


def limit_case(name):
    vm = read_dimacs(DATA / "dimacs" / f"{name}.dimacs")
    row = dict(system=name, kind="limit", features=vm.nvars, clauses=len(vm.clauses),
               cap_gib=contract.CAP_BYTES / 2**30)  # fmt: skip
    t0 = time.perf_counter()
    pl = contract.plan(vm)
    row |= dict(width_n=pl.width, peak_elements=float(pl.peak), plan_s=time.perf_counter() - t0)
    try:
        contract.count(vm, vm.nvars, pl)
        row["contracted"] = True
    except contract.TooBig as e:
        row["contracted"] = False
        row["reason"] = str(e)
    t0 = time.perf_counter()
    try:
        row["ganak"] = str(ganak.count(vm, timeout=1800))
    except Exception as e:
        row["ganak"] = f"no result: {type(e).__name__}"
    row["s_ganak"] = time.perf_counter() - t0
    return row


def ganak_case(name):
    vm = read_dimacs(DATA / "dimacs" / f"{name}.dimacs")
    t0 = time.perf_counter()
    return dict(system=name, kind="limit_ganak", ganak=str(ganak.count(vm, timeout=1800)),
                s_ganak=time.perf_counter() - t0)  # fmt: skip


def isolated(f, arg, q, layout):
    # one case at a time, with all BLAS threads on the whole chip or on the V-Cache CCD
    cpus = hw.pin_vcache() if layout == "vcache" else None
    n = min(BLAS, os.process_cpu_count() or 1)
    threadpool_limits(n)
    env = dict(layout=layout, blas=n, cpus=sorted(cpus) if cpus else None)
    try:
        q.put(f(arg) | env)
    except Exception as e:
        q.put(dict(system=arg, error=f"{type(e).__name__}: {e}") | env)


def layouts(part):
    return ("full", "vcache") if part in ("study", "dimacs") and hw.vcache_cpus() else ("full",)


def fastest(runs):
    """The exact run with the least contraction time; the times of every run are kept."""
    exact = [
        r for r in runs if "error" not in r and all(v for k, v in r.items() if k.endswith("_ok"))
    ]
    best = min(
        exact or runs, key=lambda r: sum(r.get(k) or 0 for k in ("s_count", "s_hist", "s_min"))
    )
    keep = ("layout", "blas", "s_count", "s_hist", "s_min", "error")
    return best | {"runs": [{k: r[k] for k in keep if k in r} for r in runs]}


def run(f, arg, timeout, layout="full"):
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    pr = ctx.Process(target=isolated, args=(f, arg, q, layout))
    pr.start()
    pr.join(timeout)
    if pr.is_alive():
        pr.terminate()
        pr.join()
        return dict(system=arg, error=f"no result in {timeout} s")
    return q.get()


def main(parts):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "rq4.json"
    rows = json.loads(path.read_text())["rows"] if path.exists() else []
    jobs = {
        "study": (study_case, list(SYSTEMS), "study"),
        "dimacs": (dimacs_case, DIMACS, "dimacs_synthetic"),
        "limits": (limit_case, LIMITS, "limit"),
        "ganak": (ganak_case, LIMITS, "limit_ganak"),
    }
    for part in parts:
        name, _, only = part.partition(":")
        f, names, kind = jobs[name]
        for s in [only] if only else names:
            row = fastest([run(f, s, 3600, lay) for lay in layouts(name)]) | {"kind": kind}
            rows = [r for r in rows if (r.get("kind"), r.get("system")) != (kind, s)] + [row]
            print(s, row.get("error", "ok"), flush=True)
            meta = runinfo.snapshot(
                {"max_width": MAX_WIDTH, "cap_gib": contract.CAP_BYTES / 2**30}, "0"
            )
            path.write_text(json.dumps({"rows": rows, "meta": meta}, indent=2, default=str))


if __name__ == "__main__":
    main(sys.argv[1:] or ["study", "dimacs", "limits", "ganak"])
