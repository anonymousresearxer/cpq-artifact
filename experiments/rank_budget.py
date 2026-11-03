import json
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from run import OUT, init, noise, pop

from cpq import budget, conformal, hw, labels, runinfo
from cpq.data import SYSTEMS
from cpq.predictors import fit_grid

PCT = (1, 2, 5, 10)
GRID = Path(__file__).resolve().parent / "grid.yaml"


def one(args):
    system, n, rep, alpha = args
    p = pop(system)
    sid = list(SYSTEMS).index(system)
    rng = np.random.default_rng(np.random.SeedSequence([sid, n, rep]))
    N = len(p)
    perm = rng.permutation(N)
    h = n // 2
    train, cal, u = perm[:h], perm[h:n], perm[n:]
    meas = np.zeros(N, dtype=bool)
    meas[perm[:n]] = True
    y = p.y
    g = fit_grid(p.X[train], y[train], seed=rep)
    yhat = g.predict(p.X)
    qh = conformal.qhat(np.abs(y[cal] - yhat[cal]), alpha)
    ymin = float(y.min())
    order = budget.rank(yhat, noise(sid, n, rep, N), u)
    best_m = float(y[meas].min())
    opt = np.flatnonzero(y == ymin)
    in_m = bool(meas[opt].any())
    pos = None
    if not in_m:
        r = np.empty(N, dtype=np.int64)
        r[order] = np.arange(1, len(order) + 1)
        pos = float(r[opt].min() / N)

    def uniform(k):
        s = np.random.default_rng([sid, n, rep, 7])
        s.choice(N, n, replace=False)
        return float(y[s.choice(N, min(n + k, N), replace=False)].min())

    found, extra = labels.optimise(yhat, y, meas, qh)
    rows = []
    for pct in PCT:
        k = min(-(-pct * N // 100), len(u))
        rows.append(
            dict(pct=pct, k=k, rank=min(best_m, float(y[order[:k]].min())), unif=uniform(k))
        )
    return dict(system=system, n=n, rep=rep, ymin=ymin, rows=rows, in_m=in_m, pos=pos,
                found=found, extra=extra, unif_extra=uniform(extra))  # fmt: skip


def main():
    cfg = yaml.safe_load(GRID.read_text())
    workers = int(os.environ.get("CPQ_WORKERS", hw.physical_cores()))
    jobs = [(s, n, r, cfg["alpha"]) for s in SYSTEMS for n in cfg["n"] for r in range(cfg["reps"])]
    t0 = time.time()
    if workers > 1:
        with ProcessPoolExecutor(
            workers, mp_context=mp.get_context("fork"), initializer=init
        ) as ex:
            res = list(ex.map(one, jobs, chunksize=4))
    else:
        init()
        res = [one(j) for j in jobs]
    cells = defaultdict(list)
    for r in res:
        for row in r["rows"]:
            ymin = r["ymin"]
            cells[(r["system"], r["n"], row["pct"])].append(
                (row["rank"] == ymin, (row["rank"] - ymin) / ymin,
                 row["unif"] == ymin, (row["unif"] - ymin) / ymin, row["k"])
            )  # fmt: skip
    out = [
        dict(system=s, n=n, pct=pct, k=v[0][4], hit_rank=float(np.mean([x[0] for x in v])),
             rel_rank=float(np.mean([x[1] for x in v])), hit_unif=float(np.mean([x[2] for x in v])),
             rel_unif=float(np.mean([x[3] for x in v])))
        for (s, n, pct), v in cells.items()
    ]  # fmt: skip
    pos = []
    for s in SYSTEMS:
        for n in cfg["n"]:
            rs = [r for r in res if r["system"] == s and r["n"] == n]
            ps = [r["pos"] for r in rs if r["pos"] is not None]
            median = float(np.median(ps)) if ps else None
            pos.append(dict(system=s, n=n, in_m=float(np.mean([r["in_m"] for r in rs])),
                            n_pos=len(ps), median_pos=median))  # fmt: skip
    grid = pd.concat([pd.read_parquet(f) for f in sorted((OUT / "grid" / "opt").glob("*.parquet"))])
    key = ["system", "n", "rep"]
    ours = grid[grid.method == "ours"].set_index(key)
    uni = grid[grid.method == "uniform_total"].set_index(key)
    mine = pd.DataFrame([{k: r[k] for k in (*key, "found", "extra", "unif_extra")} for r in res])
    mine = mine.set_index(key)
    d_found = (mine.found - ours.found.reindex(mine.index)).abs()
    d_extra = (mine.extra - ours.extra.reindex(mine.index)).abs()
    d_unif = (mine.unif_extra - uni.found.reindex(mine.index)).abs()
    repro = dict(cases=len(mine), found_equal=int((d_found == 0).sum()),
                 extra_equal=int((d_extra == 0).sum()), unif_equal=int((d_unif == 0).sum()),
                 max_found=float(d_found.max()), max_extra=float(d_extra.max()),
                 max_unif=float(d_unif.max()))  # fmt: skip
    meta = runinfo.snapshot(dict(cfg, pct=PCT), "SeedSequence([system index, n, rep])")
    meta |= {"seconds": time.time() - t0, "workers": workers}
    path = OUT / "rank" / "rank_budget.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(cells=out, pos=pos, repro=repro, meta=meta), indent=2))
    print(f"{len(res)} cases in {meta['seconds']:.0f} s", repro)
    sys.exit(
        0
        if all(repro[k] == repro["cases"] for k in ("found_equal", "extra_equal", "unif_equal"))
        else 1
    )


if __name__ == "__main__":
    main()
