import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import betabinom
from threadpoolctl import threadpool_limits

from cpq import conformal, hw
from cpq.data import SYSTEMS, load
from cpq.predictors import fit_grid

OUT = Path(os.environ.get("CPQ_OUT", Path(__file__).resolve().parents[1] / "results/raw")) / "p4"
ALPHA, DELTA = 0.1, 0.05
_pops = {}


def one(args):
    system, n, rep = args
    if system not in _pops:
        _pops[system] = load(system)
    p = _pops[system]
    rng = np.random.default_rng(np.random.SeedSequence([list(SYSTEMS).index(system), n, rep, 4]))
    perm = rng.permutation(len(p))
    h = n // 2
    train, cal, u = perm[:h], perm[h:n], perm[n:]
    g = fit_grid(p.X[train], p.y[train], seed=rep)
    res = np.abs(p.y - g.predict(p.X))
    rc, ru = res[cal], res[u]
    m = len(cal)
    k = int(np.ceil((m + 1) * (1 - ALPHA)))
    q = conformal.qhat(rc, ALPHA)
    xmax = conformal.pac_missed(m, len(u), ALPHA, DELTA)
    key_c, key_u = rng.random(m), rng.random(len(u))
    o = np.lexsort((key_c, rc))[k - 1]
    rk, kk = rc[o], key_c[o]
    rest = np.concatenate([rc, ru])
    return dict(
        system=system, n=n, rep=rep, m=m, n_u=len(u), xmax=int(xmax),
        x_strict=int((ru > q).sum()), x_tied=int(((ru > rk) | ((ru == rk) & (key_u > kk))).sum()),
        at_qhat=int((ru == q).sum()), dup_values=int(len(rest) - len(np.unique(rest))),
        dup_yhat=int(len(u) - len(np.unique(np.round(g.predict(p.X[u]), 9)))),
    )  # fmt: skip


def summary(df):
    rows = []
    for (system, n), d in df.groupby(["system", "n"]):
        m, n_u, xmax = int(d.m.iloc[0]), int(d.n_u.iloc[0]), int(d.xmax.iloc[0])
        k = conformal.n_missed(m, ALPHA)
        r = len(d)
        row = dict(system=system, n=n, reps=r, xmax=xmax)
        row["expected_ok"] = float(betabinom.cdf(xmax, n_u, k, m + 1 - k))
        for v in ("strict", "tied"):
            ok = float((d[f"x_{v}"] <= xmax).mean())
            row[f"ok_{v}"] = ok
            row[f"mc_{v}"] = float(np.sqrt(ok * (1 - ok) / r))
        row["mean_x_frac"] = float((d.x_strict / d.n_u).mean())
        row["target_frac"] = k / (m + 1)
        row["dup_values"] = float(d.dup_values.mean())
        row["at_qhat"] = float(d.at_qhat.mean())
        row["differs"] = float((d.x_strict != d.x_tied).mean())
        rows.append(row)
    return rows


def main(reps):
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(s, n, r) for s in ("x264", "berkeleydb") for n in (50, 100, 200) for r in range(reps)]
    workers = int(os.environ.get("CPQ_WORKERS", hw.physical_cores()))
    with ProcessPoolExecutor(
        workers, mp_context=mp.get_context("fork"), initializer=lambda: threadpool_limits(1)
    ) as ex:
        df = pd.DataFrame(list(ex.map(one, jobs, chunksize=8)))
    df.to_parquet(OUT / "p4.parquet", compression="zstd")
    (OUT / "summary.json").write_text(json.dumps(summary(df), indent=2))
    print(pd.DataFrame(summary(df)).round(3).to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
