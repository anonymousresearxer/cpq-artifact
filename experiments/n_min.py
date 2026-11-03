import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from cpq import budget, runinfo
from cpq.data import SYSTEMS, load

RAW = Path(os.environ.get("CPQ_OUT", Path(__file__).resolve().parents[1] / "results/raw"))


def main():
    meta = json.loads((RAW / "grid/meta.json").read_text())["params"]
    q, alpha = meta["q"][0], meta["alpha"]
    sel = pd.concat([pd.read_parquet(f) for f in sorted((RAW / "grid/sel").glob("*.parquet"))])
    truth = sel[sel.q == q].groupby(["system", "pct"]).truth.agg(["min", "max"])
    rows = []
    for sid, system in enumerate(SYSTEMS):
        y = load(system).y
        N = len(y)
        for pct in meta["percentiles"]:
            t = float(np.percentile(y, pct))
            total = int((y <= t).sum())
            assert truth.loc[(system, pct)].tolist() == [total, total]
            row = dict(system=system, pct=pct, size=N, count=total, empirical={})
            row["n_min"] = budget.n_min(N, total, q, alpha)
            row["n_min_fdp"] = budget.n_min(N, total, q, alpha, fdp=True)
            for n in meta["n"]:
                u = np.empty(meta["reps"])
                for r in range(meta["reps"]):
                    perm = np.random.default_rng(np.random.SeedSequence([sid, n, r])).permutation(N)
                    u[r] = (y[perm[n:]] <= t).sum()
                exact = total * (N - n) / N
                row["empirical"][n] = dict(mean=float(u.mean()), exact=exact,
                                           rel=float((u.mean() - exact) / exact))  # fmt: skip
            rows.append(row)
            print(system, pct, row["n_min"], row["n_min_fdp"], flush=True)
    meta_run = runinfo.snapshot(dict(meta, q=q), "SeedSequence([system index, n, rep])")
    path = RAW / "nmin" / "nmin.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(rows=rows, q=q, alpha=alpha, meta=meta_run), indent=2))


if __name__ == "__main__":
    main()
