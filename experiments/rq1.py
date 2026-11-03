import json
import os
import time
from pathlib import Path

import numpy as np

from cpq import bridge, instance, runinfo, spec

OUT = Path(os.environ.get("CPQ_OUT", Path(__file__).resolve().parents[1] / "results/raw")) / "rq1"
QUERIES = ["cfg", "measured", "hist", "pairs", "pairs_vc", "pred", "noise"]
D = 20


def annotations(tabs):
    yh = tabs["Pred"]["cols"]["yhat"]
    if not len(yh):
        return {}
    step = max((yh.max() - yh.min()) / 16, 1e-9)
    return {"Pred": np.rint((yh - yh.min()) / step).astype(np.int64)}


def timed(f, *a):
    t = time.perf_counter()
    r = f(*a)
    return r, time.perf_counter() - t


def pair_semiring(tabs, s):
    """N x N on the pred query: certain (hi <= t) and possible (lo <= t) unmeasured rows."""
    cols = tabs["Pred"]["cols"]
    t = float(np.median(cols["yhat"]))
    counts = []
    for mask in (cols["hi"] <= t, cols["lo"] <= t, np.ones(len(cols["yhat"]), dtype=bool)):
        sub = dict(tabs)
        sub["Pred"] = {"n": int(mask.sum()), "cols": {c: v[mask] for c, v in cols.items()}}
        counts.append(spec.evaluate(s, sub, None, 1)["N"].get((), 0))
    return counts[0] <= counts[1] <= counts[2], counts


def run_instance(name, inst, rows):
    tabs = bridge.tables(inst)
    ann = annotations(tabs)
    for q in QUERIES:
        s = bridge.query_spec(q)
        want, t_cat = timed(bridge.answer, inst, q)
        got_duck, t_duck = timed(spec.duck, s, tabs)
        res, t_sem = timed(spec.evaluate, s, tabs, ann, D)
        row = dict(instance=name, query=q, rows=len(want), catlab_s=t_cat, duckdb_s=t_duck)
        row |= dict(semiring_s=t_sem, duck_ok=got_duck == want, n_ok=res["N"] == want)
        row |= dict(identities=spec.identities(res, D))
        if q == "pred" and len(tabs["Pred"]["cols"]["yhat"]):
            row["nn_ok"], row["nn"] = pair_semiring(tabs, s)
        rows.append(row)
        print(name, q, row["duck_ok"], row["n_ok"], row["identities"], flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, chase = [], []
    for name in ("p1", "p2", "p3"):
        run_instance(name, bridge.fixture(name), rows)
    for name in ("tiny", "multi_qa", "measured_p1"):
        run_instance(f"sigma_{name}", bridge.sigma_fixture(name), rows)
        ok, secs = bridge.chase_check(name)
        chase.append(dict(fixture=name, isomorphic=ok, chase_s=secs))
        print("chase", name, ok, secs, flush=True)
    tabs, g = instance.qvm_tables()
    run_instance("berkeleydb_2560", bridge.load_instance(tabs), rows)
    meta = runinfo.snapshot({"D": D, "queries": QUERIES}, "SeedSequence([1, 200, 0])")
    (OUT / "rq1.json").write_text(
        json.dumps({"rows": rows, "chase": chase, "meta": meta}, indent=2)
    )


if __name__ == "__main__":
    main()
