import json
import sys

import numpy as np
from rq4 import MAX_WIDTH, OUT

from cpq import conformal, labels, runinfo
from cpq.data import SYSTEMS, load
from cpq.predictors import fit_grid
from cpq.tn import contract

ALPHA, PAC_DELTA = 0.1, 0.05


def index(v, delta, c0, D):
    return np.minimum(np.rint(v / delta).astype(np.int64) - c0, D + 1)


def prefix(h, k, D):
    return contract.prefix(h, min(k, D)) if k >= 0 else 0


def case(system):
    p = load(system)
    N, nf = len(p), len(p.vm.names)
    rng = np.random.default_rng(np.random.SeedSequence([list(SYSTEMS).index(system), 200, 0]))
    perm = rng.permutation(N)
    g = fit_grid(p.X[perm[:100]], p.y[perm[:100]], seed=0)
    while True:
        pl = contract.plan(p.vm, g)
        if pl.width <= MAX_WIDTH or len(g.pairs) == 0:
            break
        g = g.top_pairs(len(g.pairs) - 1)
    ymin = float(p.y[perm[:200]].min())
    c0 = min(g.low() + contract.minimum(pl, None, 0), int(np.floor(ymin / g.delta)))
    base = g.low() - c0
    cal = perm[100:200]
    yhat = g.predict(p.X)
    qh = conformal.qhat(np.abs(p.y[cal] - yhat[cal]), ALPHA)
    t = float(np.percentile(p.y, 75))
    D = max(int(np.floor((t + qh) / g.delta)) - c0, 1)
    meas = np.zeros(N, dtype=bool)
    meas[perm[:200]] = True
    u = ~meas
    h = contract.histogram(pl, D, nf, base)
    eye = np.eye(D + 2, dtype=np.int64)
    w, wm = index(yhat, g.delta, c0, D), index(p.y, g.delta, c0, D)
    pred_m, meas_m = eye[w[meas]], eye[wm[meas]]
    hc = labels.hybrid(h, pred_m, meas_m)
    hu = hc - meas_m.sum(axis=0)
    k_lo = int(np.floor((t - qh) / g.delta)) - c0
    k_hi = int(np.floor((t + qh) / g.delta)) - c0
    n_lo, n_hi = prefix(hu, k_lo, D), prefix(hu, k_hi, D)
    e_lo, e_hi = int((yhat[u] + qh <= t).sum()), int((yhat[u] - qh <= t).sum())
    xmax = conformal.pac_missed(len(cal), int(u.sum()), ALPHA, PAC_DELTA)
    c = labels.count(
        yhat, p.y, meas, np.zeros(N, dtype=bool), qh, t, len(cal), ALPHA, PAC_DELTA, 0.0
    )
    m_sat = int((p.y[meas] <= t).sum())
    low = m_sat + max(n_lo - xmax, 0)
    high = m_sat + min(n_hi + xmax, int(u.sum()))
    best_h = int(w[u].min())
    mn = contract.minimum(pl, D, base)
    found, extra = labels.optimise(yhat, p.y, meas, qh)
    return dict(
        system=system, features=nf, pairs=len(g.pairs), delta=g.delta, width=pl.width, D=D,
        t=t, qhat=qh, c0=c0, dtype_hist=str(h.dtype), dtype_corr=str(hc.dtype),
        n_minus=n_lo, n_minus_enum=e_lo, n_plus=n_hi, n_plus_enum=e_hi,
        eq_minus=n_lo == e_lo, eq_plus=n_hi == e_hi,
        low=low, low_count=c.low, high=high, high_count=c.high,
        eq_bounds=(low, high) == (c.low, c.high),
        meas_exact=m_sat, meas_rounded=int((wm[meas] <= int(np.floor(t / g.delta)) - c0).sum()),
        mindeg_corr=contract.mindeg(hc), mindeg_corr_enum=min(int(wm[meas].min()), best_h),
        mindeg_h=contract.mindeg(h), tropical=min(mn, D + 1),
        found=found, extra=extra,
    )  # fmt: skip


def main():
    ref = {
        r["system"]: r
        for r in json.loads((OUT / "rq4.json").read_text())["rows"]
        if r.get("kind") == "study"
    }
    rows = []
    for system in SYSTEMS:
        r = case(system)
        old = ref[system]
        r["same_as_rq4"] = (r["D"], r["delta"], r["pairs"]) == (
            old["D"],
            old["delta"],
            old["pairs"],
        )
        r["width_rq4"] = old["width_weighted"]
        r["eq_mindeg"] = r["mindeg_corr"] == r["mindeg_corr_enum"]
        r["eq_td"] = r["mindeg_h"] == r["tropical"]
        rows.append(r)
        print(system, {k: v for k, v in r.items() if k.startswith(("eq_", "same"))}, flush=True)
    meta = runinfo.snapshot({"n": 200, "rep": 0, "percentile": 75, "alpha": ALPHA}, "as rq4")
    (OUT / "correction.json").write_text(json.dumps({"rows": rows, "meta": meta}, indent=2))
    keys = ("same_as_rq4", "eq_minus", "eq_plus", "eq_mindeg", "eq_td")
    sys.exit(0 if all(r[k] for r in rows for k in keys) else 1)


if __name__ == "__main__":
    main()
