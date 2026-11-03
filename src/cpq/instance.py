import numpy as np

from .conformal import qhat
from .data import SYSTEMS, load
from .predictors import fit_grid


def qvm_tables(system: str = "berkeleydb", n: int = 200, seed: int = 0):
    """QVM+ tables of a real population: n measured configurations, predictions on all."""
    p = load(system)
    rng = np.random.default_rng(np.random.SeedSequence([list(SYSTEMS).index(system), n, seed]))
    perm = rng.permutation(len(p))
    meas, rest = np.sort(perm[:n]), np.sort(perm[n:])
    g = fit_grid(p.X[perm[: n // 2]], p.y[perm[: n // 2]], seed=seed)
    yhat = g.predict(p.X)
    qh = qhat(np.abs(p.y[perm[n // 2 : n]] - yhat[perm[n // 2 : n]]), 0.1)
    order = np.concatenate([meas, rest])
    cfg = np.empty(len(p), dtype=np.int64)
    cfg[order] = np.arange(1, len(p) + 1)

    def links(rows, first):
        r, f = np.nonzero(p.X[rows])
        return {"n": len(r), "cols": {first[0]: r + 1, first[1]: f + 1}}

    ms = np.rint(p.y[meas] * 1000).astype(np.int64)
    rows = np.arange(1, n + 1)
    name = lambda v, k: [v] * k  # noqa: E731
    return {
        "Fs": {"n": p.X.shape[1], "cols": {"fname": list(p.vm.names)}},
        "Cfg": {"n": len(p), "cols": {}},
        "CCs": {"n": n, "cols": {"cc_cfg": rows}},
        "VCs": {"n": len(rest), "cols": {"vc_cfg": np.arange(n + 1, len(p) + 1)}},
        "CClinks": links(meas, ("cc", "cf")),
        "VClinks": links(rest, ("vc", "vf")),
        "Qs": {
            "n": n,
            "cols": {
                "qname": name("time", n),
                "domain": name("ms", n),
                "value": ms,
                "hierarchy": name("perf", n),
            },
        },  # fmt: skip
        "QAs": {"n": n, "cols": {"qaname": name("time", n)}},
        "QAlinks": {"n": n, "cols": {"qa": rows, "ql": rows}},
        "QMC": {"n": n, "cols": {"phi": rows, "psi": rows}},
        "Pred": {
            "n": len(p),
            "cols": {
                "pcfg": cfg,
                "pname": name("time", len(p)),
                "yhat": yhat,
                "lo": yhat - qh,
                "hi": yhat + qh,
            },
        },  # fmt: skip
        "Noise": {"n": n, "cols": {"nq": rows, "eps": 0.05 * np.abs(p.y[meas])}},
    }, g
