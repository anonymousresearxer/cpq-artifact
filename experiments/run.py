import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from cpq import conformal, hw, labels, runinfo, selection
from cpq.baselines import methods
from cpq.data import SYSTEMS, load
from cpq.predictors import fit_grid

OUT = Path(os.environ.get("CPQ_OUT", Path(__file__).resolve().parents[1] / "results/raw"))
_pops = {}


def pop(system):
    if system not in _pops:
        _pops[system] = load(system)
    return _pops[system]


def init():
    threadpool_limits(1)


def listed(x):
    return list(x) if isinstance(x, list) else [x]


def fdp(sel, truth):
    return float((~truth[sel]).sum() / max(sel.sum(), 1))


def power(sel, truth):
    return float((truth & sel).sum() / truth.sum()) if truth.any() else np.nan


def optimise_rows(base, p, yhat, meas, qh, n, rng):
    y = p.y
    ymin = float(y.min())
    found, extra = labels.optimise(yhat, y, meas, qh)
    ret = float(y[np.argmin(yhat)])
    rows = {
        "ours": (found, extra),
        "measured_only": (float(y[meas].min()), 0),
        "predictor_only": (ret, 0),
        "uniform_n": (float(y[rng.choice(len(y), n, replace=False)].min()), 0),
        "uniform_total": (float(y[rng.choice(len(y), min(n + extra, len(y)), False)].min()), 0),
    }  # fmt: skip
    totals = {"uniform_total": n + extra, "ours": n + extra}
    return [
        base
        | dict(method=k, found=f, extra=e, total=totals.get(k, n), regret=f - ymin,
               rel_regret=(f - ymin) / ymin)
        for k, (f, e) in rows.items()
    ]  # fmt: skip


def compare_rows(base, p, yhat, y, meas, cal, u, sel, qh, t, eps, m, alpha, pac_delta):
    N = len(y)
    truth_mask = y <= t
    truth = int(truth_mask.sum())
    lab = labels.label(yhat, y, meas, sel, qh, t, eps)
    c = labels.count(yhat, y, meas, sel, qh, t, m, alpha, pac_delta, eps)
    hyb = labels.hybrid(
        int((yhat <= t).sum()), (yhat[meas] <= t).astype(np.int64), (y[meas] <= t).astype(np.int64)
    )
    n_meas = int(meas.sum())
    meas_sat = meas & (lab == labels.CERTAIN)
    lo_u, hi_u = u[yhat[u] + qh <= t].size, u[yhat[u] - qh <= t].size
    naive_lo = int(meas_sat.sum()) + lo_u
    naive_hi = int((meas & (lab >= labels.POSSIBLE)).sum()) + hi_u
    f_u = (yhat[u] <= t).astype(float)
    th, tl, tu = methods.ppi_mean(f_u, (yhat[cal] <= t).astype(float), (y[cal] <= t).astype(float))
    known = int((meas & truth_mask).sum())
    wl, wh = methods.wilson(int(meas_sat.sum()), n_meas)
    sel_u = np.zeros(N, dtype=bool)
    sel_u[u[yhat[u] + qh <= t]] = True
    sets = {
        "ours": lab == labels.CERTAIN,
        "measured_only": meas_sat,
        "predictor_only": yhat <= t,
        "intervals": meas_sat | sel_u,
        "ppi": None,
    }
    est = {
        "ours": (hyb, c.low, c.high),
        "measured_only": (N * meas_sat.sum() / n_meas, N * wl, N * wh),
        "predictor_only": ((yhat <= t).sum(), np.nan, np.nan),
        "intervals": (np.nan, naive_lo, naive_hi),
        "ppi": (known + len(u) * th, known + len(u) * tl, known + len(u) * tu),
    }
    rows = []
    for k, (e, lo, hi) in est.items():
        ns, prec, rec = (
            (0, np.nan, np.nan) if sets[k] is None else methods.returned(sets[k], truth_mask)
        )
        has = not np.isnan(lo)
        err = abs(e - truth) / N if not np.isnan(e) else methods.interval_error(truth, lo, hi) / N
        rows.append(
            base
            | dict(
                method=k, eps=eps, n_ret=ns, prec=prec, recall=rec, est=e, lo=lo, hi=hi, err=err,
                covers=(lo <= truth <= hi) if has else np.nan,
                width=(hi - lo) / N if has else np.nan, truth=truth,
            )
        )  # fmt: skip
    return rows


def noise(sid, n, rep, N):
    """Uniform(0, 1) per configuration, from its own stream: independent of y and of the split."""
    return np.random.default_rng([sid, n, rep, 11]).random(N)


def one(args):
    system, n, rep, cfg = args
    p = pop(system)
    alpha, pac_delta = cfg["alpha"], cfg["pac_delta"]
    qs = listed(cfg["q"])
    sens = cfg.get("q_sens")
    if sens and (sens["systems"] == "all" or system in sens["systems"]) and n in sens["n"]:
        qs = qs + listed(sens["q"])
    epss = listed(cfg["eps"])
    sid = list(SYSTEMS).index(system)
    ss = np.random.SeedSequence([sid, n, rep])
    rng = np.random.default_rng(ss)
    N = len(p)
    perm = rng.permutation(N)
    h = n // 2
    train, cal, u = perm[:h], perm[h:n], perm[n:]
    meas = np.zeros(N, dtype=bool)
    meas[perm[:n]] = True
    y = p.y
    g = fit_grid(p.X[train], y[train], seed=rep)
    yhat = g.predict(p.X)
    xi = g.delta / 2 * noise(sid, n, rep, N)
    qh = conformal.qhat(np.abs(y[cal] - yhat[cal]), alpha)
    m = len(cal)
    xmax = conformal.pac_missed(m, len(u), alpha, pac_delta)
    missed = int((np.abs(y[u] - yhat[u]) > qh).sum())
    base = dict(
        system=system, n=n, rep=rep, m=m, n_u=len(u), alpha=alpha, qhat=qh, delta=g.delta,
        n_pairs=len(g.pairs), missed=missed, missed_max=xmax,
        mae_grid=float(np.abs(y - yhat).mean()),
        mae_raw=float(np.abs(y - g.predict_raw(p.X)).mean()), n_valid=N,
    )  # fmt: skip
    opt = optimise_rows(base, p, yhat, meas, qh, n, np.random.default_rng([sid, n, rep, 7]))
    best = next(r for r in opt if r["method"] == "ours")
    sel_rows, cmp_rows = [], []
    for qi, q in enumerate(qs):
        for pct in cfg["percentiles"]:
            t = float(np.percentile(y, pct))
            sel = np.zeros(N, dtype=bool)
            msel = np.zeros(N, dtype=bool)
            sel[u] = selection.select(yhat[cal], y[cal], yhat[u], t, q, xi[cal], xi[u], g.delta)
            msel[u] = selection.select(
                yhat[cal], y[cal], yhat[u], t, q, xi[cal], xi[u], g.delta, mirror=True
            )
            ok = y[u] <= t
            s, ms = sel[u], msel[u]
            c = labels.count(yhat, y, meas, sel, qh, t, m, alpha, pac_delta, epss[0])
            truth = int((y <= t).sum())
            pred = labels.hybrid(
                int((yhat <= t).sum()),
                (yhat[meas] <= t).astype(np.int64),
                (y[meas] <= t).astype(np.int64),
            )
            lab = labels.label(yhat, y, meas, sel, qh, t, epss[0])
            sel_rows.append(
                base
                | dict(
                    q=q, pct=pct, t=t, found=best["found"], extra=best["extra"],
                    regret=best["regret"], n_sel=int(s.sum()), fdp=fdp(s, ok), power=power(s, ok),
                    n_msel=int(ms.sum()), mfdp=fdp(ms, ~ok), mpower=power(ms, ~ok),
                    both=int((sel & msel).sum()), truth=truth, certain=c.certain,
                    possible=c.possible, low=c.low, high=c.high, pred=int(pred),
                    undecided=float((lab == labels.POSSIBLE).mean()),
                )
            )  # fmt: skip
            if qi == 0:
                for eps in epss:
                    ctx = base | dict(q=q, pct=pct, t=t)
                    cmp_rows += compare_rows(
                        ctx, p, yhat, y, meas, cal, u, sel, qh, t, eps, m, alpha, pac_delta
                    )
    return dict(sel=sel_rows, cmp=cmp_rows, opt=opt)


def main(path):
    cfg = yaml.safe_load(Path(path).read_text())
    systems = list(SYSTEMS) if cfg["systems"] == "all" else cfg["systems"]
    out = OUT / cfg["name"]
    workers = int(os.environ.get("CPQ_WORKERS", hw.physical_cores()))
    jobs = [(s, n, r, cfg) for s in systems for n in cfg["n"] for r in range(cfg["reps"])]
    t0 = time.time()
    if workers > 1:
        with ProcessPoolExecutor(
            workers, mp_context=mp.get_context("fork"), initializer=init
        ) as ex:
            res = list(ex.map(one, jobs, chunksize=4))
    else:
        init()
        res = [one(j) for j in jobs]
    for kind in ("sel", "cmp", "opt"):
        d = out / kind
        d.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([r for x in res for r in x[kind]])
        for system, part in df.groupby("system"):
            part.to_parquet(d / f"{system}.parquet", compression="zstd")
    meta = runinfo.snapshot(cfg, "SeedSequence([system index, n, rep])")
    meta |= {"seconds": time.time() - t0, "peak_mb": runinfo.peak_mb(), "workers": workers}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{len(jobs)} tasks in {meta['seconds']:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1])
