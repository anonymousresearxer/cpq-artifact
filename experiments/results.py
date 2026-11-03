import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("CPQ_OUT", ROOT / "results/raw"))
OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
ORDER = ["x264", "berkeleydb", "lrzip", "7z", "polly", "javagc", "vp9", "llvm", "dune", "hipacc"]
LABEL = {"berkeleydb": "BerkeleyDB-C", "javagc": "JavaGC", "llvm": "LLVM", "hipacc": "Hipacc",
         "polly": "Polly", "dune": "Dune", "vp9": "VP9", "7z": "7z", "x264": "x264", "lrzip": "lrzip"}  # fmt: skip
METHOD = {"ours": "certain/possible", "measured_only": "measured only",
          "predictor_only": "predictor only", "intervals": "intervals", "ppi": "PPI"}  # fmt: skip


def grid(kind):
    return pd.concat([pd.read_parquet(f) for f in sorted((RAW / "grid" / kind).glob("*.parquet"))])


def at(path, needle):
    """path:line of the first line of a source file that contains needle."""
    lines = (ROOT / path).read_text().splitlines()
    return f"`{path}:{next(i for i, x in enumerate(lines, 1) if needle in x)}`"


def tabular(path, cols, rows, groups=(), breaks=()):
    """booktabs tabular* at text width; cols are (header, 'l' | 'c' | siunitx table-format)."""
    spec = "".join(
        "l" if k == "l" else "c" if k == "c" else f"S[table-format={k}]" for _, k in cols
    )
    head = " & ".join(h if k in "lc" else f"{{{h}}}" for h, k in cols)
    out = [
        "{\\footnotesize",
        f"\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}{spec}@{{}}}}",
        "\\toprule",
    ]
    if groups:
        cells, pos = [], 1
        for title, a, b in groups:
            cells += ["" for _ in range(a - pos)] + [
                f"\\multicolumn{{{b - a + 1}}}{{c}}{{{title}}}"
            ]
            pos = b + 1
        cells += ["" for _ in range(len(cols) - pos + 1)]
        out.append(" & ".join(cells).rstrip(" &") + " \\\\")
        out.append("".join(f"\\cmidrule(lr){{{a}-{b}}}" for _, a, b in groups))
    out += [head + " \\\\", "\\midrule"]
    for i, r in enumerate(rows):
        if i in breaks:
            out.append("\\addlinespace")
        out.append(" & ".join(r) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular*}}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n")


NUM = re.compile(r"(--|-?[\d.]+( \+- [\d.]+)?)")


def md(heads, rows):
    cell = lambda c: str(c).replace("|", "\\|")  # noqa: E731
    right = [all(NUM.fullmatch(str(r[i])) for r in rows) for i in range(len(heads))]
    return "\n".join(
        ["| " + " | ".join(cell(h) for h in heads) + " |",
         "|" + "|".join("---:" if r else "---" for r in right) + "|"]
        + ["| " + " | ".join(cell(c) for c in r) + " |" for r in rows]
    )  # fmt: skip


def se(x):
    """Monte Carlo standard error of the mean over repetitions."""
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def se_pooled(d, col):
    """Standard error of the mean of per-system means, each from its own repetitions."""
    v = d.groupby(["system", "rep"])[col].mean().groupby("system")
    return float(np.sqrt((v.var(ddof=1) / v.count()).sum()) / v.ngroups)


def f(x, k=2):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{k}f}"


def rq2_table(sel):
    s = sel[sel.q == 0.1]
    rows = []
    for sysname in ORDER:
        d = s[s.system == sysname]
        p50 = d[d.pct == 50].groupby("n").power.mean()
        p100 = d[(d.n == 100) & (d.pct == 50)]
        rows.append([
            LABEL[sysname], f"{int(d.n_valid.iloc[0])}", f(d.fdp.mean(), 3),
            f(se(d.groupby("rep").fdp.mean()), 3), f(p50[50]), f(p50[100]),
            f(p50[200]), f((d.missed <= d.missed_max).mean()),
            f(((p100.possible - p100.certain) / p100.n_valid).mean()),
        ])  # fmt: skip
    cols = [("System", "l"), ("$|V|$", "6.0"), ("FDR", "1.3"), ("SE", "1.3"), ("50", "1.2"),
            ("100", "1.2"), ("200", "1.2"), ("X ok", "1.2"), ("Gap", "1.2")]  # fmt: skip
    groups = [
        ("FDR at $q=0.1$", 3, 4),
        ("Power at P50, $n=$", 5, 7),
        ("PAC", 8, 8),
        ("Labels", 9, 9),
    ]
    tabular(ROOT / "paper/tables/rq2-guarantees.tex", cols, rows, groups=groups)


def rq4_table(rq4):
    dm = {r["system"]: r for r in rq4 if r.get("kind") == "dimacs_synthetic"}
    lim = {r["system"]: r for r in rq4 if r.get("kind") == "limit"}
    na = "{--}"
    rows = []
    for name, label in (("berkeleydb", "BerkeleyDB"), ("busybox-1.18.0", "BusyBox")):
        r = dm[name]
        rows.append([label, f"{r['features']}", f"{r['width_n']:.0f}", f(1000 * r["s_count"], 1),
                     f(1000 * r["s_hist"], 1), f(r["mb_hist"], 1), f(r["s_ganak"], 3)])  # fmt: skip
    breaks = [len(rows)]
    for name, label in (
        ("financial01", "Financial Services 01"),
        ("linux-2.6.33.3", "Linux 2.6.33.3"),
    ):
        r = lim[name]
        done = not r["ganak"].startswith("no result")
        rows.append([label, f"{r['features']}", f"{r['width_n']:.0f}", na, na, na,
                     f(r["s_ganak"], 3) if done else na])  # fmt: skip
    cols = [("DIMACS model", "l"), ("Vars", "4.0"), ("Width", "3.0"), ("$\\Nat$ (ms)", "3.1"),
            ("$\\Hist$ (ms)", "3.1"), ("MiB", "2.1"), ("ganak (s)", "1.3")]  # fmt: skip
    tabular(ROOT / "paper/tables/rq4-evaluator.tex", cols, rows, breaks=breaks)


def figure(opt):
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
                         "pdf.fonttype": 42, "axes.linewidth": 0.5, "axes.spines.top": False,
                         "axes.spines.right": False})  # fmt: skip
    cm = 1 / 2.54
    fig, ax = plt.subplots(1, 2, figsize=(12.2 * cm, 4.9 * cm), constrained_layout=True)
    methods = [("ours", "screen, then measure", OKABE[0], "o"),
               ("uniform_total", "best of $n'$ uniform", OKABE[1], "s"),
               ("measured_only", "measured only", OKABE[3], "D"),
               ("predictor_only", "predictor only", OKABE[4], "v")]  # fmt: skip
    ns = sorted(opt.n.unique())
    share = opt[opt.method == "ours"].assign(s=lambda d: d.total / d.n_valid).groupby("n").s.mean()
    for m, name, c, mk in methods:
        d = opt[opt.method == m].groupby("n")
        hit = d.regret.apply(lambda r: (r == 0).mean())
        ax[0].plot(ns, [hit[n] for n in ns], color=c, marker=mk, ms=4, lw=1, label=name)
        ax[1].plot(ns, [d.rel_regret.mean()[n] for n in ns], color=c, marker=mk, ms=4, lw=1)
    ax[0].set_ylabel("optimum found")
    ax[1].set_ylabel("relative regret")
    ax[1].set_yscale("log")
    ax[1].set_yticks([0.01, 0.1, 1], ["0.01", "0.1", "1"])
    ax[1].set_yticks([], minor=True)
    ax[1].set_ylim(0.01, 6)
    for a in ax:
        a.set_xlabel("initial budget $n$")
        a.set_xticks(ns, [f"{n}\n{share[n]:.2f}" for n in ns])
        a.tick_params(width=0.5, length=2, labelsize=8)
    fig.legend(
        *ax[0].get_legend_handles_labels(),
        frameon=False,
        fontsize=8,
        loc="outside upper center",
        ncol=4,
        handlelength=1.0,
        handletextpad=0.4,
        columnspacing=0.8,
    )
    path = ROOT / "paper/figs/rq3-optimise.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"Creator": None, "Producer": None, "CreationDate": None})
    plt.close(fig)


def running_case(sel, options):
    from cpq.data import load

    y = load("berkeleydb").y
    n_v, n = len(y), 50
    ps = {k: float(np.percentile(y, k)) for k in (25, 50, 75)}
    c = sel[(sel.system == "berkeleydb") & (sel.n == n) & (sel.pct == 25) & (sel.q == 0.1)]
    assert np.isclose(c.t.iloc[0], ps[25])
    below = int((y <= ps[25]).sum())
    assert below == c.truth.max()
    m, n_u, q = int(c.m.iloc[0]), int(c.n_u.iloc[0]), 0.1
    bound = math.ceil(n_u / (q * (m + 1)))
    wrong = math.floor(100 * (1 - below / bound))
    ne = c[c.n_sel > 0]
    low = int(ne.n_sel.min()) if len(ne) else None
    return [
        "### BerkeleyDB-C: running case\n",
        f"Our running case is BerkeleyDB-C as the embedded store of an information system. Its VM has {options} binary options and {n_v:,} valid configurations, and the response time of every one of them was measured~\\cite{{kaltenecker2019icse}} (Table~\\ref{{tab:bdb}}). Since the whole population is known, every answer of our model can be checked against the truth. The policy at hand is \\emph{{response time $\\le t$}}, with $t$ at the 25th, 50th and 75th percentiles of the population. After measuring only $n\\in\\{{50,100,200\\}}$ configurations, an operator must decide which unmeasured ones to declare compliant, how many possibly comply and which is fastest.\n",
        f"With $n={n}$, BH with the deterministic p-values of Theorem 6 (smallest value $1/(m+1)$) can select at the 25th percentile of BerkeleyDB-C only by returning at least {bound} configurations, while at most {below} comply, so \\emph{{certain}} would be at least {wrong} \\% wrong; there the count interval carries the answer.\n",
        "Response times in seconds over the whole population:\n",
        md(["Options", "|V|", "Measured", "Min. (s)", "P25 (s)", "P50 (s)", "P75 (s)", "Max. (s)"],
           [[options, n_v, n_v, f(y.min()), f(ps[25]), f(ps[50]), f(ps[75]), f(y.max())]]),
        "\nPercentiles are `numpy.percentile` with linear interpolation over all |V| response times; "
        f"#{{y <= P25}} = {below}, equal to the largest `truth` in the cell below.\n",
        f"Cell: BerkeleyDB-C, n = {n}, P25, q = {q}, {len(c)} repetitions (`results/raw/grid/sel/berkeleydb.parquet`, "
        "`q == 0.1`, `pct == 25`, `n == 50`). `n_sel` is the size of the BH selection of "
        "y <= t over the unmeasured set U (the configurations declared certain). `n_msel` is the size of the "
        "mirror selection (`select(..., mirror=True)` in `experiments/run.py`): BH on the scores oriented to "
        "y > t, a diagnostic of the opposite side (D-039). It is not corrected by M and does not include "
        "measured configurations. Both use the deterministic p-value (1 + #{i in Cal: S_i < S_j}) / (m + 1), "
        "with ties broken by adding to every score of Cal and U a per-configuration xi ~ Unif(0, gamma / 2), with gamma the grid step of the predictor "
        "(D-054).\n",
        md(["quantity", "value"], [
            ["repetitions with n_sel = 0", f(float((c.n_sel == 0).mean()), 2)],
            ["min n_sel over non-empty repetitions", low if low is not None else "--"],
            ["max n_sel", int(c.n_sel.max())],
            ["min n_msel (mirror selection, all repetitions)", int(c.n_msel.min())],
            ["max truth = #{y <= P25}", int(c.truth.max())],
            ["computed bound ceil(|U| / (q (m + 1))), m = n // 2, |U| = |V| - n", bound],
        ]),
        f"\nThe bound is computed from |U| = {n_u}, m = {m} and q = {q}, not observed: every p-value is "
        "at least 1 / (m + 1), so BH rejects k configurations only if k >= |U| / (q (m + 1)). "
        + (f"Observed minimum n_sel among non-empty selections: {low} "
           f"({'at or above' if low >= bound else 'below'} the bound).\n" if low is not None
           else "No repetition selects anything.\n"),
    ]  # fmt: skip


def write():
    from cpq.data import SYSTEMS

    sel, cmp_, opt = grid("sel"), grid("cmp"), grid("opt")
    meta = json.loads((RAW / "grid/meta.json").read_text())
    rq1 = json.loads((RAW / "rq1/rq1.json").read_text())
    rq4 = json.loads((RAW / "rq4/rq4.json").read_text())["rows"]
    p4 = json.loads((RAW / "p4/summary.json").read_text())
    mpath = RAW / "mutants/mutants.json"
    mut = json.loads(mpath.read_text()) if mpath.exists() else []
    rq2_table(sel)
    rq4_table(rq4)
    figure(opt)
    study = {r["system"]: r for r in rq4 if r.get("kind") == "study"}
    L = ["# Results\n"]
    L.append(
        f"Grid: commit {meta['commit'][:8]}, {meta['cpu']}, {meta['seconds']:.0f} s, "
        f"{meta['params']['reps']} repetitions, alpha = 0.1, q = 0.1, delta = 0.05.\n"
    )
    pops = sel.groupby("system").n_valid.first()
    L += ["## Subjects\n", md(["system", "|V|", "options", "response"],
          [[LABEL[x], int(pops[x]), study[x]["features"], SYSTEMS[x][1]] for x in ORDER])]  # fmt: skip
    L.append("\nVP9 ships no Performance column; its response is the encoder wall-clock time "
             "(ElapsedTime). JavaGC uses GC Time.\n")  # fmt: skip

    L += running_case(sel, study["berkeleydb"]["features"])
    L.append("## P4 check (1000 repetitions)\n")
    L.append(md(["system", "n", "X ok", "tie-broken", "expected", "tied values", "differs"],
                [[LABEL[r["system"]], r["n"], f"{r['ok_strict']:.3f} +- {r['mc_strict']:.3f}",
                  f"{r['ok_tied']:.3f} +- {r['mc_tied']:.3f}", f"{r['expected_ok']:.3f}",
                  f"{r['dup_values']:.1f}", f"{r['differs']:.3f}"] for r in p4]))  # fmt: skip
    L.append("\nX ok is the share of repetitions with at most the 0.95-quantile of mis-covered "
             "configurations; +- is the Monte Carlo error. Tied values is the mean number of equal "
             "residuals per repetition; tie-broken draws a random rank among equal residuals.\n")  # fmt: skip

    rows = rq1["rows"]
    agree = sum(r["duck_ok"] and r["n_ok"] and not r["identities"] for r in rows)
    L.append("## RQ1\n")
    L.append(f"{agree} of {len(rows)} (instance, query) cases agree exactly between Catlab `query`, "
             f"DuckDB and the semiring evaluation (N), and satisfy the homomorphism identities "
             f"(ev1, mindeg, supp). Instances: 3 QVM+ fixtures, 3 closed-form images and "
             f"BerkeleyDB-C with 2560 configurations. The chase equals the closed form on "
             f"{sum(c['isomorphic'] for c in rq1['chase'])} of {len(rq1['chase'])} fixtures "
             f"({', '.join(f'{c['chase_s']:.0f} s' for c in rq1['chase'])}); it is not run at 2560.\n")  # fmt: skip
    big = [r for r in rows if r["instance"] == "berkeleydb_2560"]
    L.append(md(["query", "rows", "Catlab (s)", "DuckDB (s)", "semiring (s)"],
                [[r["query"], r["rows"], f(r["catlab_s"], 3), f(r["duckdb_s"], 3),
                  f(r["semiring_s"], 3)] for r in big]))  # fmt: skip
    L.append(
        "\nThe times are measured on the BerkeleyDB-C instance with 2560 configurations "
        "(`berkeleydb_2560` in `experiments/rq1.py`, the last instance of the run), one call "
        "each timed with `time.perf_counter` after the six fixtures in the same process. "
        "Semiring is `spec.evaluate`, which evaluates N, H_D, T_D and the Boolean semiring "
        f"with D = {rq1['meta']['params']['D']}. Every time in this section, the chase times "
        "included, is a single measurement: there are no repetitions, so no mean, median or "
        "minimum.\n"
    )
    nn = [r for r in rows if r.get("nn_ok") is not None]
    L.append(f"\nThree N counts on the pred query, #{{hi <= t}} <= #{{lo <= t}} <= all rows "
             f"(enumerated regime): {sum(r['nn_ok'] for r in nn)} "
             f"of {len(nn)} cases.\n")  # fmt: skip

    s = sel[sel.q == 0.1]
    g = s.groupby(["n", "pct"])
    t = pd.DataFrame({"fdr": g.fdp.mean(), "power": g.power.mean(),
                      "fdr_se": g.apply(lambda d: se_pooled(d, "fdp"), include_groups=False),
                      "power_se": g.apply(lambda d: se_pooled(d, "power"), include_groups=False),
                      "xok": g.apply(lambda d: (d.missed <= d.missed_max).mean(), include_groups=False),
                      "cover": g.apply(lambda d: ((d.low <= d.truth) & (d.truth <= d.high)).mean(),
                                       include_groups=False),
                      "width": g.apply(lambda d: ((d.possible - d.certain) / d.n_valid).mean(),
                                       include_groups=False)}).reset_index()  # fmt: skip
    L.append("## RQ2 (ten systems pooled; per system in paper/tables/rq2-guarantees.tex)\n")
    L.append(md(["n", "P", "FDR", "FDR SE", "power", "power SE", "X ok", "P4 covers", "gap"],
                [[int(r.n), int(r.pct), f(r.fdr, 3), f(r.fdr_se, 3), f(r.power), f(r.power_se, 3),
                  f(r.xok), f(r.cover), f(r.width)] for r in t.itertuples()]))  # fmt: skip
    L.append(
        "\nSE is the Monte Carlo standard error of the pooled mean: the standard deviation over "
        "the repetitions of each system divided by the square root of the repetitions, combined "
        "over the ten systems as sqrt(sum of variances) / 10.\n"
    )
    ns, ps = sorted(s.n.unique()), sorted(s.pct.unique())
    reps, rows = s.rep.nunique(), len(s) // s.system.nunique()
    meta_p = meta["params"]
    L.append(
        "\nColumns of `paper/tables/rq2-guarantees.tex`, per system, from the rows with "
        f"q = {s.q.iloc[0]}, n in {{{', '.join(map(str, ns))}}}, P in {{{', '.join(map(str, ps))}}} "
        f"and {reps} repetitions, with alpha = {meta_p['alpha']} and delta = {meta_p['pac_delta']}. "
        f"FDR is the mean of the realised false discovery proportion over all {rows} rows of the "
        "system (every n, P and repetition; an empty selection counts as 0) and SE is the standard "
        f"deviation over the {reps} repetitions of the mean over the {len(ns) * len(ps)} (n, P) cells "
        f"of a repetition, divided by sqrt({reps}). Power is, for each n separately, the mean over the "
        f"{reps} repetitions of the power of the selection at P = 50 only, without an SE, and "
        "repetitions with no configuration below the threshold are left out. X ok is the share of "
        f"the {rows} rows in which the number of mis-covered unmeasured configurations is at most "
        "the PAC bound; it pools n, P and repetitions as FDR does, has no SE, and as neither "
        f"count depends on P or q it is a share of {len(ns) * reps} (n, repetition) cases. The "
        "group headers fix q for FDR and SE and P = 50 and n for the power columns; PAC (X ok) "
        "and Labels (Gap) fix nothing, and Gap is defined in the next paragraph.\n"
    )
    run = "experiments/run.py"
    nan = s[s.power.isna()].groupby(["system", "n", "pct"]).size()
    zero = (
        "none"
        if nan.empty
        else "; ".join(f"{LABEL[x]}, n = {n}, P = {pc}: {c}" for (x, n, pc), c in nan.items())
    )
    bs = sel[sel.system == "berkeleydb"].groupby("q").n.unique()
    ns_q = "; ".join(f"q = {q}: n in {{{', '.join(map(str, sorted(v)))}}}" for q, v in bs.items())
    L.append(
        "\nPower in RQ2 is |S & {y <= t} & U| / |{y <= t} & U|, `power(s, ok)` "
        f"({at(run, 'def power(')}, called at {at(run, 'power=power(s, ok)')}), where "
        f"U is V minus Train and Cal, t is the P-th percentile of y over V, "
        f"`ok = y[u] <= t` ({at(run, 'ok = y[u] <= t')}) and S is the Benjamini-Hochberg "
        f"selection in U, `selection.select` ({at(run, 'sel[u] = selection.select(')}). The "
        "measured configurations, Train and Cal, are in neither the numerator nor the "
        "denominator, and S is the selection alone: it is not the set `lab == CERTAIN`, which "
        "also holds the measured configurations that satisfy the policy.\n"
    )
    L.append(
        "For each system, the Power columns of `paper/tables/rq2-guarantees.tex` are, for each n, "
        f"the mean over the {reps} repetitions of that power in the q = {s.q.iloc[0]} rows at "
        f"P = 50 ({at('experiments/results.py', 'p50 = d[d.pct == 50].' + 'groupby(' + chr(34) + 'n' + chr(34) + ').power.mean()')}); "
        "the pooled table above averages the same values over the ten systems and the "
        f"{reps} repetitions of each (n, P) cell, with `se_pooled` for its SE. A repetition whose "
        f"denominator is 0 returns NaN ({at(run, 'if truth.any() else np.nan')}) and the means skip it; "
        f"the repetitions with denominator 0 per (system, n, P) cell, out of {reps}, are: {zero}. "
        "RQ3 does not use this power: its recall (`methods.returned`, "
        f"{at(run, 'methods.returned(sets[k], truth_mask)')}) is |certain & {{y <= t}}| / |{{y <= t}}| "
        "over V, with certain = `lab == labels.CERTAIN`, which includes the measured "
        "configurations that satisfy, and the measured configurations are in V, so the two "
        "definitions differ in the numerator and in the denominator. The q sensitivity line "
        "below is the mean of the same power over all rows of BerkeleyDB-C with each q, "
        f"every P included, without restricting n: {ns_q}.\n"
    )
    gap = (s.possible - s.certain) / s.n_valid
    cell = gap[(s.n == 100) & (s.pct == 50)].groupby(s.system).mean()
    L.append(
        "\ngap is the certain-possible gap, (possible - certain) / |V| with certain and possible "
        "the numbers of configurations labelled certain and certain-or-possible (`labels.count`), "
        "averaged over repetitions. It is not the width of the P4 count interval (RQ3). The column "
        "Gap of `paper/tables/rq2-guarantees.tex` is this gap at n = 100, P50, q = 0.1 for each "
        f"system; its mean over the ten systems, {cell.mean():.3f}, is the n = 100, P50 row above. "
        f"The mean over all nine (n, P) cells is {gap.mean():.3f}: the gap falls with n ("
        + ", ".join(f"{v:.2f}" for v in gap.groupby(s.n).mean())
        + " for n = 50, 100, 200) and with P ("
        + ", ".join(f"{v:.2f}" for v in gap.groupby(s.pct).mean())
        + " for P25, P50, P75), so the two figures average different cells.\n"
    )
    L.append(
        "\nMean gap by n and by P over the same cells as the RQ2 table (q = 0.1, ten systems "
        "pooled):\n"
    )
    L.append(md(["by", "value", "gap"],
                [["n", int(k), f(v, 3)] for k, v in gap.groupby(s.n).mean().items()]
                + [["P", int(k), f(v, 3)] for k, v in gap.groupby(s.pct).mean().items()]))  # fmt: skip
    L.append("")
    bq = sel[sel.system == "berkeleydb"].groupby("q").agg(fdr=("fdp", "mean"), pw=("power", "mean"))
    L.append(
        "\nq sensitivity (BerkeleyDB-C, n in {100, 200}): "
        + "; ".join(f"q = {q}: FDR {r.fdr:.3f}, power {r.pw:.2f}" for q, r in bq.iterrows())
        + ".\n"
    )
    qn = meta["params"]["q_sens"]["n"]
    pw = sel[(sel.system == "berkeleydb") & (sel.q == 0.1) & sel.n.isin(qn)].power.mean()
    L.append(
        f"q = 0.1 restricted to n in {{{', '.join(map(str, qn))}}}: power {pw:.2f}. This is the "
        "value comparable with those of q = 0.05 and q = 0.2, which only have those n; the "
        "q = 0.1 value above also includes the other n of the grid.\n"
    )
    ne = cmp_[cmp_.method == "ours"].groupby("eps").agg(p=("prec", "mean"), r=("recall", "mean"),
                                                         w=("width", "mean"))  # fmt: skip
    L.append("Noise bands (eps = 0, 5, 10 %): precision "
             + ", ".join(f"{v:.3f}" for v in ne.p) + "; recall " + ", ".join(f"{v:.3f}" for v in ne.r)
             + ".\n")  # fmt: skip

    eps = meta["params"]["eps"]
    L.append(
        "The noise band is epsilon(v) = eps |y(v)| "
        f"({at('src/cpq/labels.py', 'band = eps * np.abs(y)')}, `measured_labels`): a measured "
        "configuration is certain if y + epsilon <= t, violates if y - epsilon > t and is possible "
        "otherwise. eps is a fraction of the measured value y(v) of that same configuration, not "
        "of t or of the range of the population: the fraction is the same for every configuration "
        "and the absolute band is not. The main grid (RQ2 and RQ3) takes the first element of the "
        f"list `eps` of `experiments/grid.yaml` ({', '.join(str(x) for x in eps)}), here "
        f"{eps[0]} ({at('experiments/run.py', 'epss = listed(cfg')}, used at "
        f"{at('experiments/run.py', 'labels.label(yhat, y, meas, sel, qh, t, epss[0])')}), so "
        "epsilon(v) = 0 and a measured configuration complies when y <= t; the RQ3 table keeps the "
        "rows with eps = 0. The noise experiment runs `compare_rows` once for each value of the "
        f"list ({at('experiments/run.py', 'for eps in epss')}), with that eps applied to every "
        "configuration. The QVM+ table Noise of the BerkeleyDB-C instance uses the same "
        f"definition at 5 % ({at('src/cpq/instance.py', '0.05 * np.abs(p.y[meas])')}).\n"
    )

    c0 = cmp_[cmp_.eps == 0]
    gm = c0.groupby("method")
    tm = gm.agg(prec=("prec", "mean"), rec=("recall", "mean"), err=("err", "mean"),
                cov=("covers", "mean"), width=("width", "mean")).reindex(list(METHOD))  # fmt: skip
    L.append("## RQ3\n")
    L.append("Policy y <= t, all systems, n, thresholds (mean). Precision and recall are of the set "
             "each method returns; count error is |estimate - truth| / |V|, or the distance to the "
             "interval for interval-only methods.\n")  # fmt: skip
    L.append(md(["method", "precision", "recall", "count error", "interval covers", "interval width"],
                [[METHOD[m], f(r.prec), f(r.rec), f(r.err, 3), f(r["cov"]), f(r.width)]
                 for m, r in tm.iterrows()]))  # fmt: skip
    o = cmp_[(cmp_.method == "ours") & (cmp_.eps == 0)]
    sq = sel[sel.q == 0.1]
    L.append(
        "\nCount error. certain is the number of configurations labelled certain and possible the "
        "number labelled certain or possible; the point estimate of the count is the predictor's "
        "count with the measured configurations replaced by their measured values (`est`), and the "
        "P4 interval [`lo`, `hi`] is the same bracketing widened by the PAC bound on the number of "
        "mis-covered unmeasured configurations (`missed_max`). The count error of a method is "
        "|estimate - truth| / |V| (for interval-only methods, the distance from the truth to the "
        "interval over |V|). For certain/possible the point estimate errs by "
        f"{o.err.mean():.3f} of |V| on average, while the P4 interval covers the truth in "
        f"{o.covers.mean():.2f} of the cases with mean width {o.width.mean():.2f}: the interval "
        "is widened on each side by `missed_max`, "
        f"{sq.missed_max.mean():.0f} configurations on average, a PAC bound on the mis-covered "
        f"ones, of which {sq.missed.mean():.1f} occur on average. It is therefore wide and "
        "contains the truth in every repetition of the grid, whereas the point estimate carries "
        "no guarantee. The 0.77 in the certain/possible row is the width of the P4 count interval, "
        "(`hi` - `lo`) / |V| with `lo` = N- and `hi` = N+ widened by `missed_max`, averaged over "
        "all n and thresholds. It is a different quantity from the certain-possible gap "
        f"(possible - certain) / |V| of RQ2, whose mean over the same cells is {gap.mean():.2f}. "
        "For the other methods width is the width of their own interval."
    )
    run = "experiments/run.py"
    L.append(
        f"\nDefinitions behind the RQ3 table, all in `compare_rows`. Our point count is `hyb` "
        f'({at(run, "hyb = labels.hybrid(")}, used as `est["ours"]` at {at(run, '"ours": (hyb')}): '
        "the predictor's count #{v in V: yhat(v) <= t} over the |V| configurations, with the term "
        "of every v in M, the n measured configurations (Train and Cal), replaced by its measured "
        "[y(v) <= t] (`labels.hybrid`, eq:corr). The threshold t is the P-th percentile of y over "
        f"V ({at(run, 't = float(np.percentile(y, pct))')}) with P in "
        f"{{{', '.join(str(x) for x in meta['params']['percentiles'])}}}, and no noise band enters "
        f"it. Measured only is `N * meas_sat.sum() / n_meas` ({at(run, '"measured_only": (N *')}), "
        f"with `meas_sat` the measured configurations labelled certain ({at(run, 'meas_sat = meas &')}): "
        "the share of the n measured configurations with y + epsilon <= t, scaled to |V|. "
        f"Precision and recall come from `methods.returned` ({at('src/cpq/baselines/methods.py', 'def returned')}, "
        f"called at {at(run, 'methods.returned(sets[k], truth_mask)')}) and are taken over V, not over "
        "U: the returned set is a mask over all |V| configurations, measured ones included, and "
        f"the truth is y <= t over V ({at(run, 'truth_mask = y <= t')}). For certain/possible the "
        f"returned set is certain, `lab == labels.CERTAIN` ({at(run, '"ours": lab ==')}), that is, "
        "the measured configurations that satisfy plus the BH selection in U. Precision and recall "
        "of possible are not computed anywhere in `compare_rows`, so the row certain/possible of "
        "the RQ3 table reports the certain set only.\n"
    )
    om = (
        opt.assign(share=opt.total / opt.n_valid)
        .groupby("method")
        .agg(
            hit=("regret", lambda r: (r == 0).mean()),
            rel=("rel_regret", "mean"),
            share=("share", "mean"),
        )
    )
    names = {"ours": "screen, then measure", "uniform_total": "best of n + extra uniform",
             "uniform_n": "best of n uniform", "measured_only": "measured only",
             "predictor_only": "predictor only"}  # fmt: skip
    L.append("\nOPTIMISE (n + extra measurements; share of |V| measured):\n")
    L.append(
        md(
            ["method", "optimum found", "relative regret", "measured share"],
            [[names[m], f(om.hit[m]), f(om.rel[m], 4), f(om.share[m], 3)] for m in names],
        )
    )
    ov = opt[opt.method.isin(["ours", "uniform_total"])].assign(hit=lambda d: d.regret == 0)
    wins = ov.pivot_table(index="system", columns="method", values="hit")
    L.append(
        "\nRelative regret is (found - y*) / y*, with found the best measured response a "
        "method returns and y* the minimum response over the whole population V "
        "(`experiments/run.py`, `optimise_rows`); regret is found - y*, and optimum found is "
        "the share of repetitions with regret 0. Means are over all n, systems and repetitions.\n"
    )
    L.append(f"\nScreen-then-measure finds the optimum more often than the equal-budget uniform "
             f"baseline in {int((wins.ours > wins.uniform_total).sum())} of 10 systems, ties in "
             f"{int((wins.ours == wins.uniform_total).sum())}.\n")  # fmt: skip

    hr = opt.assign(hit=lambda d: d.regret == 0).pivot_table(
        index="system", columns="method", values="hit"
    )
    L.append("Hit rate (share of repetitions that find the optimum) by system, all n:\n")
    L.append(md(["system", *names.values()],
                [[LABEL[x], *[f(hr.loc[x, m]) for m in names]] for x in ORDER]))  # fmt: skip
    st = opt[opt.method == "ours"].assign(share=lambda d: d.total / d.n_valid).groupby("system")
    st = st.agg(total=("total", "mean"), share=("share", "mean"))
    L.append("\nScreen-then-measure cost by system:\n")
    L.append(md(["system", "|V|", "measured (n + extra)", "share of |V|"],
                [[LABEL[x], int(pops[x]), f"{st.total[x]:.0f}", f(st.share[x], 3)] for x in ORDER]))  # fmt: skip
    ours = opt[opt.method == "ours"].assign(
        own=lambda d: d.n / d.n_valid, scr=lambda d: d.total / d.n_valid
    )
    by = ours.pivot_table(index="system", columns="n", values=["own", "scr"])
    allsys = ours.groupby("n")[["own", "scr"]].mean()
    ns = sorted(opt.n.unique())
    L.append("\nMeasured fraction of |V| by initial budget n:\n")
    L.append(md(["system", *[f"n/|V| at n = {n}" for n in ns], *[f"(n + extra)/|V| at n = {n}" for n in ns]],
                [[LABEL[x], *[f(by.own.loc[x, n], 3) for n in ns], *[f(by.scr.loc[x, n], 3) for n in ns]]
                 for x in ORDER]
                + [["all systems", *[f(allsys.own[n], 3) for n in ns], *[f(allsys.scr[n], 3) for n in ns]]]))  # fmt: skip
    L.append(
        "\nn/|V| is the measured fraction of the series that measure only the n initial "
        "configurations (measured only, predictor only), with numerator n and denominator |V|. "
        "(n + extra)/|V| is the measured fraction of screen, then measure and of best of n' "
        "uniform, with n' = n + extra, the number of measurements that screening makes, as "
        "numerator and |V| as denominator. Each is the mean over the repetitions of a "
        "system; the row all systems is the mean over all systems and repetitions. The "
        "second line under each tick of `paper/figs/rq3-optimise.pdf` is the (n + extra)/|V| "
        "of the row all systems.\n"
    )
    L.append("\nScreen-then-measure measures the n initial configurations plus the `extra` "
             "candidates S = {v in U: lo(v) <= min(best measured, min hi)} (`labels.optimise`). "
             "measured is n + extra and share is (n + extra) / |V|, both averaged over the three "
             "values of n and all repetitions of a system.\n")  # fmt: skip
    L.append("")
    L.append("## RQ4\n")
    ok = sum(all(study[x][k] for k in ("count_ok", "hist_ok", "min_ok", "ev1_ok", "mindeg_ok", "prefix_ok"))
             for x in ORDER)  # fmt: skip
    L.append(f"Contraction equals enumeration (N, H_D, T_D) on {ok} of 10 systems; widths "
             f"{min(study[x]['width_weighted'] for x in ORDER):.0f} to "
             f"{max(study[x]['width_weighted'] for x in ORDER):.0f} with {sum(study[x]['pairs_dropped'] for x in ORDER)} "
             f"pairs dropped. Rounding to the grid changes the mean absolute error of the 15-pair "
             f"predictor by {np.mean([study[x]['mae_grid15'] / study[x]['mae_raw15'] - 1 for x in ORDER]) * 100:.1f} % "
             f"on average (maximum {max(study[x]['mae_grid15'] / study[x]['mae_raw15'] - 1 for x in ORDER) * 100:.1f} %).\n")  # fmt: skip
    L.append(
        "MiB in `paper/tables/rq4-evaluator.tex` is the peak of memory allocated during the "
        "H_D contraction (`contract.histogram`), in MiB (2^20 bytes), as traced by "
        "`tracemalloc` in `experiments/rq4.py`; it includes numpy buffers and excludes the "
        "plan and the other two evaluators.\n"
    )
    L.append(
        "The ten VMs of the study, with |O| options, the width of the network weighted by the "
        "predictor, D, the times of N, H_D and T_D in seconds and the MiB of the H_D "
        "contraction (the contents of the former rows of `rq4-evaluator.tex`): "
        + "; ".join(
            f"{LABEL[x]} (|O| = {study[x]['features']}, width {study[x]['width_weighted']:.0f}, "
            f"D = {study[x]['D']}, {f(study[x]['s_count'], 3)} / {f(study[x]['s_hist'], 3)} / "
            f"{f(study[x]['s_min'], 3)} s, {f(study[x]['mb_hist'], 1)} MiB)"
            for x in ORDER
        )
        + ".\n"
    )
    gk = {r["system"]: r for r in rq4 if r.get("kind") == "limit_ganak"}
    cap = next(r["cap_gib"] for r in rq4 if r.get("kind") == "limit")
    tout = re.search(r"after (\d+) seconds", gk["linux-2.6.33.3"]["error"]).group(1)
    L.append(
        "In `rq4-evaluator.tex` the column Width is the contraction width of the unweighted clause "
        "network (`width_n`) in the four rows; the width of the network weighted by the predictor "
        "(`width_weighted`), which H_D and T_D contract, is the one in the paragraph above and in "
        "the table below. -- in the columns N, H_D and MiB means that the contraction was not run "
        f"because its plan needs more than the {cap:.0f} GiB memory limit; -- in ganak (s) means "
        f"that ganak found no count within the {tout} s timeout.\n"
    )
    dm = [r for r in rq4 if r.get("kind") == "dimacs_synthetic"]
    lim = [r for r in rq4 if r.get("kind") == "limit"]
    nruns = {k: max(len(r["runs"]) for r in rq4 if r.get("kind") == k)
             for k in ("study", "dimacs_synthetic", "limit")}  # fmt: skip
    best = Counter(r["layout"] for r in rq4 if r.get("kind") in ("study", "dimacs_synthetic"))
    L.append(
        "Times in RQ4 (`experiments/rq4.py`). Each case runs in its own process once per layout, "
        "`full` (BLAS threads on the whole chip) and `vcache` (pinned to the CCD with the "
        f"V-Cache): {nruns['study']} runs for each study VM, {nruns['dimacs_synthetic']} for each "
        f"synthetic DIMACS model and {nruns['limit']} for each limit case. The reported times are "
        "those of the exact run with the smallest sum of the contraction times of N, H_D and T_D, "
        "that is, the minimum of the two runs by that sum; the three times come from that one run "
        f"and are not minimised separately. The chosen layout is `full` in {best['full']} and "
        f"`vcache` in {best['vcache']} of the {sum(best.values())} cases. Every evaluator is "
        "timed by a single call, with `tracemalloc` enabled (it starts before the clock in "
        "`traced`), so the times include its overhead. The ganak time is one call per run; "
        f"financial01 was also run alone in `limit_ganak`, {f(gk['financial01']['s_ganak'], 2)} s "
        f"against {f(next(r['s_ganak'] for r in lim if r['system'] == 'financial01'), 2)} s in "
        "the table.\n"
    )
    L.append(
        "The DIMACS models carry no measurements, so the predictor of their rows is synthetic "
        "(`synthetic` in `experiments/rq4.py`, called with its defaults seed 0, density 0.1 and "
        "5 pairs). A generator `numpy.random.default_rng(0)`, the same for both models, picks "
        "without replacement max(int(0.1 |O|), 2) options, "
        + " and ".join(
            f"{max(int(0.1 * r['features']), 2)} of |O| = {r['features']} in {r['system']}"
            for r in dm
        )
        + ", and gives each an integer weight uniform in {1, ..., 5}; every other option weighs 0. "
        "E, the set of pairs, is 5 draws of two distinct picked options, sorted and deduplicated, "
        "each with an integer coefficient uniform in {-3, ..., 3}; "
        + " and ".join(f"|E| = {r['pairs']} in {r['system']}" for r in dm)
        + ". The grid step is 1.0, so these integers are the grid exponents. The weights only "
        "exercise the evaluator; they come from no fit and say nothing about performance.\n"
    )
    L.append(
        md(
            ["model", "variables", "width (weighted)", "contraction (s)", "ganak (s)", "equal"],
            [
                [
                    r["system"],
                    r["features"],
                    f"{r['width_weighted']:.0f}",
                    f(r["s_count"], 3),
                    f(r["s_ganak"], 2),
                    r["count_ok"],
                ]
                for r in dm
            ],
        )
    )
    L.append("")
    for r in lim:
        g = gk.get(r["system"], {})
        if "width_n" in r:
            plan = f"width {r['width_n']:.0f}, {r['peak_elements']:.2g} elements against a cap of {r['cap_gib']:.0f} GiB"
        else:
            plan = f"no contraction order within 3600 s ({r.get('error')})"
        if "ganak" in g:
            count = f"ganak {g['ganak'][:24]} in {g['s_ganak']:.1f} s"
        else:
            count = "ganak found no count in 1800 s" if g.get("error") else "ganak not run"
        L.append(f"- {r['system']}: limit. {plan}; {count}.")

    L.append(
        "\n\nExact counts of the synthetic DIMACS models, ganak against the contraction in N "
        "(integers as stored in `rq4.json`):\n"
    )
    for r in dm:
        same = int(r["ganak"]) == int(r["count"])
        L.append(f"- {r['system']}: ganak {r['ganak']}; contraction {r['count']}; "
                 f"{'equal' if same else 'different'}.")  # fmt: skip
    corr = json.loads((RAW / "rq4/correction.json").read_text())
    yn = lambda b: "yes" if b else "no"  # noqa: E731
    L.append("\n## RQ4: correction over M on the contracted histogram\n")
    L.append(md(
        ["system", "dtype", "N-", "N+", "N-, N+ equal", "hybrid optimum", "optimum equal",
         "T_D equal", "Count bounds equal"],
        [[LABEL[r["system"]], r["dtype_hist"], r["n_minus"], r["n_plus"],
          yn(r["eq_minus"] and r["eq_plus"]), r["mindeg_corr"], yn(r["eq_mindeg"]),
          yn(r["eq_td"]), yn(r["eq_bounds"])] for r in corr["rows"]]))  # fmt: skip
    cm = corr["meta"]["params"]
    L.append(
        f"\nSame repetition as the RQ4 table (n = {cm['n']}, repetition {cm['rep']}, t at P{cm['percentile']} of "
        f"the responses, alpha = {cm['alpha']}). The contracted histogram H_D of the predictor "
        "over V is corrected over M, the n measured configurations, by eq:corr (`labels.hybrid`): "
        "the predicted monomial of every v in M is replaced by the measured one, at the grid "
        "index of the rounded y. N- = #{v in U: hi(v) <= t} and N+ = #{v in U: lo(v) <= t} are "
        "prefix sums, at floor((t -/+ qhat) / delta) - c0, of the corrected histogram minus the "
        "measured monomials of M, so they count U only; they equal the enumeration "
        "`(yhat[U] +/- qhat <= t).sum()` in "
        f"{sum(r['eq_minus'] and r['eq_plus'] for r in corr['rows'])} of {len(corr['rows'])} systems. "
        "The hybrid optimum is `mindeg` of the corrected histogram and is compared with the "
        "enumerated minimum over the hybrid population (M at the rounded y, U at yhat), "
        f"equal in {sum(r['eq_mindeg'] for r in corr['rows'])} of {len(corr['rows'])}; "
        "T_D equal compares `mindeg` of the uncorrected histogram with the contracted T_D, as in "
        "the table above. `labels.optimise` is not compared (it returns the best measured value "
        "after screening, not a grid index). Count bounds equal rebuilds Count.low and Count.high "
        "of `labels.count` from N-, N+, the exact measured count and the PAC bound. The corrected "
        "histogram has the dtype of the contracted one; the dtype column is float64 up to "
        "|O| = 52 and int64 up to 62.\n"
    )
    rk = json.loads((RAW / "rank/rank_budget.json").read_text())
    cells, rp = pd.DataFrame(rk["cells"]), rk["repro"]
    rpos = pd.DataFrame(rk["pos"])
    kp, kn = sorted(cells.pct.unique()), sorted(cells.n.unique())
    L.append("\n## Ranking under a budget\n")
    L.append(
        "After the n initial measurements of the grid, with the same partitions, seeds and "
        "predictor as `experiments/run.py` (`SeedSequence([system index, n, repetition])`, "
        f"n in {{{', '.join(map(str, kn))}}}, {rk['meta']['params']['reps']} repetitions), the "
        "unmeasured set U is ranked by yhat increasing and only its first k configurations are "
        f"measured, with k = {', '.join(map(str, kp))} % of |V| rounded up and at most |U|. Ties "
        "of yhat are broken by increasing xi, the per-configuration noise `noise(sid, n, rep, N)` "
        "that the conformal selection already uses (its scale, delta / 2, does not change the "
        "order). The result is the best of the n measured values and the k ranked ones; a hit is "
        "found = y*, with y* the minimum of the response over V, and the relative regret is "
        "(found - y*) / y*. The uniform baseline is the best of n + k configurations drawn "
        "uniformly from V with the generator of the grid series `uniform_total` "
        "(`default_rng([sid, n, rep, 7])`, after the draw of the n configurations of "
        "`uniform_n`, which is discarded). The position of the optimum is the rank in the "
        "ranking of U of the best placed configuration with y = y*, as a fraction of |V|; its "
        "median is over the repetitions in which no such configuration is among the n measured "
        "ones, and the share of repetitions in which one is, is reported apart. Means over "
        "systems give every system the same weight, and the aggregated position is the mean over "
        "systems of the per-system medians. This is not the stopping rule of Lemma 11: k is "
        "fixed in advance and nothing is certified.\n"
    )
    L.append(
        f"Reproducibility: with the same partitions, `labels.optimise` equals the `ours` rows of "
        f"`results/raw/grid/opt` (found and extra) in {rp['found_equal']} and "
        f"{rp['extra_equal']} of {rp['cases']} (system, n, repetition) cases, and the best of "
        f"n + extra uniform configurations equals `uniform_total` in {rp['unif_equal']}; the "
        f"largest differences are {rp['max_found']:g}, {rp['max_extra']:g} and "
        f"{rp['max_unif']:g}.\n"
    )
    cols = ["hit_rank", "hit_unif", "rel_rank", "rel_unif"]
    agg = cells.groupby(["n", "pct"])[cols].mean()
    L.append("Mean over the ten systems:\n")
    L.append(md(["n", "k (% of |V|)", "hit, ranking", "hit, uniform", "regret, ranking",
                 "regret, uniform"],
                [[n, pc, f(r.hit_rank), f(r.hit_unif), f(r.rel_rank, 4), f(r.rel_unif, 4)]
                 for (n, pc), r in agg.iterrows()]))  # fmt: skip
    pa = rpos.groupby("n").agg(med=("median_pos", "mean"), in_m=("in_m", "mean"))
    L.append("\nPosition of the optimum in the ranking of U, as a fraction of |V|:\n")
    L.append(md(["n", "median position", "optimum already measured"],
                [[n, f(r.med, 4), f(r.in_m, 3)] for n, r in pa.iterrows()]))  # fmt: skip
    ix = cells.set_index(["system", "n", "pct"])
    for col, name, nd in (("hit", "Hit rate", 2), ("rel", "Relative regret", 4)):
        L.append(f"\n{name} by system, ranking and uniform for each k (% of |V|):\n")
        L.append(md(
            ["system", "n", *[f"rank {pc} %" for pc in kp], *[f"uniform {pc} %" for pc in kp]],
            [[LABEL[x], n, *[f(ix.loc[(x, n, pc), f"{col}_rank"], nd) for pc in kp],
              *[f(ix.loc[(x, n, pc), f"{col}_unif"], nd) for pc in kp]]
             for x in ORDER for n in kn]))  # fmt: skip
    px = rpos.set_index(["system", "n"])
    L.append(
        "\nMedian position of the optimum (fraction of |V|) and share of repetitions with "
        "the optimum already measured, by system:\n"
    )
    L.append(md(
        ["system", *[f"median, n = {n}" for n in kn], *[f"measured, n = {n}" for n in kn]],
        [[LABEL[x], *[f(px.loc[(x, n), "median_pos"], 4) for n in kn],
          *[f(px.loc[(x, n), "in_m"], 2) for n in kn]] for x in ORDER]))  # fmt: skip
    L.append("")
    nm = json.loads((RAW / "nmin/nmin.json").read_text())
    nq, nr = nm["q"], pd.DataFrame(nm["rows"])
    nps = sorted(nr.pct.unique())
    L.append("\n## Minimum n for a necessary condition\n")
    L.append(
        "With deterministic p-values every p-value is at least 1 / (m + 1), so Benjamini-Hochberg "
        f"at level q = {nq} rejects k configurations of U only if k >= |U| / (q (m + 1)), with "
        "m = n / 2 the size of Cal. A selection whose rejections are all correct has at most N_t "
        "configurations, N_t being the number of configurations of U with y <= t, so a non-empty "
        "correct selection needs m + 1 >= |U| / (q N_t). N_t is taken as its exact expectation "
        "under the uniform random partition, N_t(n) = T_t (|V| - n) / |V|, with T_t the number "
        "of configurations of V with y <= t; |U| = |V| - n cancels and the condition is "
        "n / 2 + 1 >= |V| / (q T_t). This is a necessary condition, not a sufficient one: meeting "
        "it does not make the selection non-empty or correct. The variant for a selection with "
        "false discovery proportion at most q replaces N_t by N_t / (1 - q), since at least "
        "(1 - q) k of the k selected comply, and reads n / 2 + 1 >= (1 - q) |V| / (q T_t). The "
        "minimum n is the smallest even n that meets the condition and is at least "
        f"{2 * (math.ceil(1 / nm['alpha']) - 1)}, the smallest n for which the conformal "
        f"quantile at alpha = {nm['alpha']} is finite (m >= {math.ceil(1 / nm['alpha']) - 1}); "
        "none if it exceeds |V| / 2. The threshold t is the P-th percentile of y over V.\n"
    )
    nmx = nr.assign(a=lambda d: d.n_min.map(lambda v: "none" if v is None else v),
                    b=lambda d: d.n_min_fdp.map(lambda v: "none" if v is None else v))  # fmt: skip
    nmx = nmx.set_index(["system", "pct"])
    L.append(md(
        ["system", "|V|", *[f"necessary, P{p}" for p in nps], *[f"FDP <= q, P{p}" for p in nps]],
        [[LABEL[x], int(nmx.loc[(x, nps[0]), "size"]), *[nmx.loc[(x, p), "a"] for p in nps],
          *[nmx.loc[(x, p), "b"] for p in nps]] for x in ORDER]))  # fmt: skip
    if nr.groupby("pct")[["n_min", "n_min_fdp"]].nunique().max().max() == 1:
        L.append(
            "\nEvery system gives the same minimum n for a given P: T_t / |V| is, up to ties, "
            "P / 100 by the definition of the percentile, so the condition does not depend on the "
            "system.\n"
        )
    emp = pd.DataFrame(
        [{"system": r.system, "pct": r.pct, "n": int(n), **v} for r in nr.itertuples()
         for n, v in r.empirical.items()])  # fmt: skip
    ea = emp.assign(a=emp.rel.abs()).groupby(["system", "pct"]).a.max()
    L.append(
        f"\nInformative check of the expectation: for n in {{{', '.join(sorted(set(emp.n.astype(str)), key=int))}}} "
        f"and each P, the mean over the {meta['params']['reps']} partitions of the grid "
        "(`SeedSequence([system index, n, repetition])`) of #{v in U: y <= t} against "
        "T_t (|V| - n) / |V|. Largest absolute relative difference over those n, by system:\n"
    )
    L.append(md(["system", *[f"P{p}" for p in nps]],
                [[LABEL[x], *[f(ea.loc[(x, p)], 4) for p in nps]] for x in ORDER]))  # fmt: skip
    L.append(f"\nThe largest difference of all is {f(emp.rel.abs().max(), 4)}; it does not "
             "change the minimum n.\n")  # fmt: skip
    q2 = max(bq.index)
    L.append(f"Power at q = {q2} is only the one of the q sensitivity above (BerkeleyDB-C, n in "
             f"{{100, 200}}): {bq.loc[q2, 'pw']:.2f}; it is not computed for other systems.\n")  # fmt: skip
    if mut:
        eq = [m["id"] for m in mut if m["equivalent"]]
        live = [m for m in mut if not m["equivalent"]]
        n = len(live)
        ids = lambda xs: ", ".join(m["id"] for m in xs) or "none"  # noqa: E731
        L.append("\n## Mutants\n")
        L.append(
            f"{len(mut)} mutants of the semiring evaluators (E), the network builder (N), the spec "
            f"evaluator (S) and the UWD compiler (C); {len(eq)} are equivalent ({', '.join(eq)}) and "
            f"are excluded. Of the other {n}, the tests without the oracle kill "
            f"{sum(m['killed_by_tests'] for m in live)}, the homomorphism identities alone kill "
            f"{sum(m['killed_by_oracle'] for m in live)}, and tests plus identities kill "
            f"{sum(m['killed_by_both'] for m in live)}.\n"
        )
        L.append(
            f"- killed only by the identities: {ids([m for m in live if m['killed_by_oracle'] and not m['killed_by_tests']])}"
        )
        L.append(
            f"- killed only by the tests: {ids([m for m in live if m['killed_by_tests'] and not m['killed_by_oracle']])}"
        )
        L.append(
            f"- killed by both: {ids([m for m in live if m['killed_by_tests'] and m['killed_by_oracle']])}"
        )
        L.append(f"- survived: {ids([m for m in live if not m['killed_by_both']])}")
    (ROOT / "results/results.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    write()
