import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    d = ROOT / "results/raw" / name
    sel = pd.concat([pd.read_parquet(f) for f in sorted((d / "sel").glob("*.parquet"))])
    return sel, json.loads((d / "meta.json").read_text())


def agg(df):
    g = df.groupby(["system", "n", "pct"])
    out = g.agg(
        reps=("rep", "size"), fdr=("fdp", "mean"), power=("power", "mean"), size=("n_sel", "mean"),
        mfdr=("mfdp", "mean"), mpower=("mpower", "mean"), both=("both", "mean"),
    )  # fmt: skip
    out["x_ok"] = g.apply(lambda d: (d.missed <= d.missed_max).mean(), include_groups=False)
    out["cover"] = g.apply(
        lambda d: ((d.low <= d.truth) & (d.truth <= d.high)).mean(), include_groups=False
    )
    out["width"] = g.apply(
        lambda d: ((d.possible - d.certain) / d.n_valid).mean(), include_groups=False
    )
    out["cnt_err"] = g.apply(
        lambda d: ((d.pred - d.truth).abs() / d.n_valid).mean(), include_groups=False
    )
    return out.reset_index()


def md(df, cols, heads, fmt):
    lines = ["| " + " | ".join(heads) + " |", "|" + "|".join("---" for _ in heads) + "|"]
    for _, r in df.iterrows():
        lines.append(
            "| " + " | ".join(f.format(r[c]) for c, f in zip(cols, fmt, strict=True)) + " |"
        )
    return "\n".join(lines)


def reading(df, a, o):
    lo50 = a[(a.n == 50) & (a.pct == 25)].set_index("system")
    hi200 = a[(a.n == 200) & (a.pct == 25)].set_index("system")
    top = a.loc[a.both.idxmax()]
    return [
        f"FDR of certain lies between {a.fdr.min():.3f} and {a.fdr.max():.3f} "
        f"(mean {a.fdr.mean():.3f}) against q = {df.q.iloc[0]}.",
        "Power at P25 with n = 50 is "
        + " and ".join(f"{lo50.power[k]:.3f} ({k})" for k in lo50.index)
        + " because the p-values have resolution 1/(m+1) against BH thresholds q k / |U|; "
        "left as is. At n = 200 it is "
        + " and ".join(f"{hi200.power[k]:.3f} ({k})" for k in hi200.index)
        + ".",
        f"The P4 interval holds the true count in at least {a.cover.min():.2f} of repetitions "
        f"and X stays under its bound in at least {a.x_ok.min():.2f} (target 0.95), "
        f"but the [certain, possible] width reaches {a.width.max():.2f} of |V| at n = 50.",
        f"The mirror selection has FDR between {a.mfdr.min():.3f} and {a.mfdr.max():.3f}; "
        f"the two selections overlap in up to {top.both:.0f} configurations per repetition "
        f"({top.system}, n = {top.n:.0f}, P{top.pct:.0f}), so a violation label cannot be "
        "promoted without a rule for the overlap.",
        f"Screen-then-measure finds the optimum in {(df[df.pct == 50].regret == 0).mean():.3f} "
        f"of runs, measuring {o.extra.min():.0f} to {o.extra.max():.0f} extra configurations.",
    ]


def big(x):
    return f"{int(x):.4e}" if x.isdigit() and len(x) > 15 else x


def pilot():
    df, meta = load("pilot")
    a = agg(df)
    cfg = meta["params"]
    parts = [
        "# Pilot\n",
        f"{cfg['reps']} repetitions per cell, alpha = {cfg['alpha']}, q = {cfg['q']}, "
        f"delta = {cfg['pac_delta']}, thresholds P25/P50/P75. FDR and power are of the certain "
        "selection; m-FDR and m-pow are those of the mirror selection (violations, diagnostic "
        "only); both = configurations selected by both; X ok = share of repetitions whose "
        "mis-covered count stays under the P4 bound; cover = share where the P4 count interval "
        "holds the true count; width = (possible - certain) / |V|.\n",
    ]
    cols = ["n", "pct", "fdr", "power", "size", "mfdr", "mpower", "both", "x_ok", "cover", "width"]
    heads = ["n", "P", "FDR", "power", "|S|", "m-FDR", "m-pow", "both", "X ok", "cover", "width"]
    fmt = ["{:.0f}", "{:.0f}", "{:.3f}", "{:.3f}", "{:.0f}", "{:.3f}", "{:.3f}", "{:.0f}",
           "{:.2f}", "{:.2f}", "{:.3f}"]  # fmt: skip
    for system, d in a.groupby("system"):
        n_valid = int(df[df.system == system].n_valid.iloc[0])
        parts.append(f"\n## {system} (|V| = {n_valid})\n")
        parts.append(md(d, cols, heads, fmt))
    o = (
        df[df.pct == 50]
        .groupby(["system", "n"])
        .agg(regret=("regret", "mean"), hit=("regret", lambda r: (r == 0).mean()),
             extra=("extra", "mean"), mae=("mae_grid", "mean"), raw=("mae_raw", "mean"),
             qhat=("qhat", "mean"))
        .reset_index()
    )  # fmt: skip
    parts.append("\n## OPTIMISE and predictor\n")
    parts.append(
        md(o, ["system", "n", "regret", "hit", "extra", "mae", "raw", "qhat"],
           ["system", "n", "regret", "found", "extra", "MAE grid", "MAE raw", "qhat"],
           ["{}", "{:.0f}", "{:.3f}", "{:.2f}", "{:.0f}", "{:.3f}", "{:.3f}", "{:.3f}"])
    )  # fmt: skip
    dj = ROOT / "results/raw/pilot/dimacs.json"
    if dj.exists():
        d = pd.DataFrame(json.loads(dj.read_text()))
        d["tn"] = d.tn.fillna("over memory cap").map(big)
        d["ganak"] = d.ganak.map(big)
        parts.append("\n## DIMACS\n")
        parts.append(md(d, ["model", "vars", "clauses", "width", "tn", "ganak", "ganak_s"],
                        ["model", "vars", "clauses", "width", "contraction", "ganak", "ganak s"],
                        ["{}", "{}", "{}", "{:.0f}", "{}", "{}", "{:.1f}"]))  # fmt: skip
    parts.append("\n" + "\n".join(f"{i}. {t}" for i, t in enumerate(reading(df, a, o), 1)))
    parts.append(f"\ncommit {meta['commit'][:8]}, {meta['cpu']}, {meta['seconds']:.0f} s.\n")
    return a, o, "\n".join(parts)


if __name__ == "__main__":
    a, o, text = pilot()
    if len(sys.argv) > 1 and sys.argv[1] == "show":
        print(text)
        sys.exit()
    (ROOT / "results/pilot.md").write_text(text)
