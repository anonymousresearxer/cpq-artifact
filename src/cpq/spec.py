import cotengra as ctg
import duckdb
import numpy as np
import pandas as pd

from .tn import contract


def domains(spec: dict, tabs: dict) -> tuple[dict, dict]:
    """Size of every index and, for attribute indices, its sorted values."""
    values, size = {}, {}
    for i in spec["indices"]:
        if i["kind"] == "attr":
            cols = [
                tabs[t["table"]]["cols"][a["column"]]
                for t in spec["tensors"]
                for a in t["axes"]
                if a["index"] == i["name"]
            ]
            values[i["name"]] = np.unique(np.concatenate(cols))
            size[i["name"]] = len(values[i["name"]])
        else:
            size[i["name"]] = tabs[i["sort"]]["n"]
    return size, values


def positions(spec: dict, tabs: dict, values: dict) -> list[list[np.ndarray]]:
    """Row coordinates of every axis of every box."""
    kind = {i["name"]: i["kind"] for i in spec["indices"]}
    out = []
    for t in spec["tensors"]:
        n = tabs[t["table"]]["n"]
        pos = []
        for a in t["axes"]:
            col = tabs[t["table"]]["cols"].get(a["column"])
            if a["column"] == "_id":
                pos.append(np.arange(n))
            elif kind[a["index"]] == "ob":
                pos.append(col.astype(np.int64) - 1)
            else:
                pos.append(np.searchsorted(values[a["index"]], col))
        out.append(pos)
    return out


def keys(spec: dict, size: dict, values: dict, idx: np.ndarray) -> tuple:
    kind = {i["name"]: i["kind"] for i in spec["indices"]}
    return tuple(
        int(i) + 1 if kind[o] == "ob" else values[o][i].item()
        for o, i in zip(spec["output"], idx, strict=True)
    )


def evaluate(spec: dict, tabs: dict, ann: dict | None = None, D: int = 0) -> dict:
    """Evaluate a compiled query in N, H_D, T_D and the Boolean semiring.

    ann maps a table name to the grid index of each of its rows; unannotated rows have
    exponent 0. Each result maps an output tuple to its value.
    """
    size, values = domains(spec, tabs)
    pos = positions(spec, tabs, values)
    top, L = D + 1, D + 2
    counts, hists, trops, idx = [], [], [], []
    for t, p in zip(spec["tensors"], pos, strict=True):
        shape = tuple(size[a["index"]] for a in t["axes"])
        w = (ann or {}).get(t["table"])
        w = np.zeros(tabs[t["table"]]["n"], dtype=np.int64) if w is None else np.asarray(w)
        c = np.zeros(shape, dtype=np.int64)
        np.add.at(c, tuple(p), 1)
        h = np.zeros(shape + (L,), dtype=np.int64)
        np.add.at(h, (*p, np.minimum(w, top)), 1)
        tr = np.full(shape, top, dtype=np.int64)
        np.minimum.at(tr, tuple(p), np.minimum(w, top))
        counts.append(c)
        hists.append(h)
        trops.append(tr)
        idx.append(tuple(a["index"] for a in t["axes"]))
    n = np.einsum(spec["einsum"], *counts)
    b = np.asarray(np.einsum(spec["einsum"], *[c.astype(bool) for c in counts]))
    out = tuple(spec["output"])
    pl = _plan(idx, size, out)
    h = contract._drive(
        pl, hists, lambda A, ia, B, ib, keep: contract._conv(A, ia, B, ib, keep, L),
        lambda t, ix, keep: contract._reduce(t, ix, keep, np.sum),
        lambda u, v: contract._conv(u, (), v, (), [], L), out,
    )  # fmt: skip
    m = contract._drive(
        pl, trops, lambda A, ia, B, ib, keep: contract._pair_t(A, ia, B, ib, keep, top),
        lambda t, ix, keep: contract._reduce(t, ix, keep, np.min, initial=top),
        lambda u, v: np.minimum(u + v, top), out,
    )  # fmt: skip
    res = {"N": {}, "H": {}, "T": {}, "B": {}}
    n = np.asarray(n)
    for ix in np.argwhere(n != 0) if out else ([()] if n != 0 else []):
        k = keys(spec, size, values, ix)
        sel = tuple(ix)
        res["N"][k] = int(n[sel])
        res["H"][k] = [int(x) for x in np.asarray(h)[sel]]
        res["T"][k] = int(np.asarray(m)[sel])
    for ix in np.argwhere(b) if out else ([()] if b else []):
        res["B"][keys(spec, size, values, ix)] = True
    return res


def duck(spec: dict, tabs: dict) -> dict:
    """Answer of a compiled query by a SQL join and count in DuckDB."""
    con = duckdb.connect()
    for name, t in tabs.items():
        cols = {c: v for c, v in t["cols"].items()}
        con.register(name, pd.DataFrame({"_id": np.arange(1, t["n"] + 1), **cols}))
    froms, conds, first = [], [], {}
    for b, t in enumerate(spec["tensors"]):
        froms.append(f'"{t["table"]}" AS b{b}')
        for a in t["axes"]:
            ref = f'b{b}."{a["column"]}"'
            if a["index"] in first:
                conds.append(f"{first[a['index']]} = {ref}")
            else:
                first[a["index"]] = ref
    outs = [first[o] for o in spec["output"]]
    sql = f"SELECT {', '.join([*outs, 'COUNT(*)'])} FROM {', '.join(froms)}"
    if conds:
        sql += " WHERE " + " AND ".join(conds)
    if outs:
        sql += " GROUP BY " + ", ".join(outs)
    rows = con.execute(sql).fetchall()
    return {tuple(r[:-1]): int(r[-1]) for r in rows if r[-1]}


def _plan(idx, size, output):
    names = {x: ctg.get_symbol(k) for k, x in enumerate(dict.fromkeys(i for ix in idx for i in ix))}
    inputs = [tuple(names[x] for x in ix) for ix in idx]
    tree = ctg.array_contract_tree(
        inputs, output=tuple(names[o] for o in output),
        size_dict={names[x]: s for x, s in size.items() if x in names}, optimize="greedy",
    )  # fmt: skip
    pl = contract.Plan(tree, [], [tuple(x) for x in idx], 0, 0, 0, 0.0, 0.0)
    return pl


def identities(res: dict, D: int) -> list[str]:
    """Homomorphism identities between the four evaluations of one query."""
    bad = []
    for k, n in res["N"].items():
        h, t = res["H"][k], res["T"][k]
        if sum(h) != n:
            bad.append(f"ev1 {k}")
        if contract.mindeg(np.array(h)) != t:
            bad.append(f"mindeg {k}")
    if set(res["B"]) != set(res["N"]):
        bad.append("supp")
    return bad
