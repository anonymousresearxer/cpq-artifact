import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class VM:
    name: str
    names: list[str]
    nvars: int
    clauses: list[list[int]]
    defs: list[tuple[int, str, list[int]]] = field(default_factory=list)

    def valid(self, X: np.ndarray) -> np.ndarray:
        """Rows of the 0/1 option matrix X that satisfy the CNF."""
        cols = {i + 1: X[:, i].astype(bool) for i in range(len(self.names))}

        def lit(lt: int) -> np.ndarray:
            return cols[lt] if lt > 0 else ~cols[-lt]

        for v, op, kids in self.defs:
            parts = [lit(k) for k in kids]
            cols[v] = np.logical_and.reduce(parts) if op == "and" else np.logical_or.reduce(parts)
        ok = np.ones(len(X), dtype=bool)
        for c in self.clauses:
            ok &= np.logical_or.reduce([lit(lt) for lt in c])
        return ok


SEQ = 4


def _children(el: ET.Element, tag: str) -> list[str]:
    node = el.find(tag)
    if node is None:
        return []
    return [o.text.strip() for o in node.findall("options") if o.text and o.text.strip()]


_TOKEN = re.compile(r"\s*(!|&|\||\(|\)|[^\s!&|()]+)")


def _parse(text: str, idx: dict[str, int]):
    toks = _TOKEN.findall(text)
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else None

    def eat():
        nonlocal pos
        pos += 1
        return toks[pos - 1]

    def atom():
        t = eat()
        if t == "!":
            return ("not", atom())
        if t == "(":
            e = disj()
            if eat() != ")":
                raise ValueError(f"unbalanced parentheses in {text!r}")
            return e
        if t not in idx:
            raise ValueError(f"unknown option {t!r} in constraint {text!r}")
        return ("var", idx[t])

    def conj():
        parts = [atom()]
        while peek() == "&":
            eat()
            parts.append(atom())
        return parts[0] if len(parts) == 1 else ("and", parts)

    def disj():
        parts = [conj()]
        while peek() == "|":
            eat()
            parts.append(conj())
        return parts[0] if len(parts) == 1 else ("or", parts)

    e = disj()
    if pos != len(toks):
        raise ValueError(f"trailing tokens in {text!r}")
    return e


def _nnf(e, neg=False):
    kind = e[0]
    if kind == "var":
        return ("lit", -e[1] if neg else e[1])
    if kind == "not":
        return _nnf(e[1], not neg)
    swap = {"and": "or", "or": "and"}
    return (swap[kind] if neg else kind, [_nnf(p, neg) for p in e[1]])


class _Encoder:
    """Tseitin encoding with full equivalences, so auxiliary variables are determined."""

    def __init__(self, nvars: int):
        self.nvars = nvars
        self.clauses: list[list[int]] = []
        self.defs: list[tuple[int, str, list[int]]] = []

    def lit(self, e) -> int:
        if e[0] == "lit":
            return e[1]
        kids = [self.lit(p) for p in e[1]]
        self.nvars += 1
        a = self.nvars
        self.defs.append((a, e[0], kids))
        if e[0] == "and":
            self.clauses += [[-a, k] for k in kids] + [[a] + [-k for k in kids]]
        else:
            self.clauses += [[-a] + kids] + [[a, -k] for k in kids]
        return a

    def require(self, e) -> None:
        if e[0] == "and":
            for p in e[1]:
                self.require(p)
        elif e[0] == "lit":
            self.clauses.append([e[1]])
        else:
            self.clauses.append([self.lit(p) for p in e[1]])


def read_splc(path: str | Path, name: str | None = None) -> VM:
    """Binary options of an SPL Conqueror variability model, encoded to CNF."""
    path = Path(path)
    root = ET.parse(path).getroot()
    names, parent, optional, excl, impl = [], {}, {}, {}, {}
    for o in root.find("binaryOptions").findall("configurationOption"):
        n = o.findtext("name").strip()
        names.append(n)
        parent[n] = (o.findtext("parent") or "").strip() or None
        optional[n] = (o.findtext("optional") or "True").strip().lower() == "true"
        excl[n] = _children(o, "excludedOptions")
        impl[n] = _children(o, "impliedOptions")
    idx = {n: i + 1 for i, n in enumerate(names)}
    cl: list[list[int]] = []
    enc = _Encoder(len(names))
    groups: list[tuple[str | None, list[str]]] = []
    seen: set[str] = set()
    for n in names:
        if optional[n] or not excl[n] or n in seen:
            continue
        p = parent[n]
        grp = {n} | {e for e in excl[n] if not optional[e] and parent[e] == p}
        seen |= grp
        groups.append((p, sorted(grp)))
    # wide exclusion cliques get a sequential encoding, which keeps the contraction width low
    wide = {n: gi for gi, (_, g) in enumerate(groups) if len(g) > SEQ for n in g}
    for n in names:
        v, p = idx[n], parent[n]
        if p:
            cl.append([-v, idx[p]])
        if not optional[n] and not excl[n]:
            cl.append([-idx[p], v] if p else [v])
        cl += [[-v, -idx[e]] for e in excl[n] if not (n in wide and wide.get(e) == wide[n])]
        cl += [[-v, idx[i]] for i in impl[n]]
    for p, grp in groups:
        head = [-idx[p]] if p else []
        if len(grp) <= SEQ:
            cl.append(head + [idx[g] for g in grp])
            continue
        prev = idx[grp[0]]
        for g in grp[1:]:
            cl.append([-idx[g], -prev])
            prev = enc.lit(("or", [("lit", prev), ("lit", idx[g])]))
        cl.append(head + [prev])
    bc = root.find("booleanConstraints")
    if bc is not None:
        for c in bc.findall("constraint"):
            if c.text and c.text.strip():
                enc.require(_nnf(_parse(c.text.strip(), idx)))
    return VM(name or path.parent.name, names, enc.nvars, cl + enc.clauses, enc.defs)


def read_dimacs(path: str | Path, name: str | None = None) -> VM:
    path = Path(path)
    nvars, clauses, cur = 0, [], []
    for line in path.read_text().splitlines():
        if line.startswith("p cnf"):
            nvars = int(line.split()[2])
        elif line and line[0] not in "c%":
            for t in line.split():
                if t == "0":
                    clauses.append(cur)
                    cur = []
                else:
                    cur.append(int(t))
    return VM(name or path.stem, [f"x{i}" for i in range(1, nvars + 1)], nvars, clauses)
