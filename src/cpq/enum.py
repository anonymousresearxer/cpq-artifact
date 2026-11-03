from .vm import VM


def models(vm: VM, limit: int = 10**6):
    """All satisfying assignments of a small CNF as 0/1 tuples."""
    out = []

    def simplify(cl, v, val):
        nxt = []
        for c in cl:
            if any(abs(x) == v and (x > 0) == val for x in c):
                continue
            c2 = [x for x in c if abs(x) != v]
            if not c2:
                return None
            nxt.append(c2)
        return nxt

    def go(cl, asg):
        while True:
            unit = next((c[0] for c in cl if len(c) == 1), None)
            if unit is None:
                break
            asg = asg | {abs(unit): unit > 0}
            cl = simplify(cl, abs(unit), unit > 0)
            if cl is None:
                return
        if not cl:
            free = [v for v in range(1, vm.nvars + 1) if v not in asg]
            for bits in range(2 ** len(free)):
                full = asg | {v: bool(bits >> i & 1) for i, v in enumerate(free)}
                out.append(tuple(int(full[v]) for v in range(1, vm.nvars + 1)))
                if len(out) > limit:
                    raise OverflowError
            return
        v = abs(cl[0][0])
        for val in (True, False):
            nxt = simplify(cl, v, val)
            if nxt is not None:
                go(nxt, asg | {v: val})

    go([list(c) for c in vm.clauses], {})
    return out
