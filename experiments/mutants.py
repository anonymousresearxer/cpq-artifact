import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("CPQ_OUT", ROOT / "results/raw")) / "mutants"
PY = ROOT / ".venv/bin/python"
CON, NET, EIN, SPEC = (
    "src/cpq/tn/contract.py",
    "src/cpq/tn/network.py",
    "julia/src/einsum.jl",
    "src/cpq/spec.py",
)

# (id, component, file, original text, mutated text)
MUTANTS = [
    ("E1", "H_D product", CON, "if d < L - 1:", "if d < L - 2:"),
    ("E2", "H_D overflow", CON, "suf[..., L - 1 - d]", "suf[..., max(L - 2 - d, 0)]"),
    ("E3", "T_D saturation", CON, "S = np.minimum(align(A, ia) + align(B, ib), top)", "S = align(A, ia) + align(B, ib)", True),
    ("E4", "T_D zero", CON, "np.where(a == 1, 0, top)", "np.where(a == 1, 0, top - 1)"),
    ("E5", "H_D offset", CON, "shift = pl.const + base\n    if shift > D:", "shift = pl.const\n    if shift > D:"),
    ("E6", "mindeg", CON, "return int(nz[0]) if len(nz)", "return int(nz[-1]) if len(nz)"),
    ("E7", "ev1", CON, "return int(sum(int(x) for x in h))", "return int(sum(int(x) for x in h[:-1]))"),
    ("E8", "prefix sums", CON, "h[: k + 1]", "h[:k]"),
    ("E9", "hyperedges", CON, "counts[x] += (x in keep) - n_in", "counts[x] -= n_in"),
    ("E10", "free variables", CON, "if v not in used and v not in net.fixed", "if v not in used"),
    ("E11", "N free factor", CON, "return int(val) << pl.free", "return int(val)"),
    ("E12", "H_D unit factor", CON, "if not A[..., 1:].any():", "if not A[..., 2:].any():"),
    ("N1", "clause tensor", NET, "if any(a[pos[abs(lt)]] == (lt > 0) for lt in clause):", "if all(a[pos[abs(lt)]] == (lt > 0) for lt in clause):"),
    ("N2", "clause chain", NET, "t[a] = int(bool(hit))", "t[a] = 1"),
    ("N3", "unit propagation", NET, "fixed[abs(u)] = u > 0", "fixed[abs(u)] = u < 0"),
    ("N4", "option weights", NET, "np.array([-min(0, k), k - min(0, k)], dtype=np.int64)", "np.array([0, k], dtype=np.int64)"),
    ("N5", "pair weights", NET, "e = np.array([[0, 0], [0, k]], dtype=np.int64) - min(0, k)", "e = np.array([[0, 0], [0, k]], dtype=np.int64)"),
    ("S1", "spec evaluator", SPEC, 'int(i) + 1 if kind[o] == "ob"', 'int(i) if kind[o] == "ob"'),
    ("C1", "compiler output order", EIN, "output = vars[uwd[:outer_junction]]", "output = reverse(vars[uwd[:outer_junction]])"),
    ("C2", "compiler junctions", EIN, "Axis(String(col), vars[j])", "Axis(String(col), vars[mod1(j + 1, length(vars))])"),
    ("C3", "compiler boxes", EIN, "tensors = map(parts(uwd, :Box)) do b", "tensors = map(parts(uwd, :Box)[1:max(end-1, 1)]) do b"),
    ("C4", "compiler axes", EIN, "String([letters[a.index] for a in t.axes])", "String(reverse([letters[a.index] for a in t.axes]))"),
    ("C5", "compiler index order", EIN, "sort(collect(sorts))]", "sort(collect(sorts); rev=true)]"),
    ("C6", "compiler sort check", EIN, "get!(sorts, j, sort) == sort || error(\"junction $(vars[j]) has two sorts\")", "get!(sorts, j, sort)", True),
]  # fmt: skip

SUITE_A = [
    "tests/test_semiring.py", "tests/test_exact.py", "tests/test_vm.py", "tests/test_dimacs.py",
    "tests/test_spec.py", "tests/test_bridge.py", "tests/test_labels.py",
    "tests/test_predictors.py", "tests/test_conformal.py", "tests/test_selection.py",
    "tests/test_runner.py", "tests/test_kernels.py", "tests/test_edges.py",
]  # fmt: skip
SUITE_O = ["tests/test_oracle.py", "tests/test_oracle_zero.py"]


def pytest(paths):
    r = subprocess.run(
        [PY, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "--hypothesis-seed=0", *paths],
        cwd=ROOT, capture_output=True, text=True, timeout=1800,
    )  # fmt: skip
    return r.returncode != 0, r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""


def julia_tests():
    r = subprocess.run(
        ["julia", "--project=julia", "-e", "using Test; include(\"julia/test/queries.jl\"); include(\"julia/test/tables.jl\")"],
        cwd=ROOT, capture_output=True, text=True, timeout=1800,
    )  # fmt: skip
    return r.returncode != 0


def run_mutant(mid, comp, file, old, new, equivalent=False):
    path = ROOT / file
    text = path.read_text()
    assert text.count(old) == 1, f"{mid}: pattern occurs {text.count(old)} times"
    path.write_text(text.replace(old, new))
    try:
        row = dict(id=mid, component=comp, file=file, equivalent=equivalent)
        a_py, _ = pytest(SUITE_A)
        a_jl = julia_tests() if file.endswith(".jl") else False
        o, _ = pytest(SUITE_O)
        row |= dict(killed_by_tests=bool(a_py or a_jl), killed_by_oracle=bool(o))
        row["killed_by_both"] = row["killed_by_tests"] or row["killed_by_oracle"]
    finally:
        path.write_text(text)
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for mid, _, file, old, _, *_ in MUTANTS:
        n = (ROOT / file).read_text().count(old)
        if n != 1:
            sys.exit(f"{mid}: pattern occurs {n} times")
    base_a, tail_a = pytest(SUITE_A)
    base_o, tail_o = pytest(SUITE_O)
    if base_a or base_o:
        sys.exit(f"unmutated suites fail: {tail_a} / {tail_o}")
    rows = []
    for m in MUTANTS:
        rows.append(run_mutant(*m))
        print(rows[-1], flush=True)
        (OUT / "mutants.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
