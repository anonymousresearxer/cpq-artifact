import re
import subprocess
import tempfile
from pathlib import Path

from .data import DATA
from .vm import VM

BIN = DATA / "bin/ganak"


def write_dimacs(vm: VM, path: Path) -> None:
    lines = [f"p cnf {vm.nvars} {len(vm.clauses)}"]
    lines += [" ".join(map(str, c)) + " 0" for c in vm.clauses]
    path.write_text("\n".join(lines) + "\n")


def count(vm: VM, timeout: float = 1800) -> int:
    """Exact model count from ganak 2.8.0 with --prob 0."""
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "vm.cnf"
        write_dimacs(vm, f)
        out = subprocess.run(
            [str(BIN), "--prob", "0", str(f)], capture_output=True, text=True, timeout=timeout
        ).stdout
    m = re.search(r"^c s exact arb int (\d+)", out, re.M)
    if m is None:
        raise RuntimeError("no model count in ganak output")
    return int(m.group(1))
