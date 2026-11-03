import json
import time
from pathlib import Path

from cpq import ganak
from cpq.data import DATA
from cpq.tn import contract
from cpq.vm import read_dimacs

OUT = Path(__file__).resolve().parents[1] / "results/raw/pilot"


def main():
    rows = []
    for name in ["berkeleydb", "busybox-1.18.0", "financial01"]:
        vm = read_dimacs(DATA / "dimacs" / f"{name}.dimacs")
        t0 = time.time()
        pl = contract.plan(vm)
        tp = time.time() - t0
        row = {"model": name, "vars": vm.nvars, "clauses": len(vm.clauses),
               "width": pl.width, "plan_s": tp}  # fmt: skip
        try:
            t0 = time.time()
            row["tn"] = str(contract.count(vm, nfeat=vm.nvars, pl=pl))
            row["tn_s"] = time.time() - t0
        except contract.TooBig:
            row["tn"] = None
        t0 = time.time()
        row["ganak"] = str(ganak.count(vm))
        row["ganak_s"] = time.time() - t0
        rows.append(row)
        print(name, flush=True)
    (OUT / "dimacs.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
