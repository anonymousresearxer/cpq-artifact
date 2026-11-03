# cpq

Artifact for the paper "Certain, Possible or Predicted? Distribution-Free Answers to Quality Queries over Partially Measured Configuration Spaces".

## Requirements

- Linux x86_64 or arm64 (glibc 2.28+), macOS 12+, or WSL2 with Ubuntu
- Python 3.14 through uv, Julia 1.13.2 through juliaup
- 4 cores, 8 GB of RAM and 15 GB of disk; no GPU
- The tensor-network evaluator caps its memory at 32 GiB per process (`CPQ_MEM_GB`). On a smaller machine, lower it: the cases above the cap are reported as limits, as for the two largest DIMACS models of RQ4.

## Setup

```bash
uv python install 3.14
uv sync --extra julia
scripts/get_data.sh
julia --project=julia -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

`scripts/get_data.sh` needs network access: it downloads the data sets and the ganak 2.8.0 binary.

## Tests

```bash
uv run --extra julia pytest
uv run --extra julia pytest -m slow
```

Without `--extra julia`, uv drops juliacall and the tests that need Julia are skipped.
The Julia project pins PythonCall 0.9.36 and juliacall 0.9.36, because Catlab 0.17.6 and PythonCall 0.9.37 do not resolve together (JSON).

## Reproduction

Set `CPQ_WORKERS` to the number of cores (the commands below use 16). The raw results are not part of the repository: run the experiments first and `experiments/results.py` last, since it reads all of them and writes the tables and the figure of the paper.

| Output | Command |
| --- | --- |
| `results/pilot.md` | `CPQ_WORKERS=16 uv run python experiments/run.py experiments/pilot.yaml && uv run python experiments/report.py` |
| `results/raw/grid` (RQ2, RQ3) | `CPQ_WORKERS=16 uv run python experiments/run.py experiments/grid.yaml` |
| `results/raw/p4` | `CPQ_WORKERS=16 uv run python experiments/p4_check.py 1000` |
| `results/raw/rq1` | `uv run --extra julia python experiments/rq1.py` |
| `results/raw/rq4` | `uv run python experiments/rq4.py` |
| `results/raw/rq4/correction.json` | `uv run python experiments/rq4_corr.py` |
| `results/raw/rank/rank_budget.json` | `CPQ_WORKERS=16 uv run python experiments/rank_budget.py` |
| `results/raw/nmin/nmin.json` | `uv run python experiments/n_min.py` |
| `results/raw/mutants` | `uv run --extra julia python experiments/mutants.py` |
| `results/results.md`, `paper/tables/*.tex`, `paper/figs/rq3-optimise.pdf` | `uv run python experiments/results.py` |
| `julia/specs/*.json` | `julia --project=julia julia/specs.jl` |

The times of RQ1 and RQ4 were measured on an AMD Ryzen 9 9950X3D and depend on the machine. The widths of the tensor-network contraction orders below 2,000 tensors come from a time-limited cotengra search and can differ by one between runs; the counts, histograms and optima do not depend on them.

## License

Code: MIT (Anonymous authors). Data: Distance-Based Data (GPL-2.0) and the feature-model benchmark (MIT), downloaded by `scripts/get_data.sh` and not redistributed.
