import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .vm import VM, read_splc

DATA = Path(os.environ.get("CPQ_DATA", Path(__file__).resolve().parents[2] / "data"))
MEASURED = DATA / "Distance-Based_Data/SupplementaryWebsite/MeasuredPerformanceValues"

# Directory and response column of each subject. VP9 ships no Performance column; its response
# is the encoder wall-clock time (ElapsedTime), the other columns being CPU time, size and PSNR.
SYSTEMS = {
    "x264": ("x264", "Performance"),
    "berkeleydb": ("BerkeleyDBC", "Performance"),
    "lrzip": ("lrzip", "Performance"),
    "7z": ("7z", "Performance"),
    "polly": ("Polly", "Performance"),
    "javagc": ("JavaGC", "GC Time"),
    "vp9": ("VP9", "ElapsedTime"),
    "llvm": ("LLVM", "Performance"),
    "dune": ("Dune", "Performance"),
    "hipacc": ("Hipacc", "Performance"),
}
STUDY = list(SYSTEMS)


@dataclass
class Population:
    name: str
    vm: VM
    X: np.ndarray
    y: np.ndarray

    def __len__(self) -> int:
        return len(self.y)


def _xml_rows(path: Path, col: str):
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag != "row":
            continue
        cell = {d.get("columname") or d.get("column"): (d.text or "") for d in el.findall("data")}
        el.clear()
        yield cell["Configuration"], float(cell[col])


def _encode(rows, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    pos = {n: i for i, n in enumerate(names)}
    X = np.zeros((len(rows), len(names)), dtype=np.uint8)
    for r, (cfg, _) in enumerate(rows):
        for tok in cfg.split(","):
            tok = tok.strip()
            if tok:
                X[r, pos[tok]] = 1
    return X, np.array([y for _, y in rows])


def load(system: str) -> Population:
    sub, col = SYSTEMS[system]
    d = MEASURED / sub
    vm = read_splc(d / "FeatureModel.xml", system)
    cache = DATA / "cache" / f"{system}.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        return Population(system, vm, df[vm.names].to_numpy(np.uint8), df["y"].to_numpy())
    if system == "vp9":
        df = pd.read_csv(d / "measurements.csv", sep=";", index_col=False)
        X = df.reindex(columns=vm.names, fill_value=0).to_numpy(dtype=np.uint8)
        y = df[col].to_numpy(dtype=float)
    else:
        X, y = _encode(list(_xml_rows(d / "measurements.xml", col)), vm.names)
    # mandatory root options are implicit in some of the files
    for c in vm.clauses:
        if len(c) == 1 and c[0] > 0:
            X[:, c[0] - 1] = 1
    cache.parent.mkdir(exist_ok=True)
    out = pd.DataFrame(X, columns=vm.names).assign(y=y)
    out.to_parquet(cache, compression="zstd")
    return Population(system, vm, X, y)
