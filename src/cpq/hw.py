import os
from pathlib import Path

CPU = Path("/sys/devices/system/cpu")


def _cpus(text: str) -> set[int]:
    out = set()
    for part in text.strip().split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def _size(text: str) -> int:
    s = text.strip().upper()
    return int(s[:-1]) * 2 ** (10 * "KMG".index(s[-1]) + 10) if s[-1] in "KMG" else int(s)


def vcache_cpus() -> set[int] | None:
    """CPUs sharing the largest L3 when the L3 sizes differ, as on the X3D parts; else None."""
    dom = {}
    for d in CPU.glob("cpu[0-9]*/cache/index3"):
        try:
            if (d / "level").read_text().strip() == "3":
                dom[frozenset(_cpus((d / "shared_cpu_list").read_text()))] = _size(
                    (d / "size").read_text()
                )
        except OSError, ValueError:
            return None
    if len(set(dom.values())) < 2:
        return None
    return set(max(dom, key=dom.get))


def pin_vcache() -> set[int] | None:
    """Pin this process to the V-Cache CCD if there is one; the CPUs used, or None."""
    cpus = vcache_cpus()
    if not cpus or not hasattr(os, "sched_setaffinity"):
        return None
    cpus &= os.sched_getaffinity(0)
    if cpus:
        os.sched_setaffinity(0, cpus)
    return cpus or None


def physical_cores() -> int:
    """Physical cores this process may use, SMT siblings counted once."""
    if not hasattr(os, "sched_getaffinity"):
        return os.cpu_count() or 1
    cpus = os.sched_getaffinity(0)
    cores = set()
    for c in cpus:
        try:
            cores.add((CPU / f"cpu{c}/topology/core_cpus_list").read_text().strip())
        except OSError:
            return len(cpus)
    return len(cores)
