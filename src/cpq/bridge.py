import json

import numpy as np
from juliacall import Main as jl

jl.seval("using QVMPlus, Catlab")
qvm = jl.QVMPlus


def load_instance(tables):
    """Build a QVM+ instance from {object: {"n": rows, "cols": {column: values}}}.

    Arrows hold 1-based row indices; Name columns hold strings, Z integers and R floats.
    CCs and VCs must partition Cfg.
    """
    return qvm.load_instance(tables)


def dump_instance(inst):
    return qvm.dump_instance(inst)


def fixture(name):
    return qvm.fixture(name)


def query_spec(name):
    return json.loads(qvm.query_spec(name))


def dump_spec(spec):
    return json.loads(qvm.dump_spec(qvm.load_spec(json.dumps(spec))))


def answer(inst, name):
    return {tuple(row[:-1]): int(row[-1]) for row in qvm.answer(inst, name)}


def tables(inst):
    raw = dump_instance(inst)
    return {
        o: {"n": int(t["n"]), "cols": {c: np.asarray(list(v)) for c, v in t["cols"].items()}}
        for o, t in raw.items()
    }


def sigma_fixture(name):
    return jl.seval("n -> sigma_closed(Dict(fixtures_qvm())[n])")(name)


def chase_check(name):
    """Whether Catlab's chase is isomorphic to the closed form on a QVM fixture, and its time."""
    ok, secs = jl.seval(
        "n -> (I = Dict(fixtures_qvm())[n]; t = @elapsed(C = sigma_chase(I)); "
        "(is_isomorphic(sigma_closed(I), C), t))"
    )(name)
    return bool(ok), float(secs)
