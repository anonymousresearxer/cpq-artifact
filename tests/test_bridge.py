import numpy as np
import pytest

pytest.importorskip("juliacall")

from cpq import bridge  # noqa: E402

FIXTURES = ["p1", "p2", "p3"]
QUERIES = ["cfg", "measured", "hist", "pairs", "pairs_vc", "pred", "noise"]


def tables(inst):
    raw = bridge.dump_instance(inst)
    return {
        o: {"n": int(t["n"]), "cols": {c: np.asarray(list(v)) for c, v in t["cols"].items()}}
        for o, t in raw.items()
    }


def einsum(spec, tabs):
    sort = {i["name"]: i for i in spec["indices"]}
    values = {}
    for i in spec["indices"]:
        if i["kind"] == "attr":
            cols = [
                tabs[t["table"]]["cols"][a["column"]]
                for t in spec["tensors"]
                for a in t["axes"]
                if a["index"] == i["name"]
            ]
            values[i["name"]] = np.unique(np.concatenate(cols))
    size = {
        n: tabs[i["sort"]]["n"] if i["kind"] == "ob" else len(values[n]) for n, i in sort.items()
    }
    ops = []
    for t in spec["tensors"]:
        n = tabs[t["table"]]["n"]
        pos = []
        for a in t["axes"]:
            if a["column"] == "_id":
                pos.append(np.arange(n))
            elif sort[a["index"]]["kind"] == "ob":
                pos.append(tabs[t["table"]]["cols"][a["column"]].astype(np.int64) - 1)
            else:
                pos.append(
                    np.searchsorted(values[a["index"]], tabs[t["table"]]["cols"][a["column"]])
                )
        arr = np.zeros([size[a["index"]] for a in t["axes"]], dtype=np.int64)
        np.add.at(arr, tuple(pos), 1)
        ops.append(arr)
    out = np.einsum(spec["einsum"], *ops)
    keys = [np.arange(size[o]) for o in spec["output"]]
    res = {}
    for idx in np.argwhere(out != 0):
        key = tuple(
            int(k[i] + 1) if sort[o]["kind"] == "ob" else values[o][i].item()
            for o, k, i in zip(spec["output"], keys, idx, strict=True)
        )
        res[key] = int(out[tuple(idx)])
    if not spec["output"] and out != 0:
        res[()] = int(out)
    return res


@pytest.mark.parametrize("fx", FIXTURES)
@pytest.mark.parametrize("q", QUERIES)
def test_einsum_matches_query(fx, q):
    inst = bridge.fixture(fx)
    assert einsum(bridge.query_spec(q), tables(inst)) == bridge.answer(inst, q)


@pytest.mark.parametrize("fx", FIXTURES)
def test_instance_round_trip(fx):
    inst = bridge.fixture(fx)
    back = bridge.load_instance(tables(inst))
    assert tables(back).keys() == tables(inst).keys()
    for q in QUERIES:
        assert bridge.answer(back, q) == bridge.answer(inst, q)


@pytest.mark.parametrize("q", QUERIES)
def test_spec_round_trip(q):
    spec = bridge.query_spec(q)
    assert bridge.dump_spec(spec) == spec


def test_unpartitioned_cfg_rejected():
    t = tables(bridge.fixture("p1"))
    t["Cfg"]["n"] += 1
    with pytest.raises(Exception, match="Cfg"):
        bridge.load_instance(t)
