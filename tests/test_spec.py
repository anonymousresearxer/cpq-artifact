import numpy as np
import pytest

pytest.importorskip("juliacall")

from cpq import bridge, spec  # noqa: E402

FIXTURES = ["p1", "p2", "p3"]
QUERIES = ["cfg", "measured", "hist", "pairs", "pairs_vc", "pred", "noise"]
D = 12


def annotated(tabs):
    yh = tabs["Pred"]["cols"]["yhat"]
    return {"Pred": np.rint((yh - yh.min()) / 0.5).astype(int)} if len(yh) else {}


@pytest.mark.parametrize("fx", FIXTURES)
@pytest.mark.parametrize("q", QUERIES)
def test_engines_agree_and_identities_hold(fx, q):
    inst = bridge.fixture(fx)
    tabs = bridge.tables(inst)
    s = bridge.query_spec(q)
    res = spec.evaluate(s, tabs, annotated(tabs), D)
    want = bridge.answer(inst, q)
    assert res["N"] == want
    assert spec.duck(s, tabs) == want


def test_pred_histogram_is_the_unmeasured_prediction_histogram():
    tabs = bridge.tables(bridge.fixture("p1"))
    ann = annotated(tabs)["Pred"]
    unmeasured = np.isin(tabs["Pred"]["cols"]["pcfg"], tabs["VCs"]["cols"]["vc_cfg"])
    res = spec.evaluate(bridge.query_spec("pred"), tabs, {"Pred": ann}, D)
    assert res["H"][()] == np.bincount(np.minimum(ann[unmeasured], D + 1), minlength=D + 2).tolist()
    assert res["T"][()] == ann[unmeasured].min()


@pytest.mark.parametrize("name", ["tiny", "multi_qa", "measured_p1"])
def test_closed_form_images_agree(name):
    inst = bridge.sigma_fixture(name)
    tabs = bridge.tables(inst)
    for q in QUERIES:
        s = bridge.query_spec(q)
        assert spec.evaluate(s, tabs, None, D)["N"] == bridge.answer(inst, q) == spec.duck(s, tabs)
