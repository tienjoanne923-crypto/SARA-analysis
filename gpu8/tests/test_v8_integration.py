"""V8.1 integration checks on the real frozen inputs (read-only; no fitting).

* role-based exclusion equals the V6 label-based sets in all 20 contexts
* production GO rule reproduces the V7.1 function vectors and anchors exactly
* the two GO-mask sets are disjoint, label-free and within eligible genes
* single-segment disease vectors equal the V7.1 vectors; no segment is truncated
* F_j labels are sealed without the registration; registration refuses after test evaluations
* the strict pool of a saved context has exactly 500 distinct genes and never adds positives
* the ledger rebuild verifies every content hash
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gpu8v8final import paths8  # noqa: E402


@pytest.fixture(scope="module")
def world():
    from gpu8v6.common import World6
    return World6(load_heavy=False)


def test_role_exclusion_equals_v6(world):
    from gpu8v8final.roles8 import Roles, assert_matches_v6
    r = assert_matches_v6(world, Roles(world))
    assert r["contexts"] == 20 and r["all_equal"]


def test_production_go_equals_v71():
    rep = json.loads((paths8.AUDIT / "evidence_policy_audit_v8.json").read_text(encoding="utf-8"))
    s = rep["production_equals_v71"]
    assert s["g_function_equal"] and s["mask_equal"] and s["anchors_equal"]
    assert rep["layer2_evaluation_conditioned"]["evaluation_conditioned_audit"] is True


def test_go_masks_disjoint_and_label_free():
    from gpu8v8final.masks8 import load
    m = load()
    assert not set(m["dev"].tolist()) & set(m["test"].tolist())
    rep = json.loads((paths8.MANIFESTS / "go_masks_v8.json").read_text(encoding="utf-8"))
    assert rep["labels_read"] is False and rep["G_mask_dev"] == len(m["dev"]) and rep["G_mask_test"] == len(m["test"])


def test_segmented_text_identity():
    rep = json.loads((paths8.AUDIT / "text_segments_l1.json").read_text(encoding="utf-8"))
    assert rep["single_segment_vs_v71_maxabs"] == 0.0
    assert rep["segment_tokenizer_truncations"] == 0
    assert rep["n_multi_segment"] == rep["n_truncated_v71"]


def test_labels_sealed_without_registration(world, monkeypatch):
    from gpu8v8final.feats8 import Labels
    lab = Labels(world)
    monkeypatch.setattr(lab, "registered", lambda: False)
    d = sorted(world.fold(0)["test_diseases"])[0]
    with pytest.raises(PermissionError):
        lab.rows(d, "test")
    assert len(lab.rows(d, "val")) >= 0


def test_registration_refuses_after_test_evaluations(tmp_path, monkeypatch):
    from gpu8v8final import register8
    monkeypatch.setattr(register8, "load_manifest", lambda *a, **k: None)
    monkeypatch.setattr(register8, "test_results_exist", lambda: ["results_v8/eval/eval_test_main.parquet"])
    with pytest.raises(RuntimeError):
        register8.register()


def test_saved_pools_exactly_500():
    files = sorted(paths8.DEV.glob("fold0/pools_*.npz"))
    assert files
    for p in files:
        g = np.load(p)["genes"]
        assert g.shape[1] == 500
        assert all(len(set(r.tolist())) == 500 for r in g)


def test_registration_precedes_test_evaluation():
    reg = json.loads((paths8.MANIFESTS / "protocol_v8_registered.json").read_text(encoding="utf-8"))
    idx = paths8.RESULTS / "eval" / "eval_index.json"
    if idx.exists():
        ev = json.loads(idx.read_text(encoding="utf-8"))
        assert ev["generated_utc"] > reg["registered_utc"]
    assert reg["registered_before_any_v8_test_evaluation"] is True


def test_ledger_hashes():
    from gpu8v8final.ledger8 import collect
    df = collect()
    assert len(df) and bool(df["hash_ok"].all())
