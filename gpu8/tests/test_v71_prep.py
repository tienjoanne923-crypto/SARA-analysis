"""V7.1 preparation guardrails (plan sections 4-6, 9).

* ESM windows: every residue carries total weight exactly 1 after overlap
  normalisation, and windows never exceed 1,022 residues;
* expression: unobserved cells are zero, observed cells are centred per gene,
  and a measured zero stays observed;
* HPO aggregation: weighted mean, de-duplication by the maximum weight and a
  False mask for an entity without terms;
* frequency weights follow the registered table;
* the tissue alias table fires on organ words and does not confuse
  "intrauterine" with the endometrium or "gallbladder" with the bladder.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gpu8v71.encoders71 import esm_windows, window_weights  # noqa: E402
from gpu8v71.expression71 import ALIAS_RULES, centre  # noqa: E402
from gpu8v71.featurepack71 import aggregate_terms, freq_weight  # noqa: E402


def test_window_weights_sum_to_one():
    for L in (1, 50, 1022, 1023, 1500, 1916, 3000, 10000, 35991):
        wins = esm_windows(L)
        assert all(e - s <= 1022 for s, e in wins)
        assert wins[0][0] == 0 and wins[-1][1] == L
        tot = np.zeros(L)
        for (s, e), w in zip(wins, window_weights(L)):
            tot[s:e] += w
        assert np.allclose(tot, 1.0)


def test_expression_centring_and_masks():
    E = np.array([[0.0, 10.0, 5.0], [3.0, 0.0, 0.0]])
    o = np.array([[True, True, False], [True, True, True]])
    C = centre(E, o)
    assert C[0, 2] == 0.0
    assert abs(C[0, :2].sum()) < 1e-9 and abs(C[1].sum()) < 1e-9
    assert C[0, 0] != 0.0          # a measured zero is information, not missing


def test_aggregate_terms():
    vec = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    pos = {"A": 0, "B": 1, "C": 2}
    tab = pd.DataFrame({"entity": ["d1", "d1", "d1", "d2"], "term": ["A", "B", "A", "C"],
                        "weight": [1.0, 3.0, 0.5, 2.0]})
    X, m, info = aggregate_terms(vec, pos, tab, 3, {"d1": 0, "d2": 1, "d3": 2})
    assert np.allclose(X[0], [0.25, 0.75])          # A (max weight 1) and B (3)
    assert np.allclose(X[1], [1.0, 1.0])
    assert m.tolist() == [True, True, False] and np.all(X[2] == 0)


def test_frequency_weights():
    assert freq_weight("HP:0040281") == 0.895
    assert freq_weight("3/4") == 0.75
    assert freq_weight("12%") == 0.12
    assert freq_weight("") == 1.0
    assert freq_weight("0/5") == 0.025


def _fired(text):
    return {rid for rid, rx, _, _ in ALIAS_RULES if re.search(rx, text.lower())}


def test_alias_rules():
    assert "heart" in _fired("Dilated cardiomyopathy ; Heart failure")
    assert "endometrium" not in _fired("Intrauterine growth retardation")
    assert "endometrium" in _fired("uterine leiomyoma")
    assert "bladder" not in _fired("gallbladder dyskinesia")
    assert "gallbladder" in _fired("gallbladder dyskinesia")
    assert "thyroid" not in _fired("primary hyperparathyroidism")
    assert "parathyroid" in _fired("primary hyperparathyroidism")
    assert not _fired("xyz syndrome 3")
