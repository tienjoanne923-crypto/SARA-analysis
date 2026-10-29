"""V6 section 4.4 (second route) — disease-ontology retrieval that works without HPO.

Disease positions come from the MeSH tree numbers CTD MEDIC provides (and, for
OMIM concepts without tree numbers, from their CTD parents).  Each disease is
represented by the set of tree prefixes above it plus its own identity node.  A
node's information content is IC(t) = -log(|diseases under t| / N) over all
project diseases (structure only, no labels).

Similarity to a reference disease is the IC of the most informative shared node
(Resnik); nodes whose IC is below ``min_ic`` (very broad ancestors such as a
whole MeSH category) are ignored, so common diseases are not pooled into one
undiscriminating neighbourhood.  The top ``k`` references transfer their visible
genes exactly as the phenotype retriever does.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse

EPS = 1e-12


def disease_nodes(diseases: pd.DataFrame) -> dict[str, set]:
    own: dict[str, set] = {}
    for r in diseases.itertuples(index=False):
        nodes = set()
        for t in str(r.tree_numbers or "").split("|"):
            t = t.strip()
            if not t or t == "nan":
                continue
            path = t.split("/")[0]
            parts = path.split(".")
            for i in range(1, len(parts) + 1):
                nodes.add("T:" + ".".join(parts[:i]))
        own[r.disease_id] = nodes
    parents = {r.disease_id: [p for p in str(r.parent_ids or "").split("|") if p and p != "nan"]
               for r in diseases.itertuples(index=False)}
    out: dict[str, set] = {}
    for did, nodes in own.items():
        acc = set(nodes)
        if not acc:
            # OMIM concepts: inherit the parents' tree positions (one level only)
            for p in parents.get(did, []):
                acc |= own.get(p, set())
                acc.add("D:" + p)
        acc.add("D:" + did)
        out[did] = acc
    return out


class OntologyIndex:
    def __init__(self, diseases: pd.DataFrame, min_ic: float = 3.0):
        self.nodes = disease_nodes(diseases)
        n = len(self.nodes)
        count: dict[str, int] = defaultdict(int)
        for s in self.nodes.values():
            for t in s:
                count[t] += 1
        self.ic = {t: -math.log(c / n) for t, c in count.items()}
        self.min_ic = min_ic
        self.members: dict[str, list[str]] = defaultdict(list)
        for d, s in self.nodes.items():
            for t in s:
                self.members[t].append(d)
        self.n = n

    def informative_nodes(self, did: str) -> list[tuple[str, float]]:
        s = self.nodes.get(did, set())
        own = "D:" + did
        out = [(t, self.ic[t]) for t in s if t != own and self.ic[t] >= self.min_ic]
        return sorted(out, key=lambda x: -x[1])

    def level(self, did: str) -> str:
        s = self.nodes.get(did, set())
        if len(s) <= 1:
            return "none"
        if not self.informative_nodes(did):
            return "broad_only"
        return "informative"

    def neighbours(self, did: str, reference: set, k: int = 20) -> list[tuple[str, float]]:
        best: dict[str, float] = {}
        for t, ic in self.informative_nodes(did):
            for d2 in self.members[t]:
                if d2 == did or d2 not in reference:
                    continue
                if ic > best.get(d2, 0.0):
                    best[d2] = ic
        items = sorted(best.items(), key=lambda x: (-x[1], x[0]))[:k]
        return items


def ontology_transfer(did: str, onto: OntologyIndex, reference: set, A: sparse.csr_matrix,
                      d_ix: dict, k: int = 20) -> tuple[np.ndarray, dict]:
    nb = onto.neighbours(did, reference, k=k)
    n_g = A.shape[1]
    if not nb:
        return np.zeros(n_g, dtype=np.float32), {"onto_n_ref": 0, "onto_max_ic": 0.0}
    rows = np.array([d_ix[d] for d, _ in nb], dtype=np.int64)
    w = np.array([s for _, s in nb], dtype=np.float64)
    w = w / max(w.sum(), EPS)
    deg = np.maximum(np.asarray(A[rows].sum(axis=1)).ravel(), 1.0)
    scores = np.asarray(A[rows].T @ (w / deg)).ravel().astype(np.float32)
    return scores, {"onto_n_ref": int(len(nb)), "onto_max_ic": float(nb[0][1])}
