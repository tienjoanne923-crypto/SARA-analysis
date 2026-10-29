"""B7 KDGene adapter (Briefings in Bioinformatics 2024).

The official implementation is a knowledge-graph completion model: entity and
relation embeddings, the relation vector gated through an LSTM cell, a triple
score of ``sum(lhs * gated_rel * rhs)``, a full softmax over candidate entities
and an N3 regulariser.  Those definitions are reproduced here and retrained on
*this project's* legally visible graph, because the published numbers come from
a different disease space and cannot be compared directly.

What is kept from the original: the scoring function, the gated relation
update, the full-softmax cross-entropy objective, N3 regularisation, Adagrad,
and the practice of ranking over the gene entities only.

What necessarily differs, and is reported as such: the knowledge graph (our
CTD / HPO / ontology / PPI edges rather than DisGeNet + SymMap + SIDER +
STITCH), the disease split (disease-level cold start instead of ten-fold edge
CV), and early stopping on a legal cold validation set rather than on test MRR.
Changing the entity space is unavoidable when the protocol is the thing being
controlled; renaming the method would be worse than stating the difference.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..config import Gpu8Config
from .simple import stable_tie_noise


@dataclass
class KGSpec:
    entity_to_id: dict[str, int]
    relation_to_id: dict[str, int]
    triples: np.ndarray           # [n, 3] head, relation, tail
    gene_entity_ids: np.ndarray   # candidate entity ids (genes)
    n_entities: int
    n_relations: int


def build_kg(cfg: Gpu8Config, relations: pd.DataFrame, phenotypes: pd.DataFrame,
             hierarchy: pd.DataFrame, ppi: pd.DataFrame, visible_diseases: set[str],
             gene_ids: list[str], max_ppi_edges: int = 400_000) -> KGSpec:
    """Assemble the shared legal graph as typed triples."""
    ent: dict[str, int] = {}
    for g in gene_ids:                       # genes take the lowest ids
        ent.setdefault(f"G:{g}", len(ent))
    gene_entity_ids = np.array([ent[f"G:{g}"] for g in gene_ids], dtype=np.int64)

    rel: dict[str, int] = {}
    triples: list[tuple[int, int, int]] = []

    def add(h: str, r: str, t: str) -> None:
        hid = ent.setdefault(h, len(ent))
        tid = ent.setdefault(t, len(ent))
        rid = rel.setdefault(r, len(rel))
        triples.append((hid, rid, tid))

    # Only the disease-gene relation is restricted to the visible training
    # diseases.  Phenotype and ontology edges are legal visible knowledge for
    # every disease under this protocol, and withholding them would leave a
    # cold test disease with no entity at all — which would cripple the
    # comparator rather than test it.
    prim = relations[relations["relation_type"].isin(set(cfg.data.primary_evidence))]
    prim = prim[prim["disease_id"].isin(visible_diseases)]
    for r in prim.itertuples(index=False):
        add(f"D:{r.disease_id}", "disease_gene", f"G:{r.gene_id}")

    if phenotypes is not None and not phenotypes.empty:
        ph = phenotypes[phenotypes["negated"] == 0]
        for r in ph.itertuples(index=False):
            add(f"D:{r.disease_id}", "disease_symptom", f"P:{r.hpo_id}")

    if hierarchy is not None and not hierarchy.empty:
        for r in hierarchy.itertuples(index=False):
            add(f"D:{r.child_disease}", "disease_parent", f"D:{r.parent_disease}")

    if ppi is not None and not ppi.empty:
        sub = ppi[ppi["network_type"] == "physical"] if "network_type" in ppi.columns else ppi
        if len(sub) > max_ppi_edges:
            sub = sub.nlargest(max_ppi_edges, "score")
        gset = set(gene_ids)
        for r in sub.itertuples(index=False):
            if r.gene_a in gset and r.gene_b in gset:
                add(f"G:{r.gene_a}", "interacts_with", f"G:{r.gene_b}")

    arr = np.array(triples, dtype=np.int64) if triples else np.zeros((0, 3), dtype=np.int64)
    return KGSpec(entity_to_id=ent, relation_to_id=rel, triples=arr,
                  gene_entity_ids=gene_entity_ids, n_entities=len(ent),
                  n_relations=len(rel))


class KDGene(nn.Module):
    """Entity/relation embeddings with a gated relation update (original form)."""

    def __init__(self, n_entities: int, n_relations: int, edim: int = 1000,
                 rdim: int = 1000, init: float = 1e-3, gate: str = "LSTMCell"):
        super().__init__()
        self.edim, self.rdim = edim, rdim
        self.lhs = nn.Embedding(n_entities, edim)
        self.rhs = nn.Embedding(n_entities, edim)
        self.rel = nn.Embedding(2 * n_relations, rdim)
        for emb in (self.lhs, self.rhs, self.rel):
            nn.init.normal_(emb.weight, std=init)
        cell = {"LSTMCell": nn.LSTMCell, "GRUCell": nn.GRUCell, "RNNCell": nn.RNNCell}[gate]
        self.gate_name = gate
        self.cell = cell(rdim, edim)

    def relation_update(self, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if self.gate_name == "LSTMCell":
            out, _ = self.cell(r, (h, torch.zeros_like(h)))
        else:
            out = self.cell(r, h)
        return out

    def forward(self, triples: torch.Tensor):
        h = self.lhs(triples[:, 0])
        r = self.rel(triples[:, 1])
        upd = self.relation_update(h, r)
        logits = upd @ self.rhs.weight.t()
        factors = (h.abs(), r.abs(), self.rhs(triples[:, 2]).abs())
        return logits, factors

    def score_pairs(self, heads: torch.Tensor, rel_id: int, tails: torch.Tensor
                    ) -> torch.Tensor:
        h = self.lhs(heads)
        r = self.rel(torch.full_like(heads, rel_id))
        upd = self.relation_update(h, r)
        return (upd * self.rhs(tails)).sum(-1)


def n3_regulariser(factors, weight: float) -> torch.Tensor:
    total = 0.0
    for f in factors:
        total = total + (f ** 3).sum()
    return weight * total / max(1, factors[0].shape[0])


def train_kdgene(cfg: Gpu8Config, kg: KGSpec, device, seed: int = 42,
                 edim: int = 1000, rdim: int = 1000, lr: float = 0.05,
                 batch_size: int = 512, reg: float = 0.1, max_epochs: int = 50,
                 patience: int = 4, val_fn=None, progress: bool = True
                 ) -> tuple[KDGene, dict]:
    torch.manual_seed(seed)
    model = KDGene(kg.n_entities, kg.n_relations, edim=edim, rdim=rdim).to(device)
    opt = torch.optim.Adagrad(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss(reduction="mean")
    triples = torch.as_tensor(kg.triples, dtype=torch.long, device=device)
    # reciprocal relations, as in the original data loader
    recip = triples[:, [2, 1, 0]].clone()
    recip[:, 1] = recip[:, 1] + kg.n_relations
    allt = torch.cat([triples, recip], dim=0)

    history, best, best_epoch, since = [], -float("inf"), -1, 0
    best_state = None
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(allt.shape[0], device=device)
        total = 0.0
        for s in range(0, allt.shape[0], batch_size):
            batch = allt[perm[s:s + batch_size]]
            opt.zero_grad(set_to_none=True)
            logits, factors = model(batch)
            loss = lossf(logits, batch[:, 2]) + n3_regulariser(factors, reg)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * batch.shape[0]
        rec = {"epoch": epoch, "loss": total / max(1, allt.shape[0])}
        if val_fn is not None:
            metric = float(val_fn(model))
            rec["val"] = metric
            if metric > best + 1e-6:
                best, best_epoch, since = metric, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                since += 1
        history.append(rec)
        if progress:
            print(f"      kdgene epoch {epoch} loss={rec['loss']:.4f}"
                  + (f" val={rec.get('val', float('nan')):.4f}" if val_fn else ""), flush=True)
        if val_fn is not None and since >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"history": history, "best_epoch": best_epoch, "best_val": best,
                   "n_entities": kg.n_entities, "n_relations": kg.n_relations,
                   "n_triples": int(kg.triples.shape[0]), "edim": edim, "rdim": rdim}


@torch.no_grad()
def score_kdgene(model: KDGene, kg: KGSpec, disease_id: str, gene_ids: list[str],
                 candidate_rows: np.ndarray, gene_row_to_id: dict[int, str],
                 device) -> np.ndarray | None:
    """Score one disease over its candidate genes; unknown entity -> not applicable."""
    head = kg.entity_to_id.get(f"D:{disease_id}")
    if head is None:
        return None
    rel_id = kg.relation_to_id.get("disease_gene", 0)
    tails, keep = [], []
    for i, row in enumerate(candidate_rows):
        gid = gene_row_to_id.get(int(row))
        eid = kg.entity_to_id.get(f"G:{gid}") if gid is not None else None
        if eid is not None:
            tails.append(eid)
            keep.append(i)
    out = np.full(len(candidate_rows), -1e9, dtype=np.float64)
    if not tails:
        return stable_tie_noise(out, candidate_rows)
    h = torch.full((len(tails),), head, dtype=torch.long, device=device)
    t = torch.as_tensor(tails, dtype=torch.long, device=device)
    s = model.score_pairs(h, rel_id, t).cpu().numpy()
    out[np.array(keep, dtype=int)] = s
    return stable_tie_noise(out, candidate_rows)
