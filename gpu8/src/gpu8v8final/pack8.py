"""V8.1 feature pack: the V7.1 arrays with the V8.1 input repairs and arm overrides.

A pack is defined by an ``ArmSpec``: text version (segmented level 1 = main,
level 2 = sensitivity, short = V7.1 paired comparison), GO policy (release,
evidence variant, deletion layer), genes whose GO is masked (GO-mask arms) and
genes withdrawn from association supervision (E6).  GO changes always rebuild
g_function *and* the functional anchors.  ``key`` is the dependency key of every
cache derived from the pack.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import sparse

from . import paths8
from .common8 import array_hash, dep_key

V71_PACK = paths8.V71_CACHES / "featurepack_v71.npz"


@dataclass(frozen=True)
class ArmSpec:
    name: str = "main"
    text: str = "seg_l1"                  # seg_l1 | seg_l2 | short
    go_release: str = "current"
    go_variant: str = "full"              # evidence8.VARIANTS
    go_layer: str = "none"                # none | layer1 | layer2 | layer2_random | layer1_random | blocks | blocks_random
    go_mask: str = "none"                 # none | dev | test
    e6_cold: bool = False                 # true association-supervision cold on G_mask,test
    evidence_blocks: bool = False         # layer-3 isolation (relations + GO + HPO)
    evaluation_conditioned: bool = False
    expression: str = "hpa"               # hpa | none | random
    note: str = ""

    def key(self) -> str:
        return dep_key(**asdict(self))


class Pack8:
    def __init__(self, arm: ArmSpec, world=None, policy=None, ctx_relation_ids: list | None = None,
                 hpo_drop_rows: np.ndarray | None = None):
        from .masks8 import load as load_masks
        z = np.load(V71_PACK)
        self.a = {k: z[k] for k in z.files}
        self.arm = arm
        self.n_d = self.a["d_text"].shape[0]
        self.n_g = self.a["g_sequence"].shape[0]
        # ---- disease text
        if arm.text in ("seg_l1", "seg_l2"):
            X = np.load(paths8.CACHES / "encoders" / f"sapbert_disease_seg_{arm.text[-2:]}.npy")
            self._set_text(X, world)
        # ---- disease HPO without the withdrawn annotation rows (evidence-block arms)
        if hpo_drop_rows is not None and len(hpo_drop_rows):
            self._rebuild_hpo(hpo_drop_rows, world)
        # ---- GO / function / anchors
        masks = load_masks()
        drop = np.array([], dtype=np.int64)
        if arm.go_mask in ("dev", "test"):
            drop = masks[arm.go_mask]
        # e6_cold withdraws association supervision (handled by the chain), not GO:
        # the GO-input-missing condition is the separate go_mask arm (two tables, never merged)
        self.masked_genes = drop
        self.e6_genes = masks["test"] if arm.e6_cold else np.array([], dtype=np.int64)
        self.info = {"arm": asdict(arm), "masked_genes": int(len(drop))}
        needs_go = (arm.go_variant != "full" or arm.go_layer != "none" or len(drop) or
                    arm.go_release != "current")
        if needs_go:
            if policy is None:
                from .evidence8 import GOPolicy
                policy = GOPolicy(world, arm.go_release)
            keep = policy.variant_mask(arm.go_variant)
            dele = self.layer_delete(policy, ctx_relation_ids)
            keep = keep & ~dele
            fa = policy.function_arrays(keep=keep, drop_genes=drop)
            self.a["g_function"] = fa["g_function"]
            self.a["g_function_mask"] = fa["g_function_mask"]
            self.a["anchor_vectors"] = fa["anchor_vectors"]
            self.A = fa["A_plus"]
            self.info["go_records_kept"] = int(keep.sum())
            self.info["go_records_deleted_by_layer"] = int(dele.sum())
        else:
            self.A = sparse.load_npz(paths8.V71_CACHES / "featurepack_v71_anchors.npz").tocsr()
        # ---- expression arms (secondary feature only; the teacher keeps HPA)
        self.info["genes_with_function"] = int(self.a["g_function_mask"].sum())
        self.info["n_anchors"] = int(self.A.shape[1])
        self.hash = array_hash(self.a["d_text"], self.a["d_hpo"], self.a["g_function"], self.a["g_function_mask"],
                               self.a["anchor_vectors"], self.A.indptr, self.A.indices)

    def _set_text(self, X: np.ndarray, world) -> None:
        import pandas as pd
        from gpu8 import paths as p5
        ids = pd.read_parquet(p5.TABLES_DIR / "diseases.parquet")["disease_id"].tolist()
        if world is None:
            from gpu8v6.common import World6
            world = World6(load_heavy=False)
        d_ix = world.idx.d_ix
        tpl = pd.read_parquet(paths8.V71_TABLES / "disease_templates_v71.parquet")
        has = dict(zip(tpl["disease_id"], [bool(str(t).strip()) for t in tpl["text"]]))
        T = np.zeros_like(self.a["d_text"])
        m = np.zeros(self.n_d, dtype=bool)
        for i, did in enumerate(ids):
            if did in d_ix and has.get(did, False):
                T[d_ix[did]] = X[i]
                m[d_ix[did]] = True
        if not np.array_equal(m, self.a["d_text_mask"]):
            raise RuntimeError("segmented text mask differs from the V7.1 text mask")
        self.a["d_text"] = T

    def _rebuild_hpo(self, drop_rows: np.ndarray, world) -> None:
        import pandas as pd
        from gpu8v6 import paths6
        from gpu8v71.encoders71 import load_encoded
        from gpu8v71.featurepack71 import aggregate_terms, freq_weight
        if world is None:
            from gpu8v6.common import World6
            world = World6(load_heavy=False)
        Xh, _ = load_encoded("sapbert_hpo")
        hids = pd.read_parquet(paths8.V71_TABLES / "hpo_term_texts_v71.parquet")["term_id"].tolist()
        ph = pd.read_parquet(paths6.TABLES / "phenotypes_v6.parquet")
        keep = np.ones(len(ph), dtype=bool)
        keep[np.asarray(drop_rows, dtype=np.int64)] = False
        ph = ph[keep]
        ph = ph[(ph["keep_v6"] == 1) & (ph["negated"].astype(int) == 0)]
        tab = pd.DataFrame({"entity": ph["disease_id"], "term": ph["hpo_id"],
                            "weight": [freq_weight(f) for f in ph["frequency"]]})
        X, m, _ = aggregate_terms(Xh, {t: i for i, t in enumerate(hids)}, tab, self.n_d, world.idx.d_ix)
        self.a["d_hpo"], self.a["d_hpo_mask"] = X, m

    def layer_delete(self, policy, ctx_relation_ids) -> np.ndarray:
        arm = self.arm
        n = len(policy.records)
        if arm.go_layer == "none":
            return np.zeros(n, dtype=bool)
        if arm.go_layer in ("layer1", "layer1_random"):
            if ctx_relation_ids is None:
                raise ValueError("layer 1 needs the context's training relations")
            d = policy.training_visible_exclusion(ctx_relation_ids)
            return d if arm.go_layer == "layer1" else policy.matched_random(d, "layer1|" + str(len(ctx_relation_ids)))
        if arm.go_layer in ("layer2", "layer2_random"):
            d = policy.evaluation_conditioned()
            return d if arm.go_layer == "layer2" else policy.matched_random(d, "layer2")
        if arm.go_layer in ("blocks", "blocks_random"):
            import json
            pm = set(json.loads((paths8.TABLES / "evidence_block_pmids_v8.json").read_text(encoding="utf-8")))
            d = policy.pmid_hit_mask(pm)
            return d if arm.go_layer == "blocks" else policy.matched_random(d, "blocks")
        raise KeyError(arm.go_layer)

    def tensors(self, device: str = "cuda") -> dict:
        import torch
        t = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=device)
        a = self.a
        out = {"xT": t(a["d_text"]), "mT": t(a["d_text_mask"], torch.bool),
               "xH": t(a["d_hpo"]), "mH": t(a["d_hpo_mask"], torch.bool),
               "q": t(a["d_tissue"]), "mq": t(a["d_tissue_mask"], torch.bool),
               "xS": t(a["g_sequence"]), "mS": t(a["g_sequence_mask"], torch.bool),
               "xF": t(a["g_function"]), "mF": t(a["g_function_mask"], torch.bool),
               "e": t(a["g_expression"]), "o": t(a["g_expression_mask"]),
               "xA": t(a["anchor_vectors"])}
        out["A_plus"] = torch.as_tensor(self.A.toarray().astype(bool), device=device)
        return out
