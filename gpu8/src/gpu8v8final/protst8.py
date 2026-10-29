"""ProtST-ESM-1b as a task-adapted comparator and an open-knowledge control (§14.1, §11.3).

Only the checkpoint weights and configuration are used.  ``modeling_protst.py``
was read as text and its two computations are re-implemented here without
importing it (no ``trust_remote_code``):

    protein_feature = protein_mlp( mean over non-special residues of ESM-1b last layer )
    text_feature    = text_mlp( mean over non-special tokens of PubMedBERT last layer )
    ProtSTHead(x)   = out_proj( relu( dense(x) ) ),  1280/768 -> 512

Weights: ``torch.load(weights_only=True)`` of pytorch_model.bin, mapped onto the
standard transformers ``EsmModel`` / ``BertModel`` with ``strict=True`` (missing
and unexpected keys recorded).  Proteins longer than 1,022 residues use the same
coverage-weighted windows as the ESM-2 cache (the original zero-shot script
truncates to 1,022).  The encoder was trained on Swiss-Prot function
descriptions, so its use as a student input is an open-knowledge control, never
evidence of sequence-only transfer, and it is not used for X1/X2.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import paths8
from .common8 import dump_json, list_hash, now_utc

MDIR = paths8.EXTERNAL / "models"
PDIR = MDIR / "protst_esm1b"
WINDOW, OVERLAP = 1022, 128


class Head(nn.Module):
    def __init__(self, dim: int, out: int = 512):
        super().__init__()
        self.dense = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, out)

    def forward(self, x):
        return self.out_proj(torch.relu(self.dense(x)))


def load(device: str = "cuda", dtype=torch.float16) -> dict:
    paths8.add_libs()
    from transformers import BertConfig, BertModel, EsmConfig, EsmModel
    cfg = json.loads((PDIR / "config.json").read_text(encoding="utf-8"))
    pc = {k: v for k, v in cfg["protein_config"].items() if k not in ("architectures", "_name_or_path")}
    tc = {k: v for k, v in cfg["text_config"].items() if k not in ("architectures",)}
    esm = EsmModel(EsmConfig(**pc), add_pooling_layer=False)
    bert = BertModel(BertConfig(**tc), add_pooling_layer=False)
    pm, tm = Head(1280), Head(768)
    sd = torch.load(PDIR / "pytorch_model.bin", map_location="cpu", weights_only=True)
    groups = {"protein_model.esm.": esm, "protein_model.protein_mlp.": pm, "text_model.bert.": bert,
              "text_model.text_mlp.": tm}
    report = {"checkpoint_keys": len(sd), "unused_prefixes": {}}
    used = set()
    for pre, mod in groups.items():
        part = {k[len(pre):]: v.float() for k, v in sd.items() if k.startswith(pre)}
        used |= {pre + k for k in part}
        own = set(mod.state_dict().keys())
        missing = sorted(own - set(part))
        unexpected = sorted(set(part) - own)
        # buffers such as position_ids may be absent from a checkpoint; everything else must match
        tolerated = [k for k in missing if k.endswith("position_ids")]
        hard_missing = [k for k in missing if k not in tolerated]
        if hard_missing or unexpected:
            raise RuntimeError(f"{pre}: missing {hard_missing[:5]}, unexpected {unexpected[:5]}")
        for k in tolerated:
            part[k] = mod.state_dict()[k]
        mod.load_state_dict(part, strict=True)
        report[pre] = {"loaded": len(part) - len(tolerated), "missing_tolerated": tolerated}
    rest = sorted(set(sd) - used)
    report["not_loaded"] = rest[:20]
    report["n_not_loaded"] = len(rest)
    logit = float(sd["logit_scale"]) if "logit_scale" in sd else None
    for m in (esm, bert, pm, tm):
        m.to(device=device, dtype=dtype).eval()
    return {"esm": esm, "bert": bert, "protein_mlp": pm, "text_mlp": tm, "logit_scale": logit,
            "report": report, "device": device, "dtype": dtype}


def windows(L: int) -> list[tuple[int, int]]:
    if L <= WINDOW:
        return [(0, L)]
    stride = WINDOW - OVERLAP
    st = list(range(0, L - WINDOW + 1, stride))
    if st[-1] + WINDOW < L:
        st.append(L - WINDOW)
    return [(s, s + WINDOW) for s in st]


@torch.no_grad()
def encode_proteins(M: dict, seqs: list[str], token_budget: int = 12000) -> tuple[np.ndarray, np.ndarray]:
    """[G, 512] protein features and [G, 1280] mean residue states (zeros where no sequence)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MDIR / "esm1b_tokenizer"))
    dev = M["device"]
    G = len(seqs)
    acc = np.zeros((G, 1280), dtype=np.float64)
    items = []
    for gi, s in enumerate(seqs):
        if not s:
            continue
        L = len(s)
        cov = np.zeros(L)
        ws = windows(L)
        for a, b in ws:
            cov[a:b] += 1
        for a, b in ws:
            items.append((gi, a, b, 1.0 / cov[a:b]))
    items.sort(key=lambda x: x[2] - x[1])
    i, t0 = 0, time.time()
    while i < len(items):
        L = items[i][2] - items[i][1]
        n = max(1, token_budget // (L + 2))
        chunk = items[i:i + n]
        i += len(chunk)
        enc = tok([seqs[g][a:b] for g, a, b, _ in chunk], padding=True, return_tensors="pt").to(dev)
        h = M["esm"](**enc).last_hidden_state.float().cpu().numpy()
        for j, (g, a, b, w) in enumerate(chunk):
            acc[g] += (h[j, 1:1 + (b - a)] * w[:, None]).sum(0)
        if (i // max(n, 1)) % 100 == 0:
            print(f"    protst proteins {i}/{len(items)} windows ({time.time() - t0:.0f}s)", flush=True)
    lens = np.array([len(s) for s in seqs], dtype=np.float64)
    R = np.where(lens[:, None] > 0, acc / np.maximum(lens[:, None], 1), 0.0).astype(np.float32)
    P = np.zeros((G, 512), dtype=np.float32)
    has = lens > 0
    for b0 in range(0, G, 1024):
        x = torch.as_tensor(R[b0:b0 + 1024], device=dev, dtype=M["dtype"])
        P[b0:b0 + 1024] = M["protein_mlp"](x).float().cpu().numpy()
    P[~has] = 0.0
    return P, R


@torch.no_grad()
def encode_texts(M: dict, texts: list[str], max_len: int = 510, batch: int = 64) -> np.ndarray:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MDIR / "pubmedbert_abstract_tokenizer"))
    dev = M["device"]
    out = np.zeros((len(texts), 512), dtype=np.float32)
    for b0 in range(0, len(texts), batch):
        ids = [[tok.cls_token_id] + tok.encode(t or "", max_length=max_len, truncation=True,
                                                add_special_tokens=False) for t in texts[b0:b0 + batch]]
        L = max(len(x) for x in ids)
        arr = np.full((len(ids), L), tok.pad_token_id, dtype=np.int64)
        for k, x in enumerate(ids):
            arr[k, :len(x)] = x
        inp = torch.as_tensor(arr, device=dev)
        att = (inp != tok.pad_token_id).long()
        h = M["bert"](inp, attention_mask=att).last_hidden_state
        special = (inp == tok.cls_token_id) | (inp == tok.sep_token_id) | (inp == tok.pad_token_id)
        msk = (~special).to(h.dtype).unsqueeze(-1)
        pooled = (h * msk).sum(1) / (msk.sum(1) + 1e-6)
        out[b0:b0 + len(ids)] = M["text_mlp"](pooled).float().cpu().numpy()
    return out


def build(device: str = "cuda") -> dict:
    t0 = time.time()
    M = load(device)
    sq = pd.read_parquet(paths8.V71_TABLES / "sequences_v71.parquet")
    seqs = sq["sequence"].fillna("").tolist()
    P, R = encode_proteins(M, seqs)
    t_prot = time.time() - t0
    tpl = pd.read_parquet(paths8.V71_TABLES / "disease_templates_v71.parquet")
    T = encode_texts(M, tpl["text"].fillna("").tolist())
    enc = paths8.CACHES / "encoders"
    enc.mkdir(parents=True, exist_ok=True)
    np.save(enc / "protst_protein512.npy", P)
    np.save(enc / "protst_residue_mean1280.npy", R)
    np.save(enc / "protst_disease_text512.npy", T)
    dl = json.loads((PDIR / "download_v8.json").read_text(encoding="utf-8"))
    key = {"model": "mila-intel/ProtST-esm1b", "revision": dl["revision"],
           "protein": "coverage-weighted windowed mean of non-special residues -> protein_mlp",
           "text": "[CLS] + template (<=510 tokens), mean of non-special tokens -> text_mlp",
           "input_sha256_sequences": list_hash(sq["sha256"]), "input_sha256_texts": list_hash(tpl["text"]),
           "load_report": M["report"], "logit_scale": M["logit_scale"], "dtype": "float16",
           "seconds_protein": t_prot, "seconds_total": time.time() - t0, "written_utc": now_utc(),
           "remote_code_executed": False}
    dump_json(enc / "protst.key.json", key)
    # behaviour sanity: a protein's feature should be closer to itself than to random others
    has = np.linalg.norm(P, axis=1) > 0
    Pn = P[has] / np.linalg.norm(P[has], axis=1, keepdims=True)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Pn), size=min(2000, len(Pn)), replace=False)
    C = Pn[idx] @ Pn[idx].T
    key["sanity"] = {"mean_offdiag_cos": float((C.sum() - len(idx)) / (len(idx) ** 2 - len(idx))),
                     "n_proteins": int(has.sum())}
    dump_json(enc / "protst.key.json", key)
    return key


if __name__ == "__main__":
    paths8.ensure()
    print(json.dumps(build(), indent=2, default=str)[:3000])
