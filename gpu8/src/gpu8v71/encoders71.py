"""V7.1 sections 4-5 — frozen encoders and their caches.

SapBERT (``cambridgeltl/SapBERT-from-PubMedBERT-fulltext``): the [CLS] hidden
state of the last layer *before the pooler*, at most 128 tokens (truncation is
counted, not hidden), for

* disease templates  "name ; definition(s) ; synonyms", every segment masked
  with the V6 level-1 gene-symbol dictionary (no generated text);
* HPO terms          "name ; definition";
* GO terms           "name ; definition".

ESM-2 35M (``facebook/esm2_t12_35M_UR50D``): residues in windows of at most
1,022 with overlap 128; the last-layer residue states (CLS/EOS/padding
excluded) are summed with weight 1/c_i, c_i the number of windows covering
residue i, and divided by the length L, so every residue carries total weight
1/L.  Long-range interactions across windows are truncated: the vector is an
approximation.  Output ``[G, 480]``.

Cache keys hold the model revision, pooling rule, window rule and the SHA-256 of
the input texts / sequences; a key mismatch stops instead of silently reusing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time

import numpy as np
import pandas as pd

from gpu8 import paths as p5

from . import paths71
from .common71 import dump_json, load_manifest, now_utc

ENC_DIR = paths71.CACHES / "encoders"
WINDOW = 1022
OVERLAP = 128
MAX_TOKENS = 128


# --------------------------------------------------------------------------
# texts
# --------------------------------------------------------------------------
def _clean(x) -> str:
    s = str(x or "").strip()
    return "" if s == "nan" else s


def disease_templates() -> pd.DataFrame:
    """Masked "name ; definition(s) ; synonyms" per project disease."""
    from gpu8v6.disease_text import load_symbol_dictionary, mask_segment, parse_mondo_text
    diseases = pd.read_parquet(p5.TABLES_DIR / "diseases.parquet")
    genes = pd.read_parquet(p5.TABLES_DIR / "genes.parquet")
    sd = load_symbol_dictionary(genes)
    do = pd.read_parquet(p5.TABLES_DIR / "do_terms.parquet").set_index("term_id")
    mondo = parse_mondo_text(p5.external("mondo", "mondo.obo"))
    rows = []
    for r in diseases.itertuples(index=False):
        name = _clean(r.disease_name)
        defs, syns = [], [_clean(s) for s in str(r.synonyms or "").split("|")]
        if _clean(r.definition):
            defs.append(_clean(r.definition))
        for sid in str(r.source_ids or "").split("|"):
            sid = sid.strip()
            if sid.startswith("DO:") and sid[3:] in do.index:
                rec = do.loc[sid[3:]]
                syns.append(_clean(rec["name"]))
                syns += [_clean(s) for s in str(rec["synonyms"] or "").split("|")]
                if _clean(rec["definition"]):
                    defs.append(_clean(rec["definition"]))
        if r.mapping_type == "mondo_exact_unique" and r.mondo_ids in mondo:
            m = mondo[r.mondo_ids]
            if m.get("def"):
                defs.append(m["def"])
            syns += list(m.get("syn", []))
        seen, syn_u = {name.lower()}, []
        for s in syns:
            if s and s.lower() not in seen:
                seen.add(s.lower())
                syn_u.append(s)
        n_masked = 0
        parts = []
        for seg_list, joiner in (([name], " "), (defs, " "), (syn_u[:8], ", ")):
            out = []
            for seg in seg_list:
                t, m = mask_segment(seg, sd, level=1)
                n_masked += len(m)
                out.append(t)
            parts.append(joiner.join(x for x in out if x))
        text = " ; ".join(p for p in parts if p)
        rows.append({"disease_id": r.disease_id, "text": text, "has_definition": int(bool(defs)),
                     "n_synonyms_used": min(len(syn_u), 8), "n_masked": n_masked})
    return pd.DataFrame(rows)


def hpo_texts() -> pd.DataFrame:
    t = pd.read_parquet(p5.TABLES_DIR / "hpo_terms.parquet")
    txt = [(_clean(n) + (" ; " + _clean(d) if _clean(d) else "")) for n, d in
           zip(t["name"], t["definition"])]
    return pd.DataFrame({"term_id": t["hpo_id"], "text": txt})


def parse_go_text(path) -> pd.DataFrame:
    rows, cur, obsolete = [], None, False
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line == "[Term]" or (line.startswith("[") and line.endswith("]")):
                if cur and cur.get("id") and not obsolete:
                    rows.append(cur)
                cur, obsolete = ({"id": "", "name": "", "def": "", "ns": ""} if line == "[Term]"
                                 else None), False
                continue
            if cur is None:
                continue
            if line.startswith("id: "):
                cur["id"] = line[4:].strip()
            elif line.startswith("name: "):
                cur["name"] = line[6:].strip()
            elif line.startswith("namespace: "):
                cur["ns"] = line[11:].strip()
            elif line.startswith("def: "):
                m = re.match(r'def: "(.*?)"(\s*\[.*\])?\s*$', line)
                cur["def"] = m.group(1) if m else ""
            elif line.startswith("is_obsolete: true"):
                obsolete = True
    if cur and cur.get("id") and not obsolete:
        rows.append(cur)
    df = pd.DataFrame(rows)
    df["text"] = [n + (" ; " + d if d else "") for n, d in zip(df["name"], df["def"])]
    return df.rename(columns={"id": "term_id"})


def go_texts() -> pd.DataFrame:
    from gpu8v6 import paths6
    return parse_go_text(paths6.ext("go", "go-basic_current.obo"))


# --------------------------------------------------------------------------
# encoders
# --------------------------------------------------------------------------
def _hash_list(xs) -> str:
    h = hashlib.sha256()
    for x in xs:
        h.update(str(x).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _cache_ok(name: str, key: dict) -> bool:
    kp = ENC_DIR / f"{name}.key.json"
    if not kp.exists() or not (ENC_DIR / f"{name}.npy").exists():
        return False
    old = json.loads(kp.read_text(encoding="utf-8"))
    return all(old.get(k) == v for k, v in key.items())


def _save(name: str, X: np.ndarray, key: dict, extra: dict) -> None:
    ENC_DIR.mkdir(parents=True, exist_ok=True)
    np.save(ENC_DIR / f"{name}.npy", X.astype(np.float32))
    dump_json(ENC_DIR / f"{name}.key.json", {**key, **extra, "shape": list(X.shape),
                                              "written_utc": now_utc()})


def load_encoded(name: str) -> tuple[np.ndarray, dict]:
    X = np.load(ENC_DIR / f"{name}.npy")
    key = json.loads((ENC_DIR / f"{name}.key.json").read_text(encoding="utf-8"))
    return X, key


def sapbert_encode(texts: list[str], device: str = "cuda", batch: int = 256) -> tuple[np.ndarray, int]:
    paths71.add_libs()
    import torch
    from transformers import AutoModel, AutoTokenizer
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    d = paths71.MODELS / "sapbert"
    tok = AutoTokenizer.from_pretrained(str(d))
    model = AutoModel.from_pretrained(str(d)).to(device).eval()
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    n_trunc = 0
    order = np.argsort([len(t) for t in texts])
    with torch.no_grad():
        for b0 in range(0, len(texts), batch):
            idx = order[b0:b0 + batch]
            chunk = [texts[i] if texts[i] else "[UNK]" for i in idx]
            full = tok(chunk, add_special_tokens=True, truncation=False)["input_ids"]
            n_trunc += sum(len(x) > MAX_TOKENS for x in full)
            enc = tok(chunk, padding=True, truncation=True, max_length=MAX_TOKENS,
                      return_tensors="pt").to(device)
            h = model(**enc).last_hidden_state[:, 0, :]     # [CLS] before the pooler
            out[idx] = h.float().cpu().numpy()
    return out, int(n_trunc)


def esm_windows(L: int, window: int = WINDOW, overlap: int = OVERLAP) -> list[tuple[int, int]]:
    """[start, end) windows covering 0..L-1; the last one is aligned to the end."""
    if L <= window:
        return [(0, L)]
    stride = window - overlap
    starts = list(range(0, L - window + 1, stride))
    if starts[-1] + window < L:
        starts.append(L - window)
    return [(s, s + window) for s in starts]


def window_weights(L: int, window: int = WINDOW, overlap: int = OVERLAP) -> list[np.ndarray]:
    """Per-window residue weights 1/c_i; summed over windows every residue gets 1."""
    wins = esm_windows(L, window, overlap)
    cov = np.zeros(L, dtype=np.float64)
    for s, e in wins:
        cov[s:e] += 1.0
    return [1.0 / cov[s:e] for s, e in wins]


def esm_encode(seqs: list[str], device: str = "cuda", token_budget: int = 24000) -> np.ndarray:
    paths71.add_libs()
    import torch
    from transformers import AutoTokenizer, EsmModel
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    d = paths71.MODELS / "esm2_35m"
    tok = AutoTokenizer.from_pretrained(str(d))
    model = EsmModel.from_pretrained(str(d), add_pooling_layer=False).to(device).eval()
    H = model.config.hidden_size
    acc = np.zeros((len(seqs), H), dtype=np.float64)
    items = []                                    # (gene index, start, end, weights)
    for gi, s in enumerate(seqs):
        if not s:
            continue
        L = len(s)
        for (a, b), wts in zip(esm_windows(L), window_weights(L)):
            items.append((gi, a, b, wts))
    items.sort(key=lambda x: x[2] - x[1])
    i = 0
    t0 = time.time()
    with torch.no_grad():
        while i < len(items):
            L = items[i][2] - items[i][1]
            n = max(1, token_budget // (L + 2))
            chunk = items[i:i + n]
            i += len(chunk)
            texts = [seqs[g][a:b] for g, a, b, _ in chunk]
            enc = tok(texts, padding=True, return_tensors="pt").to(device)
            h = model(**enc).last_hidden_state.float()          # [B, T, H]
            h = h.cpu().numpy()
            for j, (g, a, b, wts) in enumerate(chunk):
                res = h[j, 1:1 + (b - a)]                       # drop CLS; EOS/pad after
                acc[g] += (res * wts[:, None]).sum(axis=0)
            if len(items) > 0 and (i // max(1, n)) % 200 == 0:
                print(f"    esm {i}/{len(items)} windows, {time.time() - t0:.0f}s", flush=True)
    lens = np.array([len(s) for s in seqs], dtype=np.float64)
    X = np.where(lens[:, None] > 0, acc / np.maximum(lens[:, None], 1.0), 0.0)
    return X.astype(np.float32)


# --------------------------------------------------------------------------
# stage entry
# --------------------------------------------------------------------------
def run(device: str = "cuda", only: list[str] | None = None) -> dict:
    models = load_manifest("models_v71.json")
    if not models:
        raise RuntimeError("models_v71.json missing: run resources71 --stage models")
    rev = {k: v["revision"] for k, v in models["models"].items()}
    rep = {"generated_utc": now_utc(), "device": device}
    todo = only or ["disease", "hpo", "go", "esm"]
    if "disease" in todo:
        dt = disease_templates()
        key = {"model": "sapbert", "revision": rev["sapbert"], "pooling": "cls_before_pooler",
               "max_tokens": MAX_TOKENS, "input_sha256": _hash_list(dt["text"])}
        t0 = time.time()
        X, n_tr = sapbert_encode(dt["text"].tolist(), device)
        dt.to_parquet(paths71.TABLES / "disease_templates_v71.parquet", index=False)
        _save("sapbert_disease", X, key, {"ids": "tables_v71/disease_templates_v71.parquet",
                                          "n_truncated": n_tr})
        rep["disease"] = {"n": len(dt), "n_truncated_at_128": n_tr, "seconds": time.time() - t0,
                          "n_with_definition": int(dt["has_definition"].sum())}
    if "hpo" in todo:
        ht = hpo_texts()
        key = {"model": "sapbert", "revision": rev["sapbert"], "pooling": "cls_before_pooler",
               "max_tokens": MAX_TOKENS, "input_sha256": _hash_list(ht["text"])}
        t0 = time.time()
        X, n_tr = sapbert_encode(ht["text"].tolist(), device)
        ht.to_parquet(paths71.TABLES / "hpo_term_texts_v71.parquet", index=False)
        _save("sapbert_hpo", X, key, {"ids": "tables_v71/hpo_term_texts_v71.parquet",
                                      "n_truncated": n_tr})
        rep["hpo"] = {"n": len(ht), "n_truncated_at_128": n_tr, "seconds": time.time() - t0}
    if "go" in todo:
        gt = go_texts()
        key = {"model": "sapbert", "revision": rev["sapbert"], "pooling": "cls_before_pooler",
               "max_tokens": MAX_TOKENS, "input_sha256": _hash_list(gt["text"])}
        t0 = time.time()
        X, n_tr = sapbert_encode(gt["text"].tolist(), device)
        gt.to_parquet(paths71.TABLES / "go_term_texts_v71.parquet", index=False)
        _save("sapbert_go", X, key, {"ids": "tables_v71/go_term_texts_v71.parquet",
                                     "n_truncated": n_tr})
        rep["go"] = {"n": len(gt), "n_truncated_at_128": n_tr, "seconds": time.time() - t0}
    if "esm" in todo:
        sq = pd.read_parquet(paths71.TABLES / "sequences_v71.parquet")
        seqs = sq["sequence"].fillna("").tolist()
        key = {"model": "esm2_35m", "revision": rev["esm2_35m"],
               "pooling": "coverage_normalised_residue_mean", "window": WINDOW,
               "overlap": OVERLAP, "input_sha256": _hash_list(sq["sha256"])}
        t0 = time.time()
        X = esm_encode(seqs, device)
        _save("esm_gene", X, key, {"rows": "gene_row order of tables_v71/sequences_v71.parquet"})
        has = sq["length"].to_numpy() > 0
        rep["esm"] = {"n_genes": len(sq), "n_encoded": int(has.sum()), "seconds": time.time() - t0,
                      "n_multi_window": int((sq["length"] > WINDOW).sum()),
                      "mean_norm": float(np.linalg.norm(X[has], axis=1).mean())}
    prev = {}
    rp = paths71.AUDIT / "encoders_v71_report.json"
    if rp.exists():
        prev = json.loads(rp.read_text(encoding="utf-8"))
    prev.update(rep)
    dump_json(rp, prev)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="V7.1 frozen encoders.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args(argv)
    paths71.ensure()
    ENC_DIR.mkdir(parents=True, exist_ok=True)
    print(json.dumps(run(args.device, args.only), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
