"""V8.1 §5.2 — segmented disease text (a uniform input repair, no new facts).

The fields are exactly the V7.1 template fields (name ; definition(s) ; at most
eight synonyms), every field masked with the V6 dictionary (level 1 main,
level 2 sensitivity) *before* segmentation, so every segment consists of masked
field text only.

Segmentation (same rule for every disease):
* if the whole template fits into 126 word pieces (128 with [CLS]/[SEP]) it is
  one segment, identical to the V7.1 input;
* otherwise the template is cut into units — the name, the definition
  sentences (sentence boundaries first), the synonym list — and the units are
  packed greedily into segments of at most 126 word pieces; a single unit that is
  still too long is split at word boundaries.  No window overlaps, so every
  word piece has coverage 1.
* the disease name is written once (first segment) and never repeated, so it is
  not re-weighted.

Each segment is encoded with the frozen SapBERT checkpoint's rule ([CLS] of the
last layer before the pooler); the disease vector is the average of the
segment vectors weighted by their number of word pieces (coverage weights all
1).  Original fields, segment texts, flags and hashes are stored.
"""
from __future__ import annotations

import json
import re
import time

import numpy as np
import pandas as pd

from gpu8 import paths as p5

from . import paths8
from .common8 import dump_json, list_hash, now_utc

MAX_TOKENS = 128
CONTENT = MAX_TOKENS - 2
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'])")
ENC_DIR = paths8.CACHES / "encoders"


def _clean(x) -> str:
    s = str(x or "").strip()
    return "" if s == "nan" else s


def disease_fields(level: int = 1) -> pd.DataFrame:
    """Masked (name, definitions, synonyms) per project disease — V7.1 template fields."""
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
        masked = {}
        n_masked = 0
        for key, seg_list in (("name", [name]), ("defs", defs), ("syns", syn_u[:8])):
            out = []
            for seg in seg_list:
                t, mm = mask_segment(seg, sd, level=level)
                n_masked += len(mm)
                out.append(t)
            masked[key] = [x for x in out if x]
        rows.append({"disease_id": r.disease_id, "name": masked["name"][0] if masked["name"] else "",
                     "defs": masked["defs"], "syns": masked["syns"], "n_masked": n_masked})
    return pd.DataFrame(rows)


def template_text(name: str, defs: list, syns: list) -> str:
    parts = [name, " ".join(defs), ", ".join(syns)]
    return " ; ".join(p for p in parts if p)


class Segmenter:
    def __init__(self, tok):
        self.tok = tok

    def n_tok(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"]) if text else 0

    @staticmethod
    def _join(units: list[tuple[str, str]]) -> str:
        out = ""
        prev = None
        for kind, txt in units:
            if prev is None:
                out = txt
            elif kind == prev:
                out += (", " if kind == "syn" else " ") + txt
            else:
                out += " ; " + txt
            prev = kind
        return out

    def _split_long(self, kind: str, txt: str) -> list[tuple[str, str]]:
        words = txt.split(" ")
        pieces, cur = [], []
        for wd in words:
            cand = " ".join(cur + [wd])
            if cur and self.n_tok(cand) > CONTENT:
                pieces.append((kind, " ".join(cur)))
                cur = [wd]
            else:
                cur.append(wd)
        if cur:
            pieces.append((kind, " ".join(cur)))
        # a single word longer than the budget is left to the tokenizer's truncation (counted)
        return pieces

    def segments(self, name: str, defs: list, syns: list) -> list[dict]:
        full = template_text(name, defs, syns)
        if self.n_tok(full) <= CONTENT:
            return [{"text": full, "n_tokens": self.n_tok(full), "first": True}]
        units: list[tuple[str, str]] = []
        if name:
            units.append(("name", name))
        for d in defs:
            for s in SENT_RE.split(d):
                s = s.strip()
                if s:
                    units.append(("def", s))
        for s in syns:
            units.append(("syn", s))
        segs: list[list[tuple[str, str]]] = []
        cur: list[tuple[str, str]] = []
        for u in units:
            if cur and self.n_tok(self._join(cur + [u])) <= CONTENT:
                cur.append(u)
                continue
            if not cur and self.n_tok(u[1]) <= CONTENT:
                cur = [u]
                continue
            if cur:
                segs.append(cur)
                cur = []
            if self.n_tok(u[1]) <= CONTENT:
                cur = [u]
            else:
                pieces = self._split_long(*u)
                for pc in pieces[:-1]:
                    segs.append([pc])
                cur = [pieces[-1]]
        if cur:
            segs.append(cur)
        out = []
        for i, s in enumerate(segs):
            t = self._join(s)
            out.append({"text": t, "n_tokens": self.n_tok(t), "first": i == 0})
        return out


def encode_segments(texts: list[str], device: str = "cuda", batch: int = 256) -> tuple[np.ndarray, int]:
    paths8.add_libs()
    import torch
    from transformers import AutoModel, AutoTokenizer
    from gpu8v71 import paths71
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
            h = model(**enc).last_hidden_state[:, 0, :]          # [CLS] before the pooler
            out[idx] = h.float().cpu().numpy()
    return out, int(n_trunc)


def build(level: int = 1, device: str = "cuda") -> dict:
    paths8.add_libs()
    from transformers import AutoTokenizer
    from gpu8v71 import paths71
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(paths71.MODELS / "sapbert"))
    seg = Segmenter(tok)
    f = disease_fields(level)
    v71 = pd.read_parquet(paths71.TABLES / "disease_templates_v71.parquet")
    v71_text = dict(zip(v71["disease_id"], v71["text"]))
    rows, seg_rows = [], []
    for i, r in enumerate(f.itertuples(index=False)):
        full = template_text(r.name, list(r.defs), list(r.syns))
        if level == 1 and v71_text.get(r.disease_id) != full:
            raise RuntimeError(f"{r.disease_id}: template differs from the V7.1 input")
        ss = seg.segments(r.name, list(r.defs), list(r.syns))
        n_full = seg.n_tok(full)
        # masked placeholders that only the segmented version reads (beyond the V7.1 cut)
        new_mask = 0
        if n_full > CONTENT:
            enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
            cut = enc["offset_mapping"][CONTENT - 1][1]
            new_mask = full[cut:].count("GENEMASK")
        for k, s in enumerate(ss):
            seg_rows.append({"disease_id": r.disease_id, "row": i, "seg": k, "text": s["text"],
                             "n_tokens": s["n_tokens"], "weight": float(max(s["n_tokens"], 1))})
        rows.append({"disease_id": r.disease_id, "n_tokens_full": n_full,
                     "truncated_v71": bool(n_full > CONTENT), "n_segments": len(ss),
                     "n_masked_fields": int(r.n_masked), "mask_in_new_text": int(new_mask),
                     "text_sha256": list_hash([full])})
    S = pd.DataFrame(seg_rows)
    X, n_tr = encode_segments(S["text"].tolist(), device)
    n_d = len(f)
    num = np.zeros((n_d, X.shape[1]), dtype=np.float64)
    den = np.zeros(n_d, dtype=np.float64)
    np.add.at(num, S["row"].to_numpy(), X * S["weight"].to_numpy()[:, None])
    np.add.at(den, S["row"].to_numpy(), S["weight"].to_numpy())
    empty = np.array([not str(template_text(r.name, list(r.defs), list(r.syns))).strip()
                      for r in f.itertuples(index=False)])
    V = (num / np.maximum(den, 1e-12)[:, None]).astype(np.float32)
    V[empty] = 0.0
    D = pd.DataFrame(rows)
    tag = f"l{level}"
    ENC_DIR.mkdir(parents=True, exist_ok=True)
    np.save(ENC_DIR / f"sapbert_disease_seg_{tag}.npy", V)
    S.to_parquet(paths8.TABLES / f"disease_segments_{tag}.parquet", index=False)
    D.to_parquet(paths8.TABLES / f"disease_segment_summary_{tag}.parquet", index=False)
    key = {"model": "sapbert", "pooling": "cls_before_pooler", "max_tokens": MAX_TOKENS,
           "segment_rule": "whole template if <=126 word pieces, else sentence-first greedy packing; "
                           "name once; weights = word pieces; no overlap",
           "mask_level": level, "input_sha256": list_hash(S["text"]),
           "rows": "diseases.parquet order (same as tables_v71/disease_templates_v71.parquet)",
           "n_segments_truncated_by_tokenizer": n_tr, "written_utc": now_utc(), "shape": list(V.shape)}
    dump_json(ENC_DIR / f"sapbert_disease_seg_{tag}.key.json", key)
    rep = {"level": level, "n_diseases": int(n_d), "n_truncated_v71": int(D["truncated_v71"].sum()),
           "share_truncated_v71": float(D["truncated_v71"].mean()),
           "n_multi_segment": int((D["n_segments"] > 1).sum()),
           "segments_total": int(len(S)), "max_segments": int(D["n_segments"].max()),
           "tokens_full_median": float(D["n_tokens_full"].median()),
           "tokens_full_p95": float(D["n_tokens_full"].quantile(0.95)),
           "segment_tokenizer_truncations": n_tr,
           "masked_placeholders_in_new_text": int(D["mask_in_new_text"].sum()),
           "diseases_with_mask_in_new_text": int((D["mask_in_new_text"] > 0).sum()),
           "seconds": time.time() - t0}
    if level == 1:
        from gpu8v71.encoders71 import load_encoded
        X71, _ = load_encoded("sapbert_disease")
        single = (D["n_segments"] == 1).to_numpy() & ~empty
        cos = (V[single] * X71[single]).sum(1) / np.maximum(
            np.linalg.norm(V[single], axis=1) * np.linalg.norm(X71[single], axis=1), 1e-12)
        rep["single_segment_vs_v71_cosine_min"] = float(cos.min())
        rep["single_segment_vs_v71_maxabs"] = float(np.abs(V[single] - X71[single]).max())
        multi = (D["n_segments"] > 1).to_numpy()
        cm = (V[multi] * X71[multi]).sum(1) / np.maximum(
            np.linalg.norm(V[multi], axis=1) * np.linalg.norm(X71[multi], axis=1), 1e-12)
        rep["multi_segment_vs_v71_cosine_median"] = float(np.median(cm))
    dump_json(paths8.AUDIT / f"text_segments_{tag}.json", rep)
    return rep


if __name__ == "__main__":
    import sys
    paths8.ensure()
    lv = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print(json.dumps(build(lv), indent=2))
