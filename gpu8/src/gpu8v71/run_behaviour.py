"""Entry point for the E5 (test-fold behaviour) and E8 (optional-channel permutation) checks,
plus the recorded full-background chunk-invariance check on real inputs."""
from __future__ import annotations

import numpy as np
import torch

from . import paths71
from .behavior71 import e8_behaviour, test_behaviour
from .common71 import dump_json, load_manifest, now_utc


def chunk_invariance() -> dict:
    from .chain71 import Runner, variant_spec
    from .train71 import D_KEYS, G_KEYS, take
    run = Runner()
    spec = variant_spec("core")
    tr, m = run.model("V71CHAIN__core__f0_outer__s42", spec)
    test = run.w.labelled(run.w.fold(0)["test_diseases"])[:50]
    rows = torch.as_tensor([run.w.idx.d_ix[d] for d in test], device=run.device)
    d = take(tr.T, D_KEYS, rows)
    with torch.no_grad():
        full = m(d, tr.g_all)["r"]
        parts = []
        G = tr.G
        for a in range(0, G, 4000):
            g = {k: v[a:a + 4000] for k, v in tr.g_all.items()}
            parts.append(m(d, g)["r"])
        chunked = torch.cat(parts, dim=1)
    diff = float((full - chunked).abs().max())
    out = {"generated_utc": now_utc(), "diseases": len(test), "genes": int(G), "chunk": 4000,
           "max_abs_difference": diff, "invariant": diff < 1e-5}
    dump_json(paths71.BEHAVIOR / "chunk_invariance_v71.json", out)
    return out


if __name__ == "__main__":
    paths71.ensure()
    if not load_manifest("protocol_v71_registered.json"):
        raise RuntimeError("protocol not registered")
    print(chunk_invariance())
    df = test_behaviour()
    print(df.groupby(["perturbation", "modality"])[["lf_drop", "core_drop"]].mean().round(4).to_string())
    e8 = e8_behaviour()
    if len(e8):
        print(e8.groupby("variant")[["normal", "permuted", "drop"]].mean().round(4).to_string())
