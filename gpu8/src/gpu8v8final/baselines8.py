"""KDGene and SIGN-style comparators on the strict 500 lists (§14.1), five folds, seed 42.

KDGene: the V5 adapter of the official architecture (1,000-d, reciprocal
relations, N3), disease-gene edges only from the outer context's role-based
reference diseases; phenotype, ontology and PPI edges as legal knowledge.  Up to
100 epochs, the checkpoint chosen on F_{j+1} cold-validation Recall@10 over the
*strict 500 lists of F_{j+1}* (patience 10); F_j lists scored once.

SIGN-style: propagation scales 0-2 of the V5 context views (gene side) and their
products with the disease view, a two-hidden-layer decoder trained on the OOF
rows of the three inner contexts (positives in the pool vs five pool negatives
per positive), epochs chosen on F_{j+1} strict-pool Recall@10.  Both are
adaptations (the originals do not define this task), labelled as such.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from . import paths8
from .common8 import dump_json, now_utc
from .ledger8 import write_result
from .models8 import recall_at

N_G = 19861


def strict_block_from_pred(ch, j: int, role: str, d_pred, seed: int = 42):
    """RowBlock-like view of the deployed strict lists (ids, genes, labels for non-test roles)."""
    from .formal8 import out_dir
    z = np.load(d_pred / f"pred_deployed__{role}.npz")
    ids = [str(x) for x in z["ids"]]
    genes = z["genes"].astype(np.int64)
    y, nq = None, None
    if role != "test" or ch.labels.registered():
        y = np.zeros(genes.shape, dtype=np.float32)
        nq = np.zeros(len(ids))
        for i, d in enumerate(ids):
            q = ch.labels.rows(d, role)
            nq[i] = len(q)
            y[i] = np.isin(genes[i], q)
    return ids, genes, y, nq


class _Blk:
    def __init__(self, ids, genes, y, nq):
        self.ids, self.genes, self.y, self.n_query = ids, genes, y, nq
        self.D, self.N = genes.shape


def run_kdgene(folds=range(5), seed: int = 42, device: str = "cuda", max_epochs: int = 100,
               patience: int = 10) -> list[dict]:
    from gpu8.baselines import kdgene as kd
    from gpu8v6.common import World6
    from .chain8 import Chain
    from .pack8 import ArmSpec
    from .roles8 import Roles
    w = World6(load_heavy=True)
    roles = Roles(w)
    ch = Chain(ArmSpec(), role="comparator", w=w)
    genes = [w.gene_id_of_row[i] for i in range(w.n_g)]
    out = []
    for j in folds:
        d_pred = paths8.FORMAL / "main" / f"s{seed}" / f"fold{j}"
        res_dir = paths8.FORMAL / "comparators" / "kdgene" / f"fold{j}"
        res_dir.mkdir(parents=True, exist_ok=True)
        if (res_dir / "done.json").exists():
            continue
        t0 = time.time()
        ref = set(roles.reference(w.contexts[f"f{j}_outer"]))
        kg = kd.build_kg(w.cfg, w.relations, w.phenotypes, w.hierarchy, w.ppi, ref, genes)
        val = _Blk(*strict_block_from_pred(ch, j, "val", d_pred))

        def score(model, b):
            s = np.full((b.D, b.N), -1e9)
            for i, d in enumerate(b.ids):
                r = kd.score_kdgene(model, kg, d, [], b.genes[i], w.gene_id_of_row, device)
                if r is not None:
                    s[i] = r
            return s

        def val_fn(model):
            model.eval()
            v = float(recall_at(val, score(model, val)).mean())
            model.train()
            return v

        model, hist = kd.train_kdgene(w.cfg, kg, device, seed=seed, edim=1000, rdim=1000,
                                      max_epochs=max_epochs, patience=patience, val_fn=val_fn, progress=True)
        model.eval()
        ids, g, _, _ = strict_block_from_pred(ch, j, "test", d_pred)
        tb = _Blk(ids, g, None, None)
        st = score(model, tb)
        np.savez_compressed(res_dir / "pred_kdgene__test.npz", ids=np.array(ids), genes=g.astype(np.int16),
                            score=st.astype(np.float32))
        info = {"fold": j, "seed": seed, "best_epoch": hist["best_epoch"], "best_val_recall10": hist["best_val"],
                "epochs_run": len(hist["history"]), "n_triples": hist["n_triples"],
                "reference_diseases": len(ref), "seconds": time.time() - t0, "lists": "deployed strict 500",
                "history": hist["history"], "generated_utc": now_utc()}
        dump_json(res_dir / "done.json", info)
        write_result(f"V8K__kdgene__f{j}__s{seed}", {"role": "comparator", "component": "kdgene", "fold": j,
                                                     "seed": seed, "status": "done", "fit": info})
        print(f"  kdgene fold {j}: best ep {hist['best_epoch']} val {hist['best_val']:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        out.append(info)
        del model
        torch.cuda.empty_cache()
    return out


def sign_views(ctx: str):
    from gpu8.contexts import load_context
    _, arrays = load_context(ctx)
    return arrays["disease_views"], arrays["gene_views"]


def sign_X(dv, gv, d_rows: np.ndarray, genes: np.ndarray) -> np.ndarray:
    dim = gv.shape[1] // 4
    g = gv[genes]                                   # [D, N, 4dim]
    d0 = dv[d_rows][:, None, :dim]
    g0, g1, g2 = g[..., :dim], g[..., dim:2 * dim], g[..., 2 * dim:3 * dim]
    return np.concatenate([g0, g1, g2, g0 * d0, g1 * d0, g2 * d0], axis=2).astype(np.float32)


def run_sign(folds=range(5), seed: int = 42, device: str = "cuda", epochs: int = 30) -> list[dict]:
    import torch.nn as nn
    from gpu8v6.common import World6
    from .chain8 import Chain, ctx_list
    from .pack8 import ArmSpec
    from .semantic8 import Semantic  # noqa: F401
    from .v6pass8 import V6Pass, V6World
    w = World6(load_heavy=False)
    ch = Chain(ArmSpec(), role="comparator", w=w)
    d_ix = w.idx.d_ix
    out = []
    for j in folds:
        res_dir = paths8.FORMAL / "comparators" / "sign" / f"fold{j}"
        res_dir.mkdir(parents=True, exist_ok=True)
        if (res_dir / "done.json").exists():
            continue
        t0 = time.time()
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        d_pred = paths8.FORMAL / "main" / f"s{seed}" / f"fold{j}"
        # training rows: inner OOF strict pools (rebuilt from the saved fold selection)
        Xs, ys = [], []
        from .formal8 import selection
        sel = selection(j)
        for c in ctx_list(j)[:3]:
            dv, gv = sign_views(c)
            z = np.load(paths8.DEV / f"fold{j}" / f"pools_{c}_oof.npz")
            ids, genes = [str(x) for x in z["ids"]], z["genes"].astype(np.int64)
            for i, d in enumerate(ids):
                q = w.query_rows(d)
                pos = np.flatnonzero(np.isin(genes[i], q))
                if len(pos) == 0:
                    continue
                neg = np.flatnonzero(~np.isin(genes[i], q))
                neg = rng.choice(neg, size=min(len(neg), 5 * len(pos)), replace=False)
                idx = np.concatenate([pos, neg])
                Xs.append(sign_X(dv, gv, np.array([d_ix[d]]), genes[i:i + 1, idx])[0])
                ys.append(np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]))
            del dv, gv
        X = np.concatenate(Xs)
        y = np.concatenate(ys)
        mu, sd = X.mean(0), np.maximum(X.std(0), 1e-6)
        Xt = torch.as_tensor((X - mu) / sd, dtype=torch.float32, device=device)
        yt = torch.as_tensor(y, dtype=torch.float32, device=device)
        model = nn.Sequential(nn.Linear(Xt.shape[1], 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(),
                              nn.Linear(128, 1)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss()
        dvo, gvo = sign_views(f"f{j}_outer")
        vids, vg, vy, vnq = strict_block_from_pred(ch, j, "val", d_pred)
        vb = _Blk(vids, vg, vy, vnq)
        vrows = np.array([d_ix[d] for d in vids])

        def score_rows(rows, genes, chunk: int = 48):
            out = np.zeros(genes.shape, dtype=np.float32)
            with torch.no_grad():
                for b0 in range(0, len(rows), chunk):
                    Xq = (sign_X(dvo, gvo, rows[b0:b0 + chunk], genes[b0:b0 + chunk]) - mu) / sd
                    t = torch.as_tensor(Xq.reshape(-1, Xq.shape[2]), dtype=torch.float32, device=device)
                    out[b0:b0 + chunk] = model(t).squeeze(-1).cpu().numpy().reshape(Xq.shape[0], Xq.shape[1])
            return out

        def score(_):
            return score_rows(vrows, vg)

        best, best_ep, best_state, curve = -1, 0, None, []
        for ep in range(1, epochs + 1):
            model.train()
            perm = torch.randperm(Xt.shape[0], device=device)
            for s in range(0, Xt.shape[0], 4096):
                idx = perm[s:s + 4096]
                opt.zero_grad(set_to_none=True)
                loss = lossf(model(Xt[idx]).squeeze(-1), yt[idx])
                loss.backward()
                opt.step()
            model.eval()
            r = float(recall_at(vb, score(None)).mean())
            curve.append(r)
            if r > best + 1e-9:
                best, best_ep = r, ep
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.eval()
        tids, tg, _, _ = strict_block_from_pred(ch, j, "test", d_pred)
        st = score_rows(np.array([d_ix[d] for d in tids]), tg)
        np.savez_compressed(res_dir / "pred_sign__test.npz", ids=np.array(tids), genes=tg.astype(np.int16),
                            score=st.astype(np.float32))
        info = {"fold": j, "seed": seed, "best_epoch": best_ep, "best_val_recall10": best, "curve": curve,
                "n_train_rows": int(len(y)), "seconds": time.time() - t0, "generated_utc": now_utc(),
                "views": "V5 context views (propagation scales 0-2) of the row's own context"}
        dump_json(res_dir / "done.json", info)
        write_result(f"V8K__sign__f{j}__s{seed}", {"role": "comparator", "component": "sign", "fold": j,
                                                   "seed": seed, "status": "done", "fit": info})
        print(f"  sign fold {j}: best ep {best_ep} val {best:.4f} ({time.time() - t0:.0f}s)", flush=True)
        out.append(info)
    return out


if __name__ == "__main__":
    import sys
    paths8.ensure()
    which = sys.argv[1] if len(sys.argv) > 1 else "sign"
    if which == "kdgene":
        run_kdgene()
    else:
        run_sign()
