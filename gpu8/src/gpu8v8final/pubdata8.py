"""Source data for the publication figures and the expanded experiment text.

Everything is recomputed from saved runs, predictions and frozen inputs after the
registration.  The additional analyses (query-level ranks, knowledge strata,
disease categories, Recall@K curves, budget curves, case studies, feature
importance, embeddings) are *descriptive*: they never feed back into any choice.
Outputs: ``results_v8/pub/*.parquet|json``.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import paths8
from .common8 import dump_json, now_utc

OUT = paths8.RESULTS / "pub"
SEEDS = (42, 52, 62)
FOLDS = range(5)

# CTD MEDIC top-level branches (MeSH tree letters/numbers) -> Chinese names
MESH_CAT = {
    "C01": "感染", "C04": "肿瘤", "C05": "肌肉骨骼疾病", "C06": "消化系统疾病", "C07": "口腔与牙齿疾病",
    "C08": "呼吸系统疾病", "C09": "耳鼻喉疾病", "C10": "神经系统疾病", "C11": "眼部疾病",
    "C12": "泌尿生殖系统疾病", "C13": "女性泌尿生殖疾病与妊娠并发症", "C14": "心血管疾病",
    "C15": "血液与淋巴系统疾病", "C16": "先天性、遗传性与新生儿疾病", "C17": "皮肤与结缔组织疾病",
    "C18": "营养与代谢性疾病", "C19": "内分泌系统疾病", "C20": "免疫系统疾病", "C21": "环境来源疾病",
    "C22": "动物疾病", "C23": "病理状态、体征与症状", "C24": "职业病", "C25": "化学物质诱发疾病",
    "C26": "创伤与损伤", "F03": "精神障碍",
}
GTYPE_ZH = {"protein": "蛋白编码", "ncrna": "非编码 RNA", "pseudo": "假基因", "other": "其他"}


def _world(heavy: bool = False):
    from gpu8v6.common import World6
    return World6(load_heavy=heavy)


def _cached(name: str, force: bool):
    p = OUT / name
    return (p, pd.read_parquet(p)) if p.exists() and not force else (p, None)


# ============================================================ gene / disease tables
def gene_table(force: bool = False) -> pd.DataFrame:
    p, df = _cached("genes.parquet", force)
    if df is not None:
        return df
    from gpu8.contexts import build_P
    from gpu8v6.gene_hpo import GeneHPOSource, HPOSpace
    from .pack8 import ArmSpec, Pack8
    w = _world(True)
    a = Pack8(ArmSpec(), w).a
    G = w.n_g
    ids = [w.gene_id_of_row[i] for i in range(G)]
    gt = w.genes.assign(gid=w.genes["gene_id"].astype(str)).set_index("gid").reindex(ids)
    lg = gt["locus_group"].fillna("").str.lower()
    gtype = np.where(lg.str.contains("protein"), "protein",
                     np.where(lg.str.contains("non-coding"), "ncrna",
                              np.where(lg.str.contains("pseudo"), "pseudo", "other")))
    rec = pd.read_parquet(paths8.TABLES / "go_records_current_v8.parquet", columns=["gene_row", "go_id"])
    npairs = rec.drop_duplicates(["gene_row", "go_id"]).groupby("gene_row").size().reindex(range(G)).fillna(0)
    nrec = rec.groupby("gene_row").size().reindex(range(G)).fillna(0)
    _, deg = build_P(w.ppi, w.idx)
    rel = w.primary_relations
    nrel = rel["gene_id"].astype(str).map(w.idx.g_ix).dropna().astype(int).value_counts().reindex(range(G)).fillna(0)
    gh = GeneHPOSource(w, HPOSpace())
    hpo_genes = set(gh.rows.loc[gh.rows["mapped"], "gene_row"].astype(int))
    df = pd.DataFrame({
        "gene_row": np.arange(G), "gene_id": ids, "symbol": gt["symbol"].fillna("").to_numpy(),
        "gtype": gtype, "has_go": a["g_function_mask"].astype(bool), "n_go_pairs": npairs.to_numpy().astype(int),
        "n_go_records": nrec.to_numpy().astype(int), "has_seq": a["g_sequence_mask"].astype(bool),
        "has_expr": a["g_expression_mask"].any(axis=1).astype(bool),
        "has_gene_hpo": np.isin(np.arange(G), list(hpo_genes)), "ppi_degree": np.asarray(deg).astype(float),
        "n_rel": nrel.to_numpy().astype(int)})
    df["has_ppi"] = df["ppi_degree"] > 0
    df["has_rel"] = df["n_rel"] > 0
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return df


def _cats(tree: str) -> list[str]:
    out = []
    for t in str(tree or "").split("|"):
        t = t.strip()
        if len(t) >= 3 and t[0] in "CF":
            k = t[:3]
            if k in MESH_CAT and k not in out:
                out.append(k)
    return out


def disease_table(force: bool = False) -> pd.DataFrame:
    p, df = _cached("diseases.parquet", force)
    if df is not None:
        return df
    w = _world()
    sp = w.spec
    fold_of = {}
    for i in range(5):
        for d in sp.folds[f"F{i}"]:
            fold_of[d] = i
    dev, ext = set(sp.dev_diseases), set(sp.external_diseases)
    seg = pd.read_parquet(paths8.TABLES / "disease_segment_summary_l1.parquet").set_index("disease_id")
    ph = w.phenotypes[~w.phenotypes["negated"].astype(bool)] if "negated" in w.phenotypes else w.phenotypes
    nhpo = ph.groupby("disease_id")["hpo_id"].nunique()
    rel = w.primary_relations
    nrel = rel.groupby("disease_id").size()
    d = w.diseases.copy()
    d["n_rel"] = d["disease_id"].map(nrel).fillna(0).astype(int)
    d["fold"] = d["disease_id"].map(fold_of).fillna(-1).astype(int)
    d["role"] = np.where(d["disease_id"].isin(dev), "dev", np.where(d["disease_id"].isin(ext), "external",
                                                                     np.where(d["fold"] >= 0, "test_fold", "none")))
    d["n_hpo"] = d["disease_id"].map(nhpo).fillna(0).astype(int)
    d["n_tokens"] = d["disease_id"].map(seg["n_tokens_full"]).fillna(0).astype(int)
    d["n_segments"] = d["disease_id"].map(seg["n_segments"]).fillna(0).astype(int)
    d["has_def"] = d["definition"].fillna("").str.len() > 0
    d["omim"] = d["omim_ids"].fillna("").astype(str).str.len() > 0
    d["cats"] = d["tree_numbers"].map(lambda t: "|".join(_cats(t)))
    keep = ["disease_id", "disease_name", "family_group", "n_rel", "fold", "role", "n_hpo", "n_tokens",
            "n_segments", "has_def", "omim", "cats"]
    df = d[keep].reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return df


# ============================================================ query-level ranks
METHODS_ALL = ["deployed", "channel_lgb_v6", "v71_direct_strict", "v71_direct_v6", "B", "F7_v6", "F7_strict",
               "standalone_sF", "standalone_r", "standalone_sF0", "standalone_sS", "C_b0", "C_b1", "C_b2"]


def _rank_lookup(pred: dict, i: int) -> dict:
    v = pred["valid"][i] if "valid" in pred else np.isfinite(pred["score"][i])
    g = pred["genes"][i][v].astype(np.int64)
    s = pred["score"][i][v].astype(np.float64)
    o = g[np.lexsort((g, -s))]
    return {int(x): r + 1 for r, x in enumerate(o.tolist())}


def query_ranks(force: bool = False) -> pd.DataFrame:
    """One row per (seed, fold, method, test disease, positive gene) with the 1-based rank (0 = not listed)."""
    p, df = _cached("query_ranks.parquet", force)
    if df is not None:
        return df
    from .evaluate8 import load_pred
    from .feats8 import Labels
    from .formal8 import sel_cfg, selection
    w = _world()
    lab = Labels(w)
    rows = []
    for seed in SEEDS:
        for j in FOLDS:
            d = paths8.FORMAL / "main" / f"s{seed}" / f"fold{j}"
            lam = sel_cfg(selection(j))["corrector"]["lam"]
            files = {m: d / f"pred_{m}__test.npz" for m in METHODS_ALL if not m.startswith("C_")}
            for b in (0, 1, 2):
                files[f"C_b{b}"] = d / f"pred_C_b{b}_l{lam:g}__test.npz"
            if seed == 42:
                for comp in ("kdgene", "sign"):
                    files[comp] = paths8.FORMAL / "comparators" / comp / f"fold{j}" / f"pred_{comp}__test.npz"
            for m, fp in files.items():
                if not fp.exists():
                    continue
                pred = load_pred(fp)
                for i, dd in enumerate(pred["ids"]):
                    dd = str(dd)
                    q = lab.rows(dd, "test")
                    if not len(q):
                        continue
                    rk = _rank_lookup(pred, i)
                    for g in q.tolist():
                        rows.append((seed, j, m, dd, int(g), rk.get(int(g), 0)))
            print(f"  query ranks s{seed} fold {j}: {len(rows)} rows", flush=True)
    df = pd.DataFrame(rows, columns=["seed", "fold", "method", "disease_id", "gene_row", "rank"])
    df["rank"] = df["rank"].astype(np.int16)
    df["seed"] = df["seed"].astype(np.int16)
    df["fold"] = df["fold"].astype(np.int8)
    df.to_parquet(p, index=False)
    return df


def query_context(force: bool = False) -> pd.DataFrame:
    """Per (fold, test disease, positive gene): evaluation-context attributes of the gene."""
    p, df = _cached("query_context.parquet", force)
    if df is not None:
        return df
    from .feats8 import Labels
    from .v6pass8 import V6Pass, V6World
    w = _world()
    lab = Labels(w)
    rows = []
    for j in FOLDS:
        v6 = V6Pass(V6World(), f"f{j}_outer")
        cold = v6.cold()
        src = np.load(paths8.FORMAL / "main" / "s42" / f"fold{j}" / f"gene_src_f{j}_outer.npy")
        for dd in w.spec.folds[f"F{j}"]:
            q = lab.rows(dd, "test")
            if not len(q) or dd not in v6.pos:
                continue
            i = v6.pos[dd]
            rB = np.asarray(v6.rB[i]).astype(np.int32)
            fu = np.asarray(v6.fused[i])
            for g in q.tolist():
                rows.append((j, dd, int(g), int(v6.ref_degree[g]), bool(v6.hpo_visible[g]), bool(cold[g]),
                             int(src[g]), int(rB[g]), bool(fu[g] > 0)))
        print(f"  query context fold {j}: {len(rows)}", flush=True)
    df = pd.DataFrame(rows, columns=["fold", "disease_id", "gene_row", "ref_degree", "hpo_visible", "cold",
                                     "sem_src", "rB", "v6_signal"])
    df.to_parquet(p, index=False)
    return df


# ============================================================ heavy: full-background ranks
def budget_ranks(force: bool = False, device: str = "cuda") -> pd.DataFrame:
    """Seed 42: for every test positive its full-background rank under the V6 order (r_B), the semantic
    order (r_S), the availability-aware RRF and the plain RRF (w_j of the context), plus the aware
    top-500 exactness check against the saved strict pools and the 500th-candidate threshold."""
    p, df = _cached("budget_ranks.parquet", force)
    if df is not None:
        return df
    import gc

    from .chain8 import Chain
    from .feats8 import Labels
    from .formal8 import selection
    from .pack8 import ArmSpec
    from .pool8 import rrf_score, top_budget
    from .teacher8 import teacher_run_id
    from .v6pass8 import V6Pass, V6World
    ch = Chain(ArmSpec(), V6World(), device=device, role="analysis")
    lab = Labels(ch.w)
    rows, thr, checks = [], [], {}
    for j in FOLDS:
        t0 = time.time()
        ctx = f"f{j}_outer"
        sel = selection(j)
        w_j = sel["w"]["w"]
        rid = teacher_run_id("main", ctx, sel["controls"], 42)
        v6 = V6Pass(V6World(), ctx)
        sem = ch.semantic(ctx, rid, v6, None)
        spos = {d: i for i, d in enumerate(sem.ids)}
        pool = np.load(paths8.FORMAL / "main" / "s42" / f"fold{j}" / f"pools_{ctx}_test.npz")
        pool_of = {str(d): set(g.astype(int).tolist()) for d, g in zip(pool["ids"], pool["genes"])}
        ok_exact, n_cmp = 0, 0
        test_ids = [d for d in ch.w.spec.folds[f"F{j}"] if d in v6.pos and d in spos and len(lab.rows(d, "test"))]
        G = v6.rB.shape[1]
        for b0 in range(0, len(test_ids), 256):
            ids = test_ids[b0:b0 + 256]
            rB = np.asarray(v6.rB[v6.rows(ids)]).astype(np.int32)
            sr = np.array([spos[d] for d in ids])
            rS, m = sem.rS[sr], sem.m[sr]
            Ra = rrf_score(rB, rS, m, w_j)
            Rp = rrf_score(rB, rS, m, w_j, plain=True)
            gid = np.arange(G)
            for k, d in enumerate(ids):
                oa = np.lexsort((gid, -Ra[k]))
                op = np.lexsort((gid, -Rp[k]))
                ra = np.empty(G, np.int32)
                ra[oa] = np.arange(1, G + 1)
                rp = np.empty(G, np.int32)
                rp[op] = np.arange(1, G + 1)
                top = set(oa[:500].tolist())
                if d in pool_of:
                    n_cmp += 1
                    ok_exact += int(top == pool_of[d])
                thr.append((j, d, float(Ra[k, oa[499]]), float(Rp[k, op[499]]), w_j,
                            int(m[k].sum()), int((rB[k] <= 500).sum())))
                for g in lab.rows(d, "test").tolist():
                    rows.append((j, d, int(g), int(rB[k, g]), int(rS[k, g]), bool(m[k, g]), int(ra[g]), int(rp[g])))
        checks[ctx] = {"compared": n_cmp, "aware_top500_equal_saved_pool": ok_exact, "w": w_j,
                       "seconds": time.time() - t0}
        print(f"  budget ranks {ctx}: {len(rows)} positives, exact {ok_exact}/{n_cmp} ({time.time() - t0:.0f}s)",
              flush=True)
        del sem
        gc.collect()
    df = pd.DataFrame(rows, columns=["fold", "disease_id", "gene_row", "rB", "rS", "m", "r_aware", "r_plain"])
    df.to_parquet(p, index=False)
    pd.DataFrame(thr, columns=["fold", "disease_id", "R500_aware", "R500_plain", "w", "n_sem_available",
                               "n_v6_top500"]).to_parquet(OUT / "budget_thresholds.parquet", index=False)
    dump_json(OUT / "budget_checks.json", {"generated_utc": now_utc(), "checks": checks})
    return df


def tsne_torch(X: np.ndarray, perplexity: float = 40.0, n_iter: int = 1000, seed: int = 42,
               device: str = "cuda") -> np.ndarray:
    """Exact-gradient t-SNE (van der Maaten & Hinton 2008) on the GPU: kNN-restricted perplexity
    calibration (k = 3u), symmetrised P, early exaggeration 12 for 250 iterations, momentum 0.5 -> 0.8,
    delta-bar-delta gains, learning rate max(n / 48, 50), PCA initialisation (std 1e-4)."""
    import torch
    g = torch.Generator(device="cpu").manual_seed(seed)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    n = Xt.shape[0]
    k = min(n - 1, int(3 * perplexity))
    nb_d, nb_i = [], []
    for b0 in range(0, n, 2048):
        d = torch.cdist(Xt[b0:b0 + 2048], Xt).pow(2)
        d[torch.arange(d.shape[0]), torch.arange(b0, b0 + d.shape[0])] = float("inf")
        v, i = torch.topk(d, k, largest=False)
        nb_d.append(v)
        nb_i.append(i)
    D, I = torch.cat(nb_d), torch.cat(nb_i)
    target = np.log(perplexity)
    lo = torch.full((n, 1), 1e-20, device=device)
    hi = torch.full((n, 1), 1e20, device=device)
    beta = torch.ones((n, 1), device=device)
    for _ in range(100):
        Pc = torch.exp(-(D - D[:, :1]) * beta)
        s = Pc.sum(1, keepdim=True)
        H = torch.log(s) + beta * ((D - D[:, :1]) * Pc).sum(1, keepdim=True) / s
        up = H > target
        lo = torch.where(up, beta, lo)
        hi = torch.where(up, hi, beta)
        beta = torch.where(hi >= 1e19, beta * 2, (lo + hi) / 2)
    Pc = torch.exp(-(D - D[:, :1]) * beta)
    Pc = Pc / Pc.sum(1, keepdim=True)
    P = torch.zeros((n, n), device=device)
    P.scatter_(1, I, Pc)
    P = (P + P.T) / (2 * n)
    P = torch.clamp(P, min=1e-12)
    Xc = Xt - Xt.mean(0)
    U, S, V = torch.pca_lowrank(Xc, q=2, center=False)
    Y = Xc @ V[:, :2]
    Y = Y / Y[:, 0].std() * 1e-4
    upd = torch.zeros_like(Y)
    gains = torch.ones_like(Y)
    lr = max(n / 12.0 / 4.0, 50.0)
    eye = torch.eye(n, device=device, dtype=torch.bool)
    for it in range(n_iter):
        exag = 12.0 if it < 250 else 1.0
        mom = 0.5 if it < 250 else 0.8
        num = 1.0 / (1.0 + torch.cdist(Y, Y).pow(2))
        num.masked_fill_(eye, 0.0)
        Q = torch.clamp(num / num.sum(), min=1e-12)
        PQ = (exag * P - Q) * num
        grad = 4.0 * (torch.diag(PQ.sum(1)) - PQ) @ Y
        inc = (grad > 0) != (upd > 0)
        gains = torch.where(inc, gains + 0.2, gains * 0.8).clamp_(min=0.01)
        upd = mom * upd - lr * gains * grad
        Y = Y + upd
        Y = Y - Y.mean(0)
    del P, num, Q, PQ
    torch.cuda.empty_cache()
    _ = g
    return Y.cpu().numpy()


def embeddings(force: bool = False, device: str = "cuda") -> pd.DataFrame:
    """t-SNE of the teacher's disease vectors z_d (fold-0 outer context, seed 42)."""
    p, df = _cached("tsne_diseases.parquet", force)
    if df is not None:
        return df
    import torch

    from .pack8 import ArmSpec, Pack8
    from .teacher8 import disease_z, gene_zF, load_teacher, teacher_run_id
    w = _world()
    pack = Pack8(ArmSpec(), w)
    T = pack.tensors(device)
    model = load_teacher(teacher_run_id("main", "f0_outer", "background", 42), T, device)
    rows = np.arange(w.n_d)
    with torch.no_grad():
        z, ok = disease_z(model, T, rows)
        zF, mF = gene_zF(model, T)
    z, ok = z.float().cpu().numpy(), ok.cpu().numpy().astype(bool)
    idx = np.flatnonzero(ok)
    t0 = time.time()
    Y = tsne_torch(z[idx], perplexity=40.0, n_iter=1000, seed=42, device=device)
    dt = disease_table()
    inv = {v: k for k, v in w.idx.d_ix.items()}
    df = pd.DataFrame({"disease_id": [inv[i] for i in idx], "x": Y[:, 0], "y": Y[:, 1]})
    df = df.merge(dt[["disease_id", "cats", "n_rel", "role", "n_hpo", "n_tokens"]], on="disease_id", how="left")
    df.to_parquet(p, index=False)
    # gene-side: cosine between each disease and the vectors of its known genes vs random genes
    zF_np, mF_np = zF.float().cpu().numpy(), mF.cpu().numpy().astype(bool)
    rel = w.primary_relations
    g_rows = rel["gene_id"].astype(str).map(w.idx.g_ix)
    d_rows = rel["disease_id"].map(w.idx.d_ix)
    keep = g_rows.notna() & d_rows.notna()
    gi, di = g_rows[keep].astype(int).to_numpy(), d_rows[keep].astype(int).to_numpy()
    ok_pair = ok[di] & mF_np[gi]
    cos_pos = (z[di[ok_pair]] * zF_np[gi[ok_pair]]).sum(axis=1)
    rng = np.random.default_rng(20260923)
    rg = rng.choice(np.flatnonzero(mF_np), size=ok_pair.sum())
    cos_rnd = (z[di[ok_pair]] * zF_np[rg]).sum(axis=1)
    pd.DataFrame({"cos_known": cos_pos, "cos_random": cos_rnd}).to_parquet(OUT / "cosine_known_random.parquet",
                                                                             index=False)
    dump_json(OUT / "tsne_meta.json", {"generated_utc": now_utc(), "n": int(len(idx)), "perplexity": 40,
                                       "seconds": time.time() - t0, "context": "f0_outer", "seed": 42,
                                       "pairs_cosine": int(ok_pair.sum())})
    return df


# ============================================================ development / training records
def dev_records() -> dict:
    out = {"w": [], "A": [], "C": [], "C_curves": [], "B": [], "teacher_controls": None}
    for j in FOLDS:
        s = json.loads((paths8.DEV / f"fold{j}" / "selection.json").read_text(encoding="utf-8"))
        for k, v in s["w"]["coverage"].items():
            out["w"].append({"fold": j, "setting": k, "coverage": v, "chosen_w": s["w"]["w"]})
        for r in s["summary"]["A"]:
            out["A"].append({"fold": j, **{k: r[k] for k in ("name", "num_leaves", "min_child_samples",
                                                               "dev_recall10", "dev_cold_recall10",
                                                               "best_iteration")},
                             "deployed": r["name"] == s["ab"]["best_A"]})
        for r in s["summary"]["corrector"]:
            out["C"].append({"fold": j, **{k: r[k] for k in ("name", "beta", "lam", "dev_recall10",
                                                               "dev_cold_recall10", "best_iteration")},
                             "cold_pos_pairs": r["diag"].get("cold_positives_in_pairs"),
                             "val_best": r["diag"].get("val_recall10_best")})
            cm = paths8.DEV / f"fold{j}" / r["name"] / "corrector_meta.json"
            if cm.exists():
                c = json.loads(cm.read_text(encoding="utf-8"))
                for it, v in enumerate(c.get("curve", []), start=1):
                    out["C_curves"].append({"fold": j, "name": r["name"], "beta": r["beta"], "lam": r["lam"],
                                            "round": it, "val_recall10": v})
        bd = s["summary"].get("B_dev")
        out["B"].append({"fold": j, "dev_recall10": bd.get("dev_recall10") if isinstance(bd, dict) else bd,
                         "dev_cold_recall10": bd.get("dev_cold_recall10") if isinstance(bd, dict) else None})
    tc = paths8.MANIFESTS / "dev_teacher_controls_v8.json"
    out["teacher_controls"] = json.loads(tc.read_text(encoding="utf-8")) if tc.exists() else None
    st = paths8.MANIFESTS / "dev_student_selection_v8.json"
    out["student"] = json.loads(st.read_text(encoding="utf-8")) if st.exists() else None
    return out


def teacher_histories() -> pd.DataFrame:
    rows = []
    for rd in sorted(paths8.RUNS.glob("V8T__main__f*_*__*__s*")):
        hp = rd / "history.json"
        if not hp.exists():
            continue
        _, _, ctx, controls, seed = rd.name.split("__")
        res = json.loads((rd / "result.json").read_text(encoding="utf-8"))
        best = (res.get("fit") or {}).get("best_epoch")
        for h in json.loads(hp.read_text(encoding="utf-8")):
            if h.get("stage") == "joint":
                rows.append({"ctx": ctx, "controls": controls, "seed": int(seed[1:]), "epoch": h["epoch"],
                             "sF_recall10": h.get("sF_recall10"), "r_recall10": h.get("r_recall10"),
                             "loss": h.get("loss"), "L_DG": h.get("L_DG"), "L_A": h.get("L_A"),
                             "best_epoch": best})
    return pd.DataFrame(rows)


FEAT_ZH = {
    "v_pheno": "表型通道分数", "v_hier": "层级通道分数", "v_onto": "本体通道分数", "v_text": "文本通道分数",
    "v_gene_hpo": "基因 HPO 通道分数", "v_rwr_pheno": "表型 RWR 分数", "v_compat": "兼容通道分数",
    "lr_pheno": "表型通道名次", "lr_hier": "层级通道名次", "lr_onto": "本体通道名次", "lr_text": "文本通道名次",
    "lr_gene_hpo": "基因 HPO 名次", "lr_rwr_pheno": "表型 RWR 名次", "lr_compat": "兼容通道名次",
    "a_pheno": "表型通道可用", "a_hier": "层级通道可用", "a_onto": "本体通道可用", "a_text": "文本通道可用",
    "a_gene_hpo": "基因 HPO 可用", "a_rwr_pheno": "RWR 可用", "a_compat": "兼容通道可用",
    "fused": "V6 融合分", "ret_rank": "V6 检索名次", "is_fill": "先验填充标记", "prior": "度先验",
    "d_hpo": "疾病 HPO 数", "d_pheno_sim": "疾病表型相似", "d_text_sim": "疾病文本相似", "d_onto_ic": "本体信息量",
    "d_onto_inf": "本体可用", "s_sem": "语义分数 $s_F$", "lr_S": "语义名次", "lr_R": "融合名次（缺失感知 RRF）",
    "m_sem": "语义可用标记", "src_go": "语义来源=GO", "src_seq": "语义来源=序列", "s_E": "表达分数",
    "a_E": "表达可用", "ref_visible": "参考边可见", "hpo_visible": "基因 HPO 可见", "lg_ref_degree": "参考度（对数）",
    "gt_protein": "蛋白编码", "gt_ncrna": "非编码 RNA", "has_seq": "有序列", "has_go": "有 GO", "has_expr": "有表达",
    "dm_text": "疾病有文本", "dm_hpo": "疾病有 HPO", "dm_tissue": "疾病有组织",
}


def feat_family(f: str) -> str:
    if f in ("s_sem", "lr_S", "m_sem", "src_go", "src_seq"):
        return "语义"
    if f == "lr_R":
        return "候选融合"
    if f in ("s_E", "a_E"):
        return "表达"
    if f.startswith(("v_", "lr_", "a_")) or f in ("fused", "ret_rank", "is_fill", "prior"):
        return "V6 检索通道"
    if f.startswith("d_") or f.startswith("dm_"):
        return "疾病描述"
    return "基因可用性"


def feature_importance() -> pd.DataFrame:
    import lightgbm as lgb
    from gpu8v6.scoring import FEATS

    from .feats8 import EXTRA
    from .formal8 import sel_cfg, selection
    names = list(FEATS) + list(EXTRA)
    rows = []
    for seed in SEEDS:
        for j in FOLDS:
            a_name = sel_cfg(selection(j))["A_name"]
            cands = [paths8.FORMAL / "main" / f"s{seed}" / f"fold{j}" / f"A_{a_name}_model.txt"]
            if seed == 42:
                cands.insert(0, paths8.DEV / f"fold{j}" / f"A_{a_name}_model.txt")
            fp = next((c for c in cands if c.exists()), None)
            if fp is None:
                continue
            b = lgb.Booster(model_str=fp.read_text(encoding="utf-8"))
            gain = b.feature_importance("gain")
            split = b.feature_importance("split")
            tot = gain.sum() or 1.0
            for k, f in enumerate(names[:len(gain)]):
                rows.append({"seed": seed, "fold": j, "feature": f, "gain_share": gain[k] / tot, "splits": split[k],
                             "model": a_name})
    return pd.DataFrame(rows)


def evidence_data() -> dict:
    rec = pd.read_parquet(paths8.TABLES / "go_records_current_v8.parquet", columns=["evidence_code", "pmids"])
    has = rec["pmids"].fillna("").astype(str).str.len() > 0
    codes = rec.assign(has_pmid=has).groupby(["evidence_code", "has_pmid"]).size().unstack(fill_value=0)
    codes.columns = ["no_pmid" if not c else "with_pmid" for c in codes.columns]
    codes = codes.reset_index()
    # PMID-relation bipartite components (primary relations with a PMID)
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components
    w = _world()
    ev = w.evidence[["relation_id", "pmid"]].dropna()
    ev = ev[ev["relation_id"].isin(set(w.primary_relations["relation_id"]))]
    ev["pmid"] = ev["pmid"].astype(str)
    ev = ev[ev["pmid"].str.len() > 0].drop_duplicates()
    r_ix = {r: i for i, r in enumerate(sorted(ev["relation_id"].unique()))}
    p_ix = {q: i for i, q in enumerate(sorted(ev["pmid"].unique()))}
    nr, npm = len(r_ix), len(p_ix)
    M = sparse.coo_matrix((np.ones(len(ev)), (ev["relation_id"].map(r_ix), ev["pmid"].map(p_ix) + nr)),
                          shape=(nr + npm, nr + npm))
    n_comp, lab = connected_components(M, directed=False)
    rel_comp = pd.Series(lab[:nr]).value_counts()
    sizes = rel_comp.value_counts().sort_index()
    audit = json.loads((paths8.AUDIT / "evidence_policy_audit_v8.json").read_text(encoding="utf-8"))
    return {"codes": codes, "block_sizes": pd.DataFrame({"relations_per_block": sizes.index.astype(int),
                                                         "n_blocks": sizes.to_numpy().astype(int)}),
            "n_components_with_pmid": int(len(rel_comp)), "giant": int(rel_comp.max()), "audit": audit}


def mask_grid() -> pd.DataFrame:
    from gpu8.contexts import build_P
    from gpu8v71.featurepack71 import FeaturePack

    from .evidence8 import GOPolicy
    from .masks8 import deciles
    w = _world(True)
    fa = GOPolicy(w, "current").function_arrays()
    seq = FeaturePack().a["g_sequence_mask"]
    elig = np.flatnonzero(seq & fa["g_function_mask"])
    _, deg = build_P(w.ppi, w.idx)
    nterm = fa["n_terms_per_gene"][elig]
    dg = np.asarray(deg)[elig]
    m = pd.read_parquet(paths8.TABLES / "go_masks_v8.parquet")
    ms = dict(zip(m["gene_row"], m["mask_set"]))
    return pd.DataFrame({"gene_row": elig, "go_decile": deciles(nterm), "ppi_decile": deciles(dg),
                         "n_go_terms": nterm, "ppi_degree": dg,
                         "mask_set": [ms.get(int(g), "none") for g in elig]})


def ledger() -> pd.DataFrame:
    led = pd.read_csv(paths8.MANIFESTS / "run_ledger_v8.csv")
    led["finished"] = pd.to_datetime(led["finished_utc"], errors="coerce", utc=True)
    led["started"] = pd.to_datetime(led["started_utc"], errors="coerce", utc=True)
    return led


def build(heavy: bool = False) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    info = {"generated_utc": now_utc()}
    gene_table()
    disease_table()
    query_ranks()
    query_context()
    fi = feature_importance()
    fi.to_parquet(OUT / "feature_importance.parquet", index=False)
    th = teacher_histories()
    th.to_parquet(OUT / "teacher_histories.parquet", index=False)
    dv = dev_records()
    for k in ("w", "A", "C", "C_curves", "B"):
        pd.DataFrame(dv[k]).to_parquet(OUT / f"dev_{k}.parquet", index=False)
    dump_json(OUT / "dev_misc.json", {"teacher_controls": dv["teacher_controls"], "student": dv["student"]})
    ev = evidence_data()
    ev["codes"].to_parquet(OUT / "go_evidence_codes.parquet", index=False)
    ev["block_sizes"].to_parquet(OUT / "evidence_block_sizes.parquet", index=False)
    dump_json(OUT / "evidence_misc.json", {"n_components_with_pmid": ev["n_components_with_pmid"],
                                           "giant": ev["giant"]})
    mask_grid().to_parquet(OUT / "mask_grid.parquet", index=False)
    ledger().to_parquet(OUT / "ledger.parquet", index=False)
    if heavy:
        budget_ranks()
        embeddings()
    info["seconds"] = time.time() - t0
    dump_json(OUT / "build_info.json", info)
    return info


if __name__ == "__main__":
    import sys
    paths8.ensure()
    what = sys.argv[1:] or ["light"]
    if "light" in what:
        print(build(False))
    if "budget" in what:
        budget_ranks(force="force" in what)
    if "tsne" in what:
        embeddings(force="force" in what)
