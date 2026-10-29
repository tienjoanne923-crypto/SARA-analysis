"""Low-cost biological support from public processed expression matrices.

Plan section 15.  No raw sequencing, no single-cell clustering, no variant
calling, no new clinical samples.  The frozen top-ranked genes of a disease are
checked against an independent kind of evidence: differential expression in a
public cohort, compared with a background matched on mean expression, PPI
degree and annotation coverage.

Design rules the plan fixes and this module follows:

* the paired skin design uses each patient's own uninvolved tissue as the pair;
  58 uninvolved samples and 64 healthy controls are never silently merged into
  "122 healthy people";
* cohorts are chosen before the model rankings are looked at;
* differential expression supports correlation only, never causation.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths
from .hashing import atomic_write_df, atomic_write_json

SERIES = {
    "GSE13355": {"disease_context": "psoriasis", "platform": "GPL570",
                 "design": "paired lesional vs uninvolved (same patient) + separate controls"},
    "GSE30999": {"disease_context": "psoriasis", "platform": "GPL570",
                 "design": "paired lesional vs non-lesional (same patient)"},
    "GSE87466": {"disease_context": "ulcerative_colitis", "platform": "GPL13158",
                 "design": "unpaired disease vs normal colon"},
    "GSE75214": {"disease_context": "ulcerative_colitis", "platform": "GPL6244",
                 "design": "unpaired active/inactive vs control, colon and ileum"},
}


# --------------------------------------------------------------------------
# GEO parsing
# --------------------------------------------------------------------------
@dataclass
class SeriesMatrix:
    accession: str
    samples: list[str]
    characteristics: dict[str, list[str]]
    titles: list[str]
    platform: str
    expr: pd.DataFrame


def parse_series_matrix(path: Path) -> SeriesMatrix:
    samples: list[str] = []
    titles: list[str] = []
    chars: dict[str, list[str]] = {}
    platform = ""
    rows: list[list[str]] = []
    header: list[str] = []
    in_table = False
    n_char = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                in_table = False
                continue
            if in_table:
                parts = [p.strip('"') for p in line.split("\t")]
                if not header:
                    header = parts
                else:
                    rows.append(parts)
                continue
            if line.startswith("!Sample_geo_accession"):
                samples = [p.strip('"') for p in line.split("\t")[1:]]
            elif line.startswith("!Sample_title"):
                titles = [p.strip('"') for p in line.split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                chars[f"ch{n_char}"] = [p.strip('"') for p in line.split("\t")[1:]]
                n_char += 1
            elif line.startswith("!Series_platform_id"):
                platform = line.split("\t")[1].strip('"')
    if not rows:
        return SeriesMatrix(path.stem, samples, chars, titles, platform, pd.DataFrame())
    df = pd.DataFrame(rows, columns=header)
    idcol = df.columns[0]
    df = df.set_index(idcol)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(how="all")
    return SeriesMatrix(path.stem.split("_")[0], samples, chars, titles, platform, df)


def parse_platform_annotation(path: Path) -> dict[str, str]:
    """probe id -> Entrez gene id, from a GEO .annot file."""
    mapping: dict[str, str] = {}
    if not path.exists():
        return mapping
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        header = None
        for line in fh:
            if line.startswith("!") or line.startswith("#") or line.startswith("^"):
                continue
            parts = line.rstrip("\n").split("\t")
            if header is None:
                header = parts
                continue
            rec = dict(zip(header, parts))
            pid = rec.get("ID", "").strip()
            gid = rec.get("Gene ID", "").strip()
            if pid and gid and gid != "":
                mapping[pid] = gid.split("///")[0].strip()
    return mapping


# --------------------------------------------------------------------------
# sample grouping
# --------------------------------------------------------------------------
def classify_samples(sm: SeriesMatrix) -> pd.DataFrame:
    rows = []
    for i, gsm in enumerate(sm.samples):
        blob = " | ".join(v[i] for v in sm.characteristics.values() if i < len(v))
        title = sm.titles[i] if i < len(sm.titles) else ""
        low = (blob + " " + title).lower()
        if sm.accession == "GSE13355":
            if "normal skin from controls" in low or re.search(r"_nn", low):
                grp, subject = "control", None
            elif "uninvolved" in low or re.search(r"_pn", low):
                grp, subject = "uninvolved", _subject(title)
            elif "involved" in low or re.search(r"_pp", low):
                grp, subject = "lesional", _subject(title)
            else:
                grp, subject = "unknown", None
        elif sm.accession == "GSE30999":
            if "non-lesion" in low or low.endswith("_nl") or "_nl" in low:
                grp, subject = "uninvolved", _subject(title)
            elif "lesion" in low or "_ls" in low:
                grp, subject = "lesional", _subject(title)
            else:
                grp, subject = "unknown", None
        elif sm.accession == "GSE87466":
            grp = "lesional" if "ulcerative colitis" in low else (
                "control" if "normal" in low else "unknown")
            subject = None
        else:  # GSE75214
            if "control" in low or "normal" in low:
                grp = "control"
            elif "ulcerative colitis" in low or low.startswith("uc"):
                grp = "lesional"
            elif "crohn" in low or "_cd" in low:
                grp = "other_disease"
            else:
                grp = "unknown"
            subject = None
        rows.append({"gsm": gsm, "title": title, "group": grp, "subject": subject,
                     "characteristics": blob})
    return pd.DataFrame(rows)


def _subject(title: str) -> str | None:
    """Patient key from a sample title.

    GSE30999 titles look like ``1019-00001_NL`` / ``1019-00001_LS``: the patient
    is everything before the final underscore, not the leading site number.
    GSE13355 titles look like ``Individual_4288_PP_sample``: the patient is the
    numeric block.
    """
    t = (title or "").strip()
    if not t:
        return None
    if re.match(r"^[\w.-]+_(NL|LS|PP|PN|NN)$", t, flags=re.I):
        return t.rsplit("_", 1)[0]
    m = re.search(r"(\d{3,})", t)
    return m.group(1) if m else None


# --------------------------------------------------------------------------
# differential expression
# --------------------------------------------------------------------------
def moderated_t(expr: pd.DataFrame, group_a: list[str], group_b: list[str],
                pairs: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    """Empirical-Bayes moderated t, the limma variance-shrinkage idea in numpy.

    With ``pairs`` the test is a paired one-sample test on within-subject
    differences, which is the correct design for lesional versus uninvolved
    tissue from the same patient.
    """
    if pairs:
        a = expr[[p[0] for p in pairs]].to_numpy(dtype=float)
        b = expr[[p[1] for p in pairs]].to_numpy(dtype=float)
        d = a - b
        n = d.shape[1]
        if n < 3:
            return pd.DataFrame()
        mean = d.mean(axis=1)
        var = d.var(axis=1, ddof=1)
        df_resid = n - 1
        se_factor = 1.0 / n
    else:
        A = expr[group_a].to_numpy(dtype=float)
        B = expr[group_b].to_numpy(dtype=float)
        na, nb = A.shape[1], B.shape[1]
        if na < 2 or nb < 2:
            return pd.DataFrame()
        mean = A.mean(axis=1) - B.mean(axis=1)
        va = A.var(axis=1, ddof=1)
        vb = B.var(axis=1, ddof=1)
        df_resid = na + nb - 2
        var = ((na - 1) * va + (nb - 1) * vb) / df_resid
        se_factor = 1.0 / na + 1.0 / nb

    var = np.maximum(var, 1e-12)
    # variance shrinkage toward the trend (a simple prior on log variance)
    log_var = np.log(var)
    prior = np.median(log_var)
    mad = np.median(np.abs(log_var - prior)) * 1.4826
    d0 = 4.0                      # modest prior degrees of freedom
    s0_sq = float(np.exp(prior + 0.5 * mad ** 2)) if mad > 0 else float(np.exp(prior))
    var_post = (d0 * s0_sq + df_resid * var) / (d0 + df_resid)
    se = np.sqrt(var_post * se_factor)
    t = mean / np.maximum(se, 1e-12)
    from scipy import stats
    dfm = d0 + df_resid
    p = 2 * stats.t.sf(np.abs(t), dfm)
    out = pd.DataFrame({"log_fc": mean, "t": t, "p_value": p,
                        "mean_expression": expr[group_a + group_b].to_numpy(dtype=float).mean(axis=1)
                        if not pairs else
                        expr[[x for pr in pairs for x in pr]].to_numpy(dtype=float).mean(axis=1)},
                       index=expr.index)
    order = np.argsort(out["p_value"].to_numpy())
    ranked = np.empty(len(out), dtype=float)
    ranked[order] = np.arange(1, len(out) + 1)
    out["q_value"] = np.minimum(1.0, out["p_value"].to_numpy() * len(out) / ranked)
    return out


# --------------------------------------------------------------------------
# matched background comparison
# --------------------------------------------------------------------------
def matched_background(candidates: list[str], universe: pd.DataFrame,
                       n_per: int = 20, seed: int = 20260922) -> list[str]:
    """Background genes matched on mean expression, PPI degree and annotation."""
    rng = np.random.default_rng(seed)
    pool = universe[~universe.index.isin(candidates)]
    if pool.empty:
        return []
    feats = ["mean_expression", "ppi_degree", "ctd_degree"]
    P = np.log1p(pool[feats].to_numpy(dtype=float))
    out: list[str] = []
    used: set[int] = set()
    for g in candidates:
        if g not in universe.index:
            continue
        target = np.log1p(universe.loc[g, feats].to_numpy(dtype=float))
        dist = np.abs(P - target).sum(axis=1)
        order = np.argsort(dist)
        taken = 0
        for i in order:
            if i in used:
                continue
            used.add(i)
            out.append(pool.index[i])
            taken += 1
            if taken >= n_per:
                break
    return out


def enrichment_test(stat: pd.Series, selected: list[str], background: list[str]
                    ) -> dict:
    from scipy import stats as sps
    a = stat.reindex(selected).dropna().to_numpy(dtype=float)
    b = stat.reindex(background).dropna().to_numpy(dtype=float)
    if len(a) < 3 or len(b) < 10:
        return {"n_selected": len(a), "n_background": len(b), "enrichment": np.nan,
                "p_value": np.nan, "note": "too few measurable genes"}
    u, p = sps.mannwhitneyu(a, b, alternative="greater")
    auc = float(u / (len(a) * len(b)))
    return {"n_selected": int(len(a)), "n_background": int(len(b)),
            "median_selected": float(np.median(a)), "median_background": float(np.median(b)),
            "enrichment": auc, "p_value": float(p),
            "note": "rank-based; supports correlation only"}


def run_series(accession: str, top_genes: dict[str, list[str]],
               gene_meta: pd.DataFrame) -> dict:
    meta = SERIES[accession]
    matrix = paths.external("geo", f"{accession}_series_matrix.txt.gz")
    annot = paths.external("geo", f"{meta['platform']}.annot.gz")
    if not matrix.exists():
        return {"accession": accession, "status": "missing_matrix"}
    sm = parse_series_matrix(matrix)
    groups = classify_samples(sm)
    probe2gene = parse_platform_annotation(annot)
    if sm.expr.empty or not probe2gene:
        return {"accession": accession, "status": "unusable",
                "n_samples": len(sm.samples), "n_probes": int(sm.expr.shape[0]),
                "annotation_entries": len(probe2gene)}

    expr = sm.expr.copy()
    expr.index = [probe2gene.get(str(i), "") for i in expr.index]
    expr = expr[expr.index != ""]
    # one probe per gene: the highest mean expression, a rule fixed in advance
    expr["_m"] = expr.mean(axis=1)
    expr = expr.sort_values("_m", ascending=False)
    expr = expr[~expr.index.duplicated(keep="first")].drop(columns=["_m"])

    les = groups[groups["group"] == "lesional"]["gsm"].tolist()
    unv = groups[groups["group"] == "uninvolved"]["gsm"].tolist()
    ctl = groups[groups["group"] == "control"]["gsm"].tolist()
    les = [g for g in les if g in expr.columns]
    unv = [g for g in unv if g in expr.columns]
    ctl = [g for g in ctl if g in expr.columns]

    pairs = None
    if unv:
        sub = groups[groups["group"].isin(["lesional", "uninvolved"])]
        by_subj: dict[str, dict] = {}
        for r in sub.itertuples(index=False):
            if r.subject:
                by_subj.setdefault(r.subject, {})[r.group] = r.gsm
        pairs = [(v["lesional"], v["uninvolved"]) for v in by_subj.values()
                 if "lesional" in v and "uninvolved" in v
                 and v["lesional"] in expr.columns and v["uninvolved"] in expr.columns]

    contrasts = {}
    if pairs and len(pairs) >= 3:
        contrasts["lesional_vs_uninvolved_paired"] = moderated_t(expr, [], [], pairs=pairs)
    if les and ctl:
        contrasts["lesional_vs_control"] = moderated_t(expr, les, ctl)

    out = {"accession": accession, "status": "ok", "platform": sm.platform,
           "design": meta["design"], "disease_context": meta["disease_context"],
           "n_samples": len(sm.samples), "n_genes": int(expr.shape[0]),
           "group_counts": groups["group"].value_counts().to_dict(),
           "n_pairs": len(pairs) if pairs else 0,
           "contrasts": {}}

    universe = pd.DataFrame(index=expr.index)
    universe["mean_expression"] = expr.mean(axis=1)
    gm = gene_meta.set_index("gene_id")
    universe["ppi_degree"] = gm["ppi_degree"].reindex(universe.index).fillna(0.0)
    universe["ctd_degree"] = gm["ctd_degree"].reindex(universe.index).fillna(0.0)

    for cname, res in contrasts.items():
        if res is None or res.empty:
            continue
        stat = res["t"].abs()
        entry = {"n_tested_genes": int(len(res)),
                 "n_q_below_0.05": int((res["q_value"] < 0.05).sum())}
        for set_name, genes in top_genes.items():
            present = [g for g in genes if g in universe.index]
            bg = matched_background(present, universe)
            entry[set_name] = {**enrichment_test(stat, present, bg),
                               "n_requested": len(genes)}
        out["contrasts"][cname] = entry
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Expression support for frozen rankings.")
    ap.add_argument("--top_genes", default="")
    ap.add_argument("--series", nargs="*", default=list(SERIES))
    args = ap.parse_args(argv)

    genes = pd.read_parquet(paths.TABLES_DIR / "genes.parquet")
    ppi = pd.read_parquet(paths.TABLES_DIR / "ppi.parquet")
    rel = pd.read_parquet(paths.TABLES_DIR / "relations.parquet")
    deg = pd.concat([ppi["gene_a"], ppi["gene_b"]]).value_counts()
    cdeg = rel["gene_id"].value_counts()
    gene_meta = pd.DataFrame({"gene_id": genes["gene_id"].astype(str)})
    gene_meta["ppi_degree"] = gene_meta["gene_id"].map(deg).fillna(0.0)
    gene_meta["ctd_degree"] = gene_meta["gene_id"].map(cdeg).fillna(0.0)

    top_genes = json.loads(Path(args.top_genes).read_text(encoding="utf-8")) \
        if args.top_genes else {}
    results = [run_series(acc, top_genes, gene_meta) for acc in args.series]
    atomic_write_json(paths.RESULTS_DIR / "expression" / "expression_report.json",
                      {"series": results})
    for r in results:
        print(f"{r['accession']}: {r['status']} "
              f"{r.get('group_counts', '')} pairs={r.get('n_pairs', 0)} "
              f"genes={r.get('n_genes', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
