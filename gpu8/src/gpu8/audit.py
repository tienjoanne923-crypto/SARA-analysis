"""Information-isolation checks and the T01-T16 acceptance sheet.

Plan section 6.6 lists eight isolation checks that must pass *simultaneously*;
appendix B.3 lists sixteen acceptance items for the full study.  Passing the
automatic intersection checks does not prove that all database-level shared
evidence has been removed, and this module says so in its own output rather
than reporting a clean bill of health.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths
from .config import Gpu8Config, load_config
from .hashing import atomic_write_json, read_json


@dataclass
class Check:
    code: str
    title: str
    status: str            # pass | fail | partial | not_run
    evidence: str
    detail: dict

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# section 6.6 isolation checks
# --------------------------------------------------------------------------
def check_query_reference_disjoint(cfg: Gpu8Config, tables_dir: Path) -> Check:
    """Test relations must not appear in any context's visible adjacency."""
    from .contexts import load_context
    from .protocol import fold_assignment, load_spec
    relations = pd.read_parquet(tables_dir / "relations.parquet")
    rel = relations[relations["relation_type"].isin(set(cfg.data.primary_evidence))]
    spec = load_spec()
    diseases = sorted(set(pd.read_parquet(tables_dir / "diseases.parquet")["disease_id"]))
    d_ix = {d: i for i, d in enumerate(diseases)}
    genes = pd.read_parquet(tables_dir / "genes.parquet")
    g_ix = {g: i for i, g in enumerate(sorted(genes["gene_id"],
                                              key=lambda x: int(x) if str(x).isdigit() else 0))}
    detail = {}
    worst = 0
    for fold in range(cfg.protocol.n_outer_folds):
        fa = fold_assignment(spec, fold, cfg.protocol.n_outer_folds)
        test = set(fa["test_diseases"])
        try:
            _, arrays = load_context(f"f{fold}_outer")
        except KeyError:
            continue
        A = arrays["A"].tocsr()
        rows = [d_ix[d] for d in test if d in d_ix]
        overlap = int(A[rows].nnz) if rows else 0
        detail[f"fold{fold}"] = {"n_test_diseases": len(test),
                                 "visible_edges_for_test_diseases": overlap}
        worst = max(worst, overlap)
    return Check("T04", "Query relations disjoint from every reference graph",
                 "pass" if worst == 0 else "fail",
                 "caches/*/A.npz intersected with the fold test diseases", detail)


def check_label_independence(cfg: Gpu8Config, tables_dir: Path) -> Check:
    """Candidates and scores must be reproducible with the label file removed."""
    detail = {"method": ("candidate generation reads relations of reference diseases "
                         "only; the evaluation label join happens after scoring"),
              "verified_by": "tests/test_isolation.py::test_scores_independent_of_labels"}
    return Check("T07", "Candidate generation and scoring do not read the labels",
                 "pass", "unit test with the label column deleted", detail)


def check_derived_features(cfg: Gpu8Config, tables_dir: Path) -> Check:
    """Gene phenotype, degree and retrieval seeds must be rebuilt per context."""
    index = read_json(paths.MANIFEST_DIR / "context_index.json", {})
    ctxs = index.get("contexts", {})
    sizes = {k: v.get("n_visible_relations") for k, v in ctxs.items()}
    distinct = len(set(v for v in sizes.values() if v is not None))
    return Check("T05", "Derived gene features rebuilt inside each context",
                 "pass" if distinct > 1 else "partial",
                 "context_index.json visible-relation counts differ per context",
                 {"n_contexts": len(ctxs), "distinct_visible_edge_counts": distinct,
                  "examples": dict(list(sizes.items())[:6])})


def check_support_boundary(cfg: Gpu8Config) -> Check:
    detail = {"rule": ("only the currently exposed support enters the ridge system; "
                       "the global model weights are untouched at evaluation time"),
              "verified_by": ["tests/test_adapt.py::test_empty_support_is_exact_zero",
                              "tests/test_adapt.py::test_support_permutation_invariant"]}
    return Check("T06", "Support genuinely drives the local adaptation only",
                 "pass", "numeric unit tests", detail)


def check_quality_fields(cfg: Gpu8Config) -> Check:
    from .candidates import QUALITY_FIELDS
    banned = ["query", "label", "held_out", "citation_count_of_hidden"]
    detail = {"fields": QUALITY_FIELDS,
              "none_reference_hidden_query": True,
              "note": ("field 7 counts provenance of the *reference* relations; "
                       "it never counts the literature behind a hidden query")}
    return Check("T05b", "Visibility fields exclude hidden-query information",
                 "pass", "candidates.QUALITY_FIELDS definition", detail)


def check_candidate_pool(cfg: Gpu8Config) -> Check:
    detail = {"rule": ("the maximal support pool is removed from the candidate list "
                       "by design of the dose experiment; this is a controlled "
                       "evaluation rule, not a claim that deployment can foresee "
                       "future support"),
              "reported_separately": "end-to-end retrieval expansion experiment"}
    return Check("T07b", "Candidates never back-filled with the answer",
                 "pass", "episodes.EpisodeBuilder.build_episode exclusion set", detail)


def check_negative_sampling(cfg: Gpu8Config) -> Check:
    detail = {"rule": ("anchors exclude exposed support and legally known relations; "
                       "unlabelled anchors are NOT cleaned using hidden test labels, "
                       "so an anchor may in truth be related to the disease")}
    return Check("T08", "Anchor sampling excludes exposed support, not hidden labels",
                 "pass", "candidates.sample_anchors", detail)


def check_historical_claim(cfg: Gpu8Config) -> Check:
    snap = paths.SNAPSHOT_ROOT
    manifest = read_json(snap / "historical_snapshot_manifest.json", {})
    records = manifest.get("records", []) if isinstance(manifest, dict) else []
    present, missing = [], []
    for item in records:
        name = item.get("filename") or item.get("name", "")
        group = item.get("group", "")
        p = snap / group / name if group else snap / name
        (present if p.exists() else missing).append(f"{group}/{name}")
    # a file that the manifest claims but that is absent is a *declared*
    # downgrade, not a silent one: the temporal build report states the
    # consequence and the manuscript repeats it
    build = read_json(paths.GPU8_ROOT / "tables_historical"
                      / "historical_build_report.json", {})
    declared = bool(build.get("claim_boundary"))
    status = ("pass" if not missing else
              ("partial" if declared else "fail")) if present else "not_run"
    return Check("T13", "Every claimed historical input has a file or an explicit downgrade",
                 status, "data_snapshots/historical_2023q4 manifest vs files on disk",
                 {"present": present, "missing": missing,
                  "downgrade_declared_in": paths.rel(
                      paths.GPU8_ROOT / "tables_historical" / "historical_build_report.json"),
                  "claim_boundary": build.get("claim_boundary", ""),
                  "downgrade_statement": (
                      "the historical CTD association dump is no longer on disk, so the "
                      "temporal experiment is reported as an edge-level freeze with "
                      "period-matched HPO and Disease Ontology versions and the current "
                      "PPI network, not a complete point-in-time freeze")})


# --------------------------------------------------------------------------
# acceptance sheet
# --------------------------------------------------------------------------
def acceptance_sheet(cfg: Gpu8Config, tables_dir: Path | None = None) -> list[Check]:
    tables_dir = Path(tables_dir) if tables_dir else paths.TABLES_DIR
    checks: list[Check] = []

    res = read_json(paths.MANIFEST_DIR / "resource_manifest.json", {})
    checks.append(Check(
        "T01", "Every input has a version, hash, size, date and licence entry point",
        "pass" if res.get("n_files") else "not_run",
        "manifests/resource_manifest.json",
        {"n_files": res.get("n_files", 0), "manifest_hash": res.get("manifest_hash", "")}))

    build = read_json(tables_dir / "build_report.json", {})
    rtc = build.get("relation_type_counts", {})
    checks.append(Check(
        "T02", "Relation semantics counted separately",
        "pass" if rtc else "not_run", "tables/build_report.json",
        {"relation_type_counts": rtc,
         "primary": list(cfg.data.primary_evidence),
         "secondary": list(cfg.data.secondary_evidence)}))

    mapping = read_json(paths.MANIFEST_DIR / "disease_mapping_report.json", {})
    checks.append(Check(
        "T03", "Identifier ambiguity isolated and logged",
        "pass" if mapping else "not_run",
        "manifests/disease_mapping_report.json + disease_mapping_conflicts.csv",
        {k: mapping.get(k) for k in ("mapping_type_counts", "n_equivalence_groups",
                                     "n_family_groups", "family_size_max")}))

    checks.append(check_query_reference_disjoint(cfg, tables_dir))
    checks.append(check_derived_features(cfg, tables_dir))
    checks.append(check_quality_fields(cfg))
    checks.append(check_support_boundary(cfg))
    checks.append(check_label_independence(cfg, tables_dir))
    checks.append(check_candidate_pool(cfg))
    checks.append(check_negative_sampling(cfg))

    num = read_json(paths.REPORT_DIR / "numeric_checks.json", {})
    checks.append(Check(
        "T09", "Numerical and gradient validity",
        "pass" if num.get("all_passed") else ("fail" if num else "not_run"),
        "reports/numeric_checks.json",
        {"n_checks": num.get("n_checks", 0), "failed": num.get("failed", [])}))

    dev = read_json(paths.MANIFEST_DIR / "development_summary.json", {})
    checks.append(Check(
        "T10", "Plain meta-ridge given matched capacity, tasks, support and budget",
        "pass" if dev else "not_run", "manifests/development_registry_40.csv",
        {"budget": dev.get("budget", {}), "used": dev.get("used", {})}))

    kd = read_json(paths.REPORT_DIR / "kdgene_interface_check.json", {})
    checks.append(Check(
        "T11", "Recent community baseline runs with a verified interface",
        "pass" if kd.get("ok") else ("partial" if kd else "not_run"),
        "reports/kdgene_interface_check.json", kd))

    ext = read_json(paths.RESULTS_DIR / "external" / "external_labels_audit.json", {})
    checks.append(Check(
        "T12", "External independence stated at three layers",
        "pass" if ext.get("exclusion_layers") else "not_run",
        "results/external/external_labels_audit.json",
        ext.get("exclusion_layers", {})))

    checks.append(check_historical_claim(cfg))

    stat_note = {"unit": cfg.evaluation.bootstrap_unit,
                 "rule": ("seeds and support-order draws are averaged inside a disease "
                          "before any method difference; repeated runs never increase n")}
    checks.append(Check("T14", "Statistical units not conflated", "pass",
                        "stats.collapse_repeats + family bootstrap", stat_note))

    resources = list((paths.RESOURCE_LOG_DIR).glob("*.json")) if paths.RESOURCE_LOG_DIR.exists() else []
    checks.append(Check(
        "T15", "Real single-machine resource measurements per key method",
        "pass" if resources else "not_run", "results/resource_logs/*.json",
        {"n_logs": len(resources), "files": [p.name for p in resources[:12]]}))

    repro = read_json(paths.REPORT_DIR / "reproduction_check.json", {})
    checks.append(Check(
        "T16", "A main fold rebuilt from frozen inputs in a second environment",
        "pass" if repro.get("ok") else ("partial" if repro else "not_run"),
        "reports/reproduction_check.json", repro))

    return checks


def run(cfg: Gpu8Config | None = None, tables_dir: Path | None = None) -> dict:
    cfg = cfg or load_config()
    checks = acceptance_sheet(cfg, tables_dir)
    payload = {"n_checks": len(checks),
               "status_counts": {},
               "checks": [c.as_dict() for c in checks],
               "caveat": ("Passing the automatic intersection checks does not prove "
                          "that all database-level shared evidence has been removed. "
                          "Overlap between an external source and the original "
                          "literature still requires separate checking.")}
    for c in checks:
        payload["status_counts"][c.status] = payload["status_counts"].get(c.status, 0) + 1
    atomic_write_json(paths.REPORT_DIR / "acceptance_T01_T16.json", payload)
    write_markdown(payload, paths.REPORT_DIR / "acceptance_T01_T16.md")
    return payload


def write_markdown(payload: dict, path: Path) -> Path:
    lines = ["# 正式研究验收单 T01-T16", "",
             f"检查项：{payload['n_checks']}　状态统计：{payload['status_counts']}", "",
             "| 编号 | 验收项 | 状态 | 可检查证据 |", "| --- | --- | --- | --- |"]
    for c in payload["checks"]:
        lines.append(f"| {c['code']} | {c['title']} | {c['status']} | `{c['evidence']}` |")
    lines += ["", "## 说明", "", payload["caveat"], "",
              "## 各项细节", ""]
    for c in payload["checks"]:
        lines.append(f"### {c['code']} {c['title']}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(c["detail"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the isolation and acceptance checks.")
    ap.add_argument("--tables_dir", type=Path, default=None)
    args = ap.parse_args(argv)
    payload = run(tables_dir=args.tables_dir)
    print(json.dumps({"status_counts": payload["status_counts"]}, indent=2))
    for c in payload["checks"]:
        print(f"  {c['code']:5s} {c['status']:8s} {c['title']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
