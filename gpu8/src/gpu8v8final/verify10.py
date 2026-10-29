"""Checks of the second-revision SARA package (extends verify9); exits non-zero when any check fails.

  1 forbidden (internal / process) wording in the Markdown, the Word files, figure text and public CSVs; no "我们";
  2 number tracing: every decimal, percentage and thousands-separated integer was produced by fmt9 (or is a constant);
  3 formatting: no signed zero, no "nan", no ASCII hyphen as minus sign, no nested full-width brackets;
  4 evidence-deletion counts equal the saved summary;
  5 GB/T 7714—2015: entry types and fields, et al. rule, titles against Crossref, numbering by first citation,
    superscript citations only, no duplicate DOI;
  6 cross references: figures/panels/tables/notes exist, main figures and tables first cited in order, every
    "式（n）" points to an existing equation;
  7 Word files: MathType objects only (no OMML), one valid Equation.DSMT4 storage per equation of the Markdown,
    right-aligned SEQ MTEqn numbers, REF fields and citation links resolve (docx_build_report10);
  8 colour-palette checks of the figures;  9 legend panels match the figure panels;
 10 abbreviations defined at first use in the main text;  11 headings and figure/table titles are neutral noun
    phrases (no evaluative words, headings at most 24 characters).
"""
from __future__ import annotations

import json
import re
import sys

from . import verify9 as V9
from .build_docx10 import REVIEW_STEM
from .manuscript10 import STEM
from .mt_factory10 import DISPLAY_RX, INLINE_RX
from .numbers9 import FIG, ROOT, e4_summary, load_R9
from .publabel9 import FORBIDDEN_RX
from .supp10 import SUPP_STEM

ABBR = ["PPI", "HPO", "GO", "RRF", "OOF", "HGNC", "ncRNA", "CTD", "MeSH", "OMIM", "HPA", "IC", "TF-IDF", "RWR", "CI",
        "SHAP", "GEO", "PMID", "IEA", "IBA"]
EVALUATIVE = re.compile(r"显著|优于|证明|表明|提升|大幅|更好|领先|突破|卓越")
SEC_REF = re.compile(r"\d+\.\d+ ?节")


def numbers(md: str, trace: dict) -> list[str]:
    return V9.check_numbers(SEC_REF.sub(" 节", md), trace)


def citations(md: str, n_refs: int) -> list[str]:
    bad = []
    if "[@" in md:
        bad.append("unresolved [@key] citation")
    body = re.split(r"\n## (?:补充)?参考文献\n", md)[0] + ("\n" + md.split("## 表\n", 1)[1] if "## 表\n" in md else "")
    seen = []
    for m in re.finditer(r"<sup>\[([\d,\-]+)\]</sup>", body):
        for part in m.group(1).split(","):
            a, b = (part.split("-") + [part])[:2]
            for n in range(int(a), int(b) + 1):
                if n not in seen:
                    seen.append(n)
    if seen != list(range(1, n_refs + 1)):
        bad.append(f"references not numbered by first citation: {seen[:20]}")
    if re.search(r"(?<!<sup>)\[\d+(?:[,\-]\d+)*\](?!</sup>)", re.sub(r"^\[\d+\] .*$", "", body, flags=re.M)):
        bad.append("citation marker outside superscript")
    entries = re.findall(r"^\[(\d+)\] (.+)$", md, flags=re.M)
    if [int(n) for n, _ in entries] != list(range(1, n_refs + 1)):
        bad.append("reference list not numbered 1..n")
    for n, e in entries:
        if not re.search(r"\[(J|C|M|R|EB/OL|C/OL)\]", e):
            bad.append(f"[{n}] no document-type code")
        if re.search(r"\[J\]", e) and not re.search(r"\[J\]\. .+, \d{4}, \d+(\([^)]+\))?: [\w\-]+\.", e):
            bad.append(f"[{n}] journal entry not 'journal, year, volume(issue): pages'")
    dois = re.findall(r"DOI:(\S+)\.$", md, flags=re.M)
    if len(dois) != len(set(dois)):
        bad.append("duplicate DOI")
    return bad


def gbt_records(info: dict) -> list[str]:
    from .refs10 import CACHE, check, record
    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    recs = [record(k, cache) for k, _ in sorted(info["ref_order"].items(), key=lambda t: t[1])]
    return check(recs)


def equations(md: str, prefix: str) -> tuple[list[str], dict]:
    n_disp = sum(1 for line in md.split("\n") if DISPLAY_RX.match(line.strip()))
    n_inline = sum(len(INLINE_RX.findall(line)) for line in md.split("\n") if not DISPLAY_RX.match(line.strip()))
    bad = [f"式（{p}{n}） has no equation" for p, n in re.findall(r"式（(S?)(\d+)）", md)
           if p != prefix or not 1 <= int(n) <= n_disp]
    if "@eq:" in md or "{#eq:" in md:
        bad.append("unresolved equation label")
    return bad, {"display": n_disp, "inline": n_inline}


def abbreviations(md: str) -> list[str]:
    text = md.split("## 1 引言", 1)[1].split("## 参考文献", 1)[0]
    text = re.sub(r"\$[^$]+\$", " ", re.sub(r"\$\$.*?\$\$", " ", text, flags=re.S))
    bad = []
    for a in ABBR:
        m = re.search(rf"(?<![A-Za-z]){re.escape(a)}(?![A-Za-z])", text)
        if not m:
            bad.append(f"{a}: not used in the main text")
            continue
        after = text[m.end():m.end() + 1]
        before = text[max(0, m.start() - 120):m.start()]
        if not (after == "）" and "（" in before and "）" not in before[before.rfind("（"):]):
            bad.append(f"{a}: first use not defined :: {text[max(0, m.start() - 40):m.end() + 10]}")
    return bad


def titles(md: str) -> list[str]:
    bad = []
    for line in md.split("\n"):
        s = line.strip()
        m = re.match(r"^#{2,4} (?:\d+(?:\.\d+)* |补充说明 S\d+　)?(.+)$", s)
        if m:
            t = m.group(1)
            if EVALUATIVE.search(t):
                bad.append(f"evaluative heading: {t}")
            if len(t) > 24:
                bad.append(f"heading longer than 24 characters: {t}")
        m = re.match(r"^(?:\*\*)?(图|表) S?\d+　([^。*（]+)", s)
        if m and EVALUATIVE.search(m.group(2)):
            bad.append(f"evaluative {m.group(1)} title: {m.group(2)}")
    return bad


def run() -> dict:
    R = load_R9()
    main_md = (ROOT / f"{STEM}.md").read_text(encoding="utf-8")
    supp_md = (ROOT / f"{SUPP_STEM}.md").read_text(encoding="utf-8")
    info_m = json.loads((ROOT / "build" / "manuscript10_info.json").read_text(encoding="utf-8"))
    info_s = json.loads((ROOT / "build" / "supp10_info.json").read_text(encoding="utf-8"))
    rep: dict = {}
    rep["1_forbidden_main"] = V9.check_forbidden(main_md)
    rep["1_forbidden_supp"] = V9.check_forbidden(supp_md, V9.SUPP_ALLOW)
    for key, fn in (("main", f"{STEM}.docx"), ("review", f"{REVIEW_STEM}.docx"), ("supp", f"{SUPP_STEM}.docx")):
        p = ROOT / fn
        rep[f"1_forbidden_docx_{key}"] = (V9.check_forbidden(V9.docx_text(p), V9.SUPP_ALLOW if key == "supp" else frozenset())
                                          if p.exists() else ["missing"])
    rep["1_first_person"] = [f"{w} in {k}" for k, md in (("main", main_md), ("supp", supp_md)) for w in ("我们",) if w in md]
    scan = json.loads((FIG / "text_scan.json").read_text(encoding="utf-8"))
    rep["1_forbidden_figures"] = [f"{k}: {v.get('forbidden')}" for k, v in scan.items() if isinstance(v, dict) and v.get("forbidden")]
    csv_bad = []
    for p in sorted((FIG / "source_data").glob("*.csv")):
        hits = [m.group(0) for m in FORBIDDEN_RX.finditer(p.read_text(encoding="utf-8-sig"))]
        if hits:
            csv_bad.append(f"{p.name}: {sorted(set(hits))[:5]}")
    rep["1_forbidden_public_csv"] = csv_bad
    rep["2_untraced_main"] = numbers(main_md, info_m["trace"])
    rep["2_untraced_supp"] = numbers(supp_md, info_s["trace"])
    rep["3_format_main"] = [x for x in V9.check_format(main_md) if not x.startswith("no ")]
    rep["3_format_supp"] = [x for x in V9.check_format(supp_md) if not x.startswith("no ")]
    e4s = e4_summary(R)
    want = f"在 {e4s['n_retrain']} 种整链重训的删除条件中，{e4s['n_inc0']} 种条件下"
    rep["4_e4_counts"] = [] if want in main_md else [f"missing: {want}"]
    rep["5_citations_main"] = citations(main_md, info_m["n_refs"])
    rep["5_citations_supp"] = citations(supp_md, info_s["n_refs"])
    rep["5_gbt_main"] = gbt_records(info_m)
    rep["5_gbt_supp"] = gbt_records(info_s)
    bad_m, first_m = V9.xrefs(main_md, supp=False)
    bad_s, _ = V9.xrefs(supp_md, supp=True)
    eq_m, cnt_m = equations(main_md, "")
    eq_s, cnt_s = equations(supp_md, "S")
    rep["6_xrefs_main"], rep["6_xrefs_supp"] = bad_m + eq_m, bad_s + eq_s
    br = ROOT / "build" / "docx_build_report10.json"
    docx_bad = []
    if br.exists():
        b = json.loads(br.read_text(encoding="utf-8"))
        if not b.get("all_ok"):
            docx_bad.append("docx build report not ok")
        for key, cnt in (("submission", cnt_m), ("review", cnt_m), ("supplement", cnt_s)):
            d = b.get(key, {})
            if d.get("omml_objects") != 0:
                docx_bad.append(f"{key}: OMML objects present")
            if (d.get("display_equations"), d.get("inline_equations")) != (cnt["display"], cnt["inline"]):
                docx_bad.append(f"{key}: equations {d.get('display_equations')}/{d.get('inline_equations')} != Markdown "
                                f"{cnt['display']}/{cnt['inline']}")
            if d.get("ole_objects") != cnt["display"] + cnt["inline"]:
                docx_bad.append(f"{key}: {d.get('ole_objects')} MathType objects for {cnt['display'] + cnt['inline']} equations")
    else:
        docx_bad.append("docx build report missing")
    rep["7_docx"] = docx_bad
    man = json.loads((FIG / "figures_manifest.json").read_text(encoding="utf-8"))
    pal = man.get("_palette_checks", {})
    rep["8_palette"] = [k for k, v in pal.items() if isinstance(v, dict) and not v.get("pass", True)] if pal else ["no palette checks"]
    rep["9_legend_panels"] = V9.legend_panels(main_md) + V9.legend_panels(supp_md)
    rep["10_abbreviations"] = abbreviations(main_md)
    rep["11_titles"] = titles(main_md) + titles(supp_md)
    abstract = re.search(r"## 摘要\n\n(.+?)\n\n", main_md, flags=re.S).group(1)
    rep["_info"] = {"abstract_chars": len(re.sub(r"\s", "", abstract)), "main_chars": len(main_md), "supp_chars": len(supp_md),
                    "refs_main": info_m["n_refs"], "refs_supp": info_s["n_refs"], "equations_main": cnt_m,
                    "equations_supp": cnt_s, "key_points": main_md.split("## 要点", 1)[1].split("## 1 引言")[0].count("\n- "),
                    "spacing_notes": [x for x in V9.check_format(main_md) + V9.check_format(supp_md) if x.startswith("no ")][:20],
                    "first_mentions": {k: v for k, v in sorted(first_m.items(), key=lambda t: t[1])}}
    fails = {k: v for k, v in rep.items() if not k.startswith("_") and v}
    rep["ok"] = not fails
    (ROOT / "build" / "verify10_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    return rep


if __name__ == "__main__":
    r = run()
    show = {k: (v[:12] if isinstance(v, list) else v) for k, v in r.items() if k != "_info"}
    print(json.dumps(show, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in r["_info"].items() if k not in ("first_mentions", "spacing_notes")}, ensure_ascii=False))
    sys.exit(0 if r["ok"] else 1)
