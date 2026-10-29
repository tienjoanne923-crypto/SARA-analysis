"""V8.1 final layer of the GPU8 pipeline.

Implements ``DiseaseGene_V8_Final_ResearchPlan.pdf``: evidence-availability-aware
semantic retrieval (the V7.1 disease-function semantic teacher), a missing-aware
strict 500-candidate pool, an availability-gated cold-balanced pairwise corrector
on top of an out-of-fold channel base ranker, and a sequence student that only
imputes function semantics for genes without legal GO.

Separate from ``gpu8`` (V5), ``gpu8v6`` and ``gpu8v71``: nothing here writes into
their result, run, manifest, cache or table directories.  Every V8.1 artefact
lives under ``results_v8/``, ``runs_v8/``, ``manifests_v8/``, ``caches_v8/``,
``tables_v8/``, ``reports_v8/`` and ``manuscript_v8/``.
"""

PROTOCOL_ID = "gpu8_v8final"
VERSION = "8.1"
