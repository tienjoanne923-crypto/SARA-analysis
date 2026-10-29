"""V6 remodel layer of the GPU8 pipeline.

V6 follows ``DiseaseGene_V6_Remodel_RepairPlan.pdf``: the 64-dim meta-ridge path
is withdrawn as the main method, zero-support retrieval and support-seeded graph
propagation are repaired as two parallel lines, and every mechanism is checked
with an explicit, falsifiable test.

This package is deliberately separate from ``gpu8``:

* the V5 code hash (part of every V5 cache key and run record) is unchanged, so
  the V5 five-fold results stay a reproducible archive;
* every V6 artefact is written under ``results_v6/``, ``runs_v6/``,
  ``manifests_v6/``, ``caches_v6/``, ``tables_v6/`` and ``reports_v6/`` with its
  own protocol identifier, candidate hashes and version fields.
"""

PROTOCOL_ID = "gpu8_v6"
VERSION = "6.0"
