"""V7.1 multimodal layer of the GPU8 pipeline.

V7.1 follows ``DiseaseGene_V71_Multimodal_ExecutablePlan.pdf``: a typed
multimodal retrieval core (frozen SapBERT disease text / HPO / GO semantics,
frozen ESM-2 35M protein sequence, HPA normal-tissue expression; optional HPA
image features and IMPC mouse phenotypes) that first retrieves on its own over
the full gene background and is then fused late with V6 through out-of-fold
(OOF) chains.

The package is separate from ``gpu8`` and ``gpu8v6``: the V5 and V6 archives are
never written to, and every V7.1 artefact lives under ``results_v71/``,
``runs_v71/``, ``manifests_v71/``, ``caches_v71/``, ``tables_v71/``,
``reports_v71/`` and ``external/v71/`` with its own protocol identifier.
"""

PROTOCOL_ID = "gpu8_v71"
VERSION = "7.1"
