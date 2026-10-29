"""Record the as-built hash of every canonical table, after the mapping step.

``ingest.py`` writes each table and records its hash, then ``mapping.py``
rewrites ``diseases.parquet`` to add the Mondo identity, equivalence group and
family group.  The ingest-time hash of that one table is therefore stale by
design.  This tool records the final state so the provenance check compares
against what the pipeline actually produced, and names the reason.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from gpu8 import paths
from gpu8.hashing import atomic_write_json, sha256_file

TABLES = ["genes", "diseases", "relations", "evidence", "phenotypes",
          "hierarchy", "ppi", "hpo_terms", "do_terms"]

rows = {}
for name in TABLES:
    p = paths.table(name)
    if not p.exists():
        rows[name] = {"status": "missing"}
        continue
    df = pd.read_parquet(p)
    rows[name] = {"status": "present", "rows": int(len(df)),
                  "columns": list(df.columns), "bytes": p.stat().st_size,
                  "sha256": sha256_file(p), "path": paths.rel(p)}

payload = {
    "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "stage": "after ingest and mapping",
    "tables": rows,
    "note": ("diseases.parquet is written twice by design: ingest builds it from "
             "the CTD vocabulary, then mapping adds mondo_ids, mapping_type, "
             "equivalence_group, family_group and family_source. The hash below "
             "is the final one and is the reference for the provenance check."),
}
atomic_write_json(paths.MANIFEST_DIR / "table_hashes.json", payload)
print(json.dumps({k: {"rows": v.get("rows"), "sha256": str(v.get("sha256"))[:12]}
                  for k, v in rows.items()}, indent=2))
