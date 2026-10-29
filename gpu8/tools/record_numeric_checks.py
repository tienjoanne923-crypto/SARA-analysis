"""Run the test suite and record its outcome for acceptance item T09."""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gpu8 import paths
from gpu8.hashing import atomic_write_json

env = dict(os.environ)
env["PYTHONPATH"] = str(paths.GPU8_ROOT / "src")
env["PYTHONIOENCODING"] = "utf-8"
log = paths.LOG_DIR / "pytest.log"
log.parent.mkdir(parents=True, exist_ok=True)

with open(log, "w", encoding="utf-8") as fh:
    p = subprocess.run([sys.executable, "-m", "pytest", str(paths.GPU8_ROOT / "tests"),
                        "-v", "--no-header", "-rA"],
                       stdout=fh, stderr=subprocess.STDOUT, env=env,
                       cwd=str(paths.GPU8_ROOT))

text = log.read_text(encoding="utf-8", errors="replace")
passed = [l.split("::")[-1].split()[0] for l in text.splitlines()
          if "PASSED" in l and "::" in l]
failed = [l.split("::")[-1].split()[0] for l in text.splitlines()
          if "FAILED" in l and "::" in l]

payload = {
    "returncode": p.returncode,
    "all_passed": p.returncode == 0,
    "n_checks": len(passed) + len(failed),
    "passed": passed,
    "failed": failed,
    "scope": ("These verify local implementation properties only: solver "
              "correctness and invariance, gradient flow through the Cholesky "
              "solve, duplicate-record invariance, anchor reproducibility, and "
              "split/context isolation on the real built artefacts. They do NOT "
              "check historical resource versions, disease mapping correctness, "
              "or shared provenance between databases, so passing them is not a "
              "claim that the study is free of leakage."),
    "log": str(log),
}
atomic_write_json(paths.REPORT_DIR / "numeric_checks.json", payload)
print(json.dumps({"all_passed": payload["all_passed"],
                  "n_checks": payload["n_checks"],
                  "failed": payload["failed"]}, indent=2))
