"""Create a new disposable analysis workspace without overwriting existing files."""
from pathlib import Path
import argparse,hashlib,json,shutil

if __name__=='__main__':
    src=Path(__file__).resolve().parents[1]
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--destination',required=True,type=Path);a=ap.parse_args()
    dst=a.destination.resolve()
    if dst.exists():raise SystemExit('Destination exists; choose a new directory.')
    if dst.is_relative_to(src):raise SystemExit('Working copy must be outside the archive directory.')
    if not dst.parent.is_dir():raise SystemExit('Destination parent directory must exist.')
    shutil.copytree(src,dst,ignore=shutil.ignore_patterns('.git','__pycache__','.pytest_cache'))
    mf=src/'PACKAGE_MANIFEST.json'
    record={'purpose':'disposable SARA analysis workspace','source_manifest_sha256':hashlib.sha256(mf.read_bytes()).hexdigest() if mf.exists() else None,'third_party_inputs_supplied':False}
    (dst/'.sara-disposable-workspace.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps({'workspace':str(dst),'status':'copied; restore omitted inputs and verify before execution'}))
