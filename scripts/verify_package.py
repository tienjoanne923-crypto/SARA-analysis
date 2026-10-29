"""Verify every packaged file against the release's SHA-256 manifest (stdlib only)."""
from pathlib import Path
import argparse, hashlib, json

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--allow-extra',action='store_true',help='Allow additional restored data in a working copy; still report all extra paths')
    a=ap.parse_args();root=a.root.resolve()
    mf=root/'PACKAGE_MANIFEST.json'
    if not mf.is_file():raise SystemExit('Missing PACKAGE_MANIFEST.json; candidate has not been finalized.')
    doc=json.loads(mf.read_text(encoding='utf-8'));bad=[]
    sums=root/'MANIFEST.sha256'
    if not sums.is_file():raise SystemExit('Missing MANIFEST.sha256.')
    recorded={}
    for line in sums.read_text(encoding='utf-8').splitlines():
        value,name=line.split('  ',1)
        if name in recorded:raise SystemExit('Duplicate manifest path: '+name)
        recorded[name]=value
    expected={r['path']:r['sha256'] for r in doc['files']}
    if len(expected)!=len(doc['files']):bad.append({'path':'PACKAGE_MANIFEST.json','status':'duplicate file path'})
    expected['PACKAGE_MANIFEST.json']=sha(mf)
    if recorded!=expected:bad.append({'path':'MANIFEST.sha256','status':'manifest disagreement'})
    actual=set()
    for p in root.rglob('*'):
        rel=p.relative_to(root)
        if any(x in ('.git','__pycache__','.pytest_cache') for x in rel.parts):continue
        if p.is_file() and rel.as_posix()!='.sara-disposable-workspace.json':actual.add(rel.as_posix())
    extras=sorted(actual-(set(expected)|{'MANIFEST.sha256'}))
    if extras and not a.allow_extra:bad.append({'path':'<package>','status':'unlisted extra files','files':extras})
    for r in doc['files']:
        p=(root/r['path']).resolve()
        if not p.is_relative_to(root):bad.append({'path':r['path'],'status':'unsafe path'});continue
        if not p.is_file():bad.append({'path':r['path'],'status':'missing'});continue
        if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:
            bad.append({'path':r['path'],'status':'changed'})
    if doc.get('file_count')!=len(doc['files']):bad.append({'path':'PACKAGE_MANIFEST.json','status':'wrong file count'})
    print(json.dumps({'checked_files':len(doc['files']),'extra_files':extras,'failed':bad,'passed':not bad},ensure_ascii=False,indent=2))
    return int(bool(bad))

if __name__=='__main__':raise SystemExit(main())
