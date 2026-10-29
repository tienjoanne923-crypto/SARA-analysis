"""Check omitted research inputs against recorded content identities, without changing them."""
from pathlib import Path
import argparse, hashlib, json

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    base=Path(__file__).resolve().parents[1]
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',required=True,type=Path,help='Research workspace containing gpu8 and upstream resource folders')
    ap.add_argument('--manifest',type=Path,default=base/'provenance/INPUTS_SHA256.json')
    ap.add_argument('--report',type=Path)
    a=ap.parse_args();root=a.root.resolve();doc=json.loads(a.manifest.read_text(encoding='utf-8'));rows=[]
    for r in doc['files']:
        p=(root/r['path']).resolve();status='matched'
        if not p.is_relative_to(root):status='unsafe_path'
        elif not p.is_file():status='missing'
        elif p.stat().st_size!=r['bytes']:status='size_mismatch'
        elif sha(p)!=r['sha256']:status='hash_mismatch'
        rows.append({'path':r['path'],'status':status})
    from collections import Counter
    counts=dict(Counter(x['status'] for x in rows))
    report={'selected_files':len(rows),'counts':counts,'all_selected_matched':counts.get('matched',0)==len(rows),'files':rows,
            'coverage_note':doc.get('coverage_note','Only selected existing files are checked. Known missing originals are separately documented.')}
    if a.report:
        rp=a.report.resolve()
        if rp.is_relative_to(root) and rp in [(root/r['path']).resolve() for r in doc['files']]:raise SystemExit('Report must not overwrite an input.')
        rp.parent.mkdir(parents=True,exist_ok=True);rp.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))
    return int(not report['all_selected_matched'])

if __name__=='__main__':raise SystemExit(main())
