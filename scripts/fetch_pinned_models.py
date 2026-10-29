"""Plan/download the exact archived model revisions and check individual bytes."""
from pathlib import Path
import argparse, hashlib, json, re, urllib.request, urllib.parse

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    root=Path(__file__).resolve().parents[1]
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--download',action='store_true',help='Actually download; default only prints fixed-revision URLs')
    ap.add_argument('--check-existing',action='store_true',help='Verify existing model files in destination, offline')
    ap.add_argument('--destination',type=Path,help='Root of disposable research workspace')
    a=ap.parse_args()
    if a.download and a.check_existing:raise SystemExit('Choose download or offline checking.')
    if (a.download or a.check_existing) and not a.destination:raise SystemExit('--destination is required.')
    dest=a.destination.resolve() if a.destination else None
    if a.download and dest==root:raise SystemExit('Download into a separate working copy, not the archive.')
    doc=json.loads((root/'gpu8/manifests_v71/models_v71.json').read_text(encoding='utf-8'));rows=[]
    for key,m in doc['models'].items():
        rev=m['revision']
        if not re.fullmatch(r'[0-9a-f]{40}',rev):raise SystemExit('Unpinned model revision: '+key)
        for name,r in m['files'].items():
            url='https://huggingface.co/'+m['repo']+'/resolve/'+rev+'/'+urllib.parse.quote(name,safe='/')
            row={'model':key,'revision':rev,'path':r['path'],'url':url,'sha256':r['sha256'],'bytes':r['bytes'],'status':'planned'}
            if dest:
                p=(dest/r['path']).resolve()
                if not p.is_relative_to(dest):raise SystemExit('Unsafe model path.')
                if p.is_file():
                    row['status']='matched' if p.stat().st_size==r['bytes'] and sha(p)==r['sha256'] else 'existing_mismatch'
                elif a.check_existing:row['status']='missing'
                elif a.download:
                    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.download-part')
                    if tmp.exists():raise SystemExit('Partial download exists; inspect before retrying: '+str(tmp))
                    req=urllib.request.Request(url,headers={'User-Agent':'SARA-pinned-archive-helper/1.0'})
                    with urllib.request.urlopen(req,timeout=120) as src,tmp.open('xb') as out:
                        for b in iter(lambda:src.read(8*1024*1024),b''):out.write(b)
                    if tmp.stat().st_size!=r['bytes'] or sha(tmp)!=r['sha256']:
                        raise SystemExit('Downloaded bytes do not match; partial retained: '+str(tmp))
                    tmp.replace(p);row['status']='downloaded_and_matched'
            rows.append(row)
    print(json.dumps({'files':rows,'model_licences':'See archived metadata and original model cards; no weights redistributed here.'},indent=2))
    return int(any(x['status'] in ('missing','existing_mismatch') for x in rows))

if __name__=='__main__':raise SystemExit(main())
