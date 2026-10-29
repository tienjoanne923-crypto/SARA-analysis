"""Read-only SHA-256 verification of separately acquired SARA inputs."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import sys

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root',type=Path,required=True,help='Root of the original relative input layout')
    parser.add_argument('--manifest',type=Path,default=Path(__file__).with_name('input_manifest.json'))
    args=parser.parse_args()
    root=args.input_root.resolve(strict=True)
    manifest=json.loads(args.manifest.read_text(encoding='utf8'))
    result=[]
    for row in manifest['files']:
        rel=PurePosixPath(row['path'])
        if rel.is_absolute() or '..' in rel.parts or ':' in row['path'] or '\\' in row['path']:
            raise ValueError('Unsafe manifest path')
        p=root.joinpath(*rel.parts).resolve()
        if not p.is_relative_to(root):raise ValueError('Input resolves outside root')
        status='missing'
        if p.is_file():
            if p.stat().st_size!=row['bytes']:
                status='size_mismatch'
            else:
                h=hashlib.sha256()
                with p.open('rb') as f:
                    for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
                status='match' if h.hexdigest()==row['sha256'] else 'hash_mismatch'
        result.append({'path':row['path'],'status':status})
    counts={s:sum(r['status']==s for r in result) for s in ['match','missing','size_mismatch','hash_mismatch']}
    print(json.dumps({'counts':counts,'files':result,'training_or_performance_reproduction':False},indent=2))
    return 0 if counts['match']==len(result) else 2

if __name__=='__main__':sys.exit(main())
