"""List/preflight original preprocessing stages; execute only in a disposable copy."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    package=Path(__file__).resolve().parents[1]
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--list',action='store_true')
    ap.add_argument('--stage',help='One stage id, as shown by --list')
    ap.add_argument('--workspace',type=Path,help='Separate working copy created by make_working_copy.py')
    ap.add_argument('--execute',action='store_true')
    a=ap.parse_args();plan=json.loads((package/'docs/PREPROCESS_STAGES.json').read_text(encoding='utf-8'))
    if a.list or not a.stage:
        if a.execute:raise SystemExit('--execute requires --stage and --workspace.')
        print(json.dumps({'stages':[{'id':s['id'],'title':s['title'],'depends_on':s['depends_on']} for s in plan['stages']],
            'default':'preflight only; no downloads, dependency auto-run, fitting or registration','execution_verified':False},indent=2));return 0
    found=[s for s in plan['stages'] if s['id']==a.stage]
    if not found:raise SystemExit('Unknown stage. Use --list.')
    s=found[0]
    if not a.workspace:
        if a.execute:raise SystemExit('--workspace is required for execution.')
        print(json.dumps({'stage':s,'status':'plan only; workspace not provided'},indent=2));return 0
    root=a.workspace.resolve()
    if root==package:raise SystemExit('Use a separate disposable working copy.')
    if not (root/'gpu8/src').is_dir():raise SystemExit('Workspace lacks gpu8/src.')
    hashes={r['path']:r for r in json.loads((package/'provenance/INPUTS_SHA256.json').read_text(encoding='utf-8'))['files']}
    missing=[];mismatch=[]
    for rel in s['requires']:
        p=(root/rel).resolve()
        if not p.is_relative_to(root):raise SystemExit('Unsafe required-input path.')
        if not p.exists():missing.append(rel)
        elif p.is_file() and rel in hashes:
            r=hashes[rel]
            if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:mismatch.append(rel)
    report={'stage':s['id'],'commands':s['commands'],'depends_on':s['depends_on'],'writes_to':s['writes_to'],
            'missing_required_inputs':missing,'changed_known_inputs':mismatch,
            'preflight_scope':'Selected prerequisite existence and recorded hashes where available; not a complete semantic dependency audit.',
            'executed':False}
    print(json.dumps(report,indent=2))
    if missing or mismatch:return 2
    if not a.execute:return 0
    marker=root/'.sara-disposable-workspace.json'
    if not marker.is_file():raise SystemExit('Missing disposable-workspace marker. Create the copy with scripts/make_working_copy.py.')
    if s.get('network_required') or s.get('research_development_or_registration'):raise SystemExit('This wrapper excludes downloads and research selection.')
    env=os.environ.copy();env['PYTHONPATH']=str(root/'gpu8/src')+os.pathsep+env.get('PYTHONPATH','');env['PYTHONDONTWRITEBYTECODE']='1'
    for args in s['commands']:
        if args[0]!='python':raise SystemExit('Unsupported non-Python stage.')
        rc=subprocess.call([sys.executable,'-B',*args[1:]],cwd=root,env=env)
        if rc:return rc
    print(json.dumps({'stage':s['id'],'execution_exit_code':0,'note':'Compare new outputs and hashes; execution alone does not establish original-result reproduction.'}))
    return 0

if __name__=='__main__':raise SystemExit(main())
