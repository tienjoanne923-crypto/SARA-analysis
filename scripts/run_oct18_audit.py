"""Adapt only workspace-path string literals in original October 18 audit copies."""
from pathlib import Path
import argparse, ast, hashlib, json, os, subprocess, sys

AUDITS=['external_family_audit.py','clinvar_raw_audit.py','equivalence_training_audit.py',
        'verify_teacher_equivalence_fit_hashes.py','external_strict_sensitivity.py','equivalence_main_sensitivity.py']

def adapt_paths(source,workspace):
    raw=source.encode('utf-8');lines=raw.splitlines(keepends=True);starts=[0]
    for line in lines:starts.append(starts[-1]+len(line))
    edits=[]
    for n in ast.walk(ast.parse(source)):
        if not isinstance(n,ast.Constant) or not isinstance(n.value,str):continue
        prefix=next((p for p in ('D:\\信','D:/信') if n.value.startswith(p)),None)
        if prefix is None:continue
        suffix=n.value[len(prefix):].replace('\\','/').lstrip('/')
        value=str(workspace/suffix)
        start=starts[n.lineno-1]+n.col_offset;end=starts[n.end_lineno-1]+n.end_col_offset
        edits.append((start,end,repr(value).encode('utf-8')))
    for start,end,value in sorted(edits,reverse=True):raw=raw[:start]+value+raw[end:]
    result=raw.decode('utf-8');ast.parse(result)
    return result,len(edits)

def main():
    package=Path(__file__).resolve().parents[1]
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace',required=True,type=Path,help='Disposable copied workspace with original required inputs/predictions')
    ap.add_argument('--audit',choices=AUDITS+['audit_training_mechanism.py'])
    ap.add_argument('--execute',action='store_true',help='Execute selected adapted audit; otherwise only plan/stage copies')
    a=ap.parse_args();workspace=a.workspace.resolve()
    if workspace==package:raise SystemExit('Use a separate disposable workspace.')
    if not (workspace/'gpu8/src').is_dir():raise SystemExit('Workspace must contain gpu8/src.')
    if a.execute and not a.audit:raise SystemExit('--execute requires a single --audit name.')
    sources=[Path('sara_revision_20261018/audit')/x for x in AUDITS]+[Path('sara_revision_20261018/audit_training_mechanism.py')]
    changes=[]
    for rel in sources:
        src=package/rel;old=src.read_text(encoding='utf-8');new,n=adapt_paths(old,workspace);dest=workspace/rel
        if dest.exists() and dest.read_text(encoding='utf-8') not in (old,new):raise SystemExit('Refuse to overwrite unrelated working file: '+str(rel))
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(new,encoding='utf-8',newline='\n')
        changes.append({'path':rel.as_posix(),'path_literals_adapted':n,'original_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'adapted_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
    record={'purpose':'workspace path adaptation only; originals unchanged; no scientific expression edited','files':changes,'executed':a.audit if a.execute else None}
    record_path=workspace/'sara_revision_20261018/audit/path_adapter_record.json';record_path.write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))
    if a.execute:
        rel=next(x for x in sources if x.name==a.audit)
        env=os.environ.copy();env['PYTHONPATH']=str(workspace/'gpu8/src')+os.pathsep+env.get('PYTHONPATH','');env['PYTHONDONTWRITEBYTECODE']='1'
        return subprocess.call([sys.executable,'-B',str(workspace/rel)],cwd=workspace,env=env)
    return 0

if __name__=='__main__':raise SystemExit(main())
