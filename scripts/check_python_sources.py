"""Check Python syntax without importing research modules or creating bytecode."""
from pathlib import Path
import argparse, ast, json

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    a=ap.parse_args();root=a.root.resolve();bad=[];count=0
    for p in sorted(root.rglob('*.py')):
        if any(x in ('.git','__pycache__','validation_runtime') or x.startswith('.venv') for x in p.relative_to(root).parts):continue
        count+=1
        try:ast.parse(p.read_text(encoding='utf-8-sig'),filename=p.relative_to(root).as_posix())
        except (SyntaxError,UnicodeError) as e:bad.append({'path':p.relative_to(root).as_posix(),'error':str(e)})
    print(json.dumps({'python_files':count,'failed':bad,'passed':not bad},indent=2))
    return int(bool(bad))

if __name__=='__main__':raise SystemExit(main())
