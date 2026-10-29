"""Run an original research module from an explicitly selected workspace."""
from pathlib import Path
import argparse, os, subprocess, sys

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace',type=Path,required=True,help='Disposable research working copy, not the archival source tree')
    ap.add_argument('module',help='gpu8.*, gpu8v6.*, gpu8v71.* or gpu8v8final.*')
    ap.add_argument('arguments',nargs=argparse.REMAINDER)
    a=ap.parse_args();root=a.workspace.resolve();src=root/'gpu8/src'
    if not src.is_dir():raise SystemExit('Workspace does not contain gpu8/src.')
    if a.module.split('.')[0] not in ('gpu8','gpu8v6','gpu8v71','gpu8v8final'):raise SystemExit('Module outside archived research packages.')
    package=Path(__file__).resolve().parents[1]
    if root==package:raise SystemExit('Use a disposable working copy; archival source tree execution is refused.')
    env=os.environ.copy();env['PYTHONPATH']=str(src)+os.pathsep+env.get('PYTHONPATH','');env['PYTHONDONTWRITEBYTECODE']='1'
    return subprocess.call([sys.executable,'-B','-m',a.module,*a.arguments],cwd=root,env=env)

if __name__=='__main__':raise SystemExit(main())
