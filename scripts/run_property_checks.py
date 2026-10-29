"""Run the original synthetic tests in this extracted source package."""
from pathlib import Path
import os, subprocess, sys, tempfile

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    env=os.environ.copy();env['PYTHONPATH']=str(root/'gpu8/src')+os.pathsep+env.get('PYTHONPATH','');env['PYTHONDONTWRITEBYTECODE']='1'
    with tempfile.TemporaryDirectory(prefix='sara_property_checks_') as td:
        raise SystemExit(subprocess.call([sys.executable,'-B','-m','pytest',str(root/'gpu8/tests/test_v8_properties.py'),'-q','-p','no:cacheprovider','--basetemp',str(Path(td)/'pytest')],cwd=root,env=env))
