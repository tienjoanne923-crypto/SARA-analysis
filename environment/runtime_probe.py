"""Bounded dependency import probe. No models, network access, training or data reads."""
from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--overlay')
args = ap.parse_args()
if args.overlay:
    # Match paths8.add_libs: append the overlay, preserving primary packages first.
    sys.path.append(args.overlay)
out = {'python': sys.version, 'platform': platform.platform(), 'packages': {},
       'network_access': False, 'training': False}
for name in ('numpy', 'scipy', 'pandas', 'pyarrow', 'lightgbm', 'torch', 'pytest',
             'transformers', 'tokenizers', 'huggingface_hub', 'safetensors'):
    try:
        mod = importlib.import_module(name)
        out['packages'][name] = {'imported': True, 'version': getattr(mod, '__version__', None)}
        if name == 'torch':
            out['torch_cuda'] = {'build': mod.version.cuda, 'available': mod.cuda.is_available()}
    except Exception as exc:
        out['packages'][name] = {'imported': False, 'error_type': type(exc).__name__, 'error': str(exc)}
print(json.dumps(out, ensure_ascii=False, indent=2))
