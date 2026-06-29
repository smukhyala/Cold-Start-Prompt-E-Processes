#!/usr/bin/env python3
"""Inspect the WebArena Shopping task bank without starting a sweep."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
root = Path(os.environ.get('WEBARENA_INFINITY_ROOT', '../webarena-infinity')).resolve()
app = root / 'apps' / 'shopping'
suite = 'real-tasks'
if not app.exists():
    print(f'ERROR: shopping app not found at {app}. Set WEBARENA_INFINITY_ROOT.', file=sys.stderr); sys.exit(1)
if (root / 'evaluation').exists():
    sys.path.insert(0, str(root / 'evaluation'))
    import tasks as tasks_mod  # type: ignore
else:
    sys.path.insert(0, str(root))
    from webarena_infinity.core import tasks as tasks_mod  # type: ignore
items = tasks_mod.load_tasks(str(app), suite)
print(f'count={len(items)}')
for i,t in enumerate(items):
    meta={k:t.get(k,'') for k in ('id','difficulty','type','category')}
    print(json.dumps({'bank_index':i, **meta}, ensure_ascii=False))
if len(items) < 60: print('decision=use_all_available')
elif len(items) == 60: print('decision=use_exact_60')
elif len(items) == 80: print('decision=use_exact_80_for_shopping_240_comparisons')
else: print('decision=inspect_bank_size_before_choosing_paired_budget')
