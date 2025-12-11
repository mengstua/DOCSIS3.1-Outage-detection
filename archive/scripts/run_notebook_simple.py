"""
Original archived: run_notebook_simple.py
"""
import json
import sys
import os
import traceback

NB = 'MLM.ipynb'
if len(sys.argv) > 1:
    NB = sys.argv[1]

if not os.path.exists(NB):
    print(f'Notebook not found: {NB}')
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
except Exception:
    pass

with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb.get('cells', [])

glob = {'__name__': '__main__'}

print(f'Executing notebook: {NB}  (code cells: {sum(1 for c in cells if c.get("cell_type")=="code")})')

for i, cell in enumerate(cells, start=1):
    if cell.get('cell_type') != 'code':
        continue
    source = cell.get('source', [])
    if isinstance(source, list):
        code = ''.join(source)
    else:
        code = source
    code = code.replace('\r\n', '\n')
    if not code.strip():
        continue
    print('\n--- Executing cell', i, '---')
    try:
        exec(compile(code, f'<cell {i}>', 'exec'), glob)
    except SystemExit as e:
        print(f'Cell {i} called SystemExit({e}). Continuing...')
    except Exception:
        print(f'ERROR executing cell {i}:')
        traceback.print_exc()

print('\nNotebook execution finished.')
