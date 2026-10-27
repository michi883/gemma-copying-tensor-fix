#!/usr/bin/env python3
"""Aggregate the same-context replays: python3 summarize_replays.py > replay_summary.md
Cell = (context taken from arm X) x (model Y).  Temperature-0 table: did the single decisive edit apply and make the tests pass.  Sampled table (temperature 0.2, 10 seeds; exploratory): rates."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
tasks = [t['id'] for t in json.loads((HERE.parent / 'tasks.json').read_text())]


def load(d, server, ctx, t):
    f = HERE / d / f'{server}__ctx-{ctx}__{t}.json'
    return json.loads(f.read_text()) if f.exists() else None


def cell0(r):
    if r is None or 'samples' not in r:
        return 'n/a (no edit call in that trajectory)'
    s = r['samples'][0]
    if not s.get('produced_edit_call'):
        return 'no edit call'
    return 'applied, tests PASS' if s['tests_pass_after_this_call'] else ('applied, tests fail' if s['applied'] else 'NOT applied (' + (s['tool_error'] or '')[:30] + ')')


print('## A. Temperature 0: decisive edit replayed in a fixed context\n')
print('| task | context: original trajectory | | context: LoRA trajectory | |')
print('|---|---|---|---|---|')
print('| | model=original | model=LoRA | model=original | model=LoRA |')
tot = {(c, m): 0 for c in ('original', 'lora') for m in ('original', 'lora')}
for t in tasks:
    row = []
    for ctx in ('original', 'lora'):
        for m in ('original', 'lora'):
            r = load('replays', m, ctx, t)
            row.append(cell0(r))
            if r and 'samples' in r and r['samples'][0].get('tests_pass_after_this_call'):
                tot[(ctx, m)] += 1
    print(f'| {t} | ' + ' | '.join(row) + ' |')
print('| **tests pass (of 6 tasks with an edit)** | ' + ' | '.join(str(tot[(c, m)]) for c in ('original', 'lora') for m in ('original', 'lora')) + ' |')

print('\n## B. Lab sampling (temperature 0.2, top_p 0.95), 10 seeds per cell, exploratory\n')
print('| task | ctx=original: model=original | ctx=original: model=LoRA | ctx=lora: model=original | ctx=lora: model=LoRA |')
print('|---|---|---|---|---|')
agg = {(c, m): [0, 0, 0] for c in ('original', 'lora') for m in ('original', 'lora')}
for t in tasks:
    row = []
    for ctx in ('original', 'lora'):
        for m in ('original', 'lora'):
            r = load('replays_sampled', m, ctx, t)
            if r is None or 'samples' not in r:
                row.append('n/a'); continue
            n = len(r['samples'])
            ap = sum(bool(s.get('applied')) for s in r['samples'])
            ps = sum(bool(s.get('tests_pass_after_this_call')) for s in r['samples'])
            row.append(f'applied {ap}/{n}, pass {ps}/{n}')
            agg[(ctx, m)][0] += ap; agg[(ctx, m)][1] += ps; agg[(ctx, m)][2] += n
    print(f'| {t} | ' + ' | '.join(row) + ' |')
print('| **all samples** | ' + ' | '.join(f"applied {agg[k][0]}/{agg[k][2]}, pass {agg[k][1]}/{agg[k][2]}" for k in (('original', 'original'), ('original', 'lora'), ('lora', 'original'), ('lora', 'lora'))) + ' |')
