#!/usr/bin/env python3
"""Summarise runs/<arm>/<task>/ for the arms found on disk and pair them task by task.

    python3 analyze.py [--arms original lora] > analysis.md        (also writes analysis.json)

Mechanical facts only (what was read, called, sent, returned, left in the files, what the tests said, where the two arms first diverge).  The verdict "copying-related failure"
is a human judgement written in REPORT.md from these facts; this script only raises flags for it:
  - identifier counts in the pristine file(s) vs the final file(s): whole-word TARGET (e.g. `directs`) and whole-word bare STEM (target without its final `s`, e.g. `direct`);
  - for every edit_file call that failed with "not found": would the old_string have matched the file's state at that moment if one identifier were repaired by <= 2 characters (a copying slip)?
  - first divergence of the two arms' assistant outputs, with the differing span.
Tasks whose tasks.json `target_identifier` is not an identifier (04_treats lists "rules.py") use the identifier in the task name; the file itself is not modified.
"""
import argparse
import difflib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK_DIR = HERE.parent / 'tasks'
TARGET_OVERRIDE = {'04_treats': 'treats'}                      # tasks.json says "rules.py" (a metadata slip); the identifier under test is the Rule attribute `treats`
WORD = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')


def target_of(task, tasks_json):
    return TARGET_OVERRIDE.get(task, tasks_json[task]['target_identifier'])


def lev(a, b, cap=3):
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def count_word(text, word):
    return len(re.findall(rf'(?<![A-Za-z0-9_]){re.escape(word)}(?![A-Za-z0-9_])', text))


def load_events(path):
    return [json.loads(l) for l in open(path)]


def apply_edit(state, old, new, allow_multiple=False):
    """Replay a successful edit_file call on the analysis copy of a file (the tool matched it, so the plain exact replace is what happened except for flexible matches, flagged)."""
    if old in state:
        return state.replace(old, new) if allow_multiple else state.replace(old, new, 1), 'exact'
    from swegemma.edit import apply_replacement                 # only importable in the shell venv; falls back to unchanged state
    return apply_replacement(state, old, new, allow_multiple=allow_multiple).new_content, 'flexible'


def repaired_match(old, state):
    """If `old` is not in `state`, try repairing ONE identifier of old (<= 2 chars away from an identifier of the state) so that it matches.  -> (repaired old, [(was, now)]) or None."""
    words = set(WORD.findall(state))
    for w in sorted(set(WORD.findall(old)), key=len, reverse=True):
        if w in words:
            continue
        for s in words:
            if s != w and lev(w, s) <= 2:
                cand = re.sub(rf'(?<![A-Za-z0-9_]){re.escape(w)}(?![A-Za-z0-9_])', s, old)
                if cand in state:
                    return cand, [(w, s)]
    return None


def summarise(arm, task, tasks_json):
    run = HERE / 'runs' / arm / task
    ev = load_events(run / 'events.jsonl')
    res = json.loads((run / 'result.json').read_text())
    target = target_of(task, tasks_json)
    stem = target[:-1] if target.endswith('s') else target
    pristine = {p.name: p.read_text() for p in (TASK_DIR / task).glob('*.py')}
    state = dict(pristine)
    calls, errors, edits, reads = [], [], [], []
    pending = None
    for e in ev:
        if e['kind'] == 'tool_call':
            pending = e
            calls.append(e['name'])
            if e['name'] == 'read_file' and isinstance(e['args'], dict):
                reads.append(e['args'].get('filepath'))
        elif e['kind'] == 'tool_result' and pending is not None:
            r, a = e['result'], pending['args']
            if isinstance(r, dict) and r.get('status') == 'error':
                errors.append({'turn': e['turn'], 'tool': pending['name'], 'type': r.get('error_type'), 'message': (r.get('error_message') or '')[:200]})
            if pending['name'] == 'edit_file' and isinstance(a, dict):
                fp = a.get('filepath', '')
                rec = {'turn': e['turn'], 'filepath': fp, 'old_string': a.get('old_string'), 'new_string': a.get('new_string'), 'ok': r.get('status') == 'ok'}
                if rec['ok'] and fp in state:
                    state[fp], rec['match'] = apply_edit(state[fp], a['old_string'], a['new_string'], bool(a.get('allow_multiple')))
                elif not rec['ok'] and fp in state and 'not found' in (r.get('error_message') or ''):
                    fix = repaired_match(a.get('old_string') or '', state[fp])
                    if fix:
                        rec['slip_repair'] = {'would_match_if': fix[1], 'repaired_old_string': fix[0]}
                edits.append(rec)
            elif pending['name'] == 'write_file' and isinstance(a, dict) and r.get('status') == 'ok':
                state[a['filepath']] = a['content']
                edits.append({'turn': e['turn'], 'filepath': a['filepath'], 'write_file': True, 'content': a['content'], 'ok': True})
            pending = None
    final = res['final_files']
    src = {k: v for k, v in pristine.items() if not k.startswith('test_')}
    ident = {'target': target, 'stem': stem,
             'target_before': sum(count_word(v, target) for v in src.values()), 'target_after': sum(count_word(final.get(k, ''), target) for k in src),
             'stem_before': sum(count_word(v, stem) for v in src.values()), 'stem_after': sum(count_word(final.get(k, ''), stem) for k in src)}
    added = [l[1:] for l in res['final_patch'].splitlines() if l.startswith('+') and not l.startswith('+++')]
    ident['target_in_added_lines'] = sum(count_word(l, target) for l in added)
    ident['stem_in_added_lines'] = sum(count_word(l, stem) for l in added)
    replies = [e for e in ev if e['kind'] == 'model_reply']
    texts = ' '.join((e['message'].get('content') or '') for e in replies)
    ident['stem_in_assistant_text'] = count_word(texts, stem)
    ident['target_in_assistant_text'] = count_word(texts, target)
    return {'arm': arm, 'task': task, 'end_reason': res['end_reason'], 'submitted': res['submit_patch_called'], 'turns': res['turns'], 'tool_calls_used': res['tool_calls_used'],
            'seconds': res['seconds'], 'tests': res['tests']['summary'], 'tests_passed': res['tests']['passed'], 'test_files_modified': res['tests']['test_files_modified_by_agent'],
            'files_read': reads, 'tool_call_names': calls, 'edits': edits, 'errors': errors, 'identifiers': ident, 'final_patch': res['final_patch'],
            'reasoning_chars_total': sum(e['reasoning_chars'] for e in replies), 'first_request_sha256': replies[0]['request_sha256'] if replies else None,
            'prompt_tokens_first': replies[0]['usage']['prompt_tokens'] if replies else None}


def outputs(arm, task):
    """The sequence of model outputs: per turn (text, [(tool, canonical arguments)])."""
    out = []
    for e in load_events(HERE / 'runs' / arm / task / 'events.jsonl'):
        if e['kind'] == 'model_reply':
            m = e['message']
            out.append((e['turn'], m.get('content') or '', [(c['function']['name'], c['function']['arguments']) for c in (m.get('tool_calls') or [])]))
    return out


def first_divergence(a_arm, b_arm, task):
    A, B = outputs(a_arm, task), outputs(b_arm, task)
    for (ta, xa, ca), (tb, xb, cb) in zip(A, B):
        if xa == xb and ca == cb:
            continue
        what, sa, sb = ('text', xa, xb) if xa != xb else ('tool call', json.dumps(ca), json.dumps(cb))
        sm = difflib.SequenceMatcher(None, sa, sb, autojunk=False)
        ops = [(tag, sa[i1:i2], sb[j1:j2], sa[max(0, i1 - 30):i2 + 30], sb[max(0, j1 - 30):j2 + 30]) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != 'equal']
        t, ra, rb, ctxa, ctxb = ops[0]
        return {'turn': ta, 'in': what, 'a_span': ra, 'b_span': rb, 'a_context': ctxa, 'b_context': ctxb, 'n_diff_ops': len(ops)}
    if len(A) != len(B):
        return {'turn': min(len(A), len(B)) + 1, 'in': 'length', 'a_span': '', 'b_span': '', 'a_context': f'{len(A)} turns', 'b_context': f'{len(B)} turns', 'n_diff_ops': 0}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', nargs=2, default=['original', 'lora'])
    a = ap.parse_args()
    tasks_json = {t['id']: t for t in json.loads((HERE.parent / 'tasks.json').read_text())}
    summary = {}
    lines = ['# Mechanical analysis', '']
    lines.append('| task | arm | tests | end | turns | tool calls | edit_file ok/err | write_file | tool errors | target/stem before→after | stem in added lines | stem in assistant text | reasoning chars |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for task in tasks_json:
        for arm in a.arms:
            if not (HERE / 'runs' / arm / task / 'result.json').exists():
                continue
            s = summarise(arm, task, tasks_json)
            summary.setdefault(task, {})[arm] = s
            ed = [e for e in s['edits'] if not e.get('write_file')]
            i = s['identifiers']
            lines.append(f"| {task} | {arm} | {s['tests']} | {s['end_reason']}{'' if s['submitted'] else ' (no submit)'} | {s['turns']} | {s['tool_calls_used']} | {sum(e['ok'] for e in ed)}/{sum(not e['ok'] for e in ed)} | "
                         f"{sum(bool(e.get('write_file')) for e in s['edits'])} | {len(s['errors'])} | {i['target']} {i['target_before']}→{i['target_after']} / {i['stem']} {i['stem_before']}→{i['stem_after']} | {i['stem_in_added_lines']} | {i['stem_in_assistant_text']} | {s['reasoning_chars_total']} |")
    lines.append('')
    for task, arms in summary.items():
        lines += [f'## {task}', '']
        shas = {arm: s['first_request_sha256'] for arm, s in arms.items()}
        lines.append(f"first request identical across arms: {len(set(shas.values())) == 1} ({', '.join(f'{k}={v[:10]}' for k, v in shas.items())})")
        for arm, s in arms.items():
            lines += ['', f"### {arm}: tests {s['tests']}", f"files read: {s['files_read']}", f"tool calls: {' '.join(s['tool_call_names'])}"]
            for e in s['edits']:
                if e.get('write_file'):
                    lines.append(f"- turn {e['turn']} write_file {e['filepath']}:\n```\n{e['content']}\n```")
                else:
                    extra = f" **slip repair would match: {e['slip_repair']['would_match_if']}**" if e.get('slip_repair') else ''
                    lines.append(f"- turn {e['turn']} edit_file {e['filepath']} {'OK' if e['ok'] else 'ERROR'}{extra}\n  - old_string: `{e['old_string']!r}`\n  - new_string: `{e['new_string']!r}`")
            for er in s['errors']:
                lines.append(f"- tool error turn {er['turn']} {er['tool']} {er['type']}: {er['message'][:160]}")
            lines += ['', 'final patch:', '```diff', s['final_patch'].rstrip('\n'), '```']
        if len(arms) == 2:
            d = first_divergence(a.arms[0], a.arms[1], task)
            if d:
                lines += ['', f"first divergence: turn {d['turn']} in {d['in']}, {d['n_diff_ops']} differing spans; {a.arms[0]}: `{d['a_span']!r}` vs {a.arms[1]}: `{d['b_span']!r}`",
                          f"  {a.arms[0]} context: `{d['a_context']!r}`", f"  {a.arms[1]} context: `{d['b_context']!r}`"]
                summary[task]['first_divergence'] = d
            else:
                lines += ['', 'the two arms produced identical outputs on every turn']
        lines.append('')
    (HERE / 'analysis.json').write_text(json.dumps(summary, indent=1))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
