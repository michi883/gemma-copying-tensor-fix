#!/usr/bin/env python3
"""Same-context replay of each trajectory's decisive edit on the server that is up on :8110.

    python -I cross_replay.py SERVER_LABEL [--temperature T --samples N --tag NAME] [TASK ...]

Why: the two arms' trajectories drift apart early for reasons unrelated to identifiers (different first sentences), so comparing whole runs mixes the effect of the weights with
the effect of a different context.  Here the CONTEXT is held fixed: for every task and every recorded arm X, the request that X sent at the turn of its first edit_file / write_file call
is rebuilt from runs/X/<task>/transcript.json (byte-identical to the recorded request: the hash is checked) and sent to the server that is up (model Y).  The produced call is then applied
by swegemma's real edit_file / write_file to a fresh pristine copy of the task repository and the task's tests are run.  Result per (context X, model Y): did the call apply, did the tests pass,
the old_string / new_string.  Y == X must reproduce the recorded call (determinism check at temperature 0).

Default is the main protocol (temperature 0, seed 1, thinking off).  --temperature/--samples/--tag give the sampled variant used only as an exploratory supplement (labelled as such in REPORT.md).
"""
import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness                                                                   # noqa: E402  (module import has no side effects)

URL = 'http://127.0.0.1:8110/v1'
ap = argparse.ArgumentParser()
ap.add_argument('server')
ap.add_argument('tasks', nargs='*')
ap.add_argument('--temperature', type=float, default=0.0)
ap.add_argument('--samples', type=int, default=1)
ap.add_argument('--tag', default='replays')
a = ap.parse_args()
tasks = a.tasks or [t['id'] for t in json.loads((HERE.parent / 'tasks.json').read_text())]
tools_decl = json.loads((HERE / 'tools.json').read_text())
out_dir = HERE / a.tag
out_dir.mkdir(exist_ok=True)
WORK = Path('/tmp/gemma-edit-work_replay')
WORK.mkdir(exist_ok=True)

from swegemma.budget import EvaluationBudget, HarnessLimits                      # noqa: E402
from swegemma.context import SwegemmaContext                                     # noqa: E402
from swegemma.models import Task                                                 # noqa: E402
from swegemma.tools import create_tools                                          # noqa: E402


def decisive_turn(arm, task):
    """first model turn with an edit_file or write_file call -> (turn, messages sent at that turn)"""
    msgs = json.loads((HERE / 'runs' / arm / task / 'transcript.json').read_text())
    n = 0
    for i, m in enumerate(msgs):
        if m['role'] == 'assistant':
            n += 1
            if any(c['function']['name'] in ('edit_file', 'write_file') for c in (m.get('tool_calls') or [])):
                return n, msgs[:i], m
    return None, None, None


def score(task, call):
    """Apply one edit_file / write_file call to a pristine copy with the real tools, then run the task's tests."""
    wd = WORK / f'{a.server}_{task}'
    harness.prepare_workdir(task, wd)
    sandbox = harness.LocalSandbox(wd)
    t = Task(instance_id=task, repo=f'gemma-edit-tasks/{task}', base_commit='baseline', problem_statement='', hints_text='', created_at='')
    ctx = SwegemmaContext(docker_manager=sandbox, container_id='local', task=t, budget=EvaluationBudget(**harness.BUDGET), harness=HarnessLimits(**harness.LIMITS))
    ctx.start_agent_session()
    name, raw = call['function']['name'], call['function']['arguments']
    try:
        args = json.loads(raw)
        result = create_tools(ctx)[name](**args)
    except Exception as e:                                                       # malformed arguments
        args, result = raw, {'status': 'error', 'error_type': type(e).__name__, 'error_message': str(e)}
    g = harness.grade(wd, task)
    return {'tool': name, 'args': args, 'applied': result.get('status') == 'ok', 'tool_error': None if result.get('status') == 'ok' else (result.get('error_message') or '')[:160],
            'tests_pass_after_this_call': g['passed'], 'tests': g['summary']}


for task in tasks:
    for ctx_arm in ('original', 'lora'):
        turn, msgs, rec_msg = decisive_turn(ctx_arm, task)
        if turn is None:
            (out_dir / f'{a.server}__ctx-{ctx_arm}__{task}.json').write_text(json.dumps({'task': task, 'server': a.server, 'context_arm': ctx_arm, 'note': 'no edit_file/write_file call in this trajectory'}))
            continue
        gen = {**harness.GEN, 'temperature': a.temperature}
        body = {'model': 'gemma-4-e4b', 'messages': msgs, 'tools': tools_decl, **gen, **harness.EXTRA_BODY}
        rebuilt = hashlib.sha256(json.dumps({**body, **harness.GEN, 'temperature': 0}, sort_keys=True).encode()).hexdigest()
        recorded = next(json.loads(l)['request_sha256'] for l in open(HERE / 'runs' / ctx_arm / task / 'events.jsonl') if json.loads(l)['kind'] == 'model_reply' and json.loads(l)['turn'] == turn)
        rec = {'task': task, 'server': a.server, 'context_arm': ctx_arm, 'decisive_turn': turn, 'temperature': a.temperature, 'request_matches_recorded': (rebuilt == recorded) if a.temperature == 0 else None, 'samples': []}
        rec_call = next((c for c in rec_msg['tool_calls'] if c['function']['name'] in ('edit_file', 'write_file')))
        for k in range(a.samples):
            body_k = dict(body, seed=harness.GEN['seed'] + k)
            req = urllib.request.Request(URL + '/chat/completions', data=json.dumps(body_k).encode(), headers={'Content-Type': 'application/json'})
            msg = json.loads(urllib.request.urlopen(req, timeout=900).read())['choices'][0]['message']
            call = next((c for c in (msg.get('tool_calls') or []) if c['function']['name'] in ('edit_file', 'write_file')), None)
            s = {'sample': k, 'produced_edit_call': call is not None, 'other_tools': [c['function']['name'] for c in (msg.get('tool_calls') or [])], 'text_head': (msg.get('content') or '')[:200]}
            if call:
                s.update(score(task, call))
                s['same_call_as_recorded_context_arm'] = call['function']['arguments'] == rec_call['function']['arguments']
            rec['samples'].append(s)
        (out_dir / f'{a.server}__ctx-{ctx_arm}__{task}.json').write_text(json.dumps(rec, indent=1))
        s0 = rec['samples'][0]
        print(f"{task} ctx={ctx_arm} model={a.server} turn={turn} matches_recorded_request={rec['request_matches_recorded']} samples={len(rec['samples'])} "
              f"applied={sum(x.get('applied', False) for x in rec['samples'])} pass={sum(x.get('tests_pass_after_this_call', False) for x in rec['samples'])}"
              + (f" same_as_recorded={s0.get('same_call_as_recorded_context_arm')}" if a.temperature == 0 else ''), flush=True)
