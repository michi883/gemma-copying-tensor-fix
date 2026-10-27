#!/usr/bin/env python3
"""Next-token distribution after the bare stem inside the decisive edit call, in an IDENTICAL context, on the server that is up on :8110.

    python -I probe_edit.py SERVER_LABEL CONTEXT_ARM TASK [TASK ...]

The request is the one CONTEXT_ARM sent at its first edit_file / write_file turn (rebuilt from its transcript, hash checked against the recorded one), with logprobs on.  The generated token
stream (Gemma's native tool-call text, greedy) is scanned for every position where the token is the task's bare stem (target identifier without its final `s`, e.g. `diagram`); for each,
the chosen next token and the probability of the lone `s` token (read from the top-20, or "< p20" when it is not among them) are written to probes/edit_<server>__ctx-<arm>__<task>.json.
"""
import hashlib
import json
import math
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze                                                                   # noqa: E402
import harness                                                                   # noqa: E402

URL = 'http://127.0.0.1:8110/v1'
server, ctx_arm, tasks = sys.argv[1], sys.argv[2], sys.argv[3:]
tools_decl = json.loads((HERE / 'tools.json').read_text())
tasks_json = {t['id']: t for t in json.loads((HERE.parent / 'tasks.json').read_text())}
(HERE / 'probes').mkdir(exist_ok=True)

for task in tasks:
    target = analyze.target_of(task, tasks_json)
    stem = target[:-1]
    msgs = json.loads((HERE / 'runs' / ctx_arm / task / 'transcript.json').read_text())
    n, cut = 0, None
    for i, m in enumerate(msgs):
        if m['role'] == 'assistant':
            n += 1
            if any(c['function']['name'] in ('edit_file', 'write_file') for c in (m.get('tool_calls') or [])):
                cut = i
                break
    body = {'model': 'gemma-4-e4b', 'messages': msgs[:cut], 'tools': tools_decl, **harness.GEN, **harness.EXTRA_BODY}
    sha = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    rec_sha = next(json.loads(l)['request_sha256'] for l in open(HERE / 'runs' / ctx_arm / task / 'events.jsonl') if json.loads(l)['kind'] == 'model_reply' and json.loads(l)['turn'] == n)
    body.update(logprobs=True, top_logprobs=20)
    req = urllib.request.Request(URL + '/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    resp = json.loads(urllib.request.urlopen(req, timeout=900).read())
    ch = resp['choices'][0]
    toks = ch['logprobs']['content']
    hits = []
    for k, t in enumerate(toks[:-1]):
        if t['token'].strip() == stem or re.fullmatch(r'.*[^A-Za-z]' + re.escape(stem), t['token']):
            nxt = toks[k + 1]
            top = {x['token']: x['logprob'] for x in nxt['top_logprobs']}
            lone_s = top.get('s')
            hits.append({'token_index': k, 'context': ''.join(x['token'] for x in toks[max(0, k - 5):k + 1]), 'chosen_next': nxt['token'], 'p_chosen_next': round(math.exp(nxt['logprob']), 4),
                         'p_lone_s': round(math.exp(lone_s), 4) if lone_s is not None else None, 'lone_s_in_top20': lone_s is not None, 'top5': [(x['token'], round(math.exp(x['logprob']), 4)) for x in nxt['top_logprobs'][:5]]})
    call = (ch['message'].get('tool_calls') or [{}])[0].get('function', {})
    rec = {'task': task, 'server': server, 'context_arm': ctx_arm, 'decisive_turn': n, 'request_matches_recorded': sha == rec_sha, 'stem': stem, 'target': target, 'generated_call': call,
           'positions_after_stem': hits,
           'stream': [{'token': t['token'], 'lp': round(t['logprob'], 4), 'top': [(x['token'], round(x['logprob'], 4)) for x in t['top_logprobs']]} for t in toks]}
    (HERE / 'probes' / f'edit_{server}__ctx-{ctx_arm}__{task}.json').write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(task, 'request matches recorded:', rec['request_matches_recorded'], '|', len(hits), 'positions after stem |', [(h['chosen_next'], h['p_chosen_next'], h['p_lone_s']) for h in hits])
