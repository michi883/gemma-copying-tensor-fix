#!/usr/bin/env python3
"""Root-cause probes for the `directs` -> `direct` copy failure.  Standard library only; talks to the llama-server of ./lab.sh (:8101).

Same request as the copy page (copy_page/server.py: PROMPT + text, temperature 0, seed 1, no system prompt, no history, cache_prompt off).  Three tools:

  gen TEXT...      free greedy generation (chat endpoint, logprobs on): the generated token ids as returned
  scan TEXT...     teacher-forced scan: the prompt, then the CORRECT token sequence of TEXT as the prompt itself tokenizes it, one whole-vocabulary scoring request per
                   position (/completion with the prompt as token ids, n_predict 1, n_probs = n_vocab).  Every position of every text is therefore measured the same way:
                   same prompt, same server, the preceding answer tokens are the correct ones.  Reported: p(correct token), its rank, p(<turn|>), the top alternatives.
  ban TEXT...      the same prompt generated greedily with <turn|> (id 106) banned (logit_bias false) -- a diagnostic intervention, not a fix

Results are appended to rc_data/<tool>.jsonl (one line per run, request + compact response); nothing is simulated.
"""
import json
import math
import os
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'copy_page'))
from server import PROMPT, SETTINGS  # noqa: E402  (the copy page's own prompt and settings, so this is the same request)

GEMMA = os.environ.get('RC_SERVER', 'http://127.0.0.1:8101')       # RC_SERVER=http://127.0.0.1:8102 to ask another llama-server
EOT = 106
WATCH = {'s': 236751, ' s': 503, '_': 236779, 'S': 236773, 'ing': 522, 'ed': 524, '<turn|>': EOT}     # tokens whose probability is recorded at every step, wherever they rank
DATA = HERE / os.environ.get('RC_DATA', 'rc_data')
DATA.mkdir(parents=True, exist_ok=True)
_names, _nv = {}, []


def post(path, body=None):
    req = urllib.request.Request(GEMMA + path, data=None if body is None else json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())


def n_vocab():
    if not _nv:
        _nv.append(post('/v1/models')['data'][0]['meta']['n_vocab'])
    return _nv[0]


def tokenize(s, add_special):
    """[(id, bytes of the piece)]; special tokens typed in the text (<turn|>) are always parsed"""
    ts = post('/tokenize', {'content': s, 'add_special': add_special, 'parse_special': True, 'with_pieces': True})['tokens']
    return [(t['id'], t['piece'].encode() if isinstance(t['piece'], str) else bytes(t['piece'])) for t in ts]


def name(i, b=None):
    """printable form of a token: its bytes, or the server's name for tokens without bytes (<turn|>)"""
    if b:
        return bytes(b).decode(errors='replace')
    if i not in _names:
        _names[i] = post('/detokenize', {'tokens': [i]})['content'] or f'<id {i}>'
    return _names[i]


def prompt_ids(text):
    """(prompt token ids, [(id, bytes)] of the text's tokens).  The text's tokens are the ones the prompt itself holds, found by tokenizing the rendered chat prompt in three
    pieces and checking that nothing merges across the seams."""
    rendered = post('/apply-template', {'messages': [{'role': 'user', 'content': PROMPT + text}]})['prompt']
    at = rendered.rindex(PROMPT + text) + len(PROMPT)
    full, head, tail = tokenize(rendered, True), tokenize(rendered[:at], True), tokenize(rendered[at + len(text):], False)
    a, b = len(head), len(full) - len(tail)
    assert full[:a] == head and full[b:] == tail, f'tokens merge across the seam of {text!r}'
    assert b''.join(p for _, p in full[a:b]) == text.encode(), f'the prompt holds {full[a:b]} for {text!r}'
    return [i for i, _ in full], full[a:b]


def vocab_dist(ids):
    """whole next-token distribution after `ids`: {token id: (probability, bytes)} in rank order"""
    r = post('/completion', {'prompt': ids, 'n_predict': 1, 'temperature': 0, 'seed': 1, 'cache_prompt': False, 'n_probs': n_vocab()})
    assert r['tokens_evaluated'] == len(ids), (r['tokens_evaluated'], len(ids))
    return [(e['id'], math.exp(e['logprob']), e['bytes']) for e in r['completion_probabilities'][0]['top_logprobs']]


def step_report(ids, want_id, remaining):
    """one scoring request at the prefix `ids`; `want_id` is the correct token there (None: the correct move is to end the turn), `remaining` the source bytes still to be copied"""
    dist = vocab_dist(ids)
    pos = {i: (n + 1, p) for n, (i, p, _) in enumerate(dist)}
    top = [{'id': i, 'tok': name(i, b), 'p': round(p, 5)} for i, p, b in dist[:6]]
    tcorrect = sum(p for i, p, b in dist if (b and remaining.startswith(bytes(b))) or (not remaining and i == EOT))        # every token whose text continues the copy correctly
    return {'want': want_id, 'want_tok': name(want_id) if want_id is not None else '<turn|>', 'p_want': pos[want_id if want_id is not None else EOT][1],
            'rank_want': pos[want_id if want_id is not None else EOT][0], 'p_eot': pos[EOT][1], 'rank_eot': pos[EOT][0], 'p_text_correct': round(tcorrect, 5),
            'sum_p': round(sum(p for _, p, _ in dist), 4), 'top': top, 'watch': {k: [round(pos[i][1], 5), pos[i][0]] for k, i in WATCH.items()}}


def scan(text):
    ids, ref = prompt_ids(text)
    out = {'text': text, 'prompt_tokens': len(ids), 'ref_ids': [i for i, _ in ref], 'ref_toks': [name(i, b) for i, b in ref], 'steps': []}
    src, done = text.encode(), 0
    for k in range(len(ref) + 1):                       # k correct answer tokens already written; the last step is the one that has to end the turn
        r = step_report(ids + out['ref_ids'][:k], ref[k][0] if k < len(ref) else None, src[done:])
        r.update(step=k, after=''.join(out['ref_toks'][:k]))
        out['steps'].append(r)
        if k < len(ref):
            done += len(ref[k][1])
    return out


def gen(text, ban=False, n=None):
    body = {'messages': [{'role': 'user', 'content': PROMPT + text}], **SETTINGS, 'logprobs': True, 'top_logprobs': 5}
    if n:
        body['max_tokens'] = n
    if ban:
        body['logit_bias'] = {str(EOT): -100}
    r = post('/v1/chat/completions', body)
    ch = r['choices'][0]
    toks = [{'id': t['id'], 'tok': name(t['id'], t['bytes']), 'p': round(math.exp(t['logprob']), 5), 'top': [(name(e['id'], e['bytes']), round(math.exp(e['logprob']), 4)) for e in t['top_logprobs'][:4]]}
            for t in (ch.get('logprobs') or {}).get('content', [])]
    return {'text': text, 'ban_eot': ban, 'answer': ch['message'].get('content'), 'finish': ch['finish_reason'], 'prompt_tokens': r['usage']['prompt_tokens'], 'generated': toks}


def ban(text, n=8):
    """greedy generation from the prompt (as token ids) with <turn|> banned by logit_bias false -- exact -inf, and /completion reports the generated token ids"""
    ids, _ = prompt_ids(text)
    r = post('/completion', {'prompt': ids, 'n_predict': n, 'temperature': 0, 'seed': 1, 'cache_prompt': False, 'logit_bias': [[EOT, False]], 'return_tokens': True, 'n_probs': 5, 'post_sampling_probs': False})
    gt = r['tokens']
    return {'text': text, 'generated_ids': gt, 'generated_toks': [name(i) for i in gt], 'content': r['content'], 'steps': [
        [(e['id'], name(e['id'], e['bytes']), round(math.exp(e['logprob']), 4)) for e in c['top_logprobs'][:5]] for c in r['completion_probabilities']]}


def save(tool, rec):
    with open(DATA / f'{tool}.jsonl', 'a') as f:
        f.write(json.dumps({'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'server': GEMMA, **rec}, ensure_ascii=False) + '\n')


def show_scan(s):
    print(f'\n{s["text"]!r}   prompt {s["prompt_tokens"]} tokens   correct answer tokens {list(zip(s["ref_ids"], s["ref_toks"]))}')
    for r in s['steps']:
        want = f'{r["want_tok"]!r}'
        print(f'  after {r["after"]!r:>20}  want {want:>10} p={r["p_want"]:.4f} (rank {r["rank_want"]})  p(<turn|>)={r["p_eot"]:.4f} (rank {r["rank_eot"]})  text-correct mass={r["p_text_correct"]:.3f}  '
              f'top: ' + '  '.join(f'{t["tok"]!r}={t["p"]:.3f}' for t in r['top'][:5]))


if __name__ == '__main__':
    tool, texts = sys.argv[1], sys.argv[2:]
    for t in texts:
        if tool == 'gen':
            r = gen(t)
            print(f'{t!r} -> {r["answer"]!r}  ids {[g["id"] for g in r["generated"]]}  ' + ' '.join(f'{g["tok"]!r}:{g["p"]}' for g in r['generated']))
        elif tool == 'scan':
            r = scan(t)
            show_scan(r)
        elif tool == 'ban':
            r = ban(t)
            print(f'{t!r} (<turn|> banned) -> {r["content"]!r}  ids {r["generated_ids"]}  {r["generated_toks"]}')
        else:
            sys.exit(f'unknown tool {tool}')
        save(tool, r)
