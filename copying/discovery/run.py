#!/usr/bin/env python3
"""Run a plan (plan_words.json / plan_blocks.json) against ONE llama-server and append one jsonl line per item.  Standard library only; sequential; resumable; no retries.

    python3 run.py --plan plan_words.json --server orig    --mode nothink     # :8101, results_words_orig_nothink.jsonl
    python3 run.py --plan plan_words.json --server patched --mode page        # :8102, results_words_patched_page.jsonl

Request = copy_page/server.py build_request(text) (PROMPT, temperature 0, seed 1, max_tokens, cache_prompt false) PLUS `logprobs`/`top_logprobs` 5 (verified to leave the answer unchanged on 15 words)
PLUS, in mode `nothink`, top-level `chat_template_kwargs.enable_thinking=false` (the setting the brief names; the copy page itself sends no kwargs, which is mode `page`: default template, system turn with <|think|>).
Pass = reply content byte-for-byte equal to the text (the exact check of the copy page).  For multi-line blocks a reply wrapped in one ``` fence is recorded as `fenced` and judged on the unwrapped text too.
Per item the file stores: output, exact, category, tokens with the chosen log-probability and the top-5 alternatives, the first position where the output stops being a prefix of the text, and timing.
"""
import argparse
import difflib
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'copy_page'))
import server                                                                               # PROMPT, SETTINGS, build_request

SERVERS = {'orig': ('http://127.0.0.1:8101', 'gemma-4-E4B_q4_0-it.gguf', '676c35070db6dbe52f93e9c864ee0fba4eddea94b9c875d9cb10daff453fbaee'),
           'patched': ('http://127.0.0.1:8102', 'gemma-4-E4B_q4_0-it-plproj-x2.gguf', '3a6ad54ed7a46e5fb9b6c6ccd6b16060f105fcc77ad4d7936c721dcdb4e9dc7c')}
TIMEOUT = 180


def classify(exp, out):
    """Category of a failed copy; first rule that applies wins (same rules as exact_copy_probe/probe.py, plus line-level ones for blocks)."""
    if out == '':
        return 'empty_output'
    if out.strip() == exp:
        return 'whitespace_added'
    if out.startswith(exp):
        return 'text_appended'
    if exp.startswith(out):
        missing = exp[len(out):]
        return 'dropped_final_s' if missing == 's' else 'dropped_last_char' if len(missing) == 1 else 'truncated_early_stop'
    if len(out) == len(exp):
        return 'case_changed' if out.lower() == exp.lower() else 'chars_substituted'
    if '\n' in exp:
        el, ol = exp.split('\n'), out.split('\n')
        if len(ol) < len(el):
            return 'lines_dropped_or_merged'
        if len(ol) > len(el):
            return 'lines_added_or_split'
    p = 0
    while p < min(len(exp), len(out)) and exp[p] == out[p]:
        p += 1
    if p >= 1 and exp[-1] != out[-1] and '\n' not in exp:
        return 'suffix_altered'
    if out.replace(' ', '').replace('_', '') == exp.replace('_', ''):
        return 'separator_changed'
    ops = {o[0] for o in difflib.SequenceMatcher(None, exp, out, autojunk=False).get_opcodes() if o[0] != 'equal'}
    return 'chars_dropped_inside' if ops == {'delete'} else 'chars_inserted' if ops == {'insert'} else 'mixed_edit'


def unfence(out):
    lines = out.split('\n')
    if len(lines) >= 3 and lines[0].startswith('```') and lines[-1].strip() == '```':
        return '\n'.join(lines[1:-1])
    return None


def first_divergence(text, toks):
    """Index of the first generated token after which the output is no longer a prefix of `text` (None if the output is a prefix of text or equal), with that token's data."""
    acc = ''
    for i, t in enumerate(toks):
        piece = bytes(t['bytes']).decode('utf-8', 'replace') if t.get('bytes') is not None else t['token']
        acc += piece
        if not text.startswith(acc):
            return i
    return None


def slim(entry):
    return {'tok': entry['token'], 'id': entry.get('id'), 'lp': round(entry['logprob'], 4)}


def call(url, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url + '/v1/chat/completions', data=data, headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=TIMEOUT)), hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True)
    ap.add_argument('--server', choices=SERVERS, required=True)
    ap.add_argument('--mode', choices=('nothink', 'page'), required=True)
    ap.add_argument('--tag', default='')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--max-tokens', type=int, default=0, help='override max_tokens (blocks need more than words)')
    args = ap.parse_args()
    url, fname, sha = SERVERS[args.server]
    props = json.load(urllib.request.urlopen(url + '/props', timeout=10))
    assert props['model_path'].endswith('/' + fname), f'{url} serves {props["model_path"]}, expected {fname}'
    kind = Path(args.plan).stem.replace('plan_', '')
    out_path = HERE / f'results_{kind}_{args.server}_{args.mode}{args.tag}.jsonl'
    plan = json.loads((HERE / args.plan).read_text())['planned']
    if args.limit:
        plan = plan[:args.limit]
    done = set()
    if out_path.exists():
        done = {json.loads(l)['id'] for l in out_path.read_text().splitlines() if l.strip()}
    meta_path = out_path.with_suffix('.meta.json')
    if not meta_path.exists():
        meta_path.write_text(json.dumps({'start': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'url': url, 'model_path': props['model_path'], 'expected_sha256': sha, 'build': props.get('build_info'),
                                         'chat_template_sha256': hashlib.sha256((props.get('chat_template') or '').encode()).hexdigest(), 'mode': args.mode, 'plan': args.plan,
                                         'plan_sha256': hashlib.sha256((HERE / args.plan).read_bytes()).hexdigest(), 'prompt': server.PROMPT, 'settings': server.SETTINGS, 'n_planned': len(plan)}, indent=1))
    print(f'{out_path.name}: {len(plan)} planned, {len(done)} already done', flush=True)
    t_start, n_run, n_fail = time.time(), 0, 0
    with open(out_path, 'a') as f:
        for item in plan:
            if item['id'] in done:
                continue
            body = server.build_request(item['text'])
            body.update(logprobs=True, top_logprobs=5)
            if args.max_tokens:
                body['max_tokens'] = args.max_tokens
            if args.mode == 'nothink':
                body['chat_template_kwargs'] = {'enable_thinking': False}
            rec = {**item, 'server': args.server, 'mode': args.mode, 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
            t0 = time.time()
            try:
                resp, rec['request_sha256'] = call(url, body)
                ch = resp['choices'][0]
                out = ch['message'].get('content') or ''
                toks = (ch.get('logprobs') or {}).get('content') or []
                rec.update(output=out, finish_reason=ch['finish_reason'], prompt_tokens=resp['usage']['prompt_tokens'], completion_tokens=resp['usage']['completion_tokens'],
                           reasoning=bool(ch['message'].get('reasoning_content')))
                rec['exact'] = out == item['text']
                un = unfence(out)
                rec['fenced'] = un is not None
                rec['exact_unfenced'] = rec['exact'] or (un == item['text'])
                rec['category'] = 'exact' if rec['exact'] else ('fenced_only' if rec['exact_unfenced'] else classify(item['text'], un if un is not None else out))
                rec['tokens'] = [{**slim(t), 'top': [slim(a) for a in t.get('top_logprobs', [])]} for t in toks]
                d = first_divergence(item['text'], toks) if not rec['exact'] else None
                rec['diverge_at'] = d
            except Exception as e:                                                          # recorded, never retried
                rec['error'] = f'{type(e).__name__}: {e}'[:300]
            rec['seconds'] = round(time.time() - t0, 3)
            f.write(json.dumps(rec) + '\n')
            f.flush()
            n_run += 1
            n_fail += ('error' not in rec) and not rec.get('exact_unfenced', True)
            if n_run % 250 == 0:
                el = time.time() - t_start
                print(f'{n_run} run, {n_fail} failures, {el:.0f}s, ~{(len(plan) - len(done) - n_run) * el / n_run:.0f}s left', flush=True)
    print(f'done: {n_run} run, {n_fail} failures, {time.time() - t_start:.0f}s', flush=True)


if __name__ == '__main__':
    main()
