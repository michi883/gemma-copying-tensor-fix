#!/usr/bin/env python3
"""Copy test, side by side: one page that asks the ORIGINAL and the PATCHED local Gemma 4 E4B Q4_0 server to copy the same text and shows both answers next to the input.  Standard library only.

    ./lab.sh start model && ./lab.sh start patched && ./lab.sh start copy        ->  http://127.0.0.1:8600     (./lab.sh stop copy to stop it; log of the process: logs/copy.log)
    python3 experiments/identifier_fidelity/copy_page/server.py [--port 8600] [--gemma http://127.0.0.1:8101] [--patched http://127.0.0.1:8102]     (the same, by hand)

The models come from `./lab.sh start model` (original GGUF, llama-server on :8101) and `./lab.sh start patched` (the same GGUF with the one tensor per_layer_model_proj.weight multiplied by 2, :8102;
experiments/identifier_fidelity/ROOT_CAUSE.md, compare_models.py proves the difference).  A browser page cannot call them directly (CORS), so this file serves index.html and forwards requests.

One click on the page makes two independent calls, /api/copy with model=original and model=patched, so each panel has its own loading and error state and a slow or dead server holds up only its own
panel.  Both are built by build_request(text), which is the only place a request is made: the user message PROMPT + text and the SETTINGS below, nothing else: no system prompt, no history, no tools,
no chat_template_kwargs (whether the model thinks is whatever the server does for such a request; each server renders the chat template stored in its own GGUF, and the two GGUFs have byte-identical
metadata).  The answer shown is choices[0].message.content exactly as returned; `reasoning_content`, if the server ever sends it, is only logged, never shown and never substituted.  The reply carries the
sha256 of the request bytes as sent, so the page can show that both models received the identical request.

Every call appends one line to compare_log.jsonl (copy) or compare_inspect_log.jsonl (Token Inspector): which model (key, label, URL, the GGUF path the server reports in /props, its sha256 from
compare_data/models_identity.json when path, size and mtime still match, chat template hash), the request body as sent and the complete response as received (or the error), and `pair`, the id that the page
gives both calls of one click.  The logs of the single-model page (copy_log.jsonl, inspect_log.jsonl) are not written any more and not touched; its page and server are in single_model_v1/.

The page's Token Inspector (POST /api/inspect, analysis in token_inspector.py, unchanged) repeats the request of ONE model with logprobs on, tokenizes the prompt, and, where the correct token is not
among the top 20, scores that one step with a second request to the same server.  Every request it makes and the complete response goes to compare_inspect_log.jsonl (one line per inspection; the 26 MB
whole-vocabulary reply of the scoring request is cut to its top 100 plus the totals), together with the result shown on the page.
"""
import argparse
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import token_inspector

HERE = Path(__file__).resolve().parent
LOG = HERE / 'compare_log.jsonl'
INSPECT_LOG = HERE / 'compare_inspect_log.jsonl'
IDENTITY = HERE.parent / 'compare_data' / 'models_identity.json'            # written by compare_models.py
GEMMAS = {'original': 'http://127.0.0.1:8101', 'patched': 'http://127.0.0.1:8102'}
LABELS = {'original': 'Original E4B', 'patched': 'Patched E4B'}
PROMPT = 'Copy the following text exactly. Output only the copied text:\n'          # the text follows after this newline
SETTINGS = {'temperature': 0, 'seed': 1, 'max_tokens': 8192, 'stream': False, 'cache_prompt': False}     # cache_prompt off: every click recomputes from scratch, like repro.py
LOCK = threading.Lock()


def build_request(text):
    return {'messages': [{'role': 'user', 'content': PROMPT + text}], **SETTINGS}


def append_log(rec, log=LOG):
    with LOCK:
        n = sum(1 for _ in open(log)) + 1 if log.exists() else 1
        with open(log, 'a') as f:
            f.write(json.dumps({'n': n, **rec}, ensure_ascii=False) + '\n')
    return n


def recorded():
    try:
        return json.loads(IDENTITY.read_text())
    except (OSError, ValueError):
        return {}


def identity(model):
    """What the server on this model's port is serving right now (/props), plus the sha256 that compare_models.py recorded for that file, but only while path, size and mtime still match the record."""
    props = json.load(urllib.request.urlopen(GEMMAS[model] + '/props', timeout=5))
    out = {'model': model, 'label': LABELS[model], 'url': GEMMAS[model], 'path': props.get('model_path'), 'alias': props.get('model_alias'), 'ftype': props.get('model_ftype'),
           'build': props.get('build_info'), 'template_sha256': hashlib.sha256((props.get('chat_template') or '').encode()).hexdigest(), 'sha256': None}
    rec = recorded().get('models', {}).get(model)
    try:
        st = os.stat(out['path'])
        if rec and rec['path'] == out['path'] and rec['size'] == st.st_size and rec['mtime'] == time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime(st.st_mtime)):
            out['sha256'] = rec['sha256']
    except (OSError, TypeError):
        pass
    return out


def unreachable(model, e):
    return f'cannot reach the model server at {GEMMAS[model]}: {e.reason if isinstance(e, urllib.error.URLError) else e}'


def unexpected(model, e):
    return f'the model server at {GEMMAS[model]} did not answer in time: {e}' if isinstance(e, TimeoutError) else f'unexpected reply from the model server: {e!r}'


def ask(text, model, pair):
    """One real request to one model.  Returns what the page needs; the log line holds everything."""
    body = build_request(text)
    data = json.dumps(body).encode()
    url = GEMMAS[model] + '/v1/chat/completions'
    rec = {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'pair': pair, 'model': model, 'url': url, 'input': text, 'request': body, 'request_sha256': hashlib.sha256(data).hexdigest()}
    t0, raw = time.time(), None
    out = {'model': model, 'label': LABELS[model], 'request_sha256': rec['request_sha256']}
    try:
        out['identity'] = rec['identity'] = identity(model)
        req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
        raw = urllib.request.urlopen(req, timeout=900).read().decode()
        rec['response'] = json.loads(raw)
        choice = rec['response']['choices'][0]
        out.update(answer=choice['message'].get('content') or '', finish_reason=choice['finish_reason'], prompt_tokens=rec['response']['usage']['prompt_tokens'],
                   completion_tokens=rec['response']['usage']['completion_tokens'])
    except urllib.error.HTTPError as e:
        out['error'] = f'HTTP {e.code} from the model server: {e.read().decode(errors="replace")[:500]}'
    except urllib.error.URLError as e:
        out['error'] = unreachable(model, e)
    except (OSError, ValueError, KeyError, IndexError) as e:                                 # incl. timeouts and a body that is not the expected JSON
        out['error'] = unexpected(model, e)
        if raw is not None:
            rec['response_text'] = raw
    out['seconds'] = round(time.time() - t0, 2)
    rec.update(seconds=out['seconds'], **({'answer': out['answer']} if 'answer' in out else {'error': out['error']}))
    out['n'] = append_log(rec)
    print(f'#{out["n"]} {model} {out["seconds"]}s  {text!r} -> {out.get("answer", out.get("error"))!r}', flush=True)
    return out


def inspect_text(text, model, pair):
    """One Token Inspector run on one model: a handful of real requests to that model server, all of them logged."""
    calls = []
    rec = {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'pair': pair, 'model': model, 'input': text, 'calls': calls}
    base = GEMMAS[model]

    def call(name, path, body=None, trim=None):
        c = {'name': name, 'url': base + path, 'request': body}
        calls.append(c)
        req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        t0 = time.time()
        resp = json.loads(urllib.request.urlopen(req, timeout=900).read())
        c.update(seconds=round(time.time() - t0, 2), response=trim(resp) if trim else resp)
        return resp

    t0 = time.time()
    try:
        rec['identity'] = identity(model)
        out = token_inspector.inspect(text, build_request(text), PROMPT, call)
    except urllib.error.HTTPError as e:
        out = {'error': f'HTTP {e.code} from the model server: {e.read().decode(errors="replace")[:500]}'}
    except urllib.error.URLError as e:
        out = {'error': unreachable(model, e)}
    except (OSError, ValueError, KeyError, IndexError, TypeError, token_inspector.InspectError) as e:
        out = {'error': f'inspection failed: {unexpected(model, e)}'}
    out['model'], out['label'], out['identity'] = model, LABELS[model], rec.get('identity')
    out['seconds'] = round(time.time() - t0, 2)
    rec.update(seconds=out['seconds'], result=out)
    out['n'] = append_log(rec, INSPECT_LOG)
    print(f'inspect #{out["n"]} {model} {out["seconds"]}s  {text!r}  {len(calls)} requests  {out.get("error", "")}', flush=True)
    return out


def info():
    rec = recorded()
    out = {'prompt': PROMPT, 'settings': SETTINGS, 'models': {}, 'patch': {k: rec.get(k) for k in ('tensor', 'factor', 'tensors_total', 'tensors_differing', 'bytes_differing', 'all_ok', 'time')}}
    for model in GEMMAS:
        try:
            out['models'][model] = identity(model)
        except (OSError, ValueError) as e:
            out['models'][model] = {'model': model, 'label': LABELS[model], 'url': GEMMAS[model], 'error': unreachable(model, e)}
    return out


class Handler(BaseHTTPRequestHandler):
    def reply(self, code, body, ctype='application/json'):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype + '; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            self.reply(200, (HERE / 'index.html').read_bytes(), 'text/html')
        elif self.path == '/api/info':
            self.reply(200, info())
        else:
            self.reply(404, {'error': 'not found'})

    def do_POST(self):
        if self.path not in ('/api/copy', '/api/inspect'):
            return self.reply(404, {'error': 'not found'})
        try:
            size = int(self.headers.get('Content-Length', 0))
            if size > 1_000_000:
                raise ValueError('input too large')
            req = json.loads(self.rfile.read(size))
            text, model, pair = req['text'], req['model'], str(req.get('pair', ''))[:40]
            if not isinstance(text, str):
                raise ValueError('text must be a string')
            if model not in GEMMAS:
                raise ValueError(f'model must be one of {sorted(GEMMAS)}')
        except (ValueError, KeyError, TypeError) as e:
            return self.reply(400, {'error': f'bad request: {e}'})
        out = (ask if self.path == '/api/copy' else inspect_text)(text, model, pair)
        self.reply(502 if 'error' in out else 200, out)

    def log_message(self, *args):                                                           # one line per call is printed by ask()
        pass


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8600)
    ap.add_argument('--gemma', default=GEMMAS['original'], help='base URL of the llama-server with the original GGUF (default %(default)s)')
    ap.add_argument('--patched', default=GEMMAS['patched'], help='base URL of the llama-server with the patched GGUF (default %(default)s)')
    args = ap.parse_args()
    GEMMAS['original'], GEMMAS['patched'] = args.gemma.rstrip('/'), args.patched.rstrip('/')
    print(f'copy test: http://127.0.0.1:{args.port}   original {GEMMAS["original"]}   patched {GEMMAS["patched"]}   log {LOG}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
