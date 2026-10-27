#!/usr/bin/env python3
"""Dose-response of the QAT E4B GGUF to the scale of ONE tensor, per_layer_model_proj.  Run with the gemma4-playground python (numpy + mlx for gguf_scale_patch.py), llama-server and
llama-perplexity on PATH.  For each factor f: copy the GGUF, multiply per_layer_model_proj by f (gguf_scale_patch.py), start a llama-server on :8110, run the teacher-forced scan of
root_cause.py on a few texts, stop it, run llama-perplexity (60 chunks x 512 tokens of PPL_TEXT), and compute the RMSNorm gain the factor implies for 2000 random tokens (norm_input_check.py's
formula).  Appends to rc_data/dose_response.jsonl.   Usage: dose_response.py PPL_TEXT SCRATCH_DIR f1 f2 ..."""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

PORT = 8110
os.environ['RC_SERVER'] = f'http://127.0.0.1:{PORT}'
os.environ['RC_DATA'] = 'rc_data/dose_response_scans'
import root_cause as rc  # noqa: E402  (reads RC_SERVER / RC_DATA at import)

sys.path.insert(0, '/path/to/llama.cpp/gguf-py')
import gguf  # noqa: E402
from gguf import quants  # noqa: E402

HERE = Path(__file__).resolve().parent
SRC = '/path/to/models/gemma-4/gemma-4-E4B_q4_0-it.gguf'
TEXTS = ['directs', 'directs now', 'directs then forks', 'forks', 'allow_redirects', 'redirects', 'treats', 'payloads', 'pulls', 'yields']
ppl_text, scratch, factors = sys.argv[1], Path(sys.argv[2]), [float(x) for x in sys.argv[3:]]
EPS = 1e-6

tensors = {t.name: t for t in gguf.GGUFReader(SRC).tensors}
W = np.array(tensors['per_layer_model_proj.weight'].data, dtype=np.float32)
ids = [int(i) for i in np.sort(np.random.default_rng(0).choice(262144, 2000, replace=False))]
E = np.concatenate([quants.dequantize(tensors['token_embd.weight'].data[i:i + 1], gguf.GGMLQuantizationType.Q6_K).astype(np.float32).reshape(1, -1) for i in ids])
V = ((E @ W.T).reshape(len(E), 42, 256) ** 2).mean(-1)          # mean square at the norm input for f = 1


def wait(port):
    for _ in range(120):
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=2)
            return
        except OSError:
            time.sleep(1)
    raise RuntimeError('server did not come up')


for f in factors:
    path = scratch / f'dose_{f:g}.gguf'
    if f == 1.0:
        path = Path(SRC)
    else:
        subprocess.run([sys.executable, str(HERE / 'gguf_scale_patch.py'), SRC, str(path), 'per_layer_model_proj.weight', repr(f)], check=True, capture_output=True)
    srv = subprocess.Popen(['llama-server', '-m', str(path), '--host', '127.0.0.1', '--port', str(PORT), '-ngl', '99', '-c', '2048', '-np', '1', '--jinja'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait(PORT)
        steps = {}
        for t in TEXTS:
            s = rc.scan(t)
            st = s['steps'][s['ref_toks'].index('s')]
            steps[t] = {'p_s': st['watch']['s'][0], 'rank_s': st['watch']['s'][1], 'p_eot': st['p_eot'], 'top': st['top'][0]['tok']}
    finally:
        srv.terminate()
        srv.wait()
    out = subprocess.run(['llama-perplexity', '-m', str(path), '-f', ppl_text, '-c', '512', '--chunks', '60', '-ngl', '99'], capture_output=True, text=True)
    m = re.search(r'Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)', out.stdout + out.stderr)
    g = np.sqrt(V * f * f / (V * f * f + EPS))
    rec = {'factor': f, 'ppl': float(m.group(1)) if m else None, 'ppl_err': float(m.group(2)) if m else None, 'gain_median': float(np.median(g)), 'gain_p5': float(np.percentile(g, 5)),
           'share_below_eps': float(((V * f * f) < EPS).mean()), 'scan': steps}
    print(f'x{f:<7g} PPL {rec["ppl"]} +/- {rec["ppl_err"]}   RMSNorm gain median {rec["gain_median"]:.3f} (5th pct {rec["gain_p5"]:.3f}), {rec["share_below_eps"]:.0%} of (token, layer) below eps   '
          + '  '.join(f'{t}:{v["p_s"]:.3f}' for t, v in steps.items()), flush=True)
    with open(rc.DATA.parent / 'dose_response.jsonl', 'a') as fh:
        fh.write(json.dumps({'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), **rec}) + '\n')
    if f != 1.0:
        path.unlink()
