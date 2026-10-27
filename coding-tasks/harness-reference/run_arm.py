#!/usr/bin/env python3
"""Run harness.py for the given tasks against the server that is up on :8110 and file the trajectories under runs/<arm>/<task>/.

    python3 run_arm.py ARM [TASK_ID ...]        # default: all seven; ARM is a label (original | lora | ...); refuses to overwrite an existing run
"""
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = 'python'
import os
URL = os.environ.get('GEMMA_URL', 'http://127.0.0.1:8110/v1')              # default: the local llama-server of serve.sh; for the vLLM pod see REPORT.md section "Smallest vLLM experiment"
MODEL = os.environ.get('GEMMA_MODEL', 'gemma-4-e4b')                       # the `model` field of every request (vLLM: the base model id, or the LoRA module name)
EXTRA = os.environ.get('GEMMA_EXTRA_BODY')                                 # JSON; default = harness.EXTRA_BODY (llama-server)
arm, ids = sys.argv[1], sys.argv[2:]
tasks = ids or [t['id'] for t in json.loads((HERE.parent / 'tasks.json').read_text())]
if URL.startswith('http://127.0.0.1:8110'):
    props = json.load(urllib.request.urlopen('http://127.0.0.1:8110/props', timeout=10))
    expect = {'original': 'gemma-4-E4B_q4_0-it.gguf', 'lora': 'gemma-4-E4B_q4_0-it-lora-r128-merged.gguf', 'x2': 'gemma-4-E4B_q4_0-it-plproj-x2.gguf'}.get(arm.split('_')[0])
    assert expect is None or props['model_path'].endswith('/' + expect), f'arm {arm} expects {expect} but the server on :8110 serves {props["model_path"]}'
work = Path('/tmp/gemma-edit-work')
work.mkdir(exist_ok=True)
for t in tasks:
    out = HERE / 'runs' / arm / t
    if out.exists():
        sys.exit(f'{out} exists; not overwriting')
    r = subprocess.run([PY, '-I', str(HERE / 'harness.py'), t, '--arm', arm, '--url', URL, '--model', MODEL, '--out', str(out), '--work-root', str(work)] + (['--extra-body', EXTRA] if EXTRA else []), capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip()[-1500:], flush=True)
    if r.returncode:
        print(r.stderr[-3000:], file=sys.stderr)
