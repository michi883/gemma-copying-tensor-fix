#!/usr/bin/env python3
"""Check this release without running any model.

    python3 verify_release.py              # --hashes and --data: Python 3.10+ standard library only, a few seconds
    python3 verify_release.py --figures    # also rebuild the three figures; needs numpy and matplotlib (requirements.txt)
    python3 verify_release.py --all

--hashes   every file against SHA256SUMS; the adapter against adapter/SHA256SUMS; each gzip'd result file against copying/discovery/UNCOMPRESSED.sha256 (the hash of the
           unzipped file that was scrubbed of local paths, see PROVENANCE.md).
--data     recompute the headline numbers from the per-item records: copy failures (words, plurals, code blocks), the Kaggle vLLM comparison (18 inputs), the seven coding
           tasks (pass/fail), the adapter file (tensor names, shapes, rank), and the cross-references between adapter, manifest and notebook.
--figures  rebuild the original repository layout in a temporary directory, run `figures/generate_figures.py --check` there (re-extracts every plotted number from the raw
           files and compares it with figure-data.json), then regenerate the six figure files from figure-data.json and compare their hashes with the shipped ones.

Exit status 0 only if every check passes.  Nothing in the release directory is modified.  This script is new for the release; the figure script it calls is the research one.
"""
import argparse
import gzip
import hashlib
import json
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FAILED = []


def check(ok, what, detail=''):
    print(('ok   ' if ok else 'FAIL ') + what + (f'   [{detail}]' if detail else ''))
    if not ok:
        FAILED.append(what)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def read_sums(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            h, name = line.split(None, 1)
            out[name.strip().lstrip('*')] = h
    return out


def jsonl(path):
    op = gzip.open if str(path).endswith('.gz') else open
    with op(path, 'rt', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------
def check_hashes():
    print('\n== hashes')
    sums = read_sums(ROOT / 'SHA256SUMS')
    bad = [n for n, h in sums.items() if not (ROOT / n).is_file() or sha(ROOT / n) != h]
    extra = sorted(str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and p.name != 'SHA256SUMS' and str(p.relative_to(ROOT)) not in sums and '__pycache__' not in p.parts and '.git' not in p.parts)
    check(not bad, f'SHA256SUMS: {len(sums)} files match', f'mismatch/missing: {bad[:5]}' if bad else '')
    check(not extra, 'no files outside SHA256SUMS', str(extra[:5]) if extra else '')
    a = read_sums(ROOT / 'adapter' / 'SHA256SUMS')
    check(all(sha(ROOT / 'adapter' / n) == h for n, h in a.items()) and len(a) == 2, 'adapter/SHA256SUMS: weights and config match', ', '.join(f'{n} {h[:8]}' for n, h in a.items()))
    un = read_sums(ROOT / 'copying' / 'discovery' / 'UNCOMPRESSED.sha256')
    bad = []
    for name, h in un.items():
        hh = hashlib.sha256()
        with gzip.open(ROOT / (name + '.gz'), 'rb') as f:
            for part in iter(lambda: f.read(1 << 20), b''):
                hh.update(part)
        if hh.hexdigest() != h:
            bad.append(name)
    check(not bad and len(un) == 7, f'copying/discovery/UNCOMPRESSED.sha256: {len(un)} unzipped result files match', str(bad))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------
def safetensors_header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return json.loads(f.read(n))


def check_data():
    print('\n== data')
    ok = lambda r: bool(r.get('exact_unfenced'))                       # pass = byte-exact (or exact inside one enclosing ``` fence), as in copying/analyze.py
    C = ROOT / 'copying' / 'discovery'

    # --- copy failures (Figure 1) ---------------------------------------------------------------------------------------------------------------------------------
    O, P = ({r['id']: r for r in jsonl(C / f'results_words_{a}_nothink.jsonl.gz')} for a in ('orig', 'patched'))
    s = [i for i, r in O.items() if r['group'] == 'sample']
    got = (sum(not ok(O[i]) for i in s), sum(not ok(P[i]) for i in s), len(s))
    check(got == (268, 1, 7688), 'words and identifiers: original fails / x2 fails / tested = 268 / 1 / 7688', str(got))
    O, P = ({r['id']: r for r in jsonl(C / f'results_plurals_{a}_nothink.jsonl.gz')} for a in ('orig', 'patched'))
    got = (sum(not ok(r) for r in O.values()), sum(not ok(r) for r in P.values()), len(O))
    check(got == (274, 1, 2000), 'English plurals: 274 / 1 / 2000', str(got))
    O, P = ({r['id']: r for r in jsonl(C / f'results_blocks_{a}_nothink.jsonl.gz')} for a in ('orig', 'patched'))
    retry = {r['id']: r for r in jsonl(C / 'results_retry_orig_nothink_b00835.jsonl.gz')}
    check('error' in O['b00835'] and 'output' in retry['b00835'], 'block b00835 timed out in the first original run and was re-run (counted as a failure)')
    O['b00835'] = retry['b00835']
    got = (sum(not ok(r) for r in O.values()), sum(not ok(r) for r in P.values()), len(O))
    check(got == (368, 15, 1278), 'Python code blocks: 368 / 15 / 1278 (incl. the b00835 re-run)', str(got))
    lone = [i for i, r in O.items() if r['has_lone_s']]
    got = (sum(not ok(O[i]) for i in lone), len(lone))
    check(got == (358, 528), 'blocks containing a lone-s token: original fails 358 of 528', str(got))
    m = json.loads((C / 'results_words_orig_nothink.meta.json').read_text())
    mp = json.loads((C / 'results_words_patched_nothink.meta.json').read_text())
    check(m['settings']['temperature'] == 0 and m['settings']['seed'] == 1 and m['mode'] == 'nothink' and m['build'] == mp['build'] == 'b11146-7fe450e19', 'run settings: temperature 0, seed 1, thinking off, llama.cpp b11146', m['build'])
    check(m['expected_sha256'].startswith('676c3507') and mp['expected_sha256'].startswith('3a6ad54e'), 'model files: original GGUF sha256 676c3507..., x2 GGUF sha256 3a6ad54e...')

    # --- tensor audit (the 665-tensor comparison, recorded output; the comparison itself needs the full weights, see REPRODUCE.md) -------------------------------
    rows = json.loads((ROOT / 'tensor' / 'results' / 'weights_scale_audit.json').read_text())['rows']
    odd = [r for r in rows if not 0.46 <= r['ratio'] <= 1.21]
    check(len(rows) == 665 and [r['gguf'] for r in odd] == ['per_layer_model_proj.weight'] and abs(odd[0]['ratio'] - 0.0155) < 5e-4,
          'tensor audit: 665 tensors compared with the standard release; the only one outside 0.46-1.21 of the standard scale is per_layer_model_proj.weight', f'ratio {odd[0]["ratio"]:.4f} = 1/{1 / odd[0]["ratio"]:.1f}' if odd else '')

    # --- adapter file -----------------------------------------------------------------------------------------------------------------------------------------------
    A = ROOT / 'adapter'
    cfg = json.loads((A / 'adapter_config.json').read_text())
    hdr = {k: v for k, v in safetensors_header(A / 'adapter_model.safetensors').items() if k != '__metadata__'}
    pre = 'base_model.model.model.language_model.per_layer_model_projection.'
    check(cfg['r'] == cfg['lora_alpha'] == 128 and cfg['target_modules'] == ['per_layer_model_projection'] and cfg['base_model_name_or_path'] == 'google/gemma-4-e4b-it-qat-w4a16-ct',
          'adapter config: rank 128, alpha 128, one target module, base google/gemma-4-e4b-it-qat-w4a16-ct')
    check(sorted(hdr) == [pre + 'lora_A.weight', pre + 'lora_B.weight'] and hdr[pre + 'lora_A.weight']['shape'] == [128, 2560] and hdr[pre + 'lora_B.weight']['shape'] == [10752, 128]
          and all(v['dtype'] == 'BF16' for v in hdr.values()), 'adapter weights: two BF16 tensors, A [128, 2560] and B [10752, 128]')
    wsha, csha = sha(A / 'adapter_model.safetensors'), sha(A / 'adapter_config.json')

    # --- Kaggle vLLM comparison ---------------------------------------------------------------------------------------------------------------------------------
    E = ROOT / 'kaggle' / 'evidence'
    with zipfile.ZipFile(E / 'copy_lora_evidence.zip') as z:
        names = sorted(z.namelist())
        same = all(z.read(n) == (E / 'extracted' / n).read_bytes() for n in names)
        res, man = json.loads(z.read('copy_test_lora_results.json')), json.loads(z.read('manifest.json'))
    check(same and len(names) == 6, 'Kaggle evidence: archive members identical to kaggle/evidence/extracted/', ', '.join(names))
    rows = res['results']
    summ = {'total': len(rows), 'base_failures': sum(not r['base_match'] for r in rows), 'lora_failures': sum(not r['lora_match'] for r in rows),
            'fixed': sum(not r['base_match'] and r['lora_match'] for r in rows), 'regressions': sum(r['base_match'] and not r['lora_match'] for r in rows)}
    check(all(r['base_match'] == (r['base']['output'] == r['input']) and r['lora_match'] == (r['lora']['output'] == r['input']) for r in rows), 'Kaggle: match flags equal exact string comparison of output and input')
    check({k: man['summary'][k] for k in summ} == summ and res['summary'] == man['summary'], 'Kaggle: summary recomputed from the 18 paired outputs equals the manifest', str(summ))
    check((summ['total'], summ['base_failures'], summ['lora_failures'], summ['fixed'], summ['regressions']) == (18, 15, 1, 14, 0), 'Kaggle: 18 inputs, base 15 failures, adapter 1, fixed 14, regressions 0')
    check(not any(r[k]['reasoning_present'] for r in rows for k in ('base', 'lora')) and all(r[k]['finish_reason'] == 'stop' for r in rows for k in ('base', 'lora')),
          'Kaggle: thinking off (no reasoning field) and every reply finished with "stop"')
    check(man['adapter_weights_sha256'] == wsha and man['adapter_config_sha256'] == csha and man['adapter_weights_bytes'] == (A / 'adapter_model.safetensors').stat().st_size,
          'Kaggle manifest hashes equal the released adapter files', f'weights {wsha[:8]}, config {csha[:8]}')
    check(sha(E / 'extracted' / 'adapter_config.json') == csha, 'adapter_config.json inside the Kaggle evidence equals the released one')
    check(man['model_name'] == 'gemma-4-e4b-it-qat-w4a16-ct' and man['model_path'].endswith('/gemma-4-e4b-it-qat-w4a16-ct/2') and man['vllm_version'] == '0.19.1' and man['tensor_parallel_size'] == 4
          and man['served_model_ids'] == ['gemma-4-e4b-it-qat-w4a16-ct', 'ple'] and man['request_settings'] == {'chat_template_kwargs': {'enable_thinking': False}, 'temperature': 0, 'seed': 1, 'max_tokens': 128},
          'Kaggle manifest: model gemma-4-e4b-it-qat-w4a16-ct (version 2), vLLM 0.19.1, tensor parallel 4, ids [base, ple], temperature 0, seed 1, thinking off')
    log = (E / 'extracted' / 'vllm_server.log').read_text(errors='replace')
    check("Loaded new LoRA adapter: name 'ple'" in log and log.count('POST /v1/chat/completions') == 36, 'Kaggle server log: adapter "ple" loaded; 36 chat requests = 18 inputs x 2 model ids')
    nb = json.loads((ROOT / 'kaggle' / 'copy-test-lora.ipynb').read_text())
    code = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
    out = '\n'.join(''.join(o.get('text', '')) for c in nb['cells'] if c['cell_type'] == 'code' for o in c.get('outputs', []))
    check('MODEL_NAME = "gemma-4-e4b-it-qat-w4a16-ct"' in code and '/gemma-4/other/{MODEL_NAME}/2' in code and 'ple-proj-svd-r128' in code and "'--max-lora-rank', '128'" in code and 'lora-modules' in code,
          'notebook code: model gemma-4-e4b-it-qat-w4a16-ct version 2, adapter dataset ple-proj-svd-r128, --max-lora-rank 128, --lora-modules')
    check(wsha in out and '"base_failures": 15' in out and '"lora_failures": 1' in out and '"fixed": 14' in out and '"regressions": 0' in out and [c.get('execution_count') for c in nb['cells'] if c['cell_type'] == 'code'] == [1, 2, 3, 4, 5, 6],
          'notebook outputs: executed cells 1-6 in order; printed adapter sha256 and summary equal the evidence')

    # --- seven coding tasks (Figure 3) -----------------------------------------------------------------------------------------------------------------------
    T = ROOT / 'coding-tasks'
    tasks = [t['id'] for t in json.loads((T / 'tasks.json').read_text())]
    passed = {arm: [json.loads((T / 'results' / 'runs' / arm / t / 'result.json').read_text())['tests']['passed'] for t in tasks] for arm in ('original', 'lora')}
    check(len(tasks) == 7 and sum(passed['original']) == 1 and sum(passed['lora']) == 5, 'coding tasks: original passes 1 of 7, merged adapter passes 5 of 7', f'original {passed["original"]}, adapter {passed["lora"]}')
    check(sum(a is False and b is True for a, b in zip(passed['original'], passed['lora'])) == 4 and sum(a is True and b is False for a, b in zip(passed['original'], passed['lora'])) == 0,
          'coding tasks: 4 flipped fail -> pass, 0 flipped pass -> fail')
    pv = ROOT / 'adapter' / 'results'
    for name, want in (('results_orig', None), ('results_x2', None), ('results_svd_r128', None), ('results_adapter_r128_merged_file', None)):
        check(len(jsonl(pv / f'{name}.jsonl')) == 427, f'adapter/results/{name}.jsonl: 427 records')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------
LAYOUT = [                                                   # (release path, path in the research repository that generate_figures.py expects); gz files are unzipped
    ('copying/discovery/plan_plurals.json', 'experiments/identifier_fidelity/discovery/plan_plurals.json'),
    ('figures/source-data/dose_gain_recompute.json', 'docs/figures/source/dose_gain_recompute.json'),
    ('figures/source-data/norm_input_check_stdout.txt', 'docs/figures/source/norm_input_check_stdout.txt'),
    ('kaggle/evidence/copy_lora_evidence.zip', 'docs/kaggle-vllm-evidence/copy_lora_evidence.zip'),
    ('tensor/results/weights_scale_audit.json', 'experiments/tensor_audit/weights_scale_audit.json'),
    ('tensor/results/activation_audit.json', 'experiments/tensor_audit/activation_audit.json'),
    ('tensor/results/dose_response.jsonl', 'experiments/identifier_fidelity/rc_data/dose_response.jsonl'),
    ('coding-tasks/results/analysis.json', 'gemma-edit-tasks/experiments/analysis.json'),
]
TREES = [('tensor/results/rc_data', 'experiments/identifier_fidelity/rc_data'), ('coding-tasks/results/runs', 'gemma-edit-tasks/experiments/runs'),
         ('coding-tasks/results/probes', 'gemma-edit-tasks/experiments/probes'), ('coding-tasks/results/replays', 'gemma-edit-tasks/experiments/replays'),
         ('coding-tasks/results/replays_sampled', 'gemma-edit-tasks/experiments/replays_sampled')]
FIGS = ['figure1-copying-errors.png', 'figure1-copying-errors.svg', 'figure2-tensor-mechanism.png', 'figure2-tensor-mechanism.svg', 'figure3-coding-agent.png', 'figure3-coding-agent.svg']


def check_figures():
    print('\n== figures')
    try:
        import matplotlib
        import numpy
    except ImportError as e:
        check(False, 'numpy and matplotlib are installed (pip install -r requirements.txt)', str(e))
        return
    print(f'     python {sys.version.split()[0]}, numpy {numpy.__version__}, matplotlib {matplotlib.__version__}  (the shipped figures were made with 3.14.8 / 2.5.4 / 3.11.2)')
    with tempfile.TemporaryDirectory(prefix='gemma-fig-check-') as tmp:
        tmp = Path(tmp)
        disc = tmp / 'experiments/identifier_fidelity/discovery'
        disc.mkdir(parents=True)
        for gz in sorted((ROOT / 'copying' / 'discovery').glob('results_*.jsonl.gz')):
            with gzip.open(gz, 'rb') as f, open(disc / gz.stem, 'wb') as g:
                shutil.copyfileobj(f, g)
        for meta in sorted((ROOT / 'copying' / 'discovery').glob('*.meta.json')):
            shutil.copy2(meta, disc / meta.name)
        for src, dst in LAYOUT:
            (tmp / dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / src, tmp / dst)
        for src, dst in TREES:
            shutil.copytree(ROOT / src, tmp / dst, dirs_exist_ok=True)
        fig = tmp / 'docs/figures'
        for n in ('generate_figures.py', 'figure-data.json'):
            shutil.copy2(ROOT / 'figures' / n, fig / n)
        r = subprocess.run([sys.executable, str(fig / 'generate_figures.py'), '--check'], capture_output=True, text=True)
        print('     ' + ' | '.join(r.stdout.strip().splitlines()))
        check(r.returncode == 0, 'generate_figures.py --check: every plotted number re-extracted from the raw files equals figure-data.json, and all claim checks pass', r.stderr.strip()[-300:])
        r = subprocess.run([sys.executable, str(fig / 'generate_figures.py'), '--plot-only'], capture_output=True, text=True)
        check(r.returncode == 0, 'generate_figures.py --plot-only: renders the six files from figure-data.json (no raw data, no model)', r.stderr.strip()[-300:])
        for n in FIGS:
            same = (fig / n).is_file() and sha(fig / n) == sha(ROOT / 'figures' / n)
            if n.endswith('.svg') or same:
                check(same, f'regenerated {n} is byte-identical to the shipped file')
            else:
                print(f'warn {n}: regenerated PNG differs byte-wise from the shipped one (anti-aliasing may vary across platforms/library builds); compare the images visually')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--hashes', action='store_true')
    ap.add_argument('--data', action='store_true')
    ap.add_argument('--figures', action='store_true')
    ap.add_argument('--all', action='store_true')
    a = ap.parse_args()
    everything = a.all or not (a.hashes or a.data or a.figures)
    if a.hashes or a.all or everything:
        check_hashes()
    if a.data or a.all or everything:
        check_data()
    if a.figures or a.all:
        check_figures()
    print('\n' + ('ALL CHECKS PASSED' if not FAILED else f'{len(FAILED)} CHECK(S) FAILED'))
    sys.exit(1 if FAILED else 0)


if __name__ == '__main__':
    main()
