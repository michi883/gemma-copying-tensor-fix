#!/usr/bin/env python3
"""Regenerate the three figures of "Fixing Gemma's Copying Errors by Changing One Tensor".

    python3 generate_figures.py              extract every plotted number from the raw experiment files, write figure-data.json, check the numbers against the
                                             paper draft / reports, render the six files (3 x SVG + PNG)
    python3 generate_figures.py --plot-only  render from the existing figure-data.json (no raw experiment files needed)
    python3 generate_figures.py --check      extract again and compare with figure-data.json; render nothing (exit status 1 on any mismatch)

Needs Python 3.10+, numpy and matplotlib (see README.md).  Extraction reads only files inside this repository; nothing is run against a model.
Every figure number is recomputed from raw records (per-item result files, per-run results, probability probes), except where figure-data.json marks it
`report-derived`.  Nothing is interpolated: lines in the figures only connect measured points.
"""
import argparse
import json
import math
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
IDF = REPO / 'experiments' / 'identifier_fidelity'
DISC, RC = IDF / 'discovery', IDF / 'rc_data'
AUDIT = REPO / 'experiments' / 'tensor_audit'
EDIT = REPO / 'gemma-edit-tasks' / 'experiments'
KAGGLE_ZIP = REPO / 'docs' / 'kaggle-vllm-evidence' / 'copy_lora_evidence.zip'
DATA_JSON = HERE / 'figure-data.json'
EPS = 1e-6
TEXTS10 = ['directs', 'directs now', 'directs then forks', 'forks', 'allow_redirects', 'redirects', 'treats', 'payloads', 'pulls', 'yields']


def rel(p):
    return str(Path(p).resolve().relative_to(REPO))


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def tally(k0, k1, n):
    """failures of the original (k0) and of the x2 model (k1) on the same n items"""
    return {'n': n, 'original_fail': k0, 'x2_fail': k1, 'original_rate': k0 / n, 'x2_rate': k1 / n, 'original_ci95': wilson(k0, n), 'x2_ci95': wilson(k1, n)}


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Figure 1: copy failures by tokenization group (per-item results of experiments/identifier_fidelity/discovery)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
def extract_figure1():
    ok = lambda r: bool(r.get('exact_unfenced'))                       # pass = byte-exact (or exact inside one enclosing ``` fence), as in analyze.py
    by_id = lambda rows: {r['id']: r for r in rows}

    # --- words and identifiers -------------------------------------------------------------------------------------------------------------------------
    O = by_id(load_jsonl(DISC / 'results_words_orig_nothink.jsonl'))
    P = by_id(load_jsonl(DISC / 'results_words_patched_nothink.jsonl'))
    sample = sorted(i for i, r in O.items() if r['group'] == 'sample')
    assert sample == sorted(i for i, r in P.items() if r['group'] == 'sample') and all(O[i]['text'] == P[i]['text'] for i in sample)
    per_stratum = Counter(O[i]['stratum'] for i in sample)
    census = lambda i: O[i]['stratum_population'] == per_stratum[O[i]['stratum']]      # stratum tested exhaustively
    grp = lambda i: ('lone_s_literal_underscore' if O[i]['text'].endswith('_s') else 'lone_s_suffix') if O[i]['cls'] == 'S' else 'other'
    words = {}
    for g in ('lone_s_suffix', 'lone_s_literal_underscore', 'other'):
        ids = [i for i in sample if grp(i) == g]
        words[g] = tally(sum(not ok(O[i]) for i in ids), sum(not ok(P[i]) for i in ids), len(ids))
    other = [i for i in sample if grp(i) == 'other']
    words['other']['by_token_class'] = {c: tally(sum(not ok(O[i]) for i in other if O[i]['cls'] == c), sum(not ok(P[i]) for i in other if O[i]['cls'] == c),
                                                 sum(O[i]['cls'] == c for i in other)) for c in ('ONE', 'SFX', 'OTHER')}
    words['other']['items_from_exhaustively_tested_strata'] = sum(census(i) for i in other)
    words['other']['items_from_random_draws'] = sum(not census(i) for i in other)
    words['lone_s_suffix']['selection'] = 'census: every such name in the scanned corpus (Python 3.14 stdlib + site-packages)'
    assert all(census(i) for i in sample if grp(i) != 'other')
    anchors = [i for i, r in O.items() if r['group'] == 'anchor_fail']
    words_total = tally(sum(not ok(O[i]) for i in sample), sum(not ok(P[i]) for i in sample), len(sample))
    known_failure_controls = {'n': len(anchors), 'original_fail': sum(not ok(O[i]) for i in anchors), 'x2_fail': sum(not ok(P[i]) for i in anchors),
                              'original_passes': sorted(O[i]['text'] for i in anchors if ok(O[i]))}
    ex_o = next(r for r in O.values() if r['text'] == 'directs' and r['group'] == 'anchor_fail')
    ex_p = next(r for r in P.values() if r['text'] == 'directs' and r['group'] == 'anchor_fail')
    example_copy = {'input': 'directs', 'original_output': ex_o['output'], 'x2_output': ex_p['output'], 'original_exact': ok(ex_o), 'x2_exact': ok(ex_p)}
    assert example_copy['original_output'] == 'direct' and example_copy['x2_output'] == 'directs' and not example_copy['original_exact'] and example_copy['x2_exact']
    graphs = next(r for r in O.values() if r['text'] == 'graphs' and r['group'] == 'sample')
    assert graphs['n_tokens'] == 1 and graphs['cls'] == 'ONE'
    seen = {}                                                          # how the tokenizer splits the three example words (teacher-forced scan records store the reference tokens)
    for f_ in sorted(RC.glob('**/scan.jsonl')):
        for r in load_jsonl(f_):
            if r['text'] in ('directs', 'directing'):
                seen.setdefault(r['text'], r['ref_toks'])
    tokenization = {'directs': seen['directs'], 'directing': seen['directing'], 'graphs': [graphs['last_piece']]}
    assert tokenization == {'directs': ['direct', 's'], 'directing': ['direct', 'ing'], 'graphs': ['graphs']}
    meta = json.loads((DISC / 'results_words_orig_nothink.meta.json').read_text())
    meta_x2 = json.loads((DISC / 'results_words_patched_nothink.meta.json').read_text())
    assert meta['mode'] == meta_x2['mode'] == 'nothink' and meta['settings']['temperature'] == 0 and meta['settings']['seed'] == 1 and meta['build'] == meta_x2['build']
    run_settings = {'mode': meta['mode'], 'temperature': meta['settings']['temperature'], 'seed': meta['settings']['seed'], 'llama_cpp_build': meta['build'], 'prompt': meta['prompt'],
                    'original_model_sha256': meta['expected_sha256'], 'x2_model_sha256': meta_x2['expected_sha256']}

    # --- plurals -------------------------------------------------------------------------------------------------------------------------------------------
    O = by_id(load_jsonl(DISC / 'results_plurals_orig_nothink.jsonl'))
    P = by_id(load_jsonl(DISC / 'results_plurals_patched_nothink.jsonl'))
    pop = json.loads((DISC / 'plan_plurals.json').read_text())['population']
    plurals = {}
    for key, cls in (('lone_s', 'S'), ('other', 'NOT_S')):
        ids = [i for i, r in O.items() if r['cls'] == cls]
        plurals[key] = tally(sum(not ok(O[i]) for i in ids), sum(not ok(P[i]) for i in ids), len(ids))
        plurals[key]['population'] = pop[cls]
        assert {O[i]['stratum_population'] for i in ids} == {pop[cls]}
    plurals_total = tally(sum(not ok(r) for r in O.values()), sum(not ok(r) for r in P.values()), len(O))
    plurals_pop = {'candidates': pop['stems'], 'lone_s_share_of_candidates': pop['S'] / pop['stems']}

    # --- code blocks ---------------------------------------------------------------------------------------------------------------------------------------
    O = by_id(load_jsonl(DISC / 'results_blocks_orig_nothink.jsonl'))
    P = by_id(load_jsonl(DISC / 'results_blocks_patched_nothink.jsonl'))
    retry = by_id(load_jsonl(DISC / 'results_retry_orig_nothink_b00835.jsonl'))
    assert 'error' in O['b00835'] and 'output' in retry['b00835'] and retry['b00835']['text'] == O['b00835']['text']
    O['b00835'] = retry['b00835']                                      # the one block that timed out at 180 s was re-run with a longer limit (never stops copying: counted as a failure)
    assert set(O) == set(P) and all('error' not in r for r in O.values()) and all('error' not in r for r in P.values())
    blocks = {}
    for key, flag in (('lone_s', True), ('other', False)):
        ids = [i for i, r in O.items() if r['has_lone_s'] is flag]
        blocks[key] = tally(sum(not ok(O[i]) for i in ids), sum(not ok(P[i]) for i in ids), len(ids))
        blocks[key]['by_lines'] = {str(k): tally(sum(not ok(O[i]) for i in ids if O[i]['k'] == k), sum(not ok(P[i]) for i in ids if O[i]['k'] == k), sum(O[i]['k'] == k for i in ids))
                                   for k in (1, 4, 10)}
    blocks_total = tally(sum(not ok(r) for r in O.values()), sum(not ok(r) for r in P.values()), len(O))
    blocks_total['block_b00835_counted_as_failure_after_rerun'] = True
    blocks_total['paired_n_if_b00835_excluded'] = len(O) - 1           # analysis_blocks_nothink.json (analyze.py) pairs 1,277 blocks and reports 367 / 15

    src = lambda *p: [x if isinstance(x, str) else rel(x) for x in p]
    return {
        'basis': 'recomputed from per-item raw results (pass = exact_unfenced); arms: original QAT E4B Q4_0 GGUF vs the same GGUF with per_layer_model_proj.weight x2 (direct scaling)',
        'sources': src(DISC / 'results_words_orig_nothink.jsonl', DISC / 'results_words_patched_nothink.jsonl', DISC / 'plan_words.json', DISC / 'results_plurals_orig_nothink.jsonl',
                       DISC / 'results_plurals_patched_nothink.jsonl', DISC / 'plan_plurals.json', DISC / 'results_blocks_orig_nothink.jsonl',
                       DISC / 'results_blocks_patched_nothink.jsonl', DISC / 'results_retry_orig_nothink_b00835.jsonl', DISC / 'results_words_orig_nothink.meta.json',
                       DISC / 'results_words_patched_nothink.meta.json', rel(RC) + '/**/scan.jsonl (tokenization examples)', DISC / 'REPORT.md'),
        'tokenization_examples': tokenization, 'example_copy': example_copy, 'run_settings': run_settings,
        'words': {'groups': words, 'total': words_total, 'known_failure_controls': known_failure_controls,
                  'selection': 'deliberately enriched: all lone-s names (census) + stratified samples of other token types; pooled rate is not a population rate'},
        'plurals': {'groups': plurals, 'total': plurals_total, 'population': plurals_pop,
                    'selection': 'seeded random draw within each group; group sizes fixed by design (lone-s over-represented 75% vs 8.6% of the list)'},
        'blocks': {'groups': blocks, 'total': blocks_total,
                   'selection': 'random line windows within six cells (1/4/10 lines x lone-s yes/no), balanced by design (cells with too few lone-s windows are smaller); a quarter from the in-domain requests source'},
    }


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Figure 2: dose-response of the tensor multiplier; healthy-model comparison; static weight statistics
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
def s_step(rec):
    """(p, rank) of the correct lone `s` at the step that must emit it, in a teacher-forced scan record"""
    st = rec['steps'][rec['ref_toks'].index('s')]
    return st['p_want'], st['rank_want']


def scan_summary(path, variant=None):
    last, spread = {}, defaultdict(list)
    for r in load_jsonl(path):
        if (variant is not None and r.get('variant') != variant) or r['text'] not in TEXTS10 or 's' not in r['ref_toks']:
            continue
        p, rank = s_step(r)
        last[r['text']] = (p, rank)
        spread[r['text']].append(p)
    assert set(last) == set(TEXTS10), (path, variant, sorted(last))
    return {'p_s': {t: last[t][0] for t in TEXTS10}, 'rank_s': {t: last[t][1] for t in TEXTS10}, 'n_top1': sum(last[t][1] == 1 for t in TEXTS10),
            'max_rerun_spread': max(max(v) - min(v) for v in spread.values())}


def parse_norm_check(path):
    """per-model blocks of norm_input_check.py's stdout -> {label: {...}}"""
    out, label = {}, None
    for line in Path(path).read_text().splitlines():
        if re.match(r'^(E4B|E2B)\s', line):
            label = line.strip()
            out[label] = {}
        m = re.search(r'\|W\|_F\s+([\d.]+)', line)
        if m and label:
            out[label]['frobenius_norm'] = float(m.group(1))
        m = re.search(r'2000 random tokens: median over tokens of the layer-median mean\(x\^2\) ([\d.e+-]+)\s+share of \(token, layer\) pairs with mean\(x\^2\) < eps: ([\d.]+)%\s+gain: median ([\d.]+), 5th percentile ([\d.]+)', line)
        if m and label:
            out[label].update(ms_median=float(m.group(1)), share_below_eps=float(m.group(2)) / 100, gain_median=float(m.group(3)), gain_p5=float(m.group(4)))
    return out


def extract_figure2():
    dose_rows = load_jsonl(RC / 'dose_response.jsonl')
    recompute = json.loads((HERE / 'source' / 'dose_gain_recompute.json').read_text())
    rc = {round(r['factor'], 6): r for r in recompute['rows']}
    dose, max_diff = [], 0.0
    for r in dose_rows:
        ctx = {t: {'p_s': r['scan'][t]['p_s'], 'rank_s': r['scan'][t]['rank_s'], 'top_choice': r['scan'][t]['top']} for t in TEXTS10}
        chk = rc[round(r['factor'], 6)]
        for a, b in ((r['gain_median'], chk['gain_median']), (r['gain_p5'], chk['gain_p5']), (r['share_below_eps'], chk['share_below_eps'])):
            max_diff = max(max_diff, abs(a - b))
        ps = sorted(c['p_s'] for c in ctx.values())
        dose.append({'factor': r['factor'], 'perplexity': r['ppl'], 'perplexity_stderr': r['ppl_err'], 'gain_median': r['gain_median'], 'gain_p5': r['gain_p5'],
                     'share_below_floor': r['share_below_eps'], 'ms_median_recomputed': chk['ms_median_of_layer_medians'], 'contexts': ctx,
                     'n_top1': sum(c['rank_s'] == 1 for c in ctx.values()), 'p_s_median': (ps[4] + ps[5]) / 2, 'p_s_min': ps[0], 'p_s_max': ps[-1]})
    assert max_diff < 1e-3, max_diff

    cond = lambda key, label, tensor, backend, summ, src: {'key': key, 'label': label, 'tensor_scale': tensor, 'backend': backend, 'source': src, **summ}
    x1 = next(d for d in dose if d['factor'] == 1.0)
    x2 = next(d for d in dose if d['factor'] == 2.0)
    from_dose = lambda d: {'p_s': {t: d['contexts'][t]['p_s'] for t in TEXTS10}, 'rank_s': {t: d['contexts'][t]['rank_s'] for t in TEXTS10}, 'n_top1': d['n_top1'], 'max_rerun_spread': 0.0}
    mlx = RC / 'mlx_scan.jsonl'
    comparison = [
        cond('qat_x1', '4-bit-ready (QAT) E4B as shipped', '1', 'llama.cpp, Q4_0', from_dose(x1), rel(RC / 'dose_response.jsonl')),
        cond('qat_x2', 'same, tensor x2', '2', 'llama.cpp, Q4_0', from_dose(x2), rel(RC / 'dose_response.jsonl')),
        cond('std_q4_0', 'standard E4B release, 4-bit rounded', '1', 'llama.cpp, Q4_0', scan_summary(RC / 'nonqat_q4_0_llamacpp' / 'scan.jsonl'), rel(RC / 'nonqat_q4_0_llamacpp' / 'scan.jsonl')),
        cond('std_q4_0_small', 'same, tensor made 65x smaller (x0.0153)', '0.0153', 'llama.cpp, Q4_0', scan_summary(RC / 'ptq_proj_x0153' / 'scan.jsonl'), rel(RC / 'ptq_proj_x0153' / 'scan.jsonl')),
        cond('std_bf16', 'standard E4B release, full precision', '1', 'mlx, bf16', scan_summary(mlx, 'bf16 (non-QAT)'), rel(mlx)),
        cond('std_bf16_small', 'same, tensor made 65x smaller (x0.0153)', '0.0153', 'mlx, bf16', scan_summary(mlx, 'bf16 (non-QAT), per_layer_model_projection x0.0153'), rel(mlx)),
    ]
    e2b = scan_summary(RC / 'e2b_q4_0' / 'scan.jsonl')

    norm = parse_norm_check(HERE / 'source' / 'norm_input_check_stdout.txt')
    pick = lambda prefix: next(v for k, v in norm.items() if k.startswith(prefix))
    stats = {'qat_e4b_shipped': pick('E4B  QAT GGUF as shipped'), 'qat_e4b_x_sqrt2560': pick('E4B  QAT GGUF with'), 'standard_e4b': pick('E4B  non-QAT bf16'),
             'qat_e2b_shipped': pick('E2B  QAT GGUF as shipped'), 'standard_e2b': pick('E2B  HF base')}
    assert all(len(v) == 5 for v in stats.values())

    w = json.loads((AUDIT / 'weights_scale_audit.json').read_text())['rows']
    ours = next(r for r in w if r['gguf'] == 'per_layer_model_proj.weight')
    others = [r['ratio'] for r in w if r is not ours]
    a = json.loads((AUDIT / 'activation_audit.json').read_text())['norms']
    fam_min = {k: v['ms_min'] for k, v in a.items() if k not in ('per_layer_projection_norm', 'layers.N.post_per_layer_input_norm')}
    audit = {'n_tensors_compared': len(w), 'tensor_rms_ratio_qat_over_standard': ours['ratio'], 'tensor_rms_qat': ours['rms_qat'], 'tensor_rms_standard': ours['rms_base'],
             'next_smallest_ratio_other_tensors': min(others), 'largest_ratio_other_tensors': max(others),
             'norm_families_checked': len(a), 'floor_norm_ms_median_6_sequences': a['per_layer_projection_norm']['ms_median'],
             'floor_norm_share_below_floor_6_sequences': a['per_layer_projection_norm']['below_eps'],
             'other_norm_families_min_ms': fam_min, 'other_norm_families_min_ms_over_floor': min(fam_min.values()) / EPS,
             'note_excluded_family': 'layers.N.post_per_layer_input_norm has median 3.8 and one structural zero (tensor_audit/REPORT.md B), not a floor case'}
    return {
        'basis': 'dose_response.jsonl = recorded raw scan values (teacher-forced p of the correct s, one scoring request per context); gain / share below floor = computed from the stored weights '
                 'for 2,000 random tokens x 42 layers and re-verified by verify_dose_gain.py; healthy-model counts recomputed from raw scan files; Frobenius norms / mean squares from norm_input_check.py',
        'sources': [rel(RC / 'dose_response.jsonl'), rel(HERE / 'source' / 'dose_gain_recompute.json'), rel(HERE / 'source' / 'norm_input_check_stdout.txt'),
                    rel(RC / 'nonqat_q4_0_llamacpp' / 'scan.jsonl'), rel(RC / 'ptq_proj_x0153' / 'scan.jsonl'), rel(mlx), rel(RC / 'e2b_q4_0' / 'scan.jsonl'),
                    rel(AUDIT / 'weights_scale_audit.json'), rel(AUDIT / 'activation_audit.json'), rel(IDF / 'ROOT_CAUSE.md'), rel(AUDIT / 'REPORT.md')],
        'floor': EPS, 'contexts': TEXTS10, 'n_random_tokens': recompute['n_tokens'], 'n_layers': recompute['n_layers'],
        'dose_response': dose, 'dose_gain_recompute_max_abs_diff': max_diff, 'healthy_comparison': comparison, 'e2b_qat_shipped_scan': e2b,
        'weight_statistics': stats, 'tensor_audit': audit,
        'ppl_note': 'perplexity of 60 chunks x 512 tokens of one text (llama-perplexity); +/- is the tool\'s standard error; the paired difference between multipliers was not recorded',
        'p_s_note': 'p_s is the teacher-forced probability of the correct lone s at the step where it must be written; x0.25 values are recorded as 0.0 (rounded to 5 decimals; rank 179,316-262,144 of 262,144)',
    }


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Figure 3: seven edit tasks, original vs merged adapter; the `diagrams` same-context probe; Kaggle vLLM copy test (reported separately, not plotted)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
TASKS = ['01_directs', '02_aligns', '03_extracts', '04_treats', '05_diagrams', '06_payloads', '07_accepts']


def added_lines(patch):
    return [l[1:] for l in patch.splitlines() if l.startswith('+') and not l.startswith('+++')]


def extract_figure3():
    A = json.loads((EDIT / 'analysis.json').read_text())
    # what each arm wrote: checked against the final patch of every run (regex over added lines, must match)
    wrote = {
        ('01_directs', 'original'): (r'"direct"', 'direct'), ('03_extracts', 'original'): (r'extract_s\(', 'extract_s'), ('03_extracts', 'lora'): (r'extract_s\(', 'extract_s'),
        ('05_diagrams', 'original'): (r'diagram_list', 'diagram_list'), ('05_diagrams', 'lora'): (r'"diagrams": document\.get\("diagrams"', 'diagrams'),
        ('06_payloads', 'original'): (r'for item in payload:', 'payload'), ('06_payloads', 'lora'): (r'in payloads\]', 'payloads'),
        ('07_accepts', 'original'): (r'\{"accept", ', 'accept'),
    }
    tasks = {}
    for t in TASKS:
        row = {}
        for arm in ('original', 'lora'):
            res = json.loads((EDIT / 'runs' / arm / t / 'result.json').read_text())
            patch = (EDIT / 'runs' / arm / t / 'final.patch').read_text()
            a = A[t][arm]
            assert res['tests']['passed'] == a['tests_passed']
            cell = {'passed': bool(res['tests']['passed']), 'tests': res['tests']['summary'], 'end_reason': res['end_reason'], 'turns': res['turns'], 'tool_calls': res['tool_calls_used'],
                    'edit_calls': len(a['edits']), 'edit_calls_applied': sum(1 for e in a['edits'] if e.get('ok')), 'final_patch_lines_added': len(added_lines(patch)),
                    'test_files_modified_by_agent': res['tests']['test_files_modified_by_agent']}
            if (t, arm) in wrote:
                rx, shown = wrote[(t, arm)]
                assert any(re.search(rx, l) for l in added_lines(patch)), (t, arm, 'final patch does not contain', rx)
                cell['final_patch_shows'] = shown
            if t == '02_aligns':
                assert cell['final_patch_lines_added'] == 0 and res['end_reason'].startswith('finish_reason=length')
                cell['final_patch_shows'] = 'no edit (repetition loop to the 8,192-token limit)'
            row[arm] = cell
        tasks[t] = row
    assert all(A[t][arm]['reasoning_chars_total'] == 0 for t in TASKS for arm in ('original', 'lora'))      # thinking was off in all 14 runs
    n_pass = {arm: sum(tasks[t][arm]['passed'] for t in TASKS) for arm in ('original', 'lora')}
    flips = {'fail_to_pass': [t for t in TASKS if not tasks[t]['original']['passed'] and tasks[t]['lora']['passed']],
             'pass_to_fail': [t for t in TASKS if tasks[t]['original']['passed'] and not tasks[t]['lora']['passed']]}

    # --- the `diagrams` same-context probe ---------------------------------------------------------------------------------------------------------------------
    po = json.loads((EDIT / 'probes' / 'edit_original__ctx-original__05_diagrams.json').read_text())
    pl = json.loads((EDIT / 'probes' / 'edit_lora__ctx-original__05_diagrams.json').read_text())
    assert po['request_matches_recorded'] and pl['request_matches_recorded'] and po['decisive_turn'] == pl['decisive_turn'] == 4
    so, sl = po['stream'], pl['stream']
    k = next(i for i, (x, y) in enumerate(zip(so, sl)) if x['token'] != y['token'])
    assert all(x['token'] == y['token'] for x, y in zip(so[:k], sl[:k]))      # identical generated prefix up to the divergence
    assert ''.join(x['token'] for x in so[max(0, k - 5):k]).endswith('document.get("diagram')
    dist = lambda s: {t: math.exp(lp) for t, lp in s[k]['top']}
    do, dl = dist(so), dist(sl)
    toks = ['",', '_', 's']
    probe = {'prefix_tokens_before_divergence': k, 'next_token_after': 'document.get("diagram', 'tokens_shown': toks,
             'original': {'chosen': so[k]['token'], 'p': {t: do[t] for t in toks}}, 'merged_adapter': {'chosen': sl[k]['token'], 'p': {t: dl[t] for t in toks}},
             'original_edit_old_string': json.loads(po['generated_call']['arguments'])['old_string'], 'merged_adapter_edit_old_string': json.loads(pl['generated_call']['arguments'])['old_string']}
    rep = lambda arm, d, ctx='original': json.loads((EDIT / d / f'{arm}__ctx-{ctx}__05_diagrams.json').read_text())
    r0, r1 = rep('original', 'replays')['samples'][0], rep('lora', 'replays')['samples'][0]
    probe['replay_temperature_0'] = {'original': {k_: r0[k_] for k_ in ('applied', 'tool_error', 'tests')}, 'merged_adapter': {k_: r1[k_] for k_ in ('applied', 'tool_error', 'tests')}}
    assert r0['args']['old_string'] == probe['original_edit_old_string'] and r1['args']['old_string'] == probe['merged_adapter_edit_old_string']
    samp = lambda arm: rep(arm, 'replays_sampled')
    s0, s1 = samp('original'), samp('lora')
    n = lambda s, key: sum(bool(x.get(key)) for x in s['samples'])                    # a reply without any edit call has no `applied` field: counted as not applied
    probe['replay_sampled_temperature_0_2'] = {'n_seeds': len(s0['samples']), 'temperature': s0['temperature'], 'original_edit_calls_produced': n(s0, 'produced_edit_call'),
                                               'merged_adapter_edit_calls_produced': n(s1, 'produced_edit_call'), 'original_applied': n(s0, 'applied'), 'merged_adapter_applied': n(s1, 'applied'),
                                               'original_tests_pass': n(s0, 'tests_pass_after_this_call'), 'merged_adapter_tests_pass': n(s1, 'tests_pass_after_this_call')}
    isolated = next(r for r in load_jsonl(DISC / 'results_words_orig_nothink.jsonl') if r['text'] == 'diagrams' and r['group'] == 'anchor_fail')
    probe['diagrams_copied_alone_by_original'] = bool(isolated['exact_unfenced'])

    # --- all sampled replays (paper: 23/120 vs 68/120) --------------------------------------------------------------------------------------------------------
    tot = {'original': [0, 0], 'lora': [0, 0]}
    for arm in tot:
        for ctx in ('original', 'lora'):
            for t in TASKS:
                f = EDIT / 'replays_sampled' / f'{arm}__ctx-{ctx}__{t}.json'
                if t == '02_aligns':                                       # no edit call in either trajectory: nothing to replay
                    assert not f.exists()
                    continue
                for s in json.loads(f.read_text())['samples']:
                    tot[arm][0] += bool(s.get('tests_pass_after_this_call'))
                    tot[arm][1] += 1
    # --- Kaggle vLLM copy test (a third, separate arm; not plotted) -------------------------------------------------------------------------------------------
    with zipfile.ZipFile(KAGGLE_ZIP) as z:
        kr = json.loads(z.read('copy_test_lora_results.json'))
        km = json.loads(z.read('manifest.json'))
    kaggle = {'summary': kr['summary'], 'inputs': [{'input': r['input'], 'base_ok': r['base_match'], 'adapter_ok': r['lora_match']} for r in kr['results']],
              'vllm_version': km['vllm_version'], 'tensor_parallel_size': km['tensor_parallel_size'], 'adapter_weights_sha256': km['adapter_weights_sha256'],
              'adapter_rank': km['adapter_rank'], 'model_name': km['model_name'], 'selection': 'hand-picked copy inputs, not a random sample; no control set',
              'sources': [rel(KAGGLE_ZIP) + ':copy_test_lora_results.json', rel(KAGGLE_ZIP) + ':manifest.json']}
    assert kaggle['summary']['total'] == len(kaggle['inputs']) == 18
    return {
        'basis': 'recomputed from per-run result.json / analysis.json / final.patch and the same-context probes; arms: original QAT E4B Q4_0 GGUF vs the same GGUF with the rank-128 SVD adapter MERGED into '
                 'per_layer_model_proj.weight (local llama.cpp); thinking off, temperature 0, one deterministic run per task and arm',
        'sources': [rel(EDIT / 'analysis.json'), rel(EDIT / 'runs'), rel(EDIT / 'probes' / 'edit_original__ctx-original__05_diagrams.json'), rel(EDIT / 'probes' / 'edit_lora__ctx-original__05_diagrams.json'),
                    rel(EDIT / 'replays'), rel(EDIT / 'replays_sampled'), rel(EDIT / 'REPORT.md'), rel(EDIT / 'stream_divergence.md')],
        'tasks': tasks, 'n_tasks': len(TASKS), 'passed': n_pass, 'flips': flips, 'reasoning_chars_in_all_14_runs': 0,
        'diagrams_probe': probe,
        'sampled_replays_all_tasks': {'edits_per_model': tot['original'][1], 'original_tests_pass': tot['original'][0], 'merged_adapter_tests_pass': tot['lora'][0],
                                      'note': '6 tasks with an edit x 2 contexts x 10 seeds, temperature 0.2; repeated edits in fixed contexts, not independent tasks'},
        'kaggle_vllm_adapter': kaggle,
    }


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# claims: numbers printed in the paper draft / reports, compared with the recomputed values
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
def check_claims(d):
    f1, f2, f3 = d['figure1'], d['figure2'], d['figure3']
    g = f1['words']['groups']
    dose = {x['factor']: x for x in f2['dose_response']}
    hc = {c['key']: c for c in f2['healthy_comparison']}
    pr = f3['diagrams_probe']
    wt = f2['weight_statistics']
    k = d['kaggle_vllm_adapter']['summary']
    C = []
    add = lambda what, got, want, src: C.append({'claim': what, 'recomputed': got, 'stated': want, 'source': src, 'match': got == want})
    P, R = 'docs/identifier-fidelity-paper-draft-v2.md', 'experiments/identifier_fidelity/discovery/REPORT.md'
    add('words: original failures / x2 failures / tested', [f1['words']['total']['original_fail'], f1['words']['total']['x2_fail'], f1['words']['total']['n']], [268, 1, 7688], P + ' s2 table; ' + R + ' s2')
    add('words: lone-s suffix names failing / tested (original)', [g['lone_s_suffix']['original_fail'], g['lone_s_suffix']['n']], [260, 432], R + ' s2')
    add('words: lone-s share failing, rounded to whole %', round(100 * g['lone_s_suffix']['original_rate']), 60, P + ' s2 ("roughly 60%")')
    add('words: all other items failing / tested (original, x2)', [g['other']['original_fail'], g['other']['x2_fail'], g['other']['n']], [8, 1, 7224], R + ' s2 (8/7,224 = 0.11%)')
    add('words: literal "_s" names (omitted from the 432) failing', [g['lone_s_literal_underscore']['n'], g['lone_s_literal_underscore']['original_fail']], [32, 0], R + ' s2 table')
    add('plurals: failures original / x2 / tested', [f1['plurals']['total']['original_fail'], f1['plurals']['total']['x2_fail'], f1['plurals']['total']['n']], [274, 1, 2000], P + ' s2 table')
    add('plurals: lone-s failing / tested (original, x2)', [f1['plurals']['groups']['lone_s']['original_fail'], f1['plurals']['groups']['lone_s']['x2_fail'], f1['plurals']['groups']['lone_s']['n']], [273, 1, 1500], R + ' s3')
    add('plurals: lone-s population', f1['plurals']['groups']['lone_s']['population'], 12801, R + ' s3')
    add('blocks: failures original / x2 / tested', [f1['blocks']['total']['original_fail'], f1['blocks']['total']['x2_fail'], f1['blocks']['total']['n']], [368, 15, 1278], P + ' s2 table')
    add('blocks: lone-s blocks failing original / x2 / tested', [f1['blocks']['groups']['lone_s']['original_fail'], f1['blocks']['groups']['lone_s']['x2_fail'], f1['blocks']['groups']['lone_s']['n']], [358, 6, 528], R + ' s4')
    add('blocks: other blocks failing original / x2 / tested', [f1['blocks']['groups']['other']['original_fail'], f1['blocks']['groups']['other']['x2_fail'], f1['blocks']['groups']['other']['n']], [10, 9, 750], R + ' s4')
    add('dose-response: perplexity x1 / x2 (2 decimals)', [round(dose[1.0]['perplexity'], 2), round(dose[2.0]['perplexity'], 2)], [61.57, 60.57], 'experiments/identifier_fidelity/ROOT_CAUSE.md s3')
    add('dose-response: contexts (of 10) with correct s on top, x0.25 / x1 / x2 / x4', [dose[x]['n_top1'] for x in (0.25, 1.0, 2.0, 4.0)], [0, 1, 10, 10], 'ROOT_CAUSE.md s2 (shipped file: 9 of 10 flip)')
    add('dose-response: gain median x1 / x2 (2 decimals)', [round(dose[1.0]['gain_median'], 2), round(dose[2.0]['gain_median'], 2)], [0.68, 0.88], 'ROOT_CAUSE.md s3')
    add('dose-response: share of pairs below the floor x1 / x2 (whole %)', [round(100 * dose[1.0]['share_below_floor']), round(100 * dose[2.0]['share_below_floor'])], [61, 7], 'ROOT_CAUSE.md s3')
    add('healthy test: contexts with correct s on top (of 10): shipped / q4_0 shrunk / bf16 shrunk', [hc['qat_x1']['n_top1'], hc['std_q4_0_small']['n_top1'], hc['std_bf16_small']['n_top1']], [1, 4, 8], 'ROOT_CAUSE.md s2 (9 / 6 / 2 of 10 no longer arg-max)')
    add('healthy test: unshrunk standard release correct (q4_0, bf16)', [hc['std_q4_0']['n_top1'], hc['std_bf16']['n_top1']], [10, 10], 'ROOT_CAUSE.md s2 rows 6-7')
    add('tensors compared / tensor RMS ratio (3 decimals)', [f2['tensor_audit']['n_tensors_compared'], round(f2['tensor_audit']['tensor_rms_ratio_qat_over_standard'], 3)], [665, 0.016], P + ' s3 (665 tensors, ~65x smaller)')
    add('Frobenius norm shipped / standard E4B', [wt['qat_e4b_shipped']['frobenius_norm'], wt['standard_e4b']['frobenius_norm']], [2.014, 129.908], 'lora_feasibility/REPORT.md s B (2.05 W4A16, ~130 standard)')
    add('other norm families sit at least 490x above the floor', f2['tensor_audit']['other_norm_families_min_ms_over_floor'] >= 490, True, 'tensor_audit/REPORT.md B')
    add('tasks passed original / merged adapter (of 7)', [f3['passed']['original'], f3['passed']['lora'], f3['n_tasks']], [1, 5, 7], P + ' s5 table')
    add('tasks flipped fail->pass / pass->fail', [len(f3['flips']['fail_to_pass']), len(f3['flips']['pass_to_fail'])], [4, 0], 'gemma-edit-tasks/experiments/REPORT.md')
    add('diagrams: p(correct s) original / merged adapter, as % rounded to 0.1', [round(100 * pr['original']['p']['s'], 1), round(100 * pr['merged_adapter']['p']['s'], 1)], [0.9, 99.7], P + ' s5 (0.9% / 99.7%)')
    add('diagrams: original copied the word alone correctly', pr['diagrams_copied_alone_by_original'], True, 'discovery/REPORT.md s1 controls (`diagrams` passes on the original in isolation)')
    add('sampled replays: tests pass original / merged adapter (of 120)', [f3['sampled_replays_all_tasks']['original_tests_pass'], f3['sampled_replays_all_tasks']['merged_adapter_tests_pass'], f3['sampled_replays_all_tasks']['edits_per_model']], [23, 68, 120], P + ' s5')
    add('Kaggle vLLM: inputs / base failures / adapter failures / fixed / regressions', [k['total'], k['base_failures'], k['lora_failures'], k['fixed'], k['regressions']], [18, 15, 1, 14, 0], P + ' s4; docs/identifier-fidelity-paper-fable-review.md')
    return C


def extract_all():
    d = {'figure1': extract_figure1(), 'figure2': extract_figure2(), 'figure3': extract_figure3()}
    d['kaggle_vllm_adapter'] = d['figure3'].pop('kaggle_vllm_adapter')
    d['arms'] = {
        'original': 'Google gemma-4-E4B_q4_0-it.gguf (QAT, Q4_0), sha256 676c3507...; local llama.cpp b11146 (Metal)',
        'x2_tensor': 'the same GGUF with per_layer_model_proj.weight multiplied by 2 (gemma-4-E4B_q4_0-it-plproj-x2.gguf, sha256 3a6ad54e...); used in figures 1 and 2',
        'merged_adapter_local': 'the same GGUF with the rank-128 truncated-SVD adapter (3,408,208 bytes, sha256 9ce83a7c...) merged into per_layer_model_proj.weight (W + B@A, stored F16); local llama.cpp; used in figure 3',
        'kaggle_vllm_adapter': 'the same adapter file served as a LoRA next to gemma-4-e4b-it-qat-w4a16-ct by vLLM 0.19.1 on 4 x L4; 18 hand-picked copy inputs; NOT plotted, reported in the README only',
    }
    d['claim_checks'] = check_claims(d)
    return d


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# rendering: three deliberately plain graphics, one finding each.  No titles, subtitles or footnotes inside the images: the title, setup and limitations are the paper
# caption (README.md).  Every number is read from the data dict (figure-data.json).
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
INK, INK2, GRID, AXIS = '#1f2328', '#4b5563', '#e6e8eb', '#9aa1ab'
ORIG, FIX, NEUTRAL = '#D55E00', '#0072B2', '#7a828c'          # original / corrected / other: Okabe-Ito vermillion and blue (colour-blind safe, checked with the dataviz validator)
SANS, MONO = 'DejaVu Sans', 'DejaVu Sans Mono'                 # both ship with matplotlib, so the output does not depend on the machine's fonts
PANEL = '#f6f7f9'


def setup_style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': SANS, 'font.size': 13, 'text.color': INK, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': INK, 'ytick.color': INK2,
                         'axes.linewidth': 1.0, 'xtick.major.width': 1.0, 'ytick.major.width': 1.0, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'path', 'svg.hashsalt': 'gemma-copy-figures', 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    return plt


def tint(hex_color, a):
    """blend a colour with white; a = share of the colour kept"""
    rgb = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
    return '#' + ''.join(f'{round(255 - (255 - v) * a):02x}' for v in rgb)


def n_fmt(n):
    return f'{n:,}'


class Canvas:
    """a figure with a full-size overlay axes whose data units are inches measured from the TOP-left corner, for titles, boxes and labels"""

    def __init__(self, plt, w, h):
        self.plt, self.w, self.h = plt, w, h
        self.fig = plt.figure(figsize=(w, h), facecolor='white')
        self.ov = self.fig.add_axes([0, 0, 1, 1], zorder=0)
        self.ov.set_xlim(0, w)
        self.ov.set_ylim(h, 0)
        self.ov.axis('off')

    def axes(self, x, y, w, h):
        ax = self.fig.add_axes([x / self.w, 1 - (y + h) / self.h, w / self.w, h / self.h], zorder=2)
        ax.patch.set_alpha(0)
        return ax

    def text(self, x, y, s, size=13, color=INK, weight='normal', family=SANS, ha='left', va='center', **kw):
        return self.ov.text(x, y, s, fontsize=size, color=color, fontweight=weight, family=family, ha=ha, va=va, **kw)

    def box(self, x, y, w, h, fc='white', ec=AXIS, lw=1.0, r=0.08, z=1):
        from matplotlib.patches import FancyBboxPatch
        self.ov.add_patch(FancyBboxPatch((x + r, y + r), w - 2 * r, h - 2 * r, boxstyle=f'round,pad={r},rounding_size={r}', fc=fc, ec=ec, lw=lw, zorder=z))

    def save(self, base, title, description):
        for ext, meta in (('png', {'Software': None}), ('svg', {'Date': None, 'Creator': 'generate_figures.py', 'Title': title, 'Description': description})):
            self.fig.savefig(f'{base}.{ext}', dpi=300, format=ext, metadata=meta, facecolor='white')
        self.plt.close(self.fig)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Figure 1: failures before and after doubling the tensor.  Graphic only: the title, the setup and the limitations belong in the paper caption (README.md).
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
def fig1(plt, d, base):
    f = d['figure1']
    t = f['words']['total']
    ex = f['example_copy']
    c = Canvas(plt, 8.6, 4.7)
    ax = c.axes(1.25, 0.45, 4.45, 3.4)
    ax.set_xlim(-0.65, 1.65)
    ax.set_ylim(0, 345)
    ax.yaxis.grid(True, color=GRID, lw=1.0)
    ax.set_axisbelow(True)
    ax.set_yticks([0, 100, 200])
    ax.tick_params(axis='y', labelsize=12, length=0)
    ax.tick_params(axis='x', length=0, pad=10)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color(AXIS)
    ax.set_ylabel('failed copies', size=13, labelpad=8)
    for x, k, col in ((0, t['original_fail'], ORIG), (1, t['x2_fail'], FIX)):
        ax.bar(x, k, width=0.58, color=col, linewidth=0, zorder=3)
        ax.text(x, k + 24, n_fmt(k), ha='center', va='bottom', fontsize=30, fontweight='bold', zorder=5)
        ax.text(x, k + 8, f'of {n_fmt(t["n"])}', ha='center', va='bottom', fontsize=12, color=INK2, zorder=5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Original', 'Tensor ×2'], fontsize=15)

    c.box(6.15, 0.55, 2.25, 3.3, fc=PANEL, ec='none', r=0.1)
    c.text(6.4, 0.95, 'Asked to copy', size=12.5, color=INK2)
    c.text(6.4, 1.4, ex['input'], size=19, family=MONO, weight='bold')
    for y, col, lab, out, good in ((2.15, ORIG, 'Original', ex['original_output'], ex['original_exact']), (3.1, FIX, 'Tensor ×2', ex['x2_output'], ex['x2_exact'])):
        c.ov.add_patch(plt.Rectangle((6.4, y - 0.2), 0.14, 0.14, fc=col, ec='none'))
        c.text(6.65, y - 0.13, lab, size=12.5, color=INK2)
        c.text(6.4, y + 0.28, out, size=17, family=MONO)
        c.text(8.12, y + 0.28, '✓' if good else '✗', size=18, weight='bold', ha='center')
    c.save(base, 'Doubling one tensor fixed almost every copying error',
           f'Bar chart: {t["original_fail"]} failed copies for the original model and {t["x2_fail"]} after multiplying one tensor by 2, out of {t["n"]} selected words and identifiers; '
           f'example: asked to copy {ex["input"]}, the original writes {ex["original_output"]}.')


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Figure 2: dose-response of the tensor multiplier
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
def fig2(plt, d, base):
    from matplotlib.ticker import FixedLocator, NullLocator
    f = d['figure2']
    dose, ctxs = f['dose_response'], f['contexts']
    fac = [x['factor'] for x in dose]
    role = lambda fa: ORIG if fa == 1.0 else FIX if fa == 2.0 else NEUTRAL
    c = Canvas(plt, 8.6, 5.5)
    ax = c.axes(1.5, 1.2, 6.8, 3.0)
    ax.set_xscale('log', base=2)
    ax.set_xlim(0.18, 78)
    ax.xaxis.set_major_locator(FixedLocator(fac))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(-0.04, 1.05)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(['0', '0.5', '1'])
    ax.yaxis.grid(True, color=GRID, lw=1.0)
    ax.set_axisbelow(True)
    ax.spines['left'].set_color(AXIS)
    ax.spines['bottom'].set_color(AXIS)
    ax.tick_params(axis='y', labelsize=13, length=0)
    ax.tick_params(axis='x', labelsize=13, pad=6)
    ax.set_ylabel('probability of the\ncorrect “s”', size=13, labelpad=10)
    meds = [x['p_s_median'] for x in dose]
    for dd in dose:
        ax.scatter([dd['factor']] * len(ctxs), [dd['contexts'][t]['p_s'] for t in ctxs], s=48, facecolor='white', edgecolor=role(dd['factor']), linewidth=1.6, zorder=4)
    ax.plot(fac, meds, color=AXIS, lw=2.0, zorder=3)
    ax.scatter(fac, meds, s=170, color=[role(x) for x in fac], edgecolor='white', linewidth=2.2, zorder=6)
    ax.set_xticklabels([f'{x:g}' if x != fac[-1] else f'{x:.1f}' for x in fac])
    for lab in ax.get_xticklabels():
        if lab.get_text() in ('1', '2'):
            lab.set_fontweight('bold')
    tr = ax.get_xaxis_transform()
    for x, lab in ((1.0, 'shipped'), (2.0, 'our fix')):
        ax.text(x, -0.135, lab, transform=tr, ha='center', va='top', fontsize=12, color=INK2)
    c.text(1.5 + 6.8 / 2, 5.05, 'multiplier on the tensor', size=13, color=INK2, ha='center')
    for x in (0.25, 1.0, 2.0):                                         # how many of the 10 test cases have the right "s" as the model's top choice
        k = next(dd['n_top1'] for dd in dose if dd['factor'] == x)
        ax.text(x, 1.075, f'of {len(ctxs)}', ha='center', va='bottom', fontsize=12, color=INK2, clip_on=False)
        ax.text(x, 1.125, str(k), ha='center', va='bottom', fontsize=18, fontweight='bold', clip_on=False)
    c.text(1.35, 0.62, 'cases\ncorrect', size=12, color=INK2, ha='right', linespacing=1.15)
    c.save(base, 'Copying flips from failing to working at x2',
           'Probability that the model writes the correct final s for ten test cases at eight multipliers on one tensor: '
           f'{dose[1]["n_top1"]} of {len(ctxs)} test cases right at x1, {dose[2]["n_top1"]} of {len(ctxs)} at x2.')


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Figure 3: seven tasks, original vs adapter
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
def fig3(plt, d, base):
    f = d['figure3']
    tasks, passed, n = f['tasks'], f['passed'], f['n_tasks']
    c = Canvas(plt, 7.6, 6.3)
    xc, cw, ch, pitch, y0 = {'original': 2.2, 'lora': 4.9}, 2.45, 0.54, 0.66, 0.95
    for arm, col, lab in (('original', ORIG, 'Original'), ('lora', FIX, 'With adapter')):
        c.ov.add_patch(plt.Rectangle((xc[arm] + 0.1, 0.4), 0.17, 0.17, fc=col, ec='none'))
        c.text(xc[arm] + 0.37, 0.485, lab, size=14.5, weight='bold')
    for i, t in enumerate(sorted(tasks)):
        y = y0 + i * pitch
        c.text(0.4, y + ch / 2, t.split('_', 1)[1], size=15, family=MONO)
        for arm, col in (('original', ORIG), ('lora', FIX)):
            if tasks[t][arm]['passed']:
                c.box(xc[arm], y, cw, ch, fc=col, ec=col, lw=1.0, r=0.08, z=2)
                c.text(xc[arm] + cw / 2, y + ch / 2, '✓  PASS', size=14.5, weight='bold', color='white', ha='center', zorder=5)
            else:
                c.box(xc[arm], y, cw, ch, fc='white', ec=col, lw=2.0, r=0.08, z=2)
                c.text(xc[arm] + cw / 2, y + ch / 2, '✗  FAIL', size=14.5, weight='bold', ha='center', zorder=5)
    yt = y0 + n * pitch + 0.1
    c.ov.plot([0.4, 7.35], [yt - 0.06, yt - 0.06], color=GRID, lw=1.5)
    c.text(0.4, yt + 0.32, 'Passed', size=15, weight='bold')
    for arm in ('original', 'lora'):
        c.text(xc[arm] + cw / 2, yt + 0.32, f'{passed[arm]} of {n}', size=22, weight='bold', ha='center')
    c.save(base, f'Seven coding tasks: {passed["original"]} passed before, {passed["lora"]} after',
           f'Pass or fail of seven Python repair tasks for the original model ({passed["original"]} of {n} passed) and the same model with the adapter merged in ({passed["lora"]} of {n} passed).')


def render_all(data, out_dir):
    plt = setup_style()
    for fn, name in ((fig1, 'figure1-copying-errors'), (fig2, 'figure2-tensor-mechanism'), (fig3, 'figure3-coding-agent')):
        fn(plt, data, str(Path(out_dir) / name))
        print('rendered', name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plot-only', action='store_true', help='render from the existing figure-data.json')
    ap.add_argument('--check', action='store_true', help='re-extract and compare with figure-data.json; render nothing')
    args = ap.parse_args()
    if args.plot_only:
        data = json.loads(DATA_JSON.read_text())
    else:
        data = extract_all()
        text = json.dumps(data, indent=1, ensure_ascii=False) + '\n'
        if args.check:
            same = DATA_JSON.exists() and DATA_JSON.read_text() == text
            print('figure-data.json', 'matches the raw files' if same else 'DIFFERS from the raw files')
            bad = [c for c in data['claim_checks'] if not c['match']]
            print(f'{len(data["claim_checks"]) - len(bad)} of {len(data["claim_checks"])} claims match')
            sys.exit(0 if same and not bad else 1)
        DATA_JSON.write_text(text)
        print('wrote', DATA_JSON.name)
    for c in data['claim_checks']:
        print(('ok       ' if c['match'] else 'MISMATCH ') + f'{c["claim"]}: {c["recomputed"]} (stated {c["stated"]})')
    if not all(c['match'] for c in data['claim_checks']):
        sys.exit('claim check failed; not rendering')
    render_all(data, HERE)


if __name__ == '__main__':
    main()
