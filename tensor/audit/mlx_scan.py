#!/usr/bin/env python3
"""The teacher-forced scan of root_cause.py on local weights run in mlx, same prompt token ids.  Usage: mlx_scan.py VARIANT TEXT...

Run with a python that has mlx-lm (gemma4-playground/.venv) while the llama-server of ./lab.sh is up: the prompt token ids come from the server (/apply-template + /tokenize), so every
variant is fed the identical id sequence.  Output: rc_data/mlx_scan.jsonl, same shape as scan.jsonl plus the variant.

  bf16          mlx-community/gemma-4-E4B-it-bf16 as it is (= the NON-QAT google/gemma-4-E4B-it)
  bf16_proj     the same with only per_layer_model_projection multiplied by PROJ_FACTOR (nothing quantized, no QAT weights)
  ptq_q4_0      the same, but every weight matrix that the QAT GGUF stores as Q4_0 (342) is rounded to Q4_0 with the llama.cpp reference quantizer (gguf-py) and put back as bf16:
                plain post-training 4-bit rounding of the standard checkpoint (embeddings, norms and scalars untouched)
  gguf_stages   the local QAT GGUF's own tensors put into the bf16 model group by group, cumulatively, and scanned after each group:
                  matrices        the 342 Q4_0 matrices (dequantized)
                  small           norms, layer scalars, per_layer_model_proj (x PROJ_FACTOR, default sqrt(n_embd); PROJ_FACTOR=1 = as stored), per-layer projection norm, final norm
                  token_embd      the Q6_K token embedding (= the tied output head)
                  per_layer_embd  the Q6_K per-layer embedding table
                after the last group every tensor of the model's text tower comes from the GGUF (rope_freqs is a runtime constant), so this is the GGUF's model run in mlx instead of llama.cpp
"""
import glob
import json
import math
import os
import sys
import time

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import load

import root_cause as rc

sys.path.insert(0, '/path/to/llama.cpp/gguf-py')
import gguf
from gguf import quants

SNAP = glob.glob('/path/to/.cache/huggingface/hub/models--mlx-community--gemma-4-E4B-it-bf16/snapshots/*/')[0]
GGUF = '/path/to/models/gemma-4/gemma-4-E4B_q4_0-it.gguf'
T = gguf.GGMLQuantizationType
LM = 'language_model.model.'
MATRIX = {'attn_q': 'self_attn.q_proj', 'attn_k': 'self_attn.k_proj', 'attn_v': 'self_attn.v_proj', 'attn_output': 'self_attn.o_proj', 'ffn_gate': 'mlp.gate_proj', 'ffn_up': 'mlp.up_proj',
          'ffn_down': 'mlp.down_proj', 'inp_gate': 'per_layer_input_gate', 'proj': 'per_layer_projection'}
SMALL = {'attn_q_norm': 'self_attn.q_norm', 'attn_k_norm': 'self_attn.k_norm', 'attn_norm': 'input_layernorm', 'post_attention_norm': 'post_attention_layernorm', 'ffn_norm': 'pre_feedforward_layernorm',
         'post_ffw_norm': 'post_feedforward_layernorm', 'post_norm': 'post_per_layer_input_norm'}
N_EMBD = 2560
PROJ_FACTOR = float(os.environ.get('PROJ_FACTOR', math.sqrt(N_EMBD)))       # per_layer_model_proj is multiplied by this when it is put into the bf16 model; PROJ_FACTOR=1 puts it in as the GGUF has it


def to_f32(t, rows=None):
    d = t.data if rows is None else t.data[rows]
    return np.array(d, dtype=np.float32) if t.tensor_type in (T.F32, T.F16) else quants.dequantize(d, t.tensor_type).astype(np.float32)


def put(model, key, arr):
    old = dict(tree_flatten(model.parameters()))[key]
    arr = mx.array(arr).astype(mx.bfloat16) if not isinstance(arr, mx.array) else arr
    assert tuple(arr.shape) == tuple(old.shape), (key, arr.shape, old.shape)
    model.load_weights([(key, arr)], strict=False)


def rel(model, key, new):
    old = np.array(dict(tree_flatten(model.parameters()))[key].astype(mx.float32)).reshape(-1)
    new = np.array(new.astype(mx.float32) if isinstance(new, mx.array) else new, dtype=np.float32).reshape(-1)
    return float(np.linalg.norm(new - old) / np.linalg.norm(old))


def patch_chunked(model, key, t, chunk=16384):
    parts = [mx.array(to_f32(t, slice(i, i + chunk))).astype(mx.bfloat16) for i in range(0, t.data.shape[0], chunk)]
    new = mx.concatenate(parts)
    mx.eval(new)
    r = rel(model, key, new)
    put(model, key, new)
    return r


def patch(model, group, tensors):
    """put one group of GGUF tensors into the model; prints how far they are from the bf16 originals"""
    err = []
    for t in tensors.values():
        n = t.name
        if group == 'matrices' and t.tensor_type == T.Q4_0:
            _, blk, kind, _ = n.split('.')
            key = f'{LM}layers.{blk}.{MATRIX[kind]}.weight'
            new = to_f32(t)
        elif group == 'small' and (n.startswith('blk.') and n.split('.', 2)[2][:-7] in SMALL or n.endswith('layer_output_scale.weight') or n in ('output_norm.weight', 'per_layer_proj_norm.weight', 'per_layer_model_proj.weight')):
            if n.startswith('blk.'):
                _, blk, rest = n.split('.', 2)
                key = f'{LM}layers.{blk}.layer_scalar' if rest.startswith('layer_output_scale') else f'{LM}layers.{blk}.{SMALL[rest[:-7]]}.weight'
            else:
                key = LM + {'output_norm.weight': 'norm.weight', 'per_layer_proj_norm.weight': 'per_layer_projection_norm.weight', 'per_layer_model_proj.weight': 'per_layer_model_projection.weight'}[n]
            new = to_f32(t) * (PROJ_FACTOR if n == 'per_layer_model_proj.weight' else 1.0)
            new = new.reshape(dict(tree_flatten(model.parameters()))[key].shape)
        elif group == 'token_embd' and n == 'token_embd.weight':
            err.append(patch_chunked(model, LM + 'embed_tokens.weight', t))
            continue
        elif group == 'per_layer_embd' and n == 'per_layer_token_embd.weight':
            err.append(patch_chunked(model, LM + 'embed_tokens_per_layer.weight', t))
            continue
        else:
            continue
        err.append(rel(model, key, new))
        put(model, key, new)
    mx.eval(model.parameters())
    print(f'  + {group}: {len(err)} tensors, relative RMS change vs bf16: mean {np.mean(err):.3f} (min {min(err):.3f}, max {max(err):.3f})', flush=True)


def ptq(model, tensors):
    err = []
    for t in tensors.values():
        if t.tensor_type == T.Q4_0:
            _, blk, kind, _ = t.name.split('.')
            key = f'{LM}layers.{blk}.{MATRIX[kind]}.weight'
            f = np.array(dict(tree_flatten(model.parameters()))[key].astype(mx.float32))
            new = quants.dequantize(quants.quantize(f, T.Q4_0), T.Q4_0).astype(np.float32)
            err.append(float(np.linalg.norm(new - f) / np.linalg.norm(f)))
            put(model, key, new)
    mx.eval(model.parameters())
    print(f'ptq_q4_0: {len(err)} matrices rounded to Q4_0, relative RMS change: mean {np.mean(err):.3f} (min {min(err):.3f}, max {max(err):.3f})', flush=True)


def next_dist(model, ids):
    out = model(mx.array([ids]))
    p = mx.softmax((out.logits if hasattr(out, 'logits') else out)[0, -1].astype(mx.float32), axis=-1)
    mx.eval(p)
    return p


def scan(model, label, text):
    ids, ref = rc.prompt_ids(text)
    ref_ids = [i for i, _ in ref]
    toks = [rc.name(i, b) for i, b in ref]
    res = {'variant': label, 'text': text, 'prompt_tokens': len(ids), 'ref_ids': ref_ids, 'ref_toks': toks, 'steps': []}
    for k in range(len(ref) + 1):
        p = next_dist(model, ids + ref_ids[:k])
        want = ref_ids[k] if k < len(ref) else rc.EOT
        res['steps'].append({'step': k, 'after': ''.join(toks[:k]), 'want': want, 'want_tok': rc.name(want), 'p_want': float(p[want].item()), 'rank_want': int(mx.sum(p > p[want]).item()) + 1,
                             'p_eot': float(p[rc.EOT].item()),
                             'top': [{'id': i, 'tok': rc.name(i), 'p': round(float(p[i].item()), 5)} for i in mx.argsort(-p)[:6].tolist()],
                             'watch': {n: [round(float(p[i].item()), 5), int(mx.sum(p > p[i]).item()) + 1] for n, i in rc.WATCH.items()}})
    return res


def run(model, label, texts):
    for t in texts:
        r = scan(model, label, t)
        s_step = r['steps'][r['ref_toks'].index('s')] if 's' in r['ref_toks'] else r['steps'][-2 if len(r['steps']) > 2 else 0]
        print(f'{label:34} {t!r:20} p(want {s_step["want_tok"]!r} after {s_step["after"]!r}) = {s_step["p_want"]:.4f} rank {s_step["rank_want"]}  p(<turn|>) = {s_step["p_eot"]:.4f}  first token p = {r["steps"][0]["p_want"]:.4f}', flush=True)
        with open(rc.DATA / 'mlx_scan.jsonl', 'a') as f:
            f.write(json.dumps({'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), **r}, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    variant, texts = sys.argv[1], sys.argv[2:]
    model, _ = load(SNAP)
    tensors = {t.name: t for t in gguf.GGUFReader(GGUF).tensors}
    if variant == 'bf16':
        run(model, 'bf16 (non-QAT)', texts)
    elif variant == 'bf16_proj':        # the non-QAT bf16 model, nothing quantized, only per_layer_model_projection multiplied by PROJ_FACTOR (e.g. 0.0153)
        key = LM + 'per_layer_model_projection.weight'
        put(model, key, dict(tree_flatten(model.parameters()))[key].astype(mx.float32) * PROJ_FACTOR)
        run(model, f'bf16 (non-QAT), per_layer_model_projection x{PROJ_FACTOR:.4g}', texts)
    elif variant == 'ptq_q4_0':
        ptq(model, tensors)
        run(model, 'bf16 + Q4_0 rounding', texts)
    elif variant == 'gguf_stages':
        done = []
        for group in ('matrices', 'small', 'token_embd', 'per_layer_embd'):
            patch(model, group, tensors)
            done.append(group)
            run(model, 'GGUF ' + '+'.join(done) + (f' (per_layer_model_proj x{PROJ_FACTOR:.4g})' if 'small' in done else ''), texts)
    else:
        sys.exit(f'unknown variant {variant}')
