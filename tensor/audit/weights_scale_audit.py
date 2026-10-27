#!/usr/bin/env python3
"""Phase 1 / part A: scale audit of ALL text-tower tensors, QAT E4B GGUF (dequantized) vs the non-QAT bf16 checkpoint (same architecture, same shapes, same function).
Run: python weights_scale_audit.py      Output: weights_scale_audit.json + a family summary on stdout.
For each tensor: RMS in both checkpoints, ratio qat/base, least-squares scale <q,b>/<b,b>, cosine(q,b).  Read-only: no model file is written.
"""
import glob, json, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, '/path/to/llama.cpp/gguf-py')
import gguf
from gguf import quants

T = gguf.GGMLQuantizationType
SNAP = glob.glob('/path/to/.cache/huggingface/hub/models--mlx-community--gemma-4-E4B-it-bf16/snapshots/*/')[0]
GG = '/path/to/models/gemma-4/gemma-4-E4B_q4_0-it.gguf'
LM = 'language_model.model.'
KIND = {'attn_q': 'self_attn.q_proj', 'attn_k': 'self_attn.k_proj', 'attn_v': 'self_attn.v_proj', 'attn_output': 'self_attn.o_proj', 'ffn_gate': 'mlp.gate_proj', 'ffn_up': 'mlp.up_proj',
        'ffn_down': 'mlp.down_proj', 'inp_gate': 'per_layer_input_gate', 'proj': 'per_layer_projection',
        'attn_q_norm': 'self_attn.q_norm', 'attn_k_norm': 'self_attn.k_norm', 'attn_norm': 'input_layernorm', 'post_attention_norm': 'post_attention_layernorm', 'ffn_norm': 'pre_feedforward_layernorm',
        'post_ffw_norm': 'post_feedforward_layernorm', 'post_norm': 'post_per_layer_input_norm'}
GLOBAL = {'token_embd.weight': 'embed_tokens.weight', 'per_layer_token_embd.weight': 'embed_tokens_per_layer.weight', 'per_layer_model_proj.weight': 'per_layer_model_projection.weight',
          'per_layer_proj_norm.weight': 'per_layer_projection_norm.weight', 'output_norm.weight': 'norm.weight'}
idx = json.load(open(SNAP + 'model.safetensors.index.json'))['weight_map']
files = {}

def base(key):
    f = idx[key]
    if f not in files: files[f] = mx.load(SNAP + f)
    return files[f][key]

def to_f32(t):
    return np.array(t.data, dtype=np.float32) if t.tensor_type in (T.F32, T.F16) else quants.dequantize(t.data, t.tensor_type).astype(np.float32)

def hfkey(name):
    if name in GLOBAL: return LM + GLOBAL[name]
    if name.startswith('blk.'):
        _, blk, rest = name.split('.', 2)
        if rest.startswith('layer_output_scale'): return f'{LM}layers.{blk}.layer_scalar'
        return f'{LM}layers.{blk}.{KIND[rest[:-7]]}.weight' if rest[:-7] in KIND else None
    return None

rows, skipped = [], []
t0 = time.time()
CH = 16_000_000                                           # elements per chunk: the 2.8e9-element per-layer table must never exist in f32 at once
for t in gguf.GGUFReader(GG).tensors:
    k = hfkey(t.name)
    if k is None or k not in idx:
        skipped.append(t.name); continue
    b_all = base(k)
    quant = t.tensor_type not in (T.F32, T.F16)
    nrow = t.data.shape[0] if quant else 1
    step = max(1, CH // (int(np.prod(b_all.shape)) // max(nrow, 1))) if quant else 1
    qq = bb = qb = 0.0; qmin = bmin = np.inf; qmax = bmax = -np.inf; n = 0
    for i in range(0, nrow, step):
        if quant:
            q = quants.dequantize(t.data[i:i + step], t.tensor_type).astype(np.float32).reshape(-1)
            b = np.array(b_all[i:i + step].astype(mx.float32)).reshape(-1)
        else:
            q = np.array(t.data, dtype=np.float32).reshape(-1); b = np.array(b_all.astype(mx.float32)).reshape(-1)
        assert q.shape == b.shape, (t.name, q.shape, b.shape)
        qq += float((q.astype(np.float64) ** 2).sum()); bb += float((b.astype(np.float64) ** 2).sum()); qb += float(q.astype(np.float64) @ b.astype(np.float64)); n += q.size
        qmin, qmax, bmin, bmax = min(qmin, float(q.min())), max(qmax, float(q.max())), min(bmin, float(b.min())), max(bmax, float(b.max()))
    rq, rb = (qq / n) ** .5, (bb / n) ** .5
    rows.append({'gguf': t.name, 'hf': k.replace(LM, ''), 'gguf_type': t.tensor_type.name, 'shape': [int(x) for x in t.shape], 'rms_qat': rq, 'rms_base': rb, 'ratio': rq / rb,
                 'ls_scale': qb / bb, 'cos': qb / ((qq * bb) ** .5 + 1e-30), 'min_qat': qmin, 'max_qat': qmax, 'min_base': bmin, 'max_base': bmax})
json.dump({'rows': rows, 'skipped': skipped}, open('weights_scale_audit.json', 'w'), indent=1)
print(f'{len(rows)} tensors compared, {len(skipped)} skipped ({skipped[:6]}...), {time.time()-t0:.0f}s')
fam = {}
for r in rows:
    g = r['gguf']
    f = g.split('.', 2)[2][:-7] if g.startswith('blk.') and g.endswith('.weight') else (g.split('.', 2)[2] if g.startswith('blk.') else g)
    fam.setdefault(f, []).append(r)
print(f'{"family":28}{"n":>4} {"ratio min":>10}{"median":>9}{"max":>9}   {"cos min":>8}{"median":>8}   gguf type   example shape')
for f, rs in sorted(fam.items(), key=lambda kv: np.median([r['ratio'] for r in kv[1]])):
    ra, co = [r['ratio'] for r in rs], [r['cos'] for r in rs]
    print(f'{f:28}{len(rs):4d} {min(ra):10.4f}{np.median(ra):9.4f}{max(ra):9.4f}   {min(co):8.3f}{np.median(co):8.3f}   {rs[0]["gguf_type"]:8}   {rs[0]["shape"]}')
