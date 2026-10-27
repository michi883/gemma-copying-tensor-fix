#!/usr/bin/env python3
"""What does `per_layer_projection_norm` (RMSNorm, eps = rms_norm_eps = 1e-6) get as input?  Run with a python that has mlx + numpy (gemma4-playground/.venv).

Reference forward (mlx_lm/models/gemma4_text.py, `_project_per_layer_inputs`):   x = per_layer_model_projection(embed(token) * sqrt(n_embd)) * n_embd**-0.5 ; x = RMSNorm(x.reshape(L, 256))
The two scales cancel, so x = W @ e for the token's (unscaled) embedding row e.  RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight: a constant scale of W is normalized away only while mean(x^2) >> eps.
For each file this prints mean(x^2) per layer (median over the layers, and the smallest layer) for some tokens and for 2000 random tokens, next to eps, and the resulting gain
sqrt(mean(x^2) / (mean(x^2) + eps)) -- 1.0 means eps is irrelevant, 0.3 means the projected per-layer input comes out 3x smaller than intended.
"""
import glob
import json
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, '/path/to/llama.cpp/gguf-py')
import gguf
from gguf import quants

H = '/path/to/.cache/huggingface/hub/'
EPS = 1e-6
TOK = {'s': 236751, 'direct': 12293, 'fork': 45689, 'redirect': 22542, 'ing': 522, 'ed': 524, 'payload': 25839, 'pull': 37474, 'yield': 55747, 'treat': 89580, ' then': 1299, ' now': 1492}
RNG = np.random.default_rng(0)
RANDOM = np.sort(RNG.choice(262144, 2000, replace=False))


def gguf_rows(t, ids):
    return np.concatenate([quants.dequantize(t.data[i:i + 1], t.tensor_type).astype(np.float32).reshape(1, -1) if t.tensor_type not in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32)
                           else np.array(t.data[i:i + 1], dtype=np.float32).reshape(1, -1) for i in ids])


def hf_tensor(pattern, key_end, rows=None):
    for f in sorted(glob.glob(pattern)):
        w = mx.load(f)
        for k, v in w.items():
            if k.endswith(key_end) and 'language_model' in k and 'audio' not in k and 'vision' not in k:
                return np.array((v[mx.array([int(r) for r in rows])] if rows is not None else v).astype(mx.float32))
    raise KeyError(key_end)


def report(label, W, E, n_layers, p=256):
    """W: (L*p, n_embd), E: (n_tokens, n_embd) rows of the embedding table for TOK values + RANDOM"""
    x = E @ W.T                                          # (n_tokens, L*p)
    v = (x.reshape(len(E), n_layers, p) ** 2).mean(-1)   # (n_tokens, L): mean square at the norm input
    gain = np.sqrt(v / (v + EPS))
    names = list(TOK)
    k = len(names)
    print(f'{label}')
    print(f'   |W|_F {np.linalg.norm(W):8.3f}   mean row norm of the embedding rows used {np.linalg.norm(E, axis=1).mean():.3f}')
    for name, i in zip(names, range(k)):
        print(f'   token {name!r:11}: mean(x^2) per layer: median {np.median(v[i]):.2e}  min {v[i].min():.2e}  max {v[i].max():.2e}  eps {EPS:.0e}   gain median {np.median(gain[i]):.2f}  min {gain[i].min():.2f}')
    r = v[k:]
    print(f'   2000 random tokens: median over tokens of the layer-median mean(x^2) {np.median(np.median(r, 1)):.2e}   share of (token, layer) pairs with mean(x^2) < eps: {(r < EPS).mean():.1%}   gain: median {np.median(gain[k:]):.2f}, 5th percentile {np.percentile(gain[k:], 5):.2f}')


ids = [int(i) for i in list(TOK.values()) + list(RANDOM)]
# --- E4B: the local QAT GGUF, and the non-QAT bf16 weights
A = {t.name: t for t in gguf.GGUFReader('/path/to/models/gemma-4/gemma-4-E4B_q4_0-it.gguf').tensors}
Wa = np.array(A['per_layer_model_proj.weight'].data, dtype=np.float32)
report('E4B  QAT GGUF as shipped (per_layer_model_proj as stored, token_embd Q6_K)', Wa, gguf_rows(A['token_embd.weight'], ids), 42)
report('E4B  QAT GGUF with per_layer_model_proj x sqrt(2560)  [the fix]', Wa * np.sqrt(2560), gguf_rows(A['token_embd.weight'], ids), 42)
snap = glob.glob(H + 'models--mlx-community--gemma-4-E4B-it-bf16/snapshots/*/')[0]
Wb = hf_tensor(snap + '*.safetensors', 'per_layer_model_projection.weight')
Eb = hf_tensor(snap + '*.safetensors', 'embed_tokens.weight', ids)
report('E4B  non-QAT bf16 (HF/mlx)', Wb, Eb, 42)
# --- E2B (works in llama.cpp): its GGUF, its HF QAT-unquantized checkpoint, its HF base checkpoint
B = {t.name: t for t in gguf.GGUFReader('/path/to/models/gemma-4/gemma-4-E2B_q4_0-it.gguf').tensors}
report('E2B  QAT GGUF as shipped', np.array(B['per_layer_model_proj.weight'].data, dtype=np.float32), gguf_rows(B['token_embd.weight'], ids), 35)
for label, repo in [('E2B  HF QAT-unquantized', 'models--google--gemma-4-E2B-it-qat-q4_0-unquantized'), ('E2B  HF base (non-QAT)', 'models--google--gemma-4-E2B-it')]:
    pat = glob.glob(H + repo + '/snapshots/*/')[0] + '*.safetensors'
    report(label, hf_tensor(pat, 'per_layer_model_projection.weight'), hf_tensor(pat, 'embed_tokens.weight', ids), 35)
