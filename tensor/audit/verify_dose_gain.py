#!/usr/bin/env python3
"""Optional static check for Figure 2: recompute the RMSNorm "gain" at each tested multiplier from the stored weights.  No model inference, no server.

Reads the shipped QAT E4B GGUF (tensors `per_layer_model_proj.weight` and `token_embd.weight`), takes the same 2,000 random vocabulary rows as
experiments/identifier_fidelity/dose_response.py (numpy seed 0), forms x = W e for every multiplier, and computes, per (token, layer) pair,
mean(x^2) over the 256 channels and the gain sqrt(ms / (ms + 1e-6)).  Writes source/dose_gain_recompute.json, which generate_figures.py compares with the values
recorded in rc_data/dose_response.jsonl.

    E4B_GGUF=/path/to/gemma-4-E4B_q4_0-it.gguf GGUF_PY=/path/to/llama.cpp/gguf-py  python3 verify_dose_gain.py

Needs numpy and gguf-py (llama.cpp); the repository's lab used python.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

GGUF = os.environ.get('E4B_GGUF', '/path/to/models/gemma-4/gemma-4-E4B_q4_0-it.gguf')
sys.path.insert(0, os.environ.get('GGUF_PY', '/path/to/llama.cpp/gguf-py'))
import gguf  # noqa: E402
from gguf import quants  # noqa: E402

EPS = 1e-6
FACTORS = [0.25, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 50.59644256269407]   # the multipliers of rc_data/dose_response.jsonl
OUT = Path(__file__).resolve().parent / 'source' / 'dose_gain_recompute.json'

tensors = {t.name: t for t in gguf.GGUFReader(GGUF).tensors}
W = np.array(tensors['per_layer_model_proj.weight'].data, dtype=np.float32)
ids = [int(i) for i in np.sort(np.random.default_rng(0).choice(262144, 2000, replace=False))]
E = np.concatenate([quants.dequantize(tensors['token_embd.weight'].data[i:i + 1], gguf.GGMLQuantizationType.Q6_K).astype(np.float32).reshape(1, -1) for i in ids])
V = ((E @ W.T).reshape(len(E), 42, 256) ** 2).mean(-1)       # mean square at the norm input for a multiplier of 1; shape (2000 tokens, 42 layers)

rows = []
for f in FACTORS:
    ms = V * f * f                                             # a constant scale of W scales every mean square by f^2
    g = np.sqrt(ms / (ms + EPS))
    rows.append({'factor': f, 'ms_median_of_layer_medians': float(np.median(np.median(ms, axis=1))), 'gain_median': float(np.median(g)), 'gain_p5': float(np.percentile(g, 5)),
                 'share_below_eps': float((ms < EPS).mean())})
    print(f'x{f:<8.4g} median ms {rows[-1]["ms_median_of_layer_medians"]:.3e}  gain median {rows[-1]["gain_median"]:.4f} (5th pct {rows[-1]["gain_p5"]:.4f})  below eps {rows[-1]["share_below_eps"]:.4f}')
OUT.write_text(json.dumps({'source': 'static weights of ' + Path(GGUF).name, 'n_tokens': len(ids), 'n_layers': 42, 'eps': EPS, 'frobenius_norm_W': float(np.linalg.norm(W)), 'rows': rows}, indent=1) + '\n')
print('wrote', OUT)
