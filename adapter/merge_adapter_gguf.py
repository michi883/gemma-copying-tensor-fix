#!/usr/bin/env python3
"""Merge the actual rank-128 LoRA adapter file into a copy of the local E4B QAT Q4_0 GGUF.

    merge_adapter_gguf.py ADAPTER_DIR SRC_GGUF DST_GGUF

The adapter (adapter_model.safetensors + adapter_config.json) touches one tensor, per_layer_model_projection (GGUF name per_layer_model_proj.weight).
A correctly applied LoRA computes  y = W x + (lora_alpha / r) * B (A x),  so the merged weight is  W' = W + (alpha / r) * bf16(B) @ bf16(A)  (factors as stored, bf16; fp32 product;
result stored F16 like the original tensor).  The copy is the only file written.  This is the merged-weight proxy of experiments/lora_feasibility/ (REPORT.md section C), built from the
adapter FILE instead of re-deriving the SVD; it does NOT exercise vLLM's LoRA kernels, the W4A16 base weights, or tensor-parallel slicing.
"""
import json
import os
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

if os.environ.get('GGUF_PY'):                                  # path of llama.cpp's gguf-py when `pip install gguf` is not used
    sys.path.insert(0, os.environ['GGUF_PY'])
import gguf

ADAPTER, SRC, DST = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
PREFIX = 'base_model.model.model.language_model.per_layer_model_projection.'


def read_safetensors(path):
    raw = Path(path).read_bytes()
    n = struct.unpack('<Q', raw[:8])[0]
    hdr = json.loads(raw[8:8 + n])
    out = {}
    for k, v in hdr.items():
        if k == '__metadata__':
            continue
        a, b = v['data_offsets']
        buf = raw[8 + n + a:8 + n + b]
        assert v['dtype'] == 'BF16', (k, v['dtype'])
        u16 = np.frombuffer(buf, dtype=np.uint16).reshape(v['shape'])
        out[k] = (u16.astype(np.uint32) << 16).view(np.float32)          # bf16 -> fp32 is exact
    return out


cfg = json.loads((ADAPTER / 'adapter_config.json').read_text())
tens = read_safetensors(ADAPTER / 'adapter_model.safetensors')
assert sorted(tens) == [PREFIX + 'lora_A.weight', PREFIX + 'lora_B.weight'], sorted(tens)
A, B = tens[PREFIX + 'lora_A.weight'], tens[PREFIX + 'lora_B.weight']
r = cfg['r']
assert A.shape == (r, 2560) and B.shape == (10752, r), (A.shape, B.shape)
scale = cfg['lora_alpha'] / r
assert not cfg.get('use_rslora') and not cfg.get('use_dora')

shutil.copyfile(SRC, DST)
rd = gguf.GGUFReader(DST, 'r+')
t = {x.name: x for x in rd.tensors}['per_layer_model_proj.weight']
assert t.tensor_type == gguf.GGMLQuantizationType.F16 and t.data.shape == (10752, 2560), (t.tensor_type, t.data.shape)
W = np.array(t.data, dtype=np.float32)
Wn = W + scale * (B @ A)
t.data[:] = Wn.astype(np.float16)
t.data.flush()
print(f'{DST}: merged r={r} alpha={cfg["lora_alpha"]} scale={scale}  ||W||={np.linalg.norm(W):.4f}  ||W_merged||={np.linalg.norm(Wn):.4f}  '
      f'||W_merged||/||W||={np.linalg.norm(Wn) / np.linalg.norm(W):.4f}  ||W_merged - W||/||W||={np.linalg.norm(Wn - W) / np.linalg.norm(W):.4f}')
