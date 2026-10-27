#!/usr/bin/env python3
"""Build a minimal PEFT adapter (no training) whose only content is a rank-r LoRA on per_layer_model_projection that approximates  W -> 2W.
Usage: build_adapter.py W_NPY OUT_DIR RANK [svd|data]      (W_NPY: dequantised competition tensor, from stepb/fetch.py + stepb/deq_compare.py, float32 [10752, 2560])
Writes OUT_DIR/adapter_config.json and OUT_DIR/adapter_model.safetensors (BF16), key names as in the competition's sample adapter."""
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lowrank

W_NPY, OUT, R = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
KIND = sys.argv[4] if len(sys.argv) > 4 else 'svd'
assert KIND == 'svd', 'data-aware factors need embedding rows; only the svd adapter is written here'
W = np.load(W_NPY).astype(np.float32)
B, A = lowrank.factors(KIND, W, R)
P = 'base_model.model.model.language_model.per_layer_model_projection.'
tensors = {P + 'lora_A.weight': lowrank.bf16_bits(A), P + 'lora_B.weight': lowrank.bf16_bits(B)}     # A (r, in), B (out, r)
hdr, off = {'__metadata__': {'format': 'pt'}}, 0
for k, v in tensors.items():
    hdr[k] = {'dtype': 'BF16', 'shape': list(v.shape), 'data_offsets': [off, off + v.nbytes]}
    off += v.nbytes
hb = json.dumps(hdr, separators=(',', ':')).encode()
hb += b' ' * (-len(hb) % 8)
OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / 'adapter_model.safetensors', 'wb') as f:
    f.write(struct.pack('<Q', len(hb)) + hb)
    for v in tensors.values():
        f.write(np.ascontiguousarray(v).tobytes())
cfg = {"alpha_pattern": {}, "auto_mapping": None, "base_model_name_or_path": "google/gemma-4-e4b-it-qat-w4a16-ct", "bias": "none", "fan_in_fan_out": False, "inference_mode": True,
       "init_lora_weights": True, "layer_replication": None, "layers_pattern": None, "layers_to_transform": None, "loftq_config": {}, "lora_alpha": R, "lora_dropout": 0.0,
       "megatron_config": None, "megatron_core": "megatron.core", "modules_to_save": None, "peft_type": "LORA", "r": R, "rank_pattern": {}, "revision": None,
       "target_modules": ["per_layer_model_projection"], "task_type": "CAUSAL_LM", "use_dora": False, "use_rslora": False}
(OUT / 'adapter_config.json').write_text(json.dumps(cfg, indent=2))
D = lowrank.bf16(B) @ lowrank.bf16(A)
print(f'{OUT}: r={R} {KIND}  A{A.shape} B{B.shape}  safetensors {(OUT / "adapter_model.safetensors").stat().st_size} bytes  ||delta-W||/||W|| = {np.linalg.norm(D - W) / np.linalg.norm(W):.4f}')
