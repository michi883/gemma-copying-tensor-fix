#!/usr/bin/env python3
"""Copy a GGUF and multiply ONE tensor by a constant, in place in the copy.  Usage: gguf_scale_patch.py SRC DST TENSOR FACTOR

Used for the per_layer_model_proj intervention of ROOT_CAUSE.md: the tensor is [2560, 10752], F16 in the QAT GGUF and BF16 in the llama-quantize output (55 MB); everything else in the file
is byte-identical to SRC."""
import os
import shutil
import sys

import numpy as np

if os.environ.get('GGUF_PY'):                                  # path of llama.cpp's gguf-py when `pip install gguf` is not used
    sys.path.insert(0, os.environ['GGUF_PY'])
import gguf

src, dst, name, factor = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
shutil.copyfile(src, dst)
t = next(t for t in gguf.GGUFReader(dst, 'r+').tensors if t.name == name)
T = gguf.GGMLQuantizationType
assert t.tensor_type in (T.F16, T.BF16), t.tensor_type
if t.tensor_type == T.F16:
    f = t.data.astype(np.float32) * factor
    t.data[:] = f.astype(np.float16)
else:                                                   # bf16: the high 16 bits of a float32, rounded to nearest even
    raw = np.ascontiguousarray(t.data).view(np.uint16)
    f = ((raw.astype(np.uint32) << 16).view(np.float32) * factor).astype(np.float32)
    u = f.view(np.uint32)
    t.data.view(np.uint16)[:] = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
t.data.flush()
print(f'{name}: {t.tensor_type.name} {[int(x) for x in t.shape]}  norm -> {np.linalg.norm(f):.3f}  (x{factor:.5f})')
