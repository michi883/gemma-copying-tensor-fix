#!/usr/bin/env python3
"""Measure the size of Gemma 4 E4B's `per_layer_model_projection` tensor in three public checkpoints, without downloading them.

    python3 inspect_tensor.py                          # fetch ~125 MB in total and print the table
    python3 inspect_tensor.py --save-w4a16 W.npy       # also keep the dequantised competition tensor (float32 [10752, 2560]); input of adapter/build_adapter.py
    python3 inspect_tensor.py --skip-standard          # skip the 55 MB standard-release tensor

Only HTTP range requests are made: the safetensors header, then the bytes of this one tensor (nothing else is read).  Needs Python 3.10+ and numpy.  No model is run.

The three checkpoints (all public, not gated, license apache-2.0 on their model cards when this was written):
  qat-w4a16-ct   google/gemma-4-E4B-it-qat-w4a16-ct        4-bit-ready (QAT) release as served in the Kaggle competition (compressed-tensors, int4 + bf16 scales, group 32)
  qat-unquant    google/gemma-4-E4B-it-qat-q4_0-unquantized  the same QAT release in bf16
  standard       google/gemma-4-E4B-it                       the ordinary (non-QAT) release

The W4A16 unpacking is the code of experiments/lora_feasibility/stepb/{fetch,deq_compare}.py of the research repository (nibble i of int32 word k holds column 8k+i, value = nibble - 8,
times the bf16 scale of its group of 32 columns).  This file is new for the release; it was checked against the numbers recorded in the research repository (see REPRODUCE.md).
"""
import argparse
import json
import struct
import sys
import urllib.request

import numpy as np

REPOS = {
    'qat-w4a16-ct': ('google/gemma-4-E4B-it-qat-w4a16-ct', '6cd26aaa2357fb2bad8c51699a7558a4d1a965bb'),
    'qat-unquant': ('google/gemma-4-E4B-it-qat-q4_0-unquantized', '476025a01dbf99361c062bbeca3d6a76bb4c4566'),
    'standard': ('google/gemma-4-E4B-it', 'ee0ef6023621cff504d758262d4e04895a5af4a2'),
}
P = 'model.language_model.per_layer_model_projection.'
EPS = 1e-6                                                       # rms_norm_eps of the model, the "safety number" of the paper


def rng(url, a, b):
    req = urllib.request.Request(url, headers={'Range': f'bytes={a}-{b}'})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read()


def header(url):
    n = struct.unpack('<Q', rng(url, 0, 7))[0]
    return n, json.loads(rng(url, 8, 8 + n - 1))


def tensor_bytes(url, n, hdr, key):
    a, b = hdr[key]['data_offsets']
    raw = rng(url, 8 + n + a, 8 + n + b - 1)
    assert len(raw) == b - a, (key, len(raw), b - a)
    return raw


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def load(name):
    repo, rev = REPOS[name]
    url = f'https://huggingface.co/{repo}/resolve/{rev}/model.safetensors'
    n, hdr = header(url)
    if P + 'weight_packed' in hdr:                              # compressed-tensors pack-quantized
        packed = np.frombuffer(tensor_bytes(url, n, hdr, P + 'weight_packed'), dtype=np.uint32).reshape(hdr[P + 'weight_packed']['shape'])
        scale = bf16_to_f32(np.frombuffer(tensor_bytes(url, n, hdr, P + 'weight_scale'), dtype=np.uint16).reshape(hdr[P + 'weight_scale']['shape']))
        rows, cols = packed.shape[0], packed.shape[1] * 8
        nib = np.stack([(packed >> (4 * i)) & 0xF for i in range(8)], axis=-1).reshape(rows, cols).astype(np.int8) - 8
        return nib.astype(np.float32) * np.repeat(scale, cols // scale.shape[1], axis=1), hdr[P + 'weight_packed']['dtype'] + '+' + hdr[P + 'weight_scale']['dtype'] + ' (int4, dequantised)'
    assert hdr[P + 'weight']['dtype'] == 'BF16', hdr[P + 'weight']
    return bf16_to_f32(np.frombuffer(tensor_bytes(url, n, hdr, P + 'weight'), dtype=np.uint16).reshape(hdr[P + 'weight']['shape'])), 'BF16'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--save-w4a16', metavar='PATH', help='write the dequantised competition tensor as a float32 .npy')
    ap.add_argument('--skip-standard', action='store_true', help='do not fetch the standard-release tensor')
    args = ap.parse_args()
    names = [k for k in REPOS if not (args.skip_standard and k == 'standard')]
    out, W = {}, {}
    for k in names:
        print(f'fetching {REPOS[k][0]} @ {REPOS[k][1][:8]} ...', file=sys.stderr, flush=True)
        W[k], kind = load(k)
        w = W[k].astype(np.float64)
        out[k] = {'repo': REPOS[k][0], 'revision': REPOS[k][1], 'stored_as': kind, 'shape': list(W[k].shape), 'frobenius_norm': float(np.linalg.norm(w)), 'rms': float(np.sqrt((w ** 2).mean()))}
    ref = out.get('standard', {}).get('frobenius_norm')
    print(f'{"checkpoint":14}{"stored as":28}{"shape":16}{"Frobenius":>11}{"rms":>11}{"vs standard":>13}')
    for k, r in out.items():
        print(f'{k:14}{r["stored_as"]:28}{str(tuple(r["shape"])):16}{r["frobenius_norm"]:11.4f}{r["rms"]:11.3e}' + (f'{r["frobenius_norm"] / ref:13.4f}' if ref else f'{"-":>13}'))
    if ref:
        print(f'\nstandard / QAT (bf16) = {ref / out["qat-unquant"]["frobenius_norm"]:.1f}x;  standard / QAT (W4A16, dequantised) = {ref / out["qat-w4a16-ct"]["frobenius_norm"]:.1f}x')
    print(f'RMSNorm safety number eps = {EPS:g}')
    if args.save_w4a16:
        np.save(args.save_w4a16, W['qat-w4a16-ct'].astype(np.float32))
        print('saved', args.save_w4a16, file=sys.stderr)
    print(json.dumps(out, indent=1), file=sys.stderr)


if __name__ == '__main__':
    main()
