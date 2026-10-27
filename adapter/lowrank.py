"""Shared low-rank helpers for the per_layer_model_proj -> LoRA study.

LoRA target: delta = W, so that W + delta = 2W.  A rank-r LoRA is B (out x r) @ A (r x in); vLLM stores both in bf16 and multiplies B by lora_alpha / r (we use alpha = r -> 1.0).
  svd  : best rank-r approximation of W (Eckart-Young):  B = U_r sqrt(S_r),  A = sqrt(S_r) V_r^T
  data : delta = W P_r, P_r = top-r principal subspace of token-embedding rows (the inputs W actually sees):  B = W V_r^T,  A = V_r
"""
import numpy as np


def bf16(a):
    """float32 -> bfloat16 -> float32 (round to nearest even), the precision at which vLLM holds LoRA factors (lora_dtype=auto -> model dtype)"""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32).view(np.float32)


def bf16_bits(a):
    return (np.ascontiguousarray(bf16(a)).view(np.uint32) >> 16).astype(np.uint16)


def svd_factors(W, r):
    U, S, Vt = np.linalg.svd(W.astype(np.float32), full_matrices=False)
    return bf16(U[:, :r] * np.sqrt(S[:r])), bf16(np.sqrt(S[:r])[:, None] * Vt[:r])


def data_factors(W, E_fit, r):
    _, _, Vt = np.linalg.svd(E_fit.astype(np.float32), full_matrices=False)
    Vr = Vt[:r]
    return bf16(W.astype(np.float32) @ Vr.T), bf16(Vr)


def factors(kind, W, r, E_fit=None):
    if kind == 'svd':
        return svd_factors(W, r)
    if kind == 'data':
        return data_factors(W, E_fit, r)
    raise ValueError(kind)
