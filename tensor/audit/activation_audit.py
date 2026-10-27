#!/usr/bin/env python3
"""Phase 1 / part B: information flow through the QAT E4B weights run in mlx (the validated path of identifier_fidelity/mlx_scan.py gguf_stages), per_layer_model_projection x1 (as stored) vs x2.
Read-only: weights are changed in memory only.  Hooks: every RMSNorm records mean(x^2) of its input (vs eps=1e-6) and the rms of its output; every decoder layer records its output; the
per-layer-input builder records the projection part and the embedding-lookup part separately.   Run: python activation_audit.py   Output: activation_audit.json + stdout tables."""
import json, os, re, sys, time
os.environ['PROJ_FACTOR'] = '1'                       # patch the GGUF's small tensors in exactly as stored
sys.path.insert(0, '/path/to/identifier_fidelity')
sys.path.insert(0, '/path/to/identifier_fidelity/copy_page')
import numpy as np, mlx.core as mx, mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.models import gemma4_text as g4
import mlx_scan as ms

EPS, F32 = 1e-6, mx.float32
t0 = time.time()
model, tok = load(ms.SNAP)
tensors = {t.name: t for t in ms.gguf.GGUFReader(ms.GGUF).tensors}
try: mx.set_memory_limit(30 * 2**30)                       # backstop: fail in mlx rather than take the machine down
except Exception as e: print('no memory limit', e)
for grp in ('matrices', 'small', 'token_embd'):
    ms.patch(model, grp, tensors)
class RowLookup:
    """per_layer_token_embd (Q6_K, 2.8e9 elements) served row by row from the GGUF for the token ids that occur: same numbers as patching the whole table, ~11 GB less memory"""
    def __init__(self, t): self.t, self.cache = t, {}
    def __call__(self, ids):
        a = np.array(ids); flat = a.reshape(-1)
        for i in np.unique(flat):
            if int(i) not in self.cache: self.cache[int(i)] = ms.to_f32(self.t, slice(int(i), int(i) + 1))[0]
        return mx.array(np.stack([self.cache[int(i)] for i in flat]).reshape(*a.shape, -1)).astype(mx.bfloat16)
print(f'model built in {time.time()-t0:.0f}s', flush=True)
inner = model.language_model.model
dict.pop(inner, 'embed_tokens_per_layer'); inner.embed_tokens_per_layer = RowLookup(tensors['per_layer_token_embd.weight']); mx.clear_cache()
KEY = ms.LM + 'per_layer_model_projection.weight'
W1 = dict(tree_flatten(model.parameters()))[KEY]
for n, m in model.named_modules():
    if isinstance(m, nn.RMSNorm): m._spy = n.replace('language_model.model.', '')
for i, l in enumerate(inner.layers): l._li = i

ON, REC, HID, PL = [False], {}, {}, {}
orig_rms = nn.RMSNorm.__call__
def spy_rms(self, x):
    out = orig_rms(self, x)
    if ON[0]:
        REC[getattr(self, '_spy', '?')] = (mx.mean(x.astype(F32) ** 2, -1).reshape(-1), mx.sqrt(mx.mean(out.astype(F32) ** 2, -1)).reshape(-1))
    return out
nn.RMSNorm.__call__ = spy_rms
orig_layer = g4.DecoderLayer.__call__
def spy_layer(self, x, *a, **k):
    out = orig_layer(self, x, *a, **k)
    if ON[0]: HID[self._li] = (mx.sqrt(mx.mean(x.astype(F32) ** 2, -1)), out[0].astype(F32))
    return out
g4.DecoderLayer.__call__ = spy_layer
def proj_fn(input_embeddings, per_layer_inputs=None):
    pp = inner.per_layer_model_projection(input_embeddings) * inner.per_layer_projection_scale
    pp = mx.unflatten(pp, -1, (inner.config.num_hidden_layers, inner.hidden_size_per_layer_input))
    ppn = inner.per_layer_projection_norm(pp)
    res = (ppn + per_layer_inputs) * inner.per_layer_input_scale
    if ON[0]: PL.update(proj=ppn.astype(F32), lookup=per_layer_inputs.astype(F32), res=res.astype(F32))
    return res
inner._project_per_layer_inputs = proj_fn

def run(ids, rec):
    ON[0] = rec; REC.clear(); HID.clear(); PL.clear()
    out = model(mx.array([ids])); logits = (out.logits if hasattr(out, 'logits') else out)[0].astype(F32)
    mx.eval(logits, *[v for p in REC.values() for v in p], *[v for p in HID.values() for v in p], *PL.values()); ON[0] = False
    return logits

def setx(f):
    ms.put(model, KEY, W1 * f); mx.eval(model.parameters())

def render(text): return tok.encode(tok.apply_chat_template([{'role': 'user', 'content': text}], add_generation_prompt=True, tokenize=False), add_special_tokens=False)
COPY = 'Copy the following text exactly. Output only the copied text:\n'
seqs = {}
for t in ['directs', 'allow_redirects', 'pulls', 'redirects']:
    a = tok.encode(t, add_special_tokens=False); seqs['copy:' + t] = (render(COPY + t) + a, len(render(COPY + t)), a)
setx(1.0)
for name, q in [('reason', 'A train leaves at 3pm going 60 km/h. A second train leaves the same station at 4pm going 80 km/h on the same track. When does the second catch up? Think step by step.'),
                ('code', 'Write a Python function dedupe(xs) that removes duplicates from a list while keeping the first occurrence of each item. Include a short docstring.')]:
    ids = render(q)
    for _ in range(48):
        ids.append(int(mx.argmax(run(ids, False)[-1]).item()))
    seqs[name] = (ids, len(render(q)), ids[len(render(q)):])
    print(name, repr(tok.decode(seqs[name][2]))[:200], flush=True)

def stats(ms_):
    g = np.sqrt(ms_ / (ms_ + EPS))
    return dict(ms_median=float(np.median(ms_)), ms_min=float(ms_.min()), below_eps=float((ms_ < EPS).mean()), below_100eps=float((ms_ < 100 * EPS).mean()), gain_median=float(np.median(g)), gain_p5=float(np.percentile(g, 5)), gain_min=float(g.min()))

out = {'seqs': {k: {'n': len(v[0]), 'prompt': v[1]} for k, v in seqs.items()}, 'norms': {}, 'flow': {}, 'logits': {}}
res = {}
for f in (1.0, 2.0):
    setx(f); res[f] = {}
    for k, (ids, npr, ans) in seqs.items():
        lg = run(ids, True)
        res[f][k] = dict(lg=np.array(lg), REC={n: (np.array(a), np.array(b)) for n, (a, b) in REC.items()}, HID={i: (np.array(a), np.array(b)) for i, (a, b) in HID.items()}, PL={n: np.array(v) for n, v in PL.items()})
    print(f'x{f} done {time.time()-t0:.0f}s', flush=True)

# --- 1. every RMSNorm: eps regime, family aggregated over layers and sequences (x1 = as stored)
fam = {}
for k in seqs:
    for n, (m, o) in res[1.0][k]['REC'].items():
        fam.setdefault(re.sub(r'layers\.\d+\.', 'layers.N.', n), []).append(m)
print('\n== RMSNorm input mean-square vs eps=1e-6, x1 (as stored): all tokens, layers, 6 sequences ==')
print(f'{"norm (family)":50}{"ms median":>10}{"ms min":>10}{"<eps":>7}{"<100eps":>8}{"gain med":>9}{"gain p5":>8}{"gain min":>9}')
for n, ms_ in sorted(fam.items(), key=lambda kv: np.median(np.concatenate(kv[1]))):
    s = stats(np.concatenate(ms_)); out['norms'][n] = s
    print(f'{n:50}{s["ms_median"]:10.2e}{s["ms_min"]:10.2e}{s["below_eps"]:7.1%}{s["below_100eps"]:8.1%}{s["gain_median"]:9.3f}{s["gain_p5"]:8.3f}{s["gain_min"]:9.3f}')

# --- 2. per-layer input: proj part vs lookup part, x1 vs x2
print('\n== per-layer input p_l = (RMSNorm(W e) + lookup)/sqrt2 : energy of the two parts and direction change x1 -> x2 ==')
for k in ['copy:directs', 'reason', 'code']:
    a, b = res[1.0][k]['PL'], res[2.0][k]['PL']
    rp1, rp2, rl = [np.sqrt((x ** 2).mean(-1)).mean() for x in (a['proj'], b['proj'], a['lookup'])]
    cos = ((a['res'] * b['res']).sum(-1) / (np.linalg.norm(a['res'], axis=-1) * np.linalg.norm(b['res'], axis=-1))).mean()
    print(f'{k:16} rms(proj part) x1 {rp1:.3f} x2 {rp2:.3f}   rms(lookup part) {rl:.3f}   proj/lookup energy ratio x1 {rp1/rl:.3f} -> x2 {rp2/rl:.3f}   mean cos(p_l x1, p_l x2) {cos:.4f}')
    out['flow'][k + ':per_layer_input'] = dict(proj_rms_x1=float(rp1), proj_rms_x2=float(rp2), lookup_rms=float(rl), cos=float(cos))

# --- 3. what reaches the residual stream: per-layer branch (post_per_layer_input_norm output) and layer outputs, x1 vs x2
print('\n== layer-by-layer effect of x2 on the residual stream (copy:directs / reason / code, mean over positions) ==')
print(f'{"layer":>5} {"resid rms":>10} {"pl-branch pre-ms":>17} {"pl-branch rms":>14} {"branch/resid":>13} {"|dh|/|h| x1->x2":>16} {"cos(h1,h2)":>11}')
rows = []
for L in range(0, inner.config.num_hidden_layers):
    r = {}
    for k in ['copy:directs', 'reason', 'code']:
        h1, h2 = res[1.0][k]['HID'][L][1], res[2.0][k]['HID'][L][1]
        resid = res[1.0][k]['HID'][L][0].mean()
        pre, br = res[1.0][k]['REC'][f'layers.{L}.post_per_layer_input_norm']
        r.setdefault('resid', []).append(resid); r.setdefault('pre', []).append(np.median(pre)); r.setdefault('br', []).append(br.mean())
        r.setdefault('d', []).append((np.linalg.norm(h2 - h1, axis=-1) / np.linalg.norm(h1, axis=-1)).mean())
        r.setdefault('c', []).append(((h1 * h2).sum(-1) / (np.linalg.norm(h1, axis=-1) * np.linalg.norm(h2, axis=-1))).mean())
    m = {a: float(np.mean(b)) for a, b in r.items()}; rows.append(m)
    if L % 3 == 0 or L >= 38 or L < 2:
        print(f'{L:5d} {m["resid"]:10.2f} {m["pre"]:17.2e} {m["br"]:14.3f} {m["br"]/m["resid"]:13.3f} {m["d"]:16.4f} {m["c"]:11.5f}')
out['flow']['layers'] = rows

# --- 4. outputs: teacher-forced next-token behaviour x1 vs x2
def lsm(l): l = l - l.max(-1, keepdims=True); return l - np.log(np.exp(l).sum(-1, keepdims=True))
print('\n== outputs, teacher-forced (same tokens) ==')
for k, (ids, npr, ans) in seqs.items():
    l1, l2 = res[1.0][k]['lg'], res[2.0][k]['lg']
    pos = np.arange(npr - 1, len(ids) - 1)                       # positions that predict the answer tokens
    a1, a2 = lsm(l1[pos]), lsm(l2[pos]); p1, p2 = np.exp(a1), np.exp(a2)
    kl = (p1 * (a1 - a2)).sum(-1); tgt = np.array(ids[npr:])
    top1_slope = float((1 - (l1[pos].max(-1) / 30) ** 2).mean())
    row = dict(kl_mean=float(kl.mean()), kl_max=float(kl.max()), argmax_agree=float((p1.argmax(-1) == p2.argmax(-1)).mean()), p_tgt_x1=float(p1[np.arange(len(pos)), tgt].mean()), p_tgt_x2=float(p2[np.arange(len(pos)), tgt].mean()),
               nll_x1=float(-a1[np.arange(len(pos)), tgt].mean()), nll_x2=float(-a2[np.arange(len(pos)), tgt].mean()), softcap_slope_top1=top1_slope)
    if k.startswith('copy:'):
        row['p_target_per_step_x1'] = [round(float(p1[i, tgt[i]]), 4) for i in range(len(pos))]; row['p_target_per_step_x2'] = [round(float(p2[i, tgt[i]]), 4) for i in range(len(pos))]
        row['answer_tokens'] = [tok.decode([t]) for t in tgt]
    out['logits'][k] = row
    print(f'{k:22} KL(x1||x2) mean {row["kl_mean"]:.4f} max {row["kl_max"]:.3f}  argmax agree {row["argmax_agree"]:.2f}  p(target) x1 {row["p_tgt_x1"]:.3f} x2 {row["p_tgt_x2"]:.3f}  NLL x1 {row["nll_x1"]:.3f} x2 {row["nll_x2"]:.3f}  softcap slope@top1 {top1_slope:.3f}' + (f'  steps {row["answer_tokens"]} x1 {row["p_target_per_step_x1"]} x2 {row["p_target_per_step_x2"]}' if k.startswith('copy:') else ''))

# --- 5. follow-ups (same run, same tokens)
print('\n== where is post_per_layer_input_norm input < 1e-4 (x1)?  (seq, layer, position, token, ms) ==')
low = []
for k, (ids, npr, ans) in seqs.items():
    m = res[1.0][k]['REC']
    T_ = len(ids)
    for L in range(inner.config.num_hidden_layers):
        a = m[f'layers.{L}.post_per_layer_input_norm'][0]
        for pos in np.where(a < 1e-4)[0]: low.append((k, L, int(pos), tok.decode([ids[pos]]), float(a[pos])))
print(len(low), 'entries;', low[:12])
out['flow']['post_pl_norm_low_ms'] = low[:50]

print('\n== softcap (30): pre-cap top-1 logit and margin top1-top2 at the answer positions, x1 ==')
for k, (ids, npr, ans) in seqs.items():
    lg = res[1.0][k]['lg'][npr - 1:len(ids) - 1]
    pre = 30 * np.arctanh(np.clip(lg / 30, -0.999999, 0.999999)); o = np.argsort(-lg, -1)
    t1 = np.take_along_axis(lg, o[:, :1], -1)[:, 0]; t2 = np.take_along_axis(lg, o[:, 1:2], -1)[:, 0]
    p1_, p2_ = 30 * np.arctanh(t1 / 30), 30 * np.arctanh(t2 / 30)
    print(f'{k:22} top1 pre-cap logit median {np.median(p1_):6.1f} (post {np.median(t1):5.1f})  share pre-cap top1 > 30: {(p1_ > 30).mean():.0%}  margin pre-cap {np.median(p1_ - p2_):5.2f} post-cap {np.median(t1 - t2):5.2f}')
    out['logits'][k]['top1_precap_median'] = float(np.median(p1_)); out['logits'][k]['share_top1_precap_gt30'] = float((p1_ > 30).mean())

print('\n== logit lens at the position that predicts the lone `s` (layer output -> final norm -> tied head -> softcap): p(s) and rank of s ==')
S = tok.encode('s', add_special_tokens=False)[0]
lens = {}
for k in ['copy:directs', 'copy:pulls', 'copy:redirects']:
    ids, npr, ans = seqs[k]; pos = len(ids) - 2                      # token `direct`/`pull`/`redirect`: its output predicts `s`
    row = {}
    for f in (1.0, 2.0):
        r = []
        for L in range(inner.config.num_hidden_layers):
            h = mx.array(res[f][k]['HID'][L][1][:, pos:pos + 1])
            z = model.language_model.model.embed_tokens.as_linear(inner.norm(h)); z = mx.tanh(z / 30.0) * 30.0
            z = np.array(z.astype(F32))[0, 0]; pr = np.exp(z - z.max()); pr /= pr.sum()
            r.append((L, float(pr[S]), int((z > z[S]).sum()) + 1))
        row[f] = r
    lens[k] = row
    print(k, 'layer: p(s)/rank  x1 | x2')
    print('   ' + '  '.join(f'L{L}:{row[1.0][L][1]:.3f}/{row[1.0][L][2]}|{row[2.0][L][1]:.3f}/{row[2.0][L][2]}' for L in list(range(24, 42, 2)) + [41]))
out['flow']['logit_lens_s'] = lens
json.dump(out, open('activation_audit.json', 'w'), indent=1)
print(f'done {time.time()-t0:.0f}s')
