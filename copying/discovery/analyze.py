#!/usr/bin/env python3
"""Paired analysis of results_<kind>_{orig,patched}_<mode>.jsonl.  Standard library only.

    python3 analyze.py words nothink            # tables to stdout;  --json writes analysis_<kind>_<mode>.json;  --list N prints N failing items per group

Pass = `exact_unfenced` (byte-exact, or exact after removing one enclosing ``` fence; fenced replies are counted separately).  Wilson 95% intervals; McNemar exact (two-sided) for paired
original-vs-patched outcomes.  Margin = log-probability margin at the decisive step: for a failure, the step where the output stops matching (a dropped suffix: the end-of-turn step), for a
pass, the least certain generated step.
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(kind, server, mode):
    p = HERE / f'results_{kind}_{server}_{mode}.jsonl'
    return {r['id']: r for r in map(json.loads, p.read_text().splitlines()) if r.get('output') is not None or 'error' in r} if p.exists() else {}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def ok(r):
    return r.get('exact_unfenced', False)


def fmt(k, n):
    lo, hi = wilson(k, n)
    return f'{k}/{n} = {100 * k / n:.2f}% [{100 * lo:.2f}, {100 * hi:.2f}]' if n else '0/0'


def decisive_step(r):
    """(chosen token, p_chosen, p_expected or None, top-1 alternative) at the first wrong step; for passes the minimum p_chosen step."""
    toks = r.get('tokens') or []
    if not toks:
        return None
    if ok(r):
        i = min(range(len(toks)), key=lambda j: toks[j]['lp'])
        return {'step': i, 'p_chosen': math.exp(toks[i]['lp']), 'p_best_other': max([math.exp(a['lp']) for a in toks[i]['top'] if a['id'] != toks[i]['id']] or [0.0])}
    d = r.get('diverge_at')
    i = d if d is not None else len(toks) - 1                                            # no divergence = stopped early: the last step (end of turn)
    pe = None
    exp_prefix = r['text']
    acc = ''.join(t['tok'] for t in toks[:i])
    rest = exp_prefix[len(acc):] if exp_prefix.startswith(acc) else ''
    for a in toks[i]['top']:
        if rest.startswith(a['tok']) and a['tok']:
            pe = max(pe or 0, math.exp(a['lp']))
    return {'step': i, 'chosen': toks[i]['tok'], 'p_chosen': math.exp(toks[i]['lp']), 'p_expected': pe, 'expected_rest': rest[:12]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kind')
    ap.add_argument('mode')
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--list', type=int, default=0)
    ap.add_argument('--group', default='stratum', help='record field to group by (stratum, cls, kind, ends_s, ...)')
    args = ap.parse_args()
    O, P = load(args.kind, 'orig', args.mode), load(args.kind, 'patched', args.mode)
    ids = sorted(set(O) & set(P))
    err = [i for i in ids if 'error' in O[i] or 'error' in P[i]]
    ids = [i for i in ids if i not in err]
    print(f'{args.kind}/{args.mode}: {len(O)} original, {len(P)} patched results, {len(ids)} paired, {len(err)} with errors')
    res = {}
    for grp in ('anchor_fail', 'anchor_pass', 'sample'):
        g = [i for i in ids if O[i]['group'] == grp]
        if not g:
            continue
        fo, fp = sum(not ok(O[i]) for i in g), sum(not ok(P[i]) for i in g)
        print(f'\n[{grp}] n={len(g)}  original fails {fmt(fo, len(g))}   patched fails {fmt(fp, len(g))}')
        res[grp] = {'n': len(g), 'orig_fail': fo, 'patched_fail': fp}
        if grp.startswith('anchor'):
            for i in g:
                print(f'   {O[i]["text"]!r:22} orig {O[i]["output"]!r:22} patched {P[i]["output"]!r}')
    S = [i for i in ids if O[i]['group'] == 'sample']
    # strata table
    by = defaultdict(list)
    for i in S:
        by[O[i].get(args.group, '?')].append(i)
    print(f'\nBy {args.group}: original fails | patched fails | orig-fail&patched-pass (fixed) | orig-pass&patched-fail (regression) | both fail    McNemar p')
    tot = Counter()
    for key in sorted(by):
        g = by[key]
        fixed = sum((not ok(O[i])) and ok(P[i]) for i in g)
        reg = sum(ok(O[i]) and not ok(P[i]) for i in g)
        both = sum((not ok(O[i])) and (not ok(P[i])) for i in g)
        fo, fp = fixed + both, reg + both
        pop = O[g[0]].get('stratum_population')
        print(f'  {key:24} n={len(g):5d}  {fmt(fo, len(g)):34} {fmt(fp, len(g)):34} fixed {fixed:4d}  reg {reg:3d}  both {both:3d}   p={mcnemar_p(fixed, reg):.3g}' + (f'  (population {pop})' if pop else ''))
        tot.update(n=len(g), fo=fo, fp=fp, fixed=fixed, reg=reg, both=both)
        res.setdefault('groups', {})[key] = {'n': len(g), 'orig_fail': fo, 'patched_fail': fp, 'fixed': fixed, 'regressed': reg, 'both': both}
    print(f'  {"ALL SAMPLE":24} n={tot["n"]:5d}  {fmt(tot["fo"], tot["n"]):34} {fmt(tot["fp"], tot["n"]):34} fixed {tot["fixed"]:4d}  reg {tot["reg"]:3d}  both {tot["both"]:3d}   p={mcnemar_p(tot["fixed"], tot["reg"]):.3g}')
    cat_o = Counter(O[i]['category'] for i in S if not ok(O[i]))
    cat_p = Counter(P[i]['category'] for i in S if not ok(P[i]))
    print('\nFailure categories (sample): original', dict(cat_o.most_common()), '| patched', dict(cat_p.most_common()))
    fenced_o, fenced_p = sum(O[i]['fenced'] for i in S), sum(P[i]['fenced'] for i in S)
    print(f'Fenced replies: original {fenced_o}, patched {fenced_p}; replies with reasoning: original {sum(O[i]["reasoning"] for i in S)}, patched {sum(P[i]["reasoning"] for i in S)}')
    res['categories'] = {'orig': dict(cat_o), 'patched': dict(cat_p)}
    # margins
    def margins(R, which):
        m = []
        for i in S:
            if (which == 'fail') != (not ok(R[i])):
                continue
            d = decisive_step(R[i])
            if d:
                m.append(d)
        return m
    for name, R in (('original', O), ('patched', P)):
        mf = margins(R, 'fail')
        mp = margins(R, 'pass')
        if mp:
            ps = sorted(d['p_chosen'] for d in mp)
            q = lambda f: ps[int(f * (len(ps) - 1))]
            thin = sum(d['p_chosen'] < 0.5 for d in mp)
            print(f'{name}: passes {len(mp)}; least-certain-step p(chosen) median {q(.5):.4f}, 5th pct {q(.05):.4f}, min {ps[0]:.4f}; thin passes (min p<0.5): {thin} ({100 * thin / len(mp):.2f}%)')
        if mf:
            pe = [d['p_expected'] for d in mf if d.get('p_expected') is not None]
            print(f'{name}: failures {len(mf)}; p(expected) at the failing step: n={len(pe)} median {sorted(pe)[len(pe) // 2] if pe else float("nan"):.4f}; chosen tokens {Counter(d["chosen"] for d in mf).most_common(4)}')
    # failing items
    if args.list:
        for title, R, Q in (('FAIL on original, OK on patched', O, P), ('FAIL on both', O, P), ('OK on original, FAIL on patched (regressions)', P, O)):
            if title.startswith('OK'):
                pick = [i for i in S if ok(O[i]) and not ok(P[i])]
            elif 'both' in title:
                pick = [i for i in S if not ok(O[i]) and not ok(P[i])]
            else:
                pick = [i for i in S if not ok(O[i]) and ok(P[i])]
            print(f'\n{title}: {len(pick)}')
            for i in pick[:args.list]:
                print(f'   {O[i]["stratum"]:18} {O[i]["text"][:60]!r:62} orig {O[i]["output"][:60]!r:62} patched {P[i]["output"][:60]!r}   [{O[i]["category"]} | {P[i]["category"]}]')
    if args.json:
        (HERE / f'analysis_{args.kind}_{args.mode}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
