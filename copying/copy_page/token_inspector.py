"""Token Inspector for the copy test: which tokens went in, which tokens Gemma generated, and the next-token probabilities where the copy first goes wrong.  Standard library only; imported by server.py.

Everything is measured on the running llama-server, nothing is simulated:
  generated tokens   the copy request repeated with logprobs on (a separate request; /api/copy itself is untouched).  Token ids, bytes, probabilities and the top-TOP alternatives of every step are
                     choices[0].logprobs.content as returned, not re-tokenized.  The probabilities are the softmax of the model's logits before any sampling (the server's default), so under temperature 0
                     (greedy) the chosen token is the most probable one but its probability is not 1.  Adding logprobs does not change the answer; the page checks that it equals the one shown in B.
  input tokens       the prompt as the chat template renders it (/apply-template), tokenized by the server (/tokenize); the count must equal usage.prompt_tokens.  The tokens of A are located inside
                     that prompt by byte offsets, so what the model was shown is what is displayed: the template trims trailing whitespace of the message, and the result says when it did.
  correct token      the token the input's own tokenization has at the step in question.  When it is not among the top TOP, one scoring request asks for the whole distribution at exactly that step:
                     /completion with the prompt as token ids (the prompt tokens + the tokens Gemma had generated before the step, reasoning included), n_predict 1, n_probs = vocabulary size.  Its
                     distribution is compared with the generation's (same top-TOP tokens, same logprobs up to numerical noise); if it does not reproduce them the result says so.
Two kinds of first error are reported, because the same visible text can be written with different token sequences: the first generated token that differs from the input's own tokenization (token
level) and the step that writes the first byte of the answer that differs from the input (text level).  They are the same step when the model writes the wrong token and the wrong text at once.
"""
import math

TOP = 20            # alternatives requested per generated token
SHOW = 10           # rows shown per table
MAX_DELTA = 0.05    # scoring vs generation: largest tolerated difference in logprob
UNCERTAIN = 3       # a correct copy shows its least certain steps: at most this many, and only those below probability SURE
SURE = 0.95


class InspectError(Exception):
    pass


def piece_bytes(piece):                                 # /tokenize pieces are strings, or lists of byte values when the piece is not valid UTF-8
    return piece.encode() if isinstance(piece, str) else bytes(piece)


def lcp(a, b):
    i, n = 0, min(len(a), len(b))
    while i < n and a[i] == b[i]:
        i += 1
    return i


def inspect(text, body, prompt, call):
    """`body`: the copy request for `text`, `prompt`: its instruction line.  `call(name, path, body=None, trim=None)` does one request to the model server and logs it.  Returns the page's JSON."""
    names = {}

    def tv(i, b, **extra):                              # a token as the page needs it; tokens without bytes are special ones (end of turn, ...) and get the server's name for them
        if not b and i not in names:
            names[i] = call('detokenize', '/detokenize', {'tokens': [i]})['content']
        return {'id': i, 'bytes': list(b), **({'label': names[i]} if not b else {}), **extra}

    chat = call('chat', '/v1/chat/completions', {**body, 'logprobs': True, 'top_logprobs': TOP})
    choice, usage = chat['choices'][0], chat['usage']
    content, finish = choice['message'].get('content') or '', choice['finish_reason']
    gen = (choice.get('logprobs') or {}).get('content')
    if not gen:
        raise InspectError('the server returned no logprobs')
    out = {'text': text, 'answer': content, 'finish_reason': finish, 'usage': usage, 'top': TOP, 'checks': [], 'points': []}

    def check(name, ok, detail):
        out['checks'].append({'name': name, 'ok': bool(ok), 'detail': detail})
        return bool(ok)

    # --- the prompt as the model got it, and the tokens of A inside it
    rendered = call('apply-template', '/apply-template', {'messages': body['messages']})['prompt']
    toks = call('tokenize', '/tokenize', {'content': rendered, 'add_special': True, 'parse_special': True, 'with_pieces': True})['tokens']
    rb, pos, spans = rendered.encode(), 0, []
    for n, t in enumerate(toks):
        b = piece_bytes(t['piece'])
        if rb.startswith(b, pos):
            spans.append((pos, pos + len(b)))
            pos += len(b)
        elif n == 0:
            spans.append((0, 0))                        # the BOS that the tokenizer adds in front of the rendered text
        else:
            break
    prompt_ok = check('prompt tokens', len(spans) == len(toks) and pos == len(rb) and len(toks) == usage['prompt_tokens'],
                      f'{len(toks)} tokens from /tokenize on the rendered prompt; the server reports {usage["prompt_tokens"]} prompt tokens')
    typed, pb = text.encode(), prompt.encode()
    start = rb.find(prompt.rstrip('\n').encode())       # without its newline: with an empty input the template trims that too
    end = start + lcp(rb[start:], pb + typed) if start >= 0 else -1      # where the prompt stops agreeing with instruction + A: the end of what the model was shown of A
    a0 = start + len(pb)
    seen = rb[a0:end] if end > a0 else b''
    located = prompt_ok and start >= 0
    ref = []                                            # the input's own tokens, as they stand in the prompt
    if located and seen:
        ref = [tv(t['id'], piece_bytes(t['piece']), start=s - a0, straddle=s < a0) for (s, e), t in zip(spans, toks) if e > a0 and s < a0 + len(seen)]
    ref_ok = located and not any(t['straddle'] for t in ref)
    out['input'] = {'ok': located, 'tokens': ref, 'typed_bytes': len(typed), 'seen_bytes': len(seen), 'dropped': list(typed[len(seen):]),
                    'merged_with_instruction': any(t['straddle'] for t in ref)}
    eog = {gen[-1]['id']} if finish == 'stop' else set()
    nxt = next((t['id'] for (s, _), t in zip(spans, toks) if s >= end), None) if located else None
    if nxt is not None:
        eog.add(nxt)                                    # the template closes the user's turn with this token; the model is expected to close its own turn with it

    # --- the generated tokens, and which of them are the answer (the rest is reasoning)
    body_end = len(gen) - 1 if finish == 'stop' else len(gen)          # the last token of a stopped generation is the end-of-turn token, not part of the answer
    gb = [bytes(t['bytes']) for t in gen]
    cb, k, total = content.encode(), None, 0
    for j in range(body_end, -1, -1):                   # the tightest suffix of the generated tokens that spells the answer
        total += len(gb[j]) if j < body_end else 0
        if total > len(cb):
            break
        if total == len(cb) and b''.join(gb[j:body_end]) == cb:
            k = j
            break
    if k is None:
        out.update(status='unaligned', generated={'reasoning': [], 'answer': [tv(t['id'], gb[n], logprob=t['logprob']) for n, t in enumerate(gen)], 'terminator': None, 'retokenized': None})
        check('answer = generated tokens', False, 'the answer text is not the concatenation of the last generated tokens, so no step can be singled out')
        return out
    ans = gen[k:body_end]
    out['status'] = 'ok'
    out['generated'] = {'reasoning': [tv(t['id'], gb[n], logprob=t['logprob']) for n, t in enumerate(gen[:k])],
                        'answer': [tv(t['id'], gb[k + n], logprob=t['logprob']) for n, t in enumerate(ans)],
                        'terminator': tv(gen[body_end]['id'], gb[body_end], logprob=gen[body_end]['logprob']) if body_end < len(gen) else None}
    off_greedy = [n for n, t in enumerate(gen) if t['top_logprobs'][0]['id'] != t['id']]
    check('greedy', not off_greedy, 'every generated token is the most probable one' if not off_greedy else f'{len(off_greedy)} generated tokens are not the most probable one (first: #{off_greedy[0]})')
    re_toks = call('tokenize-answer', '/tokenize', {'content': content, 'add_special': False, 'parse_special': False, 'with_pieces': True})['tokens']
    out['generated']['retokenized'] = None if [t['id'] for t in re_toks] == [t['id'] for t in ans] else [tv(t['id'], piece_bytes(t['piece'])) for t in re_toks]

    # --- first errors, measured against what the model was shown (`seen`), not against what was typed
    want = seen
    offs = [0]                                          # byte offset of each answer token in the answer
    for n in range(len(ans)):
        offs.append(offs[-1] + len(gb[k + n]))
    points = {}                                         # answer-relative step -> reasons
    if cb != want:
        p = lcp(cb, want)
        points.setdefault(next((n for n in range(len(ans)) if offs[n + 1] > p), len(ans)), []).append('text')       # the step whose token covers the first differing byte
    if ref_ok:
        ids = [t['id'] for t in ref]
        d = next((n for n in range(min(len(ans), len(ids))) if ans[n]['id'] != ids[n]), None)
        if d is None and len(ans) != len(ids):
            d = min(len(ans), len(ids))
        if d is not None:
            points.setdefault(d, []).append('token')
    if not points:                                      # nothing went wrong: show the steps the model was least sure about, so a correct copy can be compared with a wrong one
        low = sorted((t['logprob'], n) for n, t in enumerate(gen[k:body_end + 1]))
        for lp, n in [x for x in low[:UNCERTAIN] if math.exp(x[0]) < SURE] or low[:1]:
            points[n] = ['least-certain']
    out['matches_seen'] = cb == want

    nv = []

    def vocab():
        if not nv:
            nv.append(call('models', '/v1/models')['data'][0]['meta']['n_vocab'])
        return nv[0]

    for step in sorted(points):
        g = k + step
        if g >= len(gen):                               # the generation was cut off before it could decide
            out['points'].append({'step': step, 'reasons': points[step], 'cutoff': True})
            continue
        entry = gen[g]
        R = want[offs[step]:]                           # what the copy still has to write at this step
        correct_id = None
        if ref_ok:
            correct_id = nxt if not R else next((t['id'] for t in ref if t['start'] == offs[step]), None)
        dist, source, verify = entry['top_logprobs'], f'the generation (top {TOP})', None
        if correct_id is not None and located and correct_id not in {e['id'] for e in dist}:
            ids = [t['id'] for t in toks] + [t['id'] for t in gen[:g]]
            sc = call('score', '/completion', {'prompt': ids, 'n_predict': 1, 'temperature': 0, 'seed': 1, 'cache_prompt': False, 'n_probs': vocab()}, trim=trim_scoring)
            dist, source = sc['completion_probabilities'][0]['top_logprobs'], 'a scoring request (whole vocabulary)'
            lp = {e['id']: e['logprob'] for e in entry['top_logprobs']}
            delta = max((abs(e['logprob'] - lp[e['id']]) for e in dist[:TOP] if e['id'] in lp), default=None)
            same_top = {e['id'] for e in dist[:TOP]} == set(lp)
            verify = {'top': TOP, 'prompt_ids': len(ids), 'evaluated': sc.get('tokens_evaluated'), 'max_delta': delta, 'same_top': same_top, 'sum_prob': sum(math.exp(e['logprob']) for e in dist)}
            check(f'scoring request, step {step}', sc.get('tokens_evaluated') == len(ids) and delta is not None and delta <= MAX_DELTA and dist[0]['id'] == entry['id'],
                  f'{len(ids)} prompt ids; same top-{TOP} tokens as the generation: {same_top}; largest difference to its logprobs: {"n/a" if delta is None else f"{delta:.4f}"}; '
                  f'most probable token {"the same" if dist[0]["id"] == entry["id"] else "DIFFERENT"}')
        by_id = {e['id']: e for e in dist}
        rank = {e['id']: n + 1 for n, e in enumerate(dist)}
        textok = lambda e: (bool(e['bytes']) and R.startswith(bytes(e['bytes']))) or (not R and e['id'] in eog)       # its text continues the copy correctly
        best = next((e['id'] for e in dist if textok(e)), None) if correct_id is None else None
        shown = [e['id'] for e in dist[:SHOW]] + [i for i in (entry['id'], correct_id, best) if i in by_id and rank[i] > SHOW]
        rows = [tv(i, bytes(by_id[i]['bytes']), rank=rank[i], logprob=by_id[i]['logprob'],
                   tags=[name for name, on in (('chosen', i == entry['id']), ('correct', i == correct_id), ('text-correct', textok(by_id[i]) and i != correct_id)) if on]) for i in shown]
        out['points'].append({'step': step, 'gen_index': g, 'reasons': points[step], 'source': source, 'verify': verify, 'rows': rows,
                              'chosen': tv(entry['id'], gb[g], logprob=entry['logprob'], rank=rank.get(entry['id'])),
                              'correct': None if correct_id is None else tv(correct_id, bytes(by_id[correct_id]['bytes']), logprob=by_id[correct_id]['logprob'], rank=rank[correct_id])
                                         if correct_id in by_id else {'id': correct_id, 'bytes': [], 'missing': True},
                              'remaining': R.decode(errors='replace')[:60], 'remaining_bytes': len(R)})
    return out


def trim_scoring(r):                                    # the whole-vocabulary reply is ~26 MB; the log keeps its top and the totals
    cp = r['completion_probabilities'][0]
    top = cp['top_logprobs']
    return {**r, 'completion_probabilities': [{**cp, 'top_logprobs': top[:100], 'top_logprobs_trimmed': {'kept': 100, 'of': len(top), 'sum_prob': sum(math.exp(e['logprob']) for e in top)}}]}
