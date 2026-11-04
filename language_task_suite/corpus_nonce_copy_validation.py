"""Three-piece BPE windows and paired edit checks for nonce-code copying."""

from __future__ import annotations

import random

import torch

from corpus_nonce_copy_dgp import draw_pair


PAD = 50256


def _span_tokens(encoding, text, span):
    start, end = span
    indices = [i for i, (a, b) in enumerate(encoding.offsets)
               if a < end and b > start]
    if not indices or indices != list(range(indices[0], indices[-1] + 1)):
        return None
    offsets = encoding.offsets
    first, last = offsets[indices[0]], offsets[indices[-1]]
    if not (first[0] == start or
            (first[0] == start - 1 and text[start - 1].isspace())):
        return None
    if last[1] != end:
        return None
    if any(offsets[i][1] != offsets[i + 1][0] for i in indices[:-1]):
        return None
    return tuple(indices)


def _row(example, tokenizer, n_ctx):
    encoding = tokenizer.encode(example.text, add_special_tokens=False)
    source = _span_tokens(encoding, example.text, example.previous_span)
    target = _span_tokens(encoding, example.text, example.answer_span)
    if source is None or target is None or len(source) != 3 or len(target) != 3:
        return None
    if source[-1] >= target[0]:
        return None
    begin = min(source[0], max(0, target[-1] - (3 * n_ctx) // 4))
    if target[-1] - begin >= n_ctx or target[0] - begin < 1:
        return None
    ids = encoding.ids[begin:begin + n_ctx + 1]
    x = torch.full((n_ctx,), PAD, dtype=torch.long)
    y = torch.full((n_ctx,), -100, dtype=torch.long)
    x[:len(ids) - 1] = torch.tensor(ids[:-1])
    y[:len(ids) - 1] = torch.tensor(ids[1:])
    site = torch.tensor([i - begin - 1 for i in target])
    target_ids = torch.tensor([encoding.ids[i] for i in target])
    assert torch.equal(y[site], target_ids)
    return {'x': x, 'y': y, 'site': site, 'target': target_ids,
            'encoding': encoding, 'begin': begin, 'source': source,
            'target_indices': target}


def aligned_pair(plan, tokenizer, pool, n_ctx=256,
                 rng: random.Random | None = None, attempts=24):
    rng = rng or random.Random(0)
    for _ in range(attempts):
        examples = draw_pair(plan, pool, rng)
        if examples is None:
            return None, 'no_nonce_pair'
        left, right = examples
        a = _row(left, tokenizer, n_ctx)
        b = _row(right, tokenizer, n_ctx)
        if a is None or b is None:
            continue
        if (not torch.equal(a['site'], b['site']) or
                torch.any(a['target'] == b['target']) or
                a['begin'] != b['begin'] or a['source'] != b['source']):
            continue
        prefix_end = a['target_indices'][-1] + 1
        if a['encoding'].offsets[:prefix_end] != \
                b['encoding'].offsets[:prefix_end]:
            continue
        # Every changed prefix token must belong to an earlier selected
        # occurrence; all three character pieces differ in the two arms.
        first_site = a['site'][0].item()
        expected = set()
        for span in plan.spans[:plan.target_occurrence]:
            ia = _span_tokens(a['encoding'], left.text, span)
            ib = _span_tokens(b['encoding'], right.text, span)
            if ia is None or ib != ia or len(ia) != 3:
                expected = set()
                break
            expected.update(i - a['begin'] for i in ia
                            if a['begin'] <= i < a['begin'] + first_site + 1)
        actual = set((a['x'][:first_site + 1] != b['x'][:first_site + 1]
                      ).nonzero().flatten().tolist())
        if not expected or actual != expected:
            continue
        assert left.text[:left.answer_span[0]] != \
               right.text[:right.answer_span[0]]
        for example in examples:
            restored = example.text
            for start, end in example.spans:
                restored = restored[:start] + plan.source_code + restored[end:]
            assert restored == plan.source_text
        return ({'left': {key: value for key, value in a.items()
                          if key not in ('encoding', 'begin', 'source',
                                         'target_indices')},
                 'right': {key: value for key, value in b.items()
                           if key not in ('encoding', 'begin', 'source',
                                          'target_indices')},
                 'left_example': left, 'right_example': right,
                 'changed_prefix_tokens': len(expected)}, 'aligned')
    return None, 'token_or_context'


def check_fixture(tokenizer, pool):
    from corpus_uppercase_copy_dgp import candidates
    text = ('The report uses ABC in the first section. ' +
            'A collection of ordinary words provides a long gap before ' +
            'the report discusses this subject again. ' +
            'Several additional sentences describe the same process and ' +
            'offer context for readers. In the final section ABC appears.')
    plan = next(candidates(text, min_gap_words=20))
    pair, reason = aligned_pair(plan, tokenizer, pool,
                                rng=random.Random(4))
    assert reason == 'aligned', reason
    assert len(pair['left']['target']) == 3
    assert torch.all(pair['left']['target'] != pair['right']['target'])
    print('nonce-code multi-token paired fixture passed')
