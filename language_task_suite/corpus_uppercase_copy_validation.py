"""BPE-local paired windows for the sampled literal-copy code rule."""

from __future__ import annotations

import random

import torch

from corpus_uppercase_copy_dgp import draw_pair


PAD = 50256


def _token_index(encoding, text, span):
    for index, (start, end) in enumerate(encoding.offsets):
        if end == span[1] and (start == span[0] or
                                (start == span[0] - 1 and
                                 text[start].isspace())):
            return index
    return None


def _row(example, tokenizer, n_ctx):
    enc = tokenizer.encode(example.text, add_special_tokens=False)
    source = _token_index(enc, example.text, example.previous_span)
    target = _token_index(enc, example.text, example.answer_span)
    if source is None or target is None or source >= target:
        return None
    begin = min(source, max(0, target - (3 * n_ctx) // 4))
    if target - begin >= n_ctx or target - begin < 1:
        return None
    ids = enc.ids[begin:begin + n_ctx + 1]
    x = torch.full((n_ctx,), PAD, dtype=torch.long)
    y = torch.full((n_ctx,), -100, dtype=torch.long)
    x[:len(ids) - 1] = torch.tensor(ids[:-1])
    y[:len(ids) - 1] = torch.tensor(ids[1:])
    site = target - begin - 1
    assert y[site].item() == enc.ids[target]
    return {'x': x, 'y': y, 'site': torch.tensor([site]),
            'target': torch.tensor([enc.ids[target]]),
            'source_token': torch.tensor([source - begin]),
            'encoding': enc, 'begin': begin}


def aligned_pair(plan, tokenizer, n_ctx=256,
                 rng: random.Random | None = None, attempts=24):
    rng = rng or random.Random(0)
    for _ in range(attempts):
        examples = draw_pair(plan, tokenizer, rng)
        if examples is None:
            return None, 'no_code_pair'
        left, right = examples
        a = _row(left, tokenizer, n_ctx)
        b = _row(right, tokenizer, n_ctx)
        if a is None or b is None:
            continue
        if (a['site'].item() != b['site'].item() or
                a['source_token'].item() != b['source_token'].item() or
                a['target'].item() == b['target'].item() or
                a['begin'] != b['begin']):
            continue
        site = a['site'].item()
        if a['encoding'].offsets[:a['begin'] + site + 2] != \
                b['encoding'].offsets[:b['begin'] + site + 2]:
            continue
        expected = set()
        for span in plan.spans[:plan.target_occurrence]:
            index = _token_index(a['encoding'], left.text, span)
            other = _token_index(b['encoding'], right.text, span)
            if index is None or other != index:
                expected = set()
                break
            if a['begin'] <= index < a['begin'] + site + 1:
                expected.add(index - a['begin'])
        if not expected or a['source_token'].item() not in expected:
            continue
        actual = set((a['x'][:site + 1] != b['x'][:site + 1]
                      ).nonzero().flatten().tolist())
        if actual != expected:
            continue
        for example in examples:
            assert example.text[example.answer_span[0]:
                                example.answer_span[1]] == example.code
            restored = example.text
            for start, end in example.spans:
                restored = restored[:start] + plan.source_code + restored[end:]
            assert restored == plan.source_text
        return ({'left': {key: value for key, value in a.items()
                          if key not in ('encoding', 'begin')},
                 'right': {key: value for key, value in b.items()
                           if key not in ('encoding', 'begin')},
                 'left_example': left, 'right_example': right,
                 'changed_prefix_tokens': len(expected)}, 'aligned')
    return None, 'token_or_context'


def check_fixture(tokenizer):
    from corpus_uppercase_copy_dgp import candidates
    text = ('The report uses ABC in the initial section. ' +
            'A collection of ordinary words provides a long gap before ' +
            'the report discusses this subject again. ' +
            'Several additional sentences describe the same process and ' +
            'offer context for readers. In the final section ABC appears.')
    plans = list(candidates(text, min_gap_words=20))
    assert plans
    pair, status = aligned_pair(plans[0], tokenizer, rng=random.Random(4))
    assert status == 'aligned', status
    assert pair['left']['target'].item() != pair['right']['target'].item()
    print('uppercase copy paired-window fixture passed')
