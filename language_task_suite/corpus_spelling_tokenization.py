"""Strict paired GPT-2 windows for the document-style spelling rule."""

from __future__ import annotations

import random
import torch

from corpus_transformed_spelling_style import candidate_indices, transform


PAD = 50256


def _whole_word_token(encoded, text, span):
    for index, (start, end) in enumerate(encoded.offsets):
        if (end == span[1] and span[0]-1 <= start <= span[0] and
                text[start:end].strip() == text[span[0]:span[1]] and
                (start == span[0] or text[start:span[0]] == ' ')):
            return index
    return None


def _row(example, tokenizer, n_ctx):
    encoded = tokenizer.encode(example.text, add_special_tokens=False)
    cue = _whole_word_token(encoded, example.text, example.cue_span)
    target = _whole_word_token(encoded, example.text, example.target_span)
    if cue is None or target is None or not 0 <= cue < target:
        return None
    start = max(0, target-(3*n_ctx)//4)
    if example.target_occurrence_index >= 2:
        earlier_end = example.transformed_spans[
            example.target_occurrence_index-2][1]
        after_earlier = next((index for index, (left, _) in
                              enumerate(encoded.offsets)
                              if left >= earlier_end), None)
        if after_earlier is None:
            return None
        start = max(start, after_earlier)
    start = min(start, cue)
    if target-start >= n_ctx or target-start < 1:
        return None
    window = encoded.ids[start:start+n_ctx+1]
    x = torch.full((n_ctx,), PAD, dtype=torch.long)
    y = torch.full((n_ctx,), -100, dtype=torch.long)
    if len(window) >= 2:
        x[:len(window)-1] = torch.tensor(window[:-1])
        y[:len(window)-1] = torch.tensor(window[1:])
    site = target-start-1
    assert y[site].item() == encoded.ids[target]
    return dict(x=x, y=y, site_position=site,
                target_id=encoded.ids[target],
                cue_position=cue-start,
                target_root=example.target_root,
                cue_root=example.cue_root)


def aligned_pair(text, tokenizer, n_ctx=256, rng: random.Random | None = None):
    _, options = candidate_indices(text)
    if not options:
        return None, 'ineligible'
    options = list(options)
    if rng is not None:
        rng.shuffle(options)
    last_reason = 'token_or_context'
    for index, _ in options:
        us = transform(text, 'us', target_occurrence_index=index)
        uk = transform(text, 'uk', target_occurrence_index=index)
        assert us is not None and uk is not None
        a = _row(us, tokenizer, n_ctx)
        b = _row(uk, tokenizer, n_ctx)
        if a is None or b is None:
            continue
        if (a['site_position'] != b['site_position'] or
                a['cue_position'] != b['cue_position'] or
                a['target_id'] == b['target_id']):
            last_reason = 'position_or_target'
            continue
        site = a['site_position']
        changed = (a['x'][:site+1] != b['x'][:site+1]
                   ).nonzero().flatten().tolist()
        if changed != [a['cue_position']]:
            last_reason = 'nonlocal_prefix_difference'
            continue
        return dict(us=a, uk=b, gap_words=us.gap_words,
                    candidate_count=len(options),
                    selected_occurrence_index=index,
                    source_us_count=sum(row.source_word == row.us
                                        for row in us.source_occurrences),
                    source_uk_count=sum(row.source_word == row.uk
                                        for row in us.source_occurrences),
                    changed_us=sum(row.source_word != row.us
                                   for row in us.source_occurrences),
                    changed_uk=sum(row.source_word != row.uk
                                   for row in us.source_occurrences)), 'aligned'
    return None, last_reason
