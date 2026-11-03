"""Corpus yield, tokenizer, and intervention audit for spelling style."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from itertools import islice
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from corpus_spelling_tokenization import aligned_pair
from corpus_transformed_spelling_style import (
    LOOKUP, occurrences, reverse_transform, transform,
)


def audit(skip, n, max_chars, n_ctx):
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    source = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT',
                          split='train', streaming=True)
    stream = iter(source)
    counts, target_roots, cue_roots, hashes = Counter(), Counter(), Counter(), set()
    gaps, token_distances = [], []
    try:
        for row in islice(stream, skip, skip+n):
            text = row['text'][:max_chars]
            hashes.add(hashlib.sha256(text.encode()).hexdigest())
            counts['source_documents'] += 1
            for occurrence in occurrences(text):
                counts['original_style_words'] += 1
                counts['original_us_words'] += occurrence.source_word == occurrence.us
                counts['original_uk_words'] += occurrence.source_word == occurrence.uk
            pair, reason = aligned_pair(text, tokenizer, n_ctx)
            counts[reason] += 1
            if pair is None:
                continue
            us = transform(text, 'us', target_occurrence_index=(
                pair['selected_occurrence_index']))
            uk = transform(text, 'uk', target_occurrence_index=(
                pair['selected_occurrence_index']))
            assert us is not None and uk is not None
            assert reverse_transform(us) == reverse_transform(uk) == text
            counts['paired_answer_flips'] += us.target_word != uk.target_word
            counts['changed_us_words'] += pair['changed_us']
            counts['changed_uk_words'] += pair['changed_uk']
            target_roots[us.target_root] += 1
            cue_roots[us.cue_root] += 1
            gaps.append(us.gap_words)
            token_distances.append(pair['us']['site_position']-
                                   pair['us']['cue_position'])
    finally:
        if hasattr(stream, 'close'):
            stream.close()
    assert counts['source_documents'] == n
    assert counts['original_style_words'] == (
        counts['original_us_words']+counts['original_uk_words'])
    assert counts['paired_answer_flips'] == counts['aligned']
    return dict(config=dict(skip=skip, n=n, max_chars=max_chars,
                            n_ctx=n_ctx, source='HuggingFaceFW/fineweb '
                            'sample-10BT train',
                            tokenizer=str(DEFAULT_TOKENIZER),
                            spelling_families=len(set(row[0] for row in
                                LOOKUP.values()))),
                counts=dict(counts), unique_exact_text_hashes=len(hashes),
                target_root_counts=dict(target_roots),
                cue_root_counts=dict(cue_roots),
                gap_words=dict(min=min(gaps), max=max(gaps),
                               mean=sum(gaps)/len(gaps)) if gaps else None,
                cue_to_target_token_distance=dict(
                    min=min(token_distances), max=max(token_distances),
                    mean=sum(token_distances)/len(token_distances)
                ) if token_distances else None,
                caveat='Only selected spelling rule has an oracle. Natural '
                       'background text does not. Balanced style prior '
                       'changes corpus spelling frequencies. Eligibility '
                       'requires lowercase whole words, different lexical '
                       'roots, and one-token paired prefix/target alignment.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--skip', type=int, default=52500)
    p.add_argument('--n', type=int, default=500)
    p.add_argument('--max-chars', type=int, default=8000)
    p.add_argument('--n-ctx', type=int, default=256)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.skip, args.n, args.max_chars, args.n_ctx)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['counts'], flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
