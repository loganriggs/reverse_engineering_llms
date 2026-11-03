"""Yield and ambiguity screen for repeated capitalized phrases in FineWeb.

No passage or phrase text is written to the JSON output. Repeated surface
strings are not assumed to denote one entity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path

from datasets import load_dataset

from corpus_transformed_name_reuse import NAME_TO_GROUP


PHRASE = re.compile(
    r'(?<![A-Za-z])(?:[A-Z][a-z]{2,}|[A-Z]{2,})'
    r'(?:[ \t]+(?:[A-Z][a-z]{2,}|[A-Z]{2,})){1,3}(?![A-Za-z])')
WORD = re.compile(r'\b[A-Za-z]+\b')
NEXT_CAPITAL = re.compile(r'\s+(?:[A-Z][a-z]{2,}|[A-Z]{2,})\b')
SENTENCE_LEADS = {'The', 'This', 'That', 'These', 'Those', 'What',
                  'When', 'How', 'Why', 'And', 'But'}


def candidates(text, min_gap_words=20, max_gap_words=300):
    hits = defaultdict(list)
    for match in PHRASE.finditer(text):
        phrase = match.group()
        if phrase.split()[0] in SENTENCE_LEADS:
            continue
        if NEXT_CAPITAL.match(text[match.end():]):
            continue
        hits[phrase].append(match.span())
    results = []
    for phrase, spans in hits.items():
        for earlier, later in zip(spans, spans[1:]):
            gap = len(WORD.findall(text[earlier[1]:later[0]]))
            if min_gap_words <= gap <= max_gap_words:
                results.append((phrase, earlier, later, gap))
    return results


def audit(skip, n, max_chars, seed, sample_size):
    source = load_dataset('HuggingFaceFW/fineweb',
                          name='sample-10BT', split='train', streaming=True)
    stream = iter(source)
    counts, items, exact_hashes = Counter(), [], set()
    try:
        for offset, row in enumerate(islice(stream, skip, skip+n), skip):
            text = row['text'][:max_chars]
            counts['documents'] += 1
            source_hash = hashlib.sha256(text.encode()).hexdigest()
            exact_hashes.add(source_hash)
            options = candidates(text)
            counts['eligible_documents'] += bool(options)
            counts['eligible_pairs'] += len(options)
            counts['curated_first_name_pairs'] += sum(
                phrase.split()[0] in NAME_TO_GROUP and
                len(phrase.split()) == 2
                for phrase, _, _, _ in options)
            counts['curated_first_name_documents'] += any(
                phrase.split()[0] in NAME_TO_GROUP and
                len(phrase.split()) == 2
                for phrase, _, _, _ in options)
            for phrase, earlier, later, gap in options:
                items.append(dict(offset=offset,
                                  source_sha256=source_hash,
                                  phrase_sha256=hashlib.sha256(
                                      phrase.encode()).hexdigest(),
                                  word_count=len(phrase.split()),
                                  gap_words=gap,
                                  curated_first_name=(
                                      phrase.split()[0] in NAME_TO_GROUP),
                                  phrase=phrase,
                                  earlier=earlier, later=later))
    finally:
        if hasattr(stream, 'close'):
            stream.close()
    if counts['documents'] != n:
        raise RuntimeError(f'Expected {n} documents, got {counts["documents"]}')
    sampled = random.Random(seed).sample(items, min(sample_size, len(items)))
    result = dict(config=dict(skip=skip, n=n, max_chars=max_chars,
                              seed=seed, sample_size=sample_size,
                              source='HuggingFaceFW/fineweb sample-10BT train'),
                  counts=dict(counts), unique_exact_text_hashes=len(exact_hashes),
                  sample=[{k: v for k, v in item.items()
                           if k not in ('phrase', 'earlier', 'later')}
                          for item in sampled],
                  caveat='Exact repeated phrase is a surface recurrence, '
                         'not evidence of one referent or a natural '
                         'counterfactual replacement distribution.')
    return result, sampled


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--skip', type=int, default=52000)
    p.add_argument('--n', type=int, default=500)
    p.add_argument('--max-chars', type=int, default=8000)
    p.add_argument('--seed', type=int, default=72991)
    p.add_argument('--sample-size', type=int, default=24)
    p.add_argument('--print-phrases', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result, sampled = audit(args.skip, args.n, args.max_chars,
                            args.seed, args.sample_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['counts'])
    if args.print_phrases:
        for index, item in enumerate(sampled):
            print(index, item['offset'], item['phrase'], item['gap_words'])
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
