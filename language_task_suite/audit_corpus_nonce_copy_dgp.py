"""Audit multi-token nonce-copy edits on independent FineWeb documents."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import random
import sys

from datasets import load_dataset
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from audit_corpus_uppercase_copy_dgp import around, crop, form
from corpus_nonce_copy_dgp import candidates, codebook
from corpus_nonce_copy_validation import aligned_pair, check_fixture


def audit(skip, n, max_chars, n_ctx, seed, print_snippets=False,
          strict=True, entity_only=False):
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    pool = codebook(tokenizer)
    check_fixture(tokenizer, pool)
    source = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT',
                          split='train', streaming=True)
    stream = iter(source)
    rng = random.Random(seed)
    counts = Counter()
    hashes = []
    gaps = []
    first100 = {key: [] for key in ('source', 'left', 'right')}
    target41 = {key: [] for key in first100}
    try:
        for offset, row in enumerate(islice(stream, skip, skip + n), skip):
            counts['source_documents'] += 1
            text = row['text'][:max_chars]
            options = list(candidates(text, strict=strict,
                                      entity_only=entity_only))
            counts['candidate_documents'] += bool(options)
            counts['candidate_pairs'] += len(options)
            rng.shuffle(options)
            selected = None
            for plan in options:
                pair, _ = aligned_pair(plan, tokenizer, pool, n_ctx, rng)
                if pair is not None:
                    selected = pair
                    break
            if selected is None:
                continue
            counts['aligned_documents'] += 1
            counts['changed_prefix_tokens'] += selected['changed_prefix_tokens']
            left, right = selected['left_example'], selected['right_example']
            plan = left.plan
            hashes.append({'offset': offset,
                           'sha256': hashlib.sha256(text.encode()).hexdigest()})
            gaps.append(plan.gap_words)
            for key, version in (('source', text), ('left', left.text),
                                 ('right', right.text)):
                first = crop(version, 0, 100)
                if first:
                    first100[key].append(first)
                site = around(version, left.answer_span[0])
                if site:
                    target41[key].append(site)
            if print_snippets and counts['aligned_documents'] <= 12:
                a, b = left.previous_span, left.answer_span
                print('offset', offset, 'old', plan.source_code,
                      'codes', left.code, right.code, 'gap', plan.gap_words)
                print('left prior', left.text[max(0, a[0] - 65):a[1] + 65]
                      .replace('\n', ' '))
                print('left later', left.text[max(0, b[0] - 65):b[1] + 65]
                      .replace('\n', ' '))
    finally:
        if hasattr(stream, 'close'):
            stream.close()
    assert counts['source_documents'] == n
    return {'source': 'HuggingFaceFW/fineweb sample-10BT train',
            'offsets': [skip, skip + n], 'max_chars': max_chars,
            'n_ctx': n_ctx, 'seed': seed, 'nonce_pool_size': len(pool),
            'strict_alias_filter': strict,
            'entity_source_prior': entity_only,
            'counts': dict(counts), 'selected_source_hashes': hashes,
            'gap_words': {'min': min(gaps), 'max': max(gaps),
                          'mean': sum(gaps) / len(gaps)} if gaps else None,
            'first100_form': {key: form(value) for key, value in first100.items()
                              if value},
            'target41_form': {key: form(value) for key, value in target41.items()
                              if value},
            'caveat': 'Fictional alternate-world code aliases, not semantic '
                      'or factual preservation. Only code-site copy rule '
                      'has a known answer; natural background prose has '
                      'no exact oracle. Manual naturalness review needed. '
                      'No source prose saved.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--skip', type=int, default=85500)
    p.add_argument('--n', type=int, default=1000)
    p.add_argument('--max-chars', type=int, default=8000)
    p.add_argument('--n-ctx', type=int, default=256)
    p.add_argument('--seed', type=int, default=71902)
    p.add_argument('--print-snippets', action='store_true')
    p.add_argument('--broad', action='store_true',
                   help='Disable measurement/currency alias exclusions')
    p.add_argument('--entity-only', action='store_true',
                   help='Restrict source acronym to curated named entities')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.skip, a.n, a.max_chars, a.n_ctx,
                   a.seed, a.print_snippets, not a.broad, a.entity_only)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(result['counts'], result['gap_words'], flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
