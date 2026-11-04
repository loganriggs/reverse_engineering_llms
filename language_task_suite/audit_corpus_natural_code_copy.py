"""Audit naturally repeated uppercase-code opportunities in ordinary chunks.

No text is edited. Optional snippet review is local-only, never aggregate JSON.
"""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import islice
import json
import os
from pathlib import Path
import random
import sys

from datasets import load_dataset
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from corpus_uppercase_copy_dgp import candidates
from corpus_uppercase_copy_validation import _token_index


def aligned_sites(text, tokenizer, n_ctx=256, max_tokens=2048):
    encoding = tokenizer.encode(text, add_special_tokens=False)
    usable = min(len(encoding.ids) - 1, max_tokens)
    for plan in candidates(text):
        source_span = plan.spans[plan.target_occurrence - 1]
        target_span = plan.spans[plan.target_occurrence]
        source = _token_index(encoding, text, source_span)
        target = _token_index(encoding, text, target_span)
        if source is None or target is None or not (source < target <= usable):
            continue
        start = ((target - 1) // n_ctx) * n_ctx
        if source < start:
            continue
        if encoding.ids[source] != encoding.ids[target]:
            continue
        yield {'code': plan.source_code, 'source_token': source,
               'target_token': target, 'gap_words': plan.gap_words,
               'source_span': source_span, 'target_span': target_span}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--start', type=int, default=373000)
    p.add_argument('--n', type=int, default=5000)
    p.add_argument('--max-chars', type=int, default=8000)
    p.add_argument('--n-ctx', type=int, default=256)
    p.add_argument('--max-tokens', type=int, default=2048)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--review-output', type=Path)
    a = p.parse_args()
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    source = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT',
                          split='train', streaming=True)
    stream = iter(source)
    counts = Counter()
    code_counts = Counter()
    gaps = []
    review, seen = [], 0
    rng = random.Random(82701)
    try:
        for offset, row in enumerate(islice(stream, a.start, a.start+a.n),
                                     a.start):
            counts['source_documents'] += 1
            text = row['text'][:a.max_chars]
            plans = list(candidates(text))
            counts['documents_with_candidate'] += bool(plans)
            counts['candidate_sites'] += len(plans)
            sites = list(aligned_sites(text, tokenizer, a.n_ctx,
                                       a.max_tokens)) if plans else []
            counts['documents_with_local_one_token_copy'] += bool(sites)
            counts['local_one_token_copy_sites'] += len(sites)
            for site in sites:
                code_counts[site['code']] += 1
                gaps.append(site['gap_words'])
                if a.review_output:
                    seen += 1
                    span = site['target_span']
                    excerpt = {'offset': offset, 'code': site['code'],
                               'gap_words': site['gap_words'],
                               'source_context': text[
                                   max(0, site['source_span'][0]-110):
                                   min(len(text), site['source_span'][1]+110)],
                               'target_context': text[
                                   max(0, span[0]-110):min(len(text),span[1]+110)]}
                    if len(review) < 24:
                        review.append(excerpt)
                    else:
                        index = rng.randrange(seen)
                        if index < len(review):
                            review[index] = excerpt
            if counts['source_documents'] % 1000 == 0:
                print('scanned', counts['source_documents'], flush=True)
    finally:
        if hasattr(stream, 'close'):
            stream.close()
    assert counts['source_documents'] == a.n
    report = {'config': {key: str(value) if isinstance(value, Path) else value
                         for key, value in vars(a).items()},
              'counts': dict(counts),
              'rates_per_1000_sources': {key: 1000*counts[key]/a.n
                                         for key in counts if key != 'source_documents'},
              'distinct_codes': len(code_counts),
              'top_20_codes': code_counts.most_common(20),
              'gap_words': {'min': min(gaps) if gaps else None,
                            'max': max(gaps) if gaps else None,
                            'mean': sum(gaps)/len(gaps) if gaps else None},
              'scope': 'Unedited FineWeb source documents. A narrow three-letter '
                       'uppercase-code regex finds repeated exact strings 20–160 '
                       'words apart. One-token BPE source/target must fit the same '
                       'ordinary nonoverlapping context chunk. This certifies '
                       'literal equality, not that the author or model used an '
                       'induction algorithm. No source prose in aggregate output.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    if a.review_output:
        a.review_output.parent.mkdir(parents=True, exist_ok=True)
        a.review_output.write_text(json.dumps(review, indent=2)+'\n')
    print(report['rates_per_1000_sources'], flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
