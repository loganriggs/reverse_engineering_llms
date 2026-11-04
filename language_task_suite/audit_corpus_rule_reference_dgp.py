"""Audit source-level rule yield and context coverage without task selection."""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations, islice
import json
import os
from pathlib import Path
import random
import sys

from datasets import load_dataset
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from corpus_nonce_copy_dgp import codebook
from corpus_rule_reference_dgp import sample_document, chunk_document


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--start', type=int, default=353000)
    p.add_argument('--n', type=int, default=5000)
    p.add_argument('--seed', type=int, default=72501)
    p.add_argument('--max-chars', type=int, default=8000)
    p.add_argument('--max-tokens', type=int, default=2048)
    p.add_argument('--n-ctx', type=int, default=256)
    p.add_argument('--list-rate', type=float, default=1.)
    p.add_argument('--bracket-rate', type=float, default=1.)
    p.add_argument('--nonce-rate', type=float, default=1.)
    p.add_argument('--list-shift-rate', type=float, default=.5)
    p.add_argument('--square-rate', type=float, default=.5)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--review-output', type=Path,
                   help='Local-only source excerpts for manual semantic review')
    a = p.parse_args()
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    codes = list(codebook(tokenizer))
    random.Random(71999).shuffle(codes)
    pool = tuple(codes[:256])
    counts = Counter()
    by_kind = ('list', 'bracket', 'nonce')
    review_rng = random.Random(a.seed + 999)
    review = {kind: [] for kind in by_kind}
    review_seen = Counter()
    dataset = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT',
                           split='train', streaming=True)
    stream = iter(dataset)
    try:
        for offset, datum in enumerate(islice(stream, a.start, a.start + a.n),
                                       a.start):
            counts['source_documents'] += 1
            document = sample_document(
                datum['text'], random.Random(a.seed + offset), pool,
                list_rate=a.list_rate, bracket_rate=a.bracket_rate,
                nonce_rate=a.nonce_rate, list_shift_rate=a.list_shift_rate,
                square_rate=a.square_rate,
                max_chars=a.max_chars)
            counts['documents_with_any_text_change'] += (
                document.text != document.source_text)
            for kind in by_kind:
                counts[f'eligible_{kind}'] += kind in document.eligible
                counts[f'applied_{kind}'] += kind in document.applied
            for first, second in combinations(sorted(document.applied), 2):
                counts[f'applied_pair_{first}_{second}'] += 1
            counts[f'documents_{len(document.applied)}_applied_families'] += 1
            chunks = chunk_document(document, tokenizer, a.n_ctx,
                                    a.max_tokens)
            counts['model_chunks'] += len(chunks)
            counts['model_valid_next_token_positions'] += sum(
                int((chunk['y'] != -100).sum()) for chunk in chunks)
            local = {site['kind'] for chunk in chunks
                     for site in chunk['sites']}
            for kind in local:
                counts[f'local_{kind}_documents'] += 1
            for first, second in combinations(sorted(local), 2):
                counts[f'local_pair_{first}_{second}'] += 1
            counts[f'documents_{len(local)}_local_families'] += 1
            if a.review_output:
                for site in document.sites:
                    review_seen[site.kind] += 1
                    index = review_seen[site.kind]
                    excerpt = {'offset': offset, 'kind': site.kind,
                               'source_context': document.text[
                                   max(0, site.source[0] - 80):
                                   min(len(document.text), site.source[1] + 80)],
                               'source_excerpt': document.source_text[
                                   max(0, site.target[0] - 140):
                                   min(len(document.text), site.target[1] + 140)],
                               'transformed_excerpt': document.text[
                                   max(0, site.target[0] - 140):
                                   min(len(document.text), site.target[1] + 140)]}
                    if len(review[site.kind]) < 8:
                        review[site.kind].append(excerpt)
                    else:
                        chosen = review_rng.randrange(index)
                        if chosen < 8:
                            review[site.kind][chosen] = excerpt
            if counts['source_documents'] % 1000 == 0:
                print('scanned', counts['source_documents'], flush=True)
    finally:
        if hasattr(stream, 'close'):
            stream.close()
    assert counts['source_documents'] == a.n
    report = {'config': {key: str(value) if isinstance(value, Path) else value
                         for key, value in vars(a).items()},
              'counts': dict(counts),
              'per_1000_source_documents': {
                  kind: {stage: 1000 * counts[f'{stage}_{kind}' +
                                              ('_documents' if stage == 'local'
                                               else '')] / a.n
                         for stage in ('eligible', 'applied', 'local')}
                  for kind in by_kind},
              'scope': 'Uniform FineWeb source offsets, no conditioning on '
                       'rule presence. Each eligible family is edited with '
                       'its configured probability; edits are equal-length '
                       'and independently sampled. Local means source and '
                       'target fit one ordinary nonoverlapping model chunk '
                       'with BPE alignment; it is a lower bound for a model '
                       'with cross-chunk memory. Natural background tokens '
                       'have no exact symbolic oracle. Reference style and '
                       'edit rates are research choices, not calibrated '
                       'natural frequencies. No source prose saved.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    if a.review_output:
        a.review_output.parent.mkdir(parents=True, exist_ok=True)
        a.review_output.write_text(json.dumps(review, indent=2) + '\n')
    print(report['per_1000_source_documents'], flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
