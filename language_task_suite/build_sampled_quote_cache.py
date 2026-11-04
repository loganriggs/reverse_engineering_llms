"""Build source-disjoint paired quote-binding data; keep prose cache local."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import spacy
import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from build_fineweb_surname_copy_cache import NearDuplicateFilter
from sampled_quote_document_dgp import paired_variants
from train_sports_entity_alias import stack


PAD = 50256


def align(sample, tokenizer, n_ctx):
    encoded = tokenizer.encode(sample.text, add_special_tokens=False)
    start = sample.target_span[0]
    found = [i for i, (left, right) in enumerate(encoded.offsets)
             if left <= start < right]
    if len(found) != 1:
        return None
    index = found[0]
    if (index < 1 or index >= n_ctx or
            encoded.offsets[index][0] != start or
            sample.text[slice(*sample.target_span)] != sample.answer):
        return None
    length = min(len(encoded.ids)-1, n_ctx)
    x = torch.full((n_ctx,), PAD, dtype=torch.long)
    y = torch.full((n_ctx,), -100, dtype=torch.long)
    x[:length] = torch.tensor(encoded.ids[:length])
    y[:length] = torch.tensor(encoded.ids[1:length+1])
    return {'x': x, 'y': y, 'site': index-1,
            'target': encoded.ids[index]}


def build(train_rows, heldout_rows, n_ctx=256, seed=73151,
          pairs_per_train_source=2):
    if set(row['sha256'] for row in train_rows) & \
            set(row['sha256'] for row in heldout_rows):
        raise ValueError('Exact source overlap')
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    nlp = spacy.load('en_core_web_sm')
    dedup = NearDuplicateFilter()
    selected = {name: [] for name in ('train', 'val', 'test')}
    counts = {name: Counter() for name in selected}
    for rows, is_train in ((train_rows, True), (heldout_rows, False)):
        for source, doc in zip(rows, nlp.pipe(
                (row['preview'] for row in rows), batch_size=32), strict=True):
            split = ('train' if is_train else
                     ('val' if int(source['sha256'][:8], 16) % 2 == 0
                      else 'test'))
            counts[split]['sources_seen'] += 1
            rng = random.Random(seed + source['index'])
            pairs = []
            for a, b in paired_variants(doc, source['index'], rng):
                rows_pair = (align(a, tokenizer, n_ctx),
                             align(b, tokenizer, n_ctx))
                if any(row is None for row in rows_pair):
                    continue
                if (rows_pair[0]['target'] == rows_pair[1]['target'] or
                        rows_pair[0]['site'] != rows_pair[1]['site']):
                    continue
                pairs.append(rows_pair)
                if len(pairs) >= (pairs_per_train_source if is_train else 1):
                    break
            if not pairs:
                continue
            counts[split]['aligned_sources'] += 1
            rejected = dedup.add(source['preview'], source['sha256'])
            if rejected:
                counts[split][rejected] += 1
                continue
            selected[split].extend(pairs)
            counts[split]['selected_sources'] += 1
            counts[split]['paired_examples'] += len(pairs)
    if any(not selected[name] for name in selected):
        raise ValueError('A split has no selected examples')
    cache = {'n_ctx': n_ctx, 'vocab_size': tokenizer.get_vocab_size(),
             'pairs': {split: stack([(None, pair) for pair in pairs])
                       for split, pairs in selected.items()},
             'counts': {split: dict(counts[split]) for split in selected},
             'task_description': 'Whole-source heldout sampled-claim '
                 'speaker binding. Full next-token CE with optional '
                 'first-BPE answer-site weight; paired arms swap speaker '
                 'assignments while keeping the natural claims fixed. '
                 'Candidate rank, full-vocabulary top1, and source-swap '
                 'response are separate behavioral gates. The selected '
                 'noun chunk is not a unique visible-prefix oracle.'}
    manifest = {
        'scope': 'Source-disjoint sampled FineWeb claim pairs. Paired arms '
                 'swap assigned speakers and change the source-specific '
                 'recap phrase. Each target is the first BPE token of the '
                 'selected noun chunk; full CE includes the rest of text. '
                 'This is a semi-parametric DGP, not an exact all-token oracle.',
        'config': {'n_ctx': n_ctx, 'seed': seed,
                   'pairs_per_train_source': pairs_per_train_source,
                   'tokenizer': str(DEFAULT_TOKENIZER)},
        'source_panels': {
            'train': [train_rows[0]['index'], train_rows[-1]['index']],
            'heldout': [heldout_rows[0]['index'], heldout_rows[-1]['index']]},
        'preview_sha256': {
            'train': hashlib.sha256(json.dumps(train_rows,
                                               sort_keys=True).encode()).hexdigest(),
            'heldout': hashlib.sha256(json.dumps(heldout_rows,
                                                 sort_keys=True).encode()).hexdigest()},
        'counts': cache['counts'],
        'duplicate_screen': 'Exact source hash and SimHash Hamming<=3 across splits.',
        'caveat': 'Raw source and generated prose remain in the local cache. '
                  'The selected phrase is one of several valid noun chunks; '
                  'speaker-swap response, candidate rank, full-vocabulary '
                  'prediction, and form are distinct evidence.'}
    return cache, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-preview', type=Path, required=True)
    parser.add_argument('--heldout-preview', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--n-ctx', type=int, default=256)
    parser.add_argument('--seed', type=int, default=73151)
    parser.add_argument('--pairs-per-train-source', type=int, default=2)
    args = parser.parse_args()
    train = json.loads(args.train_preview.read_text())
    heldout = json.loads(args.heldout_preview.read_text())
    cache, manifest = build(
        train, heldout, args.n_ctx, args.seed, args.pairs_per_train_source)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.cache)
    args.manifest.write_text(json.dumps(manifest, indent=2) + '\n')
    print(manifest['counts'], flush=True)


if __name__ == '__main__':
    main()
