"""Build document-disjoint natural-text spelling-style LM windows.

The tensor cache contains reconstructable FineWeb tokens and stays outside
Git. Only an aggregate manifest is written to the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from itertools import islice
from pathlib import Path

import torch
from datasets import load_dataset
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from build_corpus_delimiter_pilot import simhash
from corpus_spelling_tokenization import aligned_pair
from corpus_transformed_spelling_style import FAMILIES


FAMILY_ID = {family: index for index, family in
             enumerate(sorted(FAMILIES))}


def _stack(rows):
    if not rows:
        return None
    return dict(x=torch.stack([row['x'] for row in rows]),
                y=torch.stack([row['y'] for row in rows]),
                site_position=torch.tensor(
                    [row['site_position'] for row in rows]),
                target_id=torch.tensor([row['target_id'] for row in rows]),
                cue_position=torch.tensor(
                    [row['cue_position'] for row in rows]),
                cue_root_id=torch.tensor(
                    [FAMILY_ID[row['cue_root']] for row in rows]),
                target_root_id=torch.tensor(
                    [FAMILY_ID[row['target_root']] for row in rows]))


def build(skip, train_docs, val_docs, test_docs, max_chars, n_ctx,
          uk_rate, seed, cache_path):
    if not 0 <= uk_rate <= 1:
        raise ValueError('uk_rate outside [0,1]')
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    source = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT',
                          split='train', streaming=True)
    stream = iter(source)
    counts = {split: Counter() for split in ('train', 'val', 'test')}
    roots = {split: Counter() for split in counts}
    hashes = {split: set() for split in counts}
    signatures = {split: [] for split in counts}
    train_rows, val_us, val_uk, test_us, test_uk = [], [], [], [], []
    total = train_docs+val_docs+test_docs
    try:
        for local_index, row in enumerate(islice(stream, skip, skip+total)):
            offset = skip+local_index
            split = ('train' if local_index < train_docs else
                     'val' if local_index < train_docs+val_docs else
                     'test')
            counter = counts[split]
            counter['source_documents'] += 1
            text = row['text'][:max_chars]
            source_hash = hashlib.sha256(text.encode()).hexdigest()
            if source_hash in hashes[split]:
                counter['exact_duplicate_within_split'] += 1
                continue
            hashes[split].add(source_hash)
            pair, reason = aligned_pair(
                text, tokenizer, n_ctx,
                rng=random.Random(seed+offset))
            counter[reason] += 1
            if pair is None:
                continue
            signature = simhash(text)
            if any((signature ^ previous).bit_count() <= 3
                   for other_split, values in signatures.items()
                   if other_split != split for previous in values):
                counter['near_duplicate_cross_split_rejected'] += 1
                continue
            signatures[split].append(signature)
            roots[split][pair['us']['target_root']] += 1
            counter['selected_candidate_count_sum'] += pair['candidate_count']
            counter['changed_us_words'] += pair['changed_us']
            counter['changed_uk_words'] += pair['changed_uk']
            if split == 'train':
                style = ('uk' if random.Random(
                    seed+9_000_000+offset).random() < uk_rate else 'us')
                train_rows.append(pair[style])
                counter[style+'_selected'] += 1
            elif split == 'val':
                val_us.append(pair['us'])
                val_uk.append(pair['uk'])
            else:
                test_us.append(pair['us'])
                test_uk.append(pair['uk'])
    finally:
        if hasattr(stream, 'close'):
            stream.close()
    assert all(counts[split]['source_documents'] == amount
               for split, amount in (('train', train_docs),
                                     ('val', val_docs), ('test', test_docs)))
    assert train_rows and val_us and test_us
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(train=_stack(train_rows),
                    val=dict(us=_stack(val_us), uk=_stack(val_uk)),
                    test=dict(us=_stack(test_us), uk=_stack(test_uk)),
                    n_ctx=n_ctx,
                    vocab_size=tokenizer.get_vocab_size(),
                    family_id=FAMILY_ID,
                    source='FineWeb sample-10BT; token arrays local only'),
               cache_path)
    return dict(config=dict(skip=skip, train_docs=train_docs,
                            val_docs=val_docs, test_docs=test_docs,
                            max_chars=max_chars, n_ctx=n_ctx,
                            uk_rate=uk_rate, seed=seed,
                            source='HuggingFaceFW/fineweb sample-10BT train',
                            tokenizer=str(DEFAULT_TOKENIZER),
                            cache_path=str(cache_path)),
                counts={name: dict(counter) for name, counter in
                        counts.items()},
                target_root_counts={name: dict(counter) for name, counter in
                                    roots.items()},
                family_id=FAMILY_ID,
                unique_source_hashes={name: len(values) for name, values in
                                      hashes.items()},
                near_duplicate_screen='64-bit 5-gram SimHash cross-split '
                                      'Hamming <=3 rejection',
                caveat='Token cache can reconstruct source text and must '
                       'not be committed. SimHash is a near-duplicate '
                       'screen, not proof of no overlap. The style rule '
                       'is known; background language is empirical.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--skip', type=int, default=53500)
    p.add_argument('--train-docs', type=int, default=24000)
    p.add_argument('--val-docs', type=int, default=3000)
    p.add_argument('--test-docs', type=int, default=3000)
    p.add_argument('--max-chars', type=int, default=8000)
    p.add_argument('--n-ctx', type=int, default=256)
    p.add_argument('--uk-rate', type=float, default=.5)
    p.add_argument('--seed', type=int, default=31017)
    p.add_argument('--cache-path', type=Path, default=Path(
        '/home/loganriggs/Coding/neuron_circuits/artifacts/'
        'corpus_spelling_style_pilot/dataset.pt'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = build(args.skip, args.train_docs, args.val_docs,
                   args.test_docs, args.max_chars, args.n_ctx,
                   args.uk_rate, args.seed, args.cache_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['counts'], flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
