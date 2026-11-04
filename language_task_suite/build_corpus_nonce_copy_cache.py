"""Source-split three-token nonce-copy windows; keep reconstructable cache local."""

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
import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from build_corpus_delimiter_pilot import simhash
from corpus_nonce_copy_dgp import candidates, codebook
from corpus_nonce_copy_validation import aligned_pair, check_fixture


SPLITS = {'val': (95000, 105000),
          'train': (115000, 215000),
          'test': (335000, 345000)}


def stacked(rows):
    return {key: torch.stack([row[key] for row in rows])
            for key in ('x', 'y', 'site', 'target')}


def build(n_ctx, max_chars, seed, cache_path,
          train_pairs_per_source=1):
    assert train_pairs_per_source >= 1
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    pool = list(codebook(tokenizer))
    random.Random(71999).shuffle(pool)
    train_pool, unseen_pool = tuple(pool[:256]), tuple(pool[256:512])
    assert not set(train_pool) & set(unseen_pool)
    check_fixture(tokenizer, train_pool)
    source = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT',
                          split='train', streaming=True)
    stream = iter(source)
    counts = {split: Counter() for split in SPLITS}
    rows = {split: {arm: [] for arm in ('left', 'right')}
            for split in SPLITS}
    unseen_rows = {arm: [] for arm in ('left', 'right')}
    hashes = {split: [] for split in SPLITS}
    selected_hashes = set()
    signatures = []
    partial_path = cache_path.with_suffix('.partial.pt')
    config = (n_ctx, max_chars, seed, train_pairs_per_source)
    next_offset = 95000
    if partial_path.exists():
        state = torch.load(partial_path, map_location='cpu', weights_only=True)
        if tuple(state['config']) != config:
            raise ValueError(f'Partial cache has different settings: {partial_path}')
        next_offset = state['next_offset']
        counts = {split: Counter(state['counts'][split]) for split in SPLITS}
        rows = state['rows']
        unseen_rows = state['unseen_rows']
        hashes = state['hashes']
        selected_hashes = set(state['selected_hashes'])
        signatures = state['signatures']
        print('Resuming at source offset', next_offset, flush=True)

    def save_partial(offset):
        state = {'config': config, 'next_offset': offset + 1,
                 'counts': {split: dict(counts[split]) for split in SPLITS},
                 'rows': rows, 'unseen_rows': unseen_rows,
                 'hashes': hashes,
                 'selected_hashes': list(selected_hashes),
                 'signatures': signatures}
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = partial_path.with_suffix('.tmp')
        torch.save(state, temp_path)
        temp_path.replace(partial_path)
    try:
        for offset, datum in enumerate(islice(stream, next_offset, 345000),
                                       next_offset):
            split = next((name for name, (start, stop) in SPLITS.items()
                          if start <= offset < stop), None)
            if split is None:
                continue
            counter = counts[split]
            counter['source_documents'] += 1
            text = datum['text'][:max_chars]
            options = list(candidates(text, entity_only=True))
            if not options:
                continue
            counter['candidate_documents'] += 1
            rng = random.Random(seed + offset)
            rng.shuffle(options)
            selected = None
            selected_plan = None
            for plan in options:
                pair, _ = aligned_pair(plan, tokenizer, train_pool, n_ctx, rng)
                if pair is not None:
                    selected, selected_plan = pair, plan
                    break
            if selected is None:
                counter['unaligned_documents'] += 1
                continue
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest in selected_hashes:
                counter['exact_duplicate_rejected'] += 1
                continue
            signature = simhash(text)
            if any((signature ^ prior).bit_count() <= 3
                   for prior in signatures):
                counter['near_duplicate_rejected'] += 1
                continue
            if split == 'test':
                separate, _ = aligned_pair(
                    selected_plan, tokenizer, unseen_pool, n_ctx,
                    random.Random(seed ^ offset ^ 0xBEEF))
                if separate is None:
                    counter['unseen_code_unaligned'] += 1
                    continue
            selected_hashes.add(digest)
            signatures.append(signature)
            hashes[split].append({'offset': offset, 'sha256': digest})
            counter['paired_documents'] += 1
            realizations = [selected]
            if split == 'train':
                for _ in range(train_pairs_per_source - 1):
                    extra, _ = aligned_pair(selected_plan, tokenizer,
                                            train_pool, n_ctx, rng)
                    if extra is not None:
                        realizations.append(extra)
                    else:
                        counter['augmentation_pair_unaligned'] += 1
            counter['paired_realizations'] += len(realizations)
            for realization in realizations:
                for arm in ('left', 'right'):
                    rows[split][arm].append(realization[arm])
            if split == 'test':
                for arm in ('left', 'right'):
                    unseen_rows[arm].append(separate[arm])
            if counter['paired_documents'] % 250 == 0:
                print(split, 'selected', counter['paired_documents'],
                      'through offset', offset, flush=True)
            if split == 'train' and counter['paired_documents'] % 100 == 0:
                save_partial(offset)
    finally:
        if hasattr(stream, 'close'):
            stream.close()
    assert all(counts[split]['source_documents'] == stop - start
               for split, (start, stop) in SPLITS.items())
    cache = {'n_ctx': n_ctx, 'vocab_size': 50257,
             'train_codebook': train_pool, 'unseen_codebook': unseen_pool,
             'paired': {split: {arm: stacked(rows[split][arm])
                                for arm in ('left', 'right')}
                        for split in SPLITS},
             'test_unseen_codes': {arm: stacked(unseen_rows[arm])
                                   for arm in ('left', 'right')}}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    return {'source': 'HuggingFaceFW/fineweb sample-10BT train',
            'splits': {split: {'offsets': bounds,
                               'counts': dict(counts[split]),
                               'selected_source_sha256': hashes[split]}
                       for split, bounds in SPLITS.items()},
            'n_ctx': n_ctx, 'max_chars': max_chars, 'seed': seed,
            'train_pairs_per_source': train_pairs_per_source,
            'train_codebook_size': len(train_pool),
            'unseen_codebook_size': len(unseen_pool),
            'train_unseen_codebook_disjoint': True,
            'local_cache': str(cache_path),
            'caveat': 'The .pt cache reconstructs source prose and stays '
                      'local. Selected eligible sources screened for exact '
                      'duplicates and SimHash<=3 near duplicates across '
                      'splits. Codebook holdout is at full three-character '
                      'code level, not at constituent character level. '
                      'Natural background prose has no exact oracle.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-ctx', type=int, default=256)
    p.add_argument('--max-chars', type=int, default=8000)
    p.add_argument('--seed', type=int, default=71903)
    p.add_argument('--train-pairs-per-source', type=int, default=1)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = build(a.n_ctx, a.max_chars, a.seed, a.cache,
                   a.train_pairs_per_source)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    a.cache.with_suffix('.partial.pt').unlink(missing_ok=True)
    print({key: value['counts'] for key, value in result['splits'].items()},
          flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
