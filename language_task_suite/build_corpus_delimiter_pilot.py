"""Build source-document-split GPT-2 token windows for delimiter pilot.

The tensor cache contains reconstructable corpus tokens and stays outside
the Git repository. The committed JSON contains aggregates only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from dataclasses import replace
from itertools import islice
from pathlib import Path

import torch
from datasets import load_dataset
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from corpus_transformed_delimiters import candidates, transform


WORD = re.compile(r"\b[a-z]+(?:'[a-z]+)?\b")
PAD = 50256  # GPT-2 end-of-text token, ignored on the label side.


def simhash(text):
    words = WORD.findall(text.lower())
    grams = set(tuple(words[i:i+5]) for i in range(len(words)-4))
    votes = [0]*64
    for gram in grams:
        number = int.from_bytes(hashlib.blake2b(
            ' '.join(gram).encode(), digest_size=8).digest(), 'little')
        for bit in range(64):
            votes[bit] += 1 if number & (1 << bit) else -1
    return sum((1 << bit) for bit, value in enumerate(votes) if value > 0)


def _aligned_variant(text, pair_index, style, tokenizer, n_ctx,
                     prior_pair=None):
    example = transform(text, random.Random(0), style=style,
                        pair_index=pair_index)
    assert example is not None
    if prior_pair is not None:
        prior_open, prior_close, _ = prior_pair
        assert prior_close < example.open_index
        changed = (example.text[:prior_open]+'['+
                   example.text[prior_open+1:prior_close]+']'+
                   example.text[prior_close+1:])
        example = replace(example, text=changed)
    full = tokenizer.encode(example.text, add_special_tokens=False)
    prefix = tokenizer.encode(example.prefix, add_special_tokens=False)
    target = len(prefix.ids)
    if (target >= len(full.ids) or
            full.ids[:target] != prefix.ids or
            full.offsets[target][0] != example.close_index):
        return None
    # Keep at least about 3/4 of the context before the scored close and
    # retain the opener itself. Extra suffix supports ordinary LM training.
    opener = next((i for i, (start, end) in enumerate(full.offsets)
                   if start <= example.open_index < end), None)
    if opener is None:
        return None
    start = max(0, target - (3*n_ctx)//4)
    start = min(start, opener)
    if target-start >= n_ctx:
        return None
    window = full.ids[start:start+n_ctx+1]
    if target-start < 1:
        return None
    x = torch.full((n_ctx,), PAD, dtype=torch.long)
    y = torch.full((n_ctx,), -100, dtype=torch.long)
    if len(window) >= 2:
        x[:len(window)-1] = torch.tensor(window[:-1])
        y[:len(window)-1] = torch.tensor(window[1:])
    site_position = target-start-1
    assert y[site_position].item() == full.ids[target]
    return dict(x=x, y=y, site_position=site_position,
                target_id=full.ids[target],
                target_is_pure_delimiter=(full.offsets[target][1] ==
                                          example.close_index+1),
                opener_position=opener-start)


def _stack(rows):
    if not rows:
        return None
    result = dict(x=torch.stack([row['x'] for row in rows]),
                  y=torch.stack([row['y'] for row in rows]),
                  site_position=torch.tensor(
                      [row['site_position'] for row in rows]),
                  target_id=torch.tensor([row['target_id'] for row in rows]),
                  target_is_pure_delimiter=torch.tensor(
                      [row['target_is_pure_delimiter'] for row in rows]),
                  opener_position=torch.tensor(
                      [row['opener_position'] for row in rows]))
    if 'alternative_target_id' in rows[0]:
        result['alternative_target_id'] = torch.tensor(
            [row['alternative_target_id'] for row in rows])
    return result


def build(skip, train_docs, val_docs, test_docs, max_chars, n_ctx,
          square_rate, seed, cache_path):
    assert 0 <= square_rate <= 1 and n_ctx >= 64
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    source = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT',
                          split='train', streaming=True)
    stream = iter(source)
    counts = {split: Counter() for split in ('train', 'val', 'test')}
    train_rows, val_round, val_square = [], [], []
    test_round, test_square = [], []
    val_nuisance_base, val_nuisance_changed = [], []
    test_nuisance_base, test_nuisance_changed = [], []
    hashes = {split: set() for split in counts}
    signatures = {split: [] for split in counts}
    total = train_docs+val_docs+test_docs
    try:
        for local_index, row in enumerate(islice(stream, skip, skip+total)):
            if local_index < train_docs:
                split = 'train'
            elif local_index < train_docs+val_docs:
                split = 'val'
            else:
                split = 'test'
            counter = counts[split]
            counter['source_documents'] += 1
            text = row['text'][:max_chars]
            source_hash = hashlib.sha256(text.encode()).hexdigest()
            if source_hash in hashes[split]:
                counter['exact_duplicate_within_split'] += 1
                continue
            hashes[split].add(source_hash)
            signature = simhash(text)
            if any((signature ^ earlier).bit_count() <= 3
                   for other_split, values in signatures.items()
                   if other_split != split for earlier in values):
                counter['near_duplicate_cross_split_rejected'] += 1
                continue
            options = candidates(text)
            if not options:
                continue
            counter['eligible_documents'] += 1
            pair_index = random.Random(seed+skip+local_index).randrange(
                len(options))
            round_row = _aligned_variant(text, pair_index, 'round',
                                         tokenizer, n_ctx)
            square_row = _aligned_variant(text, pair_index, 'square',
                                          tokenizer, n_ctx)
            if round_row is None or square_row is None:
                counter['unaligned_pair'] += 1
                continue
            # A paired causal test should change exactly the opener token in
            # an otherwise identical visible GPT-2 prefix. Cropping can shift
            # by one token when the two full-text tokenizations differ.
            if (round_row['site_position'] != square_row['site_position'] or
                    round_row['opener_position'] != square_row['opener_position'] or
                    (round_row['x'][:round_row['site_position']+1] !=
                     square_row['x'][:square_row['site_position']+1]
                     ).nonzero().flatten().tolist() !=
                    [round_row['opener_position']]):
                counter['nonlocal_prefix_difference'] += 1
                continue
            signatures[split].append(signature)
            counter['aligned_pairs'] += 1
            counter['paired_pure_delimiter'] += (
                round_row['target_is_pure_delimiter'] and
                square_row['target_is_pure_delimiter'])
            if split == 'train':
                style = ('square' if random.Random(
                    seed+9_000_000+skip+local_index).random() < square_rate
                         else 'round')
                train_rows.append(square_row if style == 'square'
                                  else round_row)
                counter[style+'_selected'] += 1
            elif split == 'val':
                val_round.append(round_row)
                val_square.append(square_row)
            else:
                test_round.append(round_row)
                test_square.append(square_row)
            if split != 'train':
                previous = [pair for pair in options[:pair_index]
                            if pair[1] < options[pair_index][0]]
                if previous:
                    nuisance = _aligned_variant(
                        text, pair_index, 'round', tokenizer, n_ctx,
                        prior_pair=previous[-1])
                    if nuisance is not None:
                        same_site = (nuisance['site_position'] ==
                                     round_row['site_position'] and
                                     nuisance['opener_position'] ==
                                     round_row['opener_position'] and
                                     nuisance['target_id'] ==
                                     round_row['target_id'])
                        changed_positions = (
                            (round_row['x'][:round_row['site_position']+1] !=
                             nuisance['x'][:nuisance['site_position']+1]
                             ).nonzero().flatten().tolist()
                            if same_site else [])
                        if (len(changed_positions) == 2 and
                                max(changed_positions) <
                                round_row['opener_position']):
                            if split == 'val':
                                val_nuisance_base.append(dict(
                                    round_row,
                                    alternative_target_id=square_row['target_id']))
                                val_nuisance_changed.append(dict(
                                    nuisance,
                                    alternative_target_id=square_row['target_id']))
                            else:
                                test_nuisance_base.append(dict(
                                    round_row,
                                    alternative_target_id=square_row['target_id']))
                                test_nuisance_changed.append(dict(
                                    nuisance,
                                    alternative_target_id=square_row['target_id']))
                            counter['nuisance_pairs'] += 1
    finally:
        if hasattr(stream, 'close'):
            stream.close()
    assert all(counts[split]['source_documents'] == amount
               for split, amount in [('train', train_docs),
                                     ('val', val_docs), ('test', test_docs)])
    assert train_rows and val_round and test_round
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(train=_stack(train_rows),
                    val=dict(round=_stack(val_round),
                             square=_stack(val_square)),
                    test=dict(round=_stack(test_round),
                              square=_stack(test_square)),
                    val_nuisance=dict(
                        base=_stack(val_nuisance_base),
                        changed=_stack(val_nuisance_changed)),
                    test_nuisance=dict(
                        base=_stack(test_nuisance_base),
                        changed=_stack(test_nuisance_changed)),
                    n_ctx=n_ctx, vocab_size=tokenizer.get_vocab_size(),
                    source='FineWeb sample-10BT; token arrays are local only'),
               cache_path)
    result = dict(config=dict(skip=skip, train_docs=train_docs,
                              val_docs=val_docs, test_docs=test_docs,
                              max_chars=max_chars, n_ctx=n_ctx,
                              square_rate=square_rate, seed=seed,
                              source='HuggingFaceFW/fineweb sample-10BT train',
                              tokenizer=str(DEFAULT_TOKENIZER),
                              cache_path=str(cache_path)),
                  counts={k: dict(v) for k, v in counts.items()},
                  unique_source_hashes={k: len(v) for k, v in hashes.items()},
                  near_duplicate_screen='64-bit 5-gram SimHash, '
                                        'cross-split Hamming <= 3 rejected',
                  caveat='Local tensor cache is reconstructable corpus text '
                         'and must not be committed. SimHash is only a '
                         'near-duplicate screen, not a proof of no overlap.')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--skip', type=int, default=44000)
    p.add_argument('--train-docs', type=int, default=2500)
    p.add_argument('--val-docs', type=int, default=500)
    p.add_argument('--test-docs', type=int, default=500)
    p.add_argument('--max-chars', type=int, default=8000)
    p.add_argument('--n-ctx', type=int, default=256)
    p.add_argument('--square-rate', type=float, default=.5)
    p.add_argument('--seed', type=int, default=25117)
    p.add_argument('--cache-path', type=Path, default=Path(
        '/home/loganriggs/Coding/neuron_circuits/artifacts/'
        'corpus_delimiter_pilot/dataset.pt'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = build(args.skip, args.train_docs, args.val_docs,
                   args.test_docs, args.max_chars, args.n_ctx,
                   args.square_rate, args.seed, args.cache_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['counts'], flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
