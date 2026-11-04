"""Build local-only token cache with page and whole-column-pair holdouts."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

import torch
from tokenizers import Tokenizer

from audit_totto_table_description_dgp import DEFAULT_TOKENIZER, target_token_ids
from totto_table_description_dgp import check_visible, extract


TRAIN = 'totto_data/totto_train_data.jsonl'
HOLDOUT_PAIR = frozenset(('title', 'year'))


def page_split(page):
    digest = hashlib.sha256(page.encode()).digest()
    bucket = int.from_bytes(digest[:4], 'little') % 100
    return ('train' if bucket < 80 else
            'val' if bucket < 90 else 'test')


def encode(tokenizer, example, n_ctx=256):
    ids = tokenizer.encode(example.text).ids
    if not 2 <= len(ids) <= n_ctx+1:
        return None
    positions = []
    families = []
    for site in example.sites:
        target_ids, _ = target_token_ids(tokenizer, example, site)
        if not target_ids or target_ids[0] < 1 or target_ids[-1] > n_ctx:
            return None
        positions.append(tuple(index-1 for index in target_ids))
        families.append(site.column.casefold())
    x = torch.full((n_ctx,), tokenizer.token_to_id('<|endoftext|>'),
                   dtype=torch.int32)
    y = torch.full((n_ctx,), -100, dtype=torch.int32)
    n = len(ids)-1
    x[:n] = torch.tensor(ids[:-1], dtype=torch.int32)
    y[:n] = torch.tensor(ids[1:], dtype=torch.int32)
    return {'x': x, 'y': y, 'sites': tuple(positions),
            'families': tuple(families), 'n_tokens': n,
            'pair': tuple(sorted(families[:2]))}


def build(source, output, manifest, *, n_ctx=256):
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    counts = Counter()
    splits = {name: [] for name in
              ('train', 'val', 'test', 'combo_val', 'combo_test')}
    page_sets = {name: set() for name in splits}
    family_counts = {name: Counter() for name in splits}
    with zipfile.ZipFile(source) as archive, archive.open(TRAIN) as stream:
        for line in stream:
            row = json.loads(line)
            counts['records'] += 1
            example = extract(row)
            if example is None:
                continue
            check_visible(example)
            record = encode(tokenizer, example, n_ctx)
            if record is None:
                counts['tokenization_rejected'] += 1
                continue
            page = row.get('table_webpage_url') or example.page
            split = page_split(page)
            families = set(record['families'])
            header_line = next(line for line in example.text.splitlines()
                               if line.startswith('| '))
            visible_columns = {column.strip().casefold() for column in
                               header_line.strip('|').split('|')}
            if HOLDOUT_PAIR <= visible_columns and split == 'train':
                counts['visible_pair_dropped_from_train'] += 1
                continue
            if HOLDOUT_PAIR <= families:
                split = 'combo_'+split
            splits[split].append(record)
            page_sets[split].add(page)
            family_counts[split].update(families)
            counts['accepted'] += 1
    # The same page may have both regular and combo examples in one heldout
    # partition. Across train/val/test partitions, identity stays disjoint.
    assert not (page_sets['train'] & (page_sets['val'] | page_sets['test'] |
                                      page_sets['combo_val'] |
                                      page_sets['combo_test']))
    assert not ((page_sets['val'] | page_sets['combo_val']) &
                (page_sets['test'] | page_sets['combo_test']))
    payload = {'n_ctx': n_ctx, 'vocab_size': tokenizer.get_vocab_size(),
               'splits': splits, 'holdout_pair': sorted(HOLDOUT_PAIR)}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    summary = {
        'source': str(source), 'cache_local': str(output),
        'n_ctx': n_ctx, 'vocab_size': tokenizer.get_vocab_size(),
        'holdout_pair': sorted(HOLDOUT_PAIR),
        'split_rule': 'SHA256 page URL, first 4 bytes modulo 100; '
                      '80% train, 10% val, 10% test. Every training row '
                      'with both visible Title and Year columns is removed; '
                      'combo evaluation requires both selected sites.',
        'counts': dict(counts),
        'examples_per_split': {name: len(rows)
                               for name, rows in splits.items()},
        'pages_per_split': {name: len(pages)
                            for name, pages in page_sets.items()},
        'top_families_per_split': {
            name: counter.most_common(20)
            for name, counter in family_counts.items()},
        'caveat': 'The text cache contains source table prose and remains '
                  'local. Exact value alignment is a known relation site, '
                  'not an all-token symbolic oracle or full semantic '
                  'faithfulness certificate.'}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(summary, indent=2)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--n-ctx', type=int, default=256)
    args = parser.parse_args()
    out = build(args.source, args.output, args.manifest,
                n_ctx=args.n_ctx)
    print(out['counts'], out['examples_per_split'], flush=True)


if __name__ == '__main__':
    main()
