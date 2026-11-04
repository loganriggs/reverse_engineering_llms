"""Build source-document-split list and list x bracket GPT-2 windows.

The .pt cache reconstructs source prose and must remain local. Only the
aggregate manifest is suitable for Git.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import sys

from datasets import load_dataset
import torch
from tokenizers import Tokenizer

from corpus_numbered_list_validation import (
    aligned, check, token_local_factorial)
from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from build_corpus_delimiter_pilot import simhash
from corpus_numbered_list_dgp import plans, realize


PAD = 50256
ARMS = ((0, 'round'), (0, 'square'),
        (1, 'round'), (1, 'square'))
SPLITS = {'val': (95000, 105000),
          'train': (115000, 215000),
          'test': (215000, 225000)}


def _index(offsets, pos):
    return next((i for i, (start, end) in enumerate(offsets)
                 if start <= pos < end), None)


def window(doc, tokenizer, n_ctx):
    enc = tokenizer.encode(doc.text, add_special_tokens=False)
    sites = []
    for site in doc.sites:
        i = _index(enc.offsets, site.start)
        if i is None or enc.offsets[i][1] != site.end:
            return None
        sites.append(i)
    first = _index(enc.offsets, doc.plan.markers[0][0])
    if first is None:
        return None
    sources = [first]
    if doc.plan.bracket:
        opener = _index(enc.offsets, doc.plan.bracket[0])
        if opener is None:
            return None
        sources.append(opener)
    target = max(sites)
    start = min(max(0, target-(3*n_ctx)//4), min(sources))
    if target-start >= n_ctx or target-start < 1:
        return None
    ids = enc.ids[start:start+n_ctx+1]
    x = torch.full((n_ctx,), PAD, dtype=torch.long)
    y = torch.full((n_ctx,), -100, dtype=torch.long)
    x[:len(ids)-1] = torch.tensor(ids[:-1])
    y[:len(ids)-1] = torch.tensor(ids[1:])
    positions = tuple(i-start-1 for i in sites)
    targets = tuple(enc.ids[i] for i in sites)
    assert all(y[pos].item() == answer for pos, answer in
               zip(positions, targets, strict=True))
    return {'x': x, 'y': y, 'site': torch.tensor(positions),
            'target': torch.tensor(targets)}


def stacked(rows):
    return {key: torch.stack([row[key] for row in rows])
            for key in ('x', 'y', 'site', 'target')}


def make_list(plan, tokenizer, n_ctx, target_position=2):
    if len(plan.markers) <= target_position:
        return None
    arms = {(shift, 'round'): realize(
        plan, shift, target_position=target_position)
            for shift in (0, 1)}
    check(arms)
    if not all(aligned(doc, tokenizer, n_ctx) for doc in arms.values()):
        return None
    if not token_local_factorial(arms, tokenizer):
        return None
    rows = {shift: window(arms[(shift, 'round')], tokenizer, n_ctx)
            for shift in (0, 1)}
    if any(row is None for row in rows.values()):
        return None
    return rows


def make_joint(plan, tokenizer, n_ctx):
    arms = {arm: realize(plan, *arm) for arm in ARMS}
    try:
        check(arms)
    except ValueError:
        return None
    if not all(aligned(doc, tokenizer, n_ctx) for doc in arms.values()):
        return None
    if not token_local_factorial(arms, tokenizer):
        return None
    rows = {arm: window(doc, tokenizer, n_ctx)
            for arm, doc in arms.items()}
    if any(row is None for row in rows.values()):
        return None
    if len({tuple(row['site'].tolist()) for row in rows.values()}) != 1:
        return None
    return rows


def build(n_ctx, max_chars, cache_path):
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    dataset = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT',
                           split='train', streaming=True)
    stream = iter(dataset)
    counters = {name: Counter() for name in SPLITS}
    list_rows = {name: {0: [], 1: []} for name in SPLITS}
    later_rows = {name: {position: {0: [], 1: []}
                         for position in (3, 4)} for name in SPLITS}
    joint_rows = {name: {arm: [] for arm in ARMS}
                  for name in ('val', 'test')}
    hashes = {name: [] for name in SPLITS}
    selected_hashes = set()
    signatures = []
    try:
        for offset, row in enumerate(islice(stream, 95000, 225000), 95000):
            split = next((name for name, (a, b) in SPLITS.items()
                          if a <= offset < b), None)
            if split is None:
                continue
            counts = counters[split]
            counts['source_documents'] += 1
            text = row['text'][:max_chars]
            candidate_plans = list(plans(text, max_chars=max_chars))
            if not candidate_plans:
                continue
            counts['list_candidates'] += 1
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest in selected_hashes:
                counts['exact_duplicate_rejected'] += 1
                continue
            signature = simhash(text)
            if any((signature ^ prior).bit_count() <= 3 for prior in signatures):
                counts['near_duplicate_rejected'] += 1
                continue
            pair = next((result for plan in candidate_plans
                         if (result := make_list(plan, tokenizer, n_ctx))
                         is not None), None)
            if pair is None:
                counts['list_unaligned'] += 1
                continue
            selected_hashes.add(digest)
            signatures.append(signature)
            hashes[split].append(digest)
            counts['list_pairs'] += 1
            for shift, datum in pair.items():
                list_rows[split][shift].append(datum)
            for position in (3, 4):
                later = next((result for plan in candidate_plans
                              if (result := make_list(
                                  plan, tokenizer, n_ctx,
                                  target_position=position)) is not None),
                             None)
                if later is not None:
                    counts[f'ordinal_{position+1}_pairs'] += 1
                    for shift, datum in later.items():
                        later_rows[split][position][shift].append(datum)
            if split == 'train':
                continue
            for plan in plans(text, with_bracket=True,
                              max_chars=max_chars):
                joint = make_joint(plan, tokenizer, n_ctx)
                if joint is None:
                    continue
                for arm, datum in joint.items():
                    joint_rows[split][arm].append(datum)
                counts['joint_quads'] += 1
                break
    finally:
        if hasattr(stream, 'close'):
            stream.close()
    assert all(counters[split]['source_documents'] == stop-start
               for split, (start, stop) in SPLITS.items())
    cache = {
        'n_ctx': n_ctx, 'vocab_size': 50257,
        'list': {split: {shift: stacked(rows)
                         for shift, rows in shifts.items()}
                 for split, shifts in list_rows.items()},
        'list_by_ordinal': {
            split: {3: {shift: stacked(rows)
                        for shift, rows in list_rows[split].items()},
                    **{position+1: {shift: stacked(rows)
                                    for shift, rows in arms.items()}
                       for position, arms in later_rows[split].items()
                       if arms[0]}}
            for split in SPLITS},
        'joint': {split: {arm: stacked(rows)
                          for arm, rows in arms.items()}
                  for split, arms in joint_rows.items()}}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    return {'source': 'HuggingFaceFW/fineweb sample-10BT train',
            'splits': {name: {'offsets': bounds,
                              'counts': dict(counters[name]),
                              'source_sha256': hashes[name]}
                       for name, bounds in SPLITS.items()},
            'n_ctx': n_ctx, 'max_chars': max_chars,
            'local_cache': str(cache_path),
            'caveat': 'The .pt cache reconstructs source prose and stays '
                      'local. Exact and SimHash<=3 duplicate screens cover '
                      'selected eligible sources, not all corpus rows. '
                      'Validation panel informed DGP design; fresh test '
                      'offsets 215000..224999 were not inspected before '
                      'this build. No full-token oracle for source prose.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-ctx', type=int, default=256)
    p.add_argument('--max-chars', type=int, default=8000)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    a = p.parse_args()
    report = build(a.n_ctx, a.max_chars, a.cache)
    a.manifest.parent.mkdir(parents=True, exist_ok=True)
    a.manifest.write_text(json.dumps(report, indent=2)+'\n')
    print({name: row['counts'] for name, row in report['splits'].items()},
          flush=True)
    sys.stdout.flush()
    os._exit(0)  # datasets streaming can stall at interpreter exit


if __name__ == '__main__':
    main()
