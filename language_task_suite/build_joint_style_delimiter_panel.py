"""Cache a document-disjoint FineWeb style×delimiter factorial locally.

The resulting token tensors can reconstruct source text. Keep the .pt local;
only aggregate evaluation JSON belongs in Git.
"""

from __future__ import annotations

import argparse
import hashlib
from itertools import islice
import os
from pathlib import Path
import random

import torch
from datasets import load_dataset
from tokenizers import Tokenizer

from audit_corpus_joint_style_delimiter import aligned
from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from corpus_joint_style_delimiter_dgp import sample_plan, validate_factorial


ARMS = (('us', 'round'), ('uk', 'round'),
        ('us', 'square'), ('uk', 'square'))


def encode(doc, alignment, tokenizer, n_ctx):
    enc = tokenizer.encode(doc.text, add_special_tokens=False)
    start = alignment['window_start']
    site = [index-start-1 for index in alignment['site_token']]
    assert all(0 <= pos < n_ctx for pos in site)
    length = min(n_ctx, len(enc.ids)-start-1)
    assert length > max(site)
    x = torch.full((n_ctx,), 50256, dtype=torch.long)
    y = torch.full((n_ctx,), -100, dtype=torch.long)
    x[:length] = torch.tensor(enc.ids[start:start+length])
    y[:length] = torch.tensor(enc.ids[start+1:start+length+1])
    return {'x': x, 'y': y,
            'site': torch.tensor(site),
            'target': torch.tensor(alignment['target_ids'])}


def build(skip, n, n_ctx, seed):
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    source = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT',
                          split='train', streaming=True)
    rng = random.Random(seed)
    found = {arm: [] for arm in ARMS}
    hashes = []
    counts = dict(documents=0, plans=0, aligned=0)
    stream = iter(source)
    try:
        for row in islice(stream, skip, skip+n):
            counts['documents'] += 1
            text = row['text'][:8000]
            plan = sample_plan(text, rng)
            if plan is None:
                continue
            counts['plans'] += 1
            arms = validate_factorial(plan)
            positions = aligned(arms, tokenizer, n_ctx)
            if positions is None:
                continue
            counts['aligned'] += 1
            hashes.append(hashlib.sha256(text.encode()).hexdigest())
            for arm in ARMS:
                found[arm].append(encode(arms[arm], positions[arm],
                                         tokenizer, n_ctx))
    finally:
        if hasattr(stream, 'close'):
            stream.close()
    if counts['documents'] != n or counts['aligned'] < 10:
        raise RuntimeError(f'Unexpected panel yield: {counts}')
    stacked = {arm: {key: torch.stack([row[key] for row in rows])
                     for key in ('x', 'y', 'site', 'target')}
               for arm, rows in found.items()}
    for style in ('us', 'uk'):
        assert torch.equal(stacked[style, 'round']['target'][:, 0],
                           stacked[style, 'square']['target'][:, 0])
    for bracket in ('round', 'square'):
        assert torch.equal(stacked['us', bracket]['target'][:, 1],
                           stacked['uk', bracket]['target'][:, 1])
    return {'config': {'skip': skip, 'n': n, 'n_ctx': n_ctx, 'seed': seed},
            'counts': counts, 'source_sha256': hashes,
            'arms': stacked}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--skip', type=int, default=83500)
    p.add_argument('--n', type=int, default=1000)
    p.add_argument('--n-ctx', type=int, default=320)
    p.add_argument('--seed', type=int, default=98501)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = build(args.skip, args.n, args.n_ctx, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print('saved', args.output, result['counts'], flush=True)
    os._exit(0)  # datasets/pyarrow streaming may hang during interpreter exit


if __name__ == '__main__':
    main()
