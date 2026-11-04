"""Build source-disjoint natural-surname-copy pairs; cache prose locally only."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from build_corpus_delimiter_pilot import simhash
from fineweb_name_copy_dgp import NameCopyDGP, WORD
from train_sports_entity_alias import aligned, stack


class NearDuplicateFilter:
    """Exact Hamming<=3 SimHash screen via four 16-bit LSH blocks."""
    def __init__(self):
        self.signatures = []
        self.buckets = defaultdict(list)
        self.hashes = set()

    def add(self, text, source_hash):
        if source_hash in self.hashes:
            return 'exact_duplicate'
        signature = simhash(text)
        candidate_ids = set()
        for j in range(4):
            candidate_ids.update(self.buckets[(j, (signature >> (16*j)) & 65535)])
        if any((signature ^ self.signatures[i]).bit_count() <= 3
               for i in candidate_ids):
            return 'near_duplicate'
        i = len(self.signatures)
        self.signatures.append(signature)
        for j in range(4):
            self.buckets[(j, (signature >> (16*j)) & 65535)].append(i)
        self.hashes.add(source_hash)
        return None


def _align(sample, target_number, tokenizer, n_ctx):
    start, end = sample.sites[target_number]
    doc = SimpleNamespace(text=sample.text, answer_start=start,
                          answer_end=end)
    row = aligned(doc, tokenizer, n_ctx)
    if row is None:
        return None
    return {key: row[key] for key in ('x', 'y', 'site', 'target')}


def make_pair(dgp, index, tokenizer, n_ctx, rng, min_gap_words):
    sources = list(dgp.candidates[index]); rng.shuffle(sources)
    for source in sources:
        replacements = dgp.replacements(index, source)
        if len(replacements) < 2:
            continue
        a, b = rng.sample(replacements, 2)
        samples = (dgp.make(index, source, a), dgp.make(index, source, b))
        for target_number in range(min(len(item.sites) for item in samples)):
            positions = dgp.candidates[index][source]
            gap = len(WORD.findall(dgp.rows[index]['preview'][
                positions[target_number]+len(source):
                positions[target_number+1]]))
            if gap < min_gap_words:
                continue
            pair = tuple(_align(item, target_number, tokenizer, n_ctx)
                         for item in samples)
            if all(row is not None for row in pair) and \
                    pair[0]['target'] != pair[1]['target']:
                return pair, gap
    return None


def build(train_rows, test_rows, n_ctx=256, seed=45231,
          train_pairs_per_source=2, min_gap_words=20):
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    train_dgp, test_dgp = (NameCopyDGP(rows)
                           for rows in (train_rows, test_rows))
    dedup = NearDuplicateFilter()
    rows = {split: [] for split in ('train', 'val', 'test')}
    gaps = {split: [] for split in rows}
    hashes = {split: [] for split in rows}
    counts = {split: Counter() for split in rows}
    for source_rows, dgp, is_train in ((train_rows, train_dgp, True),
                                       (test_rows, test_dgp, False)):
        for index in dgp.eligible:
            source = source_rows[index]
            source_hash = source['sha256']
            if is_train:
                split = 'train'
            else:
                split = ('val' if int(source_hash[:8], 16) % 2 == 0
                         else 'test')
            counts[split]['eligible_source_documents'] += 1
            rng = random.Random(seed+source['index'])
            pairs = []
            for _ in range(train_pairs_per_source if is_train else 1):
                pair = make_pair(dgp, index, tokenizer, n_ctx, rng,
                                 min_gap_words)
                if pair is not None:
                    pairs.append(pair)
            if not pairs:
                counts[split]['unaligned_source_documents'] += 1
                continue
            rejected = dedup.add(source['preview'], source_hash)
            if rejected:
                counts[split][rejected] += 1
                continue
            rows[split].extend(pair for pair,_ in pairs)
            gaps[split].extend(gap for _,gap in pairs)
            hashes[split].append(dict(offset=source['index'],
                                      sha256=source_hash))
            counts[split]['selected_source_documents'] += 1
            counts[split]['paired_examples'] += len(pairs)
    if min(len(rows[split]) for split in rows) == 0:
        raise RuntimeError('A split has no aligned source pairs')
    cache = {'n_ctx': n_ctx, 'vocab_size': tokenizer.get_vocab_size(),
             'pairs': {split: stack([(None, pair)
                                     for pair in rows[split]])
                       for split in rows},
             'gap_words': {split: torch.tensor(gaps[split])
                           for split in rows},
             'counts': {split: dict(counts[split]) for split in rows}}
    manifest = {'source':'HuggingFaceFW/fineweb sample-10BT train',
                'config':{'n_ctx':n_ctx,'seed':seed,
                          'train_pairs_per_source':train_pairs_per_source,
                          'min_gap_words':min_gap_words,
                          'tokenizer':str(DEFAULT_TOKENIZER)},
                'source_ranges':{
                    'train':[train_rows[0]['index'],train_rows[-1]['index']],
                    'heldout':[test_rows[0]['index'],test_rows[-1]['index']]},
                'dgp_eligible':{'train':len(train_dgp.eligible),
                                'heldout':len(test_dgp.eligible)},
                'counts':cache['counts'],
                'gap_bands':{split:{
                    'under50':sum(gap < 50 for gap in values),
                    '50_99':sum(50 <= gap < 100 for gap in values),
                    '100_plus':sum(gap >= 100 for gap in values)}
                             for split,values in gaps.items()},
                'selected_source_sha256':hashes,
                'duplicate_screen':'Exact full-source hash and preview '
                    '64-bit SimHash Hamming<=3 across all splits.',
                'caveat':'Only the controlled later surname token has a '
                    'known copy rule. Person-name detection is heuristic. '
                    'The training cache contains source prose and remains '
                    'local; the aggregate manifest does not contain prose.'}
    return cache, manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-preview', type=Path, required=True)
    p.add_argument('--test-preview', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--n-ctx', type=int, default=256)
    p.add_argument('--seed', type=int, default=45231)
    p.add_argument('--train-pairs-per-source', type=int, default=2)
    p.add_argument('--min-gap-words', type=int, default=20)
    a = p.parse_args()
    train, test = (json.loads(path.read_text()) for path in
                   (a.train_preview, a.test_preview))
    assert not set(row['index'] for row in train) & \
            set(row['index'] for row in test)
    cache, manifest = build(train, test, a.n_ctx, a.seed,
                            a.train_pairs_per_source,a.min_gap_words)
    a.cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.cache.with_suffix('.tmp')
    torch.save(cache, tmp)
    tmp.replace(a.cache)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(manifest, indent=2)+'\n')
    print(manifest['counts'], flush=True)


if __name__ == '__main__':
    main()
