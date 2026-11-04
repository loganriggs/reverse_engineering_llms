"""Build a disjoint list x bracket FineWeb panel and evaluate frozen models.

The local cache contains reconstructable source prose and must not be committed.
The JSON report contains only aggregate scores and source hashes.
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

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from build_corpus_delimiter_pilot import simhash
from build_corpus_numbered_list_cache import ARMS, make_joint, stacked
from corpus_numbered_list_dgp import plans
from train_corpus_delimiter_bilinear import _model
from train_corpus_list_delimiter_mixture import eval_joint


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--start', type=int, default=235000)
    p.add_argument('--n', type=int, default=100000)
    p.add_argument('--max-chars', type=int, default=8000)
    p.add_argument('--n-ctx', type=int, default=256)
    p.add_argument('--prior-manifest', type=Path)
    p.add_argument('--checkpoints', type=Path, nargs='*', default=[])
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    torch.set_num_threads(4)
    prior_hashes = set()
    if a.prior_manifest:
        prior = json.loads(a.prior_manifest.read_text())
        prior_hashes = {digest for split in prior['splits'].values()
                        for digest in split['source_sha256']}
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    source = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT',
                          split='train', streaming=True)
    stream = iter(source)
    counts = Counter()
    arm_rows = {arm: [] for arm in ARMS}
    hashes = []
    signatures = []
    offsets = []
    try:
        for offset, row in enumerate(islice(stream, a.start, a.start + a.n),
                                     a.start):
            counts['source_documents'] += 1
            text = row['text'][:a.max_chars]
            possible = plans(text, with_bracket=True, max_chars=a.max_chars)
            joint = next((panel for plan in possible
                          if (panel := make_joint(plan, tokenizer, a.n_ctx))
                          is not None), None)
            if joint is None:
                continue
            counts['eligible_joint'] += 1
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest in prior_hashes or digest in hashes:
                counts['exact_duplicate_rejected'] += 1
                continue
            signature = simhash(text)
            if any((signature ^ old).bit_count() <= 3 for old in signatures):
                counts['within_panel_near_duplicate_rejected'] += 1
                continue
            for arm in ARMS:
                arm_rows[arm].append(joint[arm])
            hashes.append(digest)
            signatures.append(signature)
            offsets.append(offset)
            if len(hashes) % 50 == 0:
                print('selected', len(hashes), 'through offset', offset,
                      flush=True)
    finally:
        if hasattr(stream, 'close'):
            stream.close()
    assert counts['source_documents'] == a.n
    assert hashes
    panel = {arm: stacked(arm_rows[arm]) for arm in ARMS}
    a.cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(panel, a.cache)
    report = {
        'source': 'HuggingFaceFW/fineweb sample-10BT train',
        'offsets': [a.start, a.start + a.n],
        'n_ctx': a.n_ctx, 'max_chars': a.max_chars,
        'counts': dict(counts), 'selected': len(hashes),
        'source_sha256': hashes, 'source_offsets': offsets,
        'prior_manifest': str(a.prior_manifest) if a.prior_manifest else None,
        'models': {},
        'caveat': 'All four list-shift x bracket-style edit arms are '
                  'token-local and within context. When a prior manifest '
                  'is supplied, no selected source duplicates it exactly. '
                  'Selected sources are screened for near duplicates '
                  'within this panel, but cross-panel near duplicates '
                  'are not screened. The cache reconstructs source prose '
                  'and is local only. Natural background text has no '
                  'complete symbolic oracle.'}
    for path in a.checkpoints:
        saved = torch.load(path, map_location='cpu', weights_only=True)
        model = _model(50257, 256, 2, 64, 4, a.device, 'none')
        model.load_state_dict(saved['state_dict'])
        model.eval()
        score = eval_joint(model, panel, a.device)
        report['models'][path.stem] = score
        print(path.stem, 'joint exact', score['all_arms_both_rules_top1'],
              'two-way', score['all_arms_both_rules_two_way'], flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print('saved', a.output, flush=True)
    # HF streaming can leave worker threads alive after the completed report.
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
