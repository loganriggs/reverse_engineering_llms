"""Test list continuation at later ordinals absent from weighted training sites."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import sys

import torch
from tokenizers import Tokenizer

from train_corpus_delimiter_bilinear import _model
from train_corpus_list_delimiter_mixture import eval_list
from datasets import load_dataset
from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from build_corpus_numbered_list_cache import make_list, stacked
from corpus_numbered_list_dgp import plans


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--skip', type=int, default=225000)
    p.add_argument('--n', type=int, default=10000)
    p.add_argument('--n-ctx', type=int, default=256)
    p.add_argument('--max-chars', type=int, default=8000)
    p.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    p.add_argument('--training-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    seen = {digest for split in json.loads(
        a.training_manifest.read_text())['splits'].values()
        for digest in split['source_sha256']}
    rows = {position: {0: [], 1: []} for position in (2, 3, 4, 5)}
    hashes = {position: [] for position in rows}
    counts = Counter()
    dataset = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT',
                           split='train', streaming=True)
    stream = iter(dataset)
    try:
        for row in islice(stream, a.skip, a.skip+a.n):
            counts['source_documents'] += 1
            source = row['text'][:a.max_chars]
            options = list(plans(source, max_chars=a.max_chars))
            if not options:
                continue
            counts['list_candidate_documents'] += 1
            digest = hashlib.sha256(source.encode()).hexdigest()
            if digest in seen:
                counts['exact_overlap_rejected'] += 1
                continue
            for position in rows:
                pair = next((result for plan in options
                             if (result := make_list(
                                 plan, tokenizer, a.n_ctx,
                                 target_position=position)) is not None),
                            None)
                if pair is None:
                    continue
                counts[f'ordinal_{position+1}_pairs'] += 1
                hashes[position].append(digest)
                for shift, datum in pair.items():
                    rows[position][shift].append(datum)
    finally:
        if hasattr(stream, 'close'):
            stream.close()
    assert counts['source_documents'] == a.n
    panels = {position+1: {shift: stacked(items)
                           for shift, items in shifted.items()}
              for position, shifted in rows.items()
              if shifted[0]}
    torch.set_num_threads(4)
    evaluations = {}
    for path in a.checkpoints:
        saved = torch.load(path, map_location='cpu', weights_only=True)
        model = _model(50257, a.n_ctx, 2, 64, 4, a.device, 'none')
        model.load_state_dict(saved['state_dict'])
        evaluations[path.stem] = {
            str(position): eval_list(model, panel, a.device)
            for position, panel in panels.items()}
    result = {
        'config': {key: ([str(v) for v in value] if isinstance(value, list)
                         else str(value) if isinstance(value, Path) else value)
                   for key, value in vars(a).items()},
        'counts': dict(counts),
        'source_sha256': {str(position+1): values
                          for position, values in hashes.items()},
        'evaluations': evaluations,
        'scope': 'Fresh document offsets absent from training and earlier '
                 'validation/test. Exact selected-source hash overlap '
                 'rejected; approximate duplicates not ruled out. '
                 'Weighted list training targeted only item 3 (or 4 in '
                 'shifted arm); later ordinals probe transfer beyond '
                 'those labels. All-token CE still exposed other numbers.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['counts'], flush=True)
    for name, panel in evaluations.items():
        print(name, {k: round(v['both_top1'], 3)
                     for k, v in panel.items()}, flush=True)
    sys.stdout.flush()
    os._exit(0)  # datasets streaming can stall at interpreter exit


if __name__ == '__main__':
    main()
