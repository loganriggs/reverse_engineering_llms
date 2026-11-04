"""Break down a saved surname-copy model by source split and copy distance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch

from train_corpus_delimiter_bilinear import _model
from train_fineweb_surname_copy_bilinear import evaluate


def subset(pairs, indices):
    return tuple({key: value[indices] for key, value in arm.items()}
                 for arm in pairs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-train-per-band', type=int, default=120)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    a = p.parse_args()
    cache = torch.load(a.cache, map_location='cpu', weights_only=True)
    state = torch.load(a.state, map_location='cpu', weights_only=True)
    identity = state['identity']
    # The checkpoint identity is defined by the training script; verify both
    # the cache path and the architecture rather than assuming compatible files.
    if Path(identity[0]) != a.cache.resolve():
        raise ValueError('Checkpoint belongs to a different cache')
    _, depth, d_model, heads, _, _, _, _, vocab, n_ctx = identity
    if vocab != cache['vocab_size'] or n_ctx != cache['n_ctx']:
        raise ValueError('Vocabulary or context mismatch')
    model = _model(vocab, n_ctx, depth, d_model, heads, a.device, 'rms')
    model.load_state_dict(state['model'])
    rng = random.Random(46101)
    report = {'step': state['step'], 'architecture': {'depth':depth,
              'd_model':d_model,'heads':heads}, 'splits': {}}
    for split in ('train', 'val', 'test'):
        pairs = cache['pairs'][split]
        gaps = cache['gap_words'][split].tolist()
        assert len(gaps) == len(pairs[0]['x'])
        bands = {'20_49': [i for i,gap in enumerate(gaps) if gap < 50],
                 '50_99': [i for i,gap in enumerate(gaps)
                           if 50 <= gap < 100],
                 '100_plus': [i for i,gap in enumerate(gaps)
                              if gap >= 100]}
        output = {}
        for name, indices in bands.items():
            population_n = len(indices)
            if split == 'train' and len(indices) > a.max_train_per_band:
                indices = sorted(rng.sample(indices, a.max_train_per_band))
            output[name] = {'population_pairs':population_n,
                            'evaluated_pairs':len(indices),
                            'metrics':evaluate(model,subset(pairs,indices),
                                               a.device,a.batch_size)
                            if indices else None}
        report['splits'][split] = output
    report['scope'] = ('Checkpoint evaluation on source-disjoint splits. '
                       'Train bands are reproducible subsamples; validation '
                       'and test bands are complete. Long-gap test bands are '
                       'small, so their rates have high sampling uncertainty.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print({split:{band:(entry['evaluated_pairs'],
                        None if entry['metrics'] is None else
                        entry['metrics']['candidate_rank_paired'])
                  for band,entry in report['splits'][split].items()}
           for split in report['splits']}, flush=True)


if __name__ == '__main__':
    main()
