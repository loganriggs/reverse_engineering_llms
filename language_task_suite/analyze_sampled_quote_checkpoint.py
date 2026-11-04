"""Aggregate train/heldout binding and first-token familiarity for a checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train_corpus_delimiter_bilinear import _model
from train_fineweb_surname_copy_bilinear import evaluate


@torch.no_grad()
def seen_strata(model, data, seen, device, batch_size=8):
    a, b = data
    results = {'both_seen': [0, 0], 'any_unseen': [0, 0]}
    for start in range(0, len(a['x']), batch_size):
        left = {k: v[start:start+batch_size].to(device)
                for k, v in a.items()}
        right = {k: v[start:start+batch_size].to(device)
                 for k, v in b.items()}
        ix = torch.arange(len(left['x']), device=device)
        z = model(left['x']).float()[ix, left['site']]
        w = model(right['x']).float()[ix, right['site']]
        success = ((z[ix, left['target']] > z[ix, right['target']]) &
                   (w[ix, right['target']] > w[ix, left['target']]))
        for i, passed in enumerate(success.tolist()):
            key = ('both_seen' if int(left['target'][i]) in seen and
                   int(right['target'][i]) in seen else 'any_unseen')
            results[key][0] += 1
            results[key][1] += passed
    return {key: {'pairs': n, 'paired_rank': hits/n if n else None}
            for key, (n, hits) in results.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--depth', type=int, required=True)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--train-slice', type=int, default=224)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available()
                        else 'cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cache = torch.load(args.cache, map_location='cpu', weights_only=True)
    saved = torch.load(args.state, map_location='cpu', weights_only=True)
    model = _model(cache['vocab_size'], cache['n_ctx'], args.depth,
                   args.d_model, args.heads, args.device, 'rms')
    model.load_state_dict(saved['model'])
    model.eval()
    seen = set(torch.cat([arm['target']
                          for arm in cache['pairs']['train']]).tolist())
    splits = {}
    for name, data in cache['pairs'].items():
        if name == 'train':
            data = tuple({k: v[:args.train_slice] for k, v in arm.items()}
                         for arm in data)
        splits[name] = {'metrics': evaluate(model, data, args.device, 8),
                        'target_familiarity': seen_strata(
                            model, data, seen, args.device, 8)}
    out = {'checkpoint_step': saved['step'],
           'config': {'depth': args.depth, 'd_model': args.d_model,
                      'heads': args.heads, 'train_slice': args.train_slice},
           'splits': splits,
           'scope': 'Train slice is the first N pairs; validation/test are '
                    'whole-source heldout. Both-seen means the two target '
                    'BPE token IDs occurred as training targets. This does '
                    'not measure whole-phrase familiarity or complete rule '
                    'learning.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + '\n')
    print(args.depth, saved['step'],
          {k: round(v['metrics']['candidate_rank_paired'], 3)
           for k, v in splits.items()})


if __name__ == '__main__':
    main()
