"""Evaluate joint-rule checkpoints on one common fresh held-out report panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from train_sports_alias_bracket import draw, evaluate, stack
from train_sports_report_multirule import DeepModel, SPECS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--n', type=int, default=100)
    p.add_argument('--seed', type=int, default=98301)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    args = p.parse_args()
    states = [torch.load(path, map_location='cpu', weights_only=True)
              for path in args.checkpoints]
    configs = [state['config'] for state in states]
    data_fields = ('n_ctx', 'alias_prior', 'alias_exponent')
    if any(tuple(config[key] for key in data_fields) !=
           tuple(configs[0][key] for key in data_fields)
           for config in configs[1:]):
        raise ValueError('Incompatible DGP configurations')
    config = configs[0]
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    records, attempts = draw(random.Random(args.seed), args.n,
                             'heldout', tokenizer, config['n_ctx'],
                             unique=True, alias_prior=config['alias_prior'],
                             alias_exponent=config['alias_exponent'])
    batch = stack(records)
    results = []
    for path, state, cfg in zip(args.checkpoints, states, configs):
        model = DeepModel(50257, cfg['d_model'], cfg['heads'],
                          SPECS[f"block{cfg['depth']}"], cfg['n_ctx'],
                          norm='rms', residual='lerp',
                          attention='bilinear').to(args.device)
        model.load_state_dict(state['state_dict'])
        results.append({'checkpoint': str(path),
                        'checkpoint_sha256': hashlib.sha256(
                            path.read_bytes()).hexdigest(),
                        'metrics': evaluate(model, batch, args.device)})
        del model
    output = {'panel': {'n': args.n, 'seed': args.seed,
                        'draw_attempts': attempts,
                        'unique_game_sha256': [hashlib.sha256(
                            repr(key).encode()).hexdigest()
                            for key, _ in records]},
              'results': results,
              'caveat': 'Fresh paired draws share held-out game pool with '
                        'prior validation and test; game keys never overlap '
                        'training group. Human prose has no full oracle.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print('saved', args.output)


if __name__ == '__main__':
    main()
