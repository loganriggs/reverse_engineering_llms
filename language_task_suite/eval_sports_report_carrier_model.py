"""Fresh grouped-game behavioral check for a trained carrier checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path
import random

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from train_sports_report_carrier import DeepModel, SPECS, draw_pairs, evaluate, _stack


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--seed', type=int, default=95411)
    p.add_argument('--pairs', type=int, default=100)
    p.add_argument('--group', choices=('train', 'heldout'),
                   default='heldout')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    config = state['config']
    if state['tokenizer_sha256'] != hashlib.sha256(
            DEFAULT_TOKENIZER.read_bytes()).hexdigest():
        raise ValueError('Tokenizer changed')
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    rows, attempts = draw_pairs(
        random.Random(a.seed), a.pairs, a.group, tokenizer,
        config['n_ctx'], unique_games=True,
        carrier_words=config['carrier_words'],
        query_kind=config['query_kind'])
    paired = tuple(_stack([record[side] for _, record in rows])
                   for side in (0, 1))
    norm = 'rms'
    model = DeepModel(50257, config['d_model'], config['heads'],
                      SPECS[f"block{state['depth']}"], config['n_ctx'],
                      norm=norm, residual='lerp',
                      attention='bilinear').to(a.device)
    model.load_state_dict(state['state_dict'])
    result = {'config': {'checkpoint': str(a.checkpoint),
                         'seed': a.seed, 'pairs': a.pairs,
                         'group': a.group,
                         'attempts': attempts,
                         'query_kind': config['query_kind'],
                         'carrier_words': config['carrier_words']},
              'checkpoint_sha256': hashlib.sha256(
                  a.checkpoint.read_bytes()).hexdigest(),
              'score': evaluate(model, paired, a.device),
              'caveat': 'New random draw under current carrier filter. '
                        'For heldout group there is no training-game overlap, '
                        'but individual keys may overlap earlier heldout '
                        'evaluation panels. Train-group panel is diagnostic '
                        'and may contain fitted games.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['score'])


if __name__ == '__main__':
    main()
