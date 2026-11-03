"""Aggregate heldout next-token errors for the three-rule natural-text model."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from train_sports_alias_bracket_caps import ARMS, draw
from train_sports_report_multirule import DeepModel, SPECS


def word(tokenizer, token_id):
    return tokenizer.decode([int(token_id)], skip_special_tokens=False)


@torch.no_grad()
def diagnose(model, rows, tokenizer, device):
    model.eval()
    result = {kind: defaultdict(Counter) for kind in
              ('alias', 'bracket', 'capitalization')}
    for i, arm in enumerate(ARMS):
        for _, octet in rows:
            row = octet[i]
            output = model(row['x'][None].to(device)).float()[0]
            for k, kind in enumerate(result):
                target = word(tokenizer, row['target'][k])
                predicted = word(tokenizer, output[int(row['site'][k])].argmax())
                bucket = result[kind][f'{arm[k]} | {target}']
                bucket['n'] += 1
                bucket['correct'] += predicted == target
                bucket['prediction/'+predicted] += 1
    return {kind: {target: dict(counts)
                   for target, counts in sorted(groups.items())}
            for kind, groups in result.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    a = p.parse_args()
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    config = saved['config']
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    rng = random.Random(config['seed'])
    draw(rng, config['train_worlds'], 'train', tokenizer,
         config['n_ctx'])
    val, _ = draw(rng, config['val_worlds'], 'heldout', tokenizer,
                  config['n_ctx'], unique=True)
    test, _ = draw(rng, config['test_worlds'], 'heldout', tokenizer,
                   config['n_ctx'], unique=True,
                   excluded={key for key, _ in val})
    model = DeepModel(50257, config['d_model'], config['heads'],
                      SPECS[f"block{config['depth']}"], config['n_ctx'],
                      norm='rms', residual='lerp',
                      attention='bilinear').to(a.device)
    model.load_state_dict(saved['state_dict'])
    report = {'checkpoint': str(a.checkpoint), 'test_worlds': len(test),
              'aggregates': diagnose(model, test, tokenizer, a.device),
              'note': 'No source report text is saved.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print('Saved', a.output)


if __name__ == '__main__':
    main()
