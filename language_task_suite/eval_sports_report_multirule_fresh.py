"""Fresh game-disjoint behavioral and selective-edit gate for multirule LM."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from train_sports_report_multirule import (
    DeepModel, KINDS, SPECS, draw, stack,
)


@torch.no_grad()
def evaluate(model, triples, device, batch_size=8):
    model.eval()
    n = len(triples[0]['x'])
    sums = defaultdict(float)
    for start in range(0, n, batch_size):
        end = min(start+batch_size, n)
        b = end-start
        ix = torch.arange(b, device=device)
        rows = [{k: v[start:end].to(device) for k, v in variant.items()}
                for variant in triples]
        logits = [model(row['x']).float() for row in rows]
        for j, kind in enumerate(KINDS):
            score = [z[ix, r['site'][:, j]] for z, r in zip(logits, rows)]
            target = [r['target'][:, j] for r in rows]
            correct = [(s.argmax(-1) == t) for s, t in zip(score, target)]
            for variant, c in zip(('original', 'score_edit', 'player_edit'), correct):
                sums[f'{kind}/{variant}_top1'] += c.sum().item()
            edit_index = 2 if kind == 'player_points' else 1
            nuisance_index = 1 if kind == 'player_points' else 2
            sums[f'{kind}/paired_top1'] += \
                (correct[0] & correct[edit_index]).sum().item()
            old, new = target[0], target[edit_index]
            before = score[0][ix, new]-score[0][ix, old]
            after = score[edit_index][ix, new]-score[edit_index][ix, old]
            sums[f'{kind}/edit_direction'] += (after > before).sum().item()
            old_probs = score[0].softmax(-1)
            edited_probs = score[edit_index].softmax(-1)
            nuisance_probs = score[nuisance_index].softmax(-1)
            sums[f'{kind}/targeted_full_tv'] += \
                .5*(old_probs-edited_probs).abs().sum(-1).sum().item()
            sums[f'{kind}/nuisance_full_tv'] += \
                .5*(old_probs-nuisance_probs).abs().sum(-1).sum().item()
    return {'triplets': n,
            'metrics': {kind: {key.split('/', 1)[1]: value/n
                               for key, value in sums.items()
                               if key.startswith(kind+'/')}
                        for kind in KINDS}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--n', type=int, default=100)
    p.add_argument('--seed', type=int, default=96801)
    p.add_argument('--batch-size', type=int, default=8)
    a = p.parse_args()
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    records, attempts = draw(random.Random(a.seed), a.n, 'heldout', tokenizer,
                              config['n_ctx'], True,
                              carrier_words=config['carrier_words'])
    assert len({key for key, _ in records}) == len(records)
    model = DeepModel(50257, config['d_model'], config['heads'],
                      SPECS[f"block{config['depth']}"], config['n_ctx'],
                      norm='rms', residual='lerp',
                      attention='bilinear').to(device)
    model.load_state_dict(checkpoint['state_dict'])
    result = {'config': {**vars(a), 'checkpoint': str(a.checkpoint),
                         'output': str(a.output), 'device': device},
              'draw_attempts': attempts,
              'model_training_config': config,
              'result': evaluate(model, stack(records), device, a.batch_size),
              'caveat': 'Fresh query/form draws from 100 unique held-out '
                        'game keys. May overlap earlier held-out eval games; '
                        'never overlaps train-group games. Candidate TV is '
                        'full-vocabulary softmax TV at the answer position.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['result'], flush=True)


if __name__ == '__main__':
    main()
