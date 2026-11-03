"""Cross-placement transfer on paired fixed and interleaved reports."""

import argparse
import hashlib
import json
from pathlib import Path
import random

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from eval_sports_report_multirule_fresh import evaluate
from sports_report_multirule_dgp import sample_plan, realize as fixed_realize
from sports_report_multirule_interleaved import realize as spread_realize
from train_sports_report_multirule import DeepModel, SPECS, stack, triple


def panel(seed, n, tokenizer, n_ctx):
    rng = random.Random(seed)
    records = {'fixed': [], 'spread': []}
    seen = set()
    attempts = 0
    while len(seen) < n:
        attempts += 1
        if attempts > 500*n:
            raise RuntimeError('Too few paired aligned held-out games')
        plan = sample_plan(rng, 'heldout')
        key = plan.carrier_plan.source_game_key
        if key in seen:
            continue
        fixed = triple(plan, tokenizer, n_ctx, fixed_realize)
        spread = triple(plan, tokenizer, n_ctx, spread_realize)
        if fixed is None or spread is None:
            continue
        seen.add(key)
        records['fixed'].append((key, fixed))
        records['spread'].append((key, spread))
    assert [key for key, _ in records['fixed']] == \
        [key for key, _ in records['spread']]
    return {name: stack(rows) for name, rows in records.items()}, \
        [hashlib.sha256(repr(key).encode()).hexdigest()
         for key, _ in records['fixed']], attempts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=97201)
    p.add_argument('--n', type=int, default=100)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    states = [torch.load(path, map_location='cpu', weights_only=True)
              for path in a.checkpoints]
    contexts = {state['config']['n_ctx'] for state in states}
    if len(contexts) != 1:
        raise ValueError('Models have different context widths')
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    data, hashes, attempts = panel(a.seed, a.n, tokenizer, contexts.pop())
    results = {}
    for path, state in zip(a.checkpoints, states):
        c = state['config']
        model = DeepModel(50257, c['d_model'], c['heads'],
                          SPECS[f"block{c['depth']}"], c['n_ctx'],
                          norm='rms', residual='lerp',
                          attention='bilinear').to(a.device)
        model.load_state_dict(state['state_dict'])
        results[str(path)] = {
            'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'training_seed': c['seed'],
            'training_placement': c.get('placement', 'fixed'),
            'fixed': evaluate(model, data['fixed'], a.device),
            'spread': evaluate(model, data['spread'], a.device),
        }
    result = {'config': {**vars(a),
                         'checkpoints': [str(p) for p in a.checkpoints],
                         'output': str(a.output)},
              'panel': {'unique_heldout_games': a.n,
                        'draw_attempts': attempts,
                        'game_sha256': hashes,
                        'paired_worlds_and_sentence_multisets': True},
              'models': results,
              'caveat': 'Fixed and spread placements are evaluated on the '
                        'same sampled worlds and sentence forms. Check each '
                        'model training_placement before calling a condition '
                        'OOD. Semantic and selective-edit audit is separate.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    for row in results.values():
        print(row['training_seed'], row['training_placement'],
              {name: {kind: round(values['paired_top1'], 3)
                      for kind, values in value['metrics'].items()}
               for name, value in (('fixed', row['fixed']),
                                   ('spread', row['spread']))},
              flush=True)


if __name__ == '__main__':
    main()
