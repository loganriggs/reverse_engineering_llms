"""Fresh, game-disjoint paired alias evaluation for saved bilinear models."""

import argparse
import hashlib
import json
from pathlib import Path
import random

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from sports_entity_alias_dgp import nuisance_rename, realize, sample_plan
from train_sports_entity_alias import aligned, evaluate, stack
from train_sports_report_multirule import DeepModel, SPECS


def panel(seed, n, tokenizer, n_ctx, alias_prior='roster',
          alias_exponent=0.):
    rng = random.Random(seed)
    rows, nuisance_rows, matched_target_rows, seen, attempts = \
        [], [], [], set(), 0
    while len(rows) < n:
        attempts += 1
        if attempts > 500*n:
            raise RuntimeError('Too few unique aligned held-out games')
        plan = sample_plan(rng, 'heldout', alias_prior=alias_prior,
                           alias_exponent=alias_exponent)
        if plan.game_key in seen:
            continue
        pair = tuple(aligned(realize(plan, arm), tokenizer, n_ctx)
                     for arm in ('a', 'b'))
        if any(row is None for row in pair):
            continue
        seen.add(plan.game_key)
        rows.append((plan.game_key, pair))
        if nuisance_rename(plan) is not None:
            control = aligned(realize(plan, 'a', nuisance=True),
                              tokenizer, n_ctx)
            if control is not None:
                assert control['target'] == pair[0]['target']
                nuisance_rows.append((plan.game_key, (pair[0], control)))
                matched_target_rows.append((plan.game_key, pair))
    return stack(rows), stack(nuisance_rows), stack(matched_target_rows), \
        [hashlib.sha256(repr(key).encode()).hexdigest()
         for key, _ in rows], attempts, len(nuisance_rows)


@torch.no_grad()
def evaluate_edit_comparison(model, data, device, batch_size=8):
    model.eval()
    n = len(data[0]['x'])
    tv = target_abs = target_signed_drop = correct_both = 0.
    for start in range(0, n, batch_size):
        rows = [{key: value[start:start+batch_size].to(device)
                 for key, value in arm.items()} for arm in data]
        logits = []
        for row in rows:
            output = model(row['x']).float()
            ix = torch.arange(len(row['x']), device=device)
            logits.append(output[ix, row['site']])
        targets = rows[0]['target']
        ix = torch.arange(len(targets), device=device)
        p, q = [z.softmax(-1) for z in logits]
        tv += .5*(p-q).abs().sum(-1).sum().item()
        change = p[ix, targets]-q[ix, targets]
        target_abs += change.abs().sum().item()
        target_signed_drop += change.sum().item()
        correct_both += ((logits[0].argmax(-1) == targets) &
                         (logits[1].argmax(-1) ==
                          rows[1]['target'])).sum().item()
    return {'pairs': n, 'mean_edit_tv': tv/n,
            'mean_target_probability_change_abs': target_abs/n,
            'mean_original_target_probability_drop': target_signed_drop/n,
            'paired_top1': correct_both/n}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=97701)
    p.add_argument('--n', type=int, default=100)
    p.add_argument('--alias-prior', choices=('roster', 'eligible_target'),
                   default='roster')
    p.add_argument('--alias-exponent', type=float, default=0.)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    states = [torch.load(path, map_location='cpu', weights_only=True)
              for path in args.checkpoints]
    contexts = {state['config']['n_ctx'] for state in states}
    if len(contexts) != 1:
        raise ValueError('Checkpoint context widths differ')
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    data, nuisance_data, matched_target_data, hashes, attempts, nuisance_n = panel(
        args.seed, args.n, tokenizer, contexts.pop(), args.alias_prior,
        args.alias_exponent)
    results = {}
    for path, state in zip(args.checkpoints, states):
        config = state['config']
        model = DeepModel(50257, config['d_model'], config['heads'],
                          SPECS[f"block{config['depth']}"], config['n_ctx'],
                          norm='rms', residual='lerp',
                          attention='bilinear').to(args.device)
        model.load_state_dict(state['state_dict'])
        result = evaluate(model, data, args.device)
        results[str(path)] = {'checkpoint_sha256': hashlib.sha256(
                                  path.read_bytes()).hexdigest(),
                              'training_seed': config['seed'],
                              'training_steps': config['steps'],
                              'metrics': result,
                              'target_on_nuisance_subset':
                                  evaluate_edit_comparison(
                                      model, matched_target_data,
                                      args.device),
                              'nuisance': evaluate_edit_comparison(
                                  model, nuisance_data, args.device)}
        print(config['seed'], result, flush=True)
    output = {'config': {**vars(args),
                         'checkpoints': [str(p) for p in args.checkpoints],
                         'output': str(args.output)},
              'panel': {'unique_heldout_games': args.n,
                        'draw_attempts': attempts,
                        'game_sha256': hashes,
                        'nuisance_subset_games': nuisance_n,
                        'same_worlds_and_forms_for_all_models': True},
              'models': results,
              'caveat': 'Coreference is surface-screened, not fully annotated. '
                        'Full-vocabulary paired top1 and two-alias candidate '
                        'ranking answer different behavioral questions.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
