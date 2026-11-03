"""Balanced held-out alias-distance panels for the joint natural-rule pilot."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from sports_alias_bracket_dgp import sample_plan
from train_sports_alias_bracket import evaluate, quad, stack
from train_sports_report_multirule import DeepModel, SPECS


def distance_bin(plan):
    gap = plan.alias.target_token-plan.alias.source_token
    return 'near_25_49' if gap < 50 else 'middle_50_99' if gap < 100 \
        else 'far_100_plus'


def draw_strata(rng, n, tokenizer, n_ctx, alias_prior, alias_exponent):
    names = ('near_25_49', 'middle_50_99', 'far_100_plus')
    records = {name: [] for name in names}
    seen = {name: set() for name in names}
    attempts = Counter()
    max_attempts = n*30
    while any(len(records[name]) < n for name in names) and \
            sum(attempts.values()) < max_attempts:
        plan = sample_plan(rng, 'heldout', alias_prior, alias_exponent)
        name = distance_bin(plan)
        attempts[name] += 1
        if len(records[name]) >= n or plan.alias.game_key in seen[name]:
            continue
        rows = quad(plan, tokenizer, n_ctx)
        if rows is None:
            continue
        records[name].append((plan.alias.game_key, rows, plan))
        seen[name].add(plan.alias.game_key)
    print('sampled', {name: len(group) for name, group in records.items()},
          'from', sum(attempts.values()), 'draws', flush=True)
    if any(len(group) < 5 for group in records.values()):
        raise RuntimeError('Fewer than five aligned examples in a distance bin')
    return records, dict(attempts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--n-per-bin', type=int, default=30)
    p.add_argument('--seed', type=int, default=98401)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    args = p.parse_args()
    states = [torch.load(path, map_location='cpu', weights_only=True)
              for path in args.checkpoints]
    configs = [state['config'] for state in states]
    fields = ('n_ctx', 'alias_prior', 'alias_exponent')
    if any(tuple(c[k] for k in fields) != tuple(configs[0][k]
           for k in fields) for c in configs[1:]):
        raise ValueError('DGP configurations differ')
    config = configs[0]
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    records, attempts = draw_strata(
        random.Random(args.seed), args.n_per_bin, tokenizer,
        config['n_ctx'], config['alias_prior'], config['alias_exponent'])
    panels = {name: stack([(key, rows) for key, rows, _ in group])
              for name, group in records.items()}
    output = {'target_n_per_bin': args.n_per_bin, 'seed': args.seed,
              'draw_attempts': attempts,
              'panels': {name: {
                  'game_sha256': [hashlib.sha256(repr(key).encode()).hexdigest()
                                  for key, _, _ in group],
                  'alias_gap_min': min(p.alias.target_token-p.alias.source_token
                                       for _, _, p in group),
                  'alias_gap_max': max(p.alias.target_token-p.alias.source_token
                                       for _, _, p in group),
                  'bracket_before_alias': sum(p.closer_token < p.alias.source_token
                                              for _, _, p in group),
                  'alias_before_bracket': sum(p.alias.target_token < p.opener_token
                                              for _, _, p in group),
                  'intervening_alias_mentions': sum(any(
                      word == p.alias.original_surname for word in
                      p.alias.source_tokens[p.alias.source_token+2:
                                            p.alias.target_token])
                      for _, _, p in group),
              } for name, group in records.items()},
              'results': [],
              'caveat': 'Stratified sample conditions on BPE alignment and '
                        'natural-report site availability; it does not '
                        'estimate task prevalence. Games can overlap '
                        'across bins, but never overlap the training split.'}
    for path, state, cfg in zip(args.checkpoints, states, configs):
        model = DeepModel(50257, cfg['d_model'], cfg['heads'],
                          SPECS[f"block{cfg['depth']}"], cfg['n_ctx'],
                          norm='rms', residual='lerp',
                          attention='bilinear').to(args.device)
        model.load_state_dict(state['state_dict'])
        output['results'].append({
            'checkpoint': str(path),
            'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'metrics': {name: evaluate(model, panel, args.device)
                        for name, panel in panels.items()}})
        del model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print('saved', args.output)


if __name__ == '__main__':
    main()
