"""Evaluate RotoWire-trained joint-rule models on SBNation held-out games."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from audit_rotowire_document_source import game_key, load
from sports_alias_bracket_dgp import Plan, bracket_pairs
from sports_document_dgp import game_partition
from sports_entity_alias_dgp import (
    Plan as AliasPlan, candidates, single_bpe_surname_pool,
)
from train_sports_alias_bracket import evaluate, quad, stack
from train_sports_report_multirule import DeepModel, SPECS


SOURCE = Path('/home/loganriggs/Coding/neuron_circuits/artifacts/'
              'rotowire/sbnation.tar.bz2')


def draw(n, seed, tokenizer, n_ctx, source=SOURCE):
    rng = random.Random(seed)
    dataset = source.stem.removesuffix('.tar')
    assert dataset in ('rotowire', 'sbnation')
    rows = [row for row in load(source, dataset)['train']
            if game_partition(game_key(row)) == 'heldout']
    # One report per game. All choices below depend on source text and seed,
    # never on the model's outputs.
    by_game = {}
    for row in rows:
        by_game.setdefault(game_key(row), row)
    keys = sorted(by_game)
    rng.shuffle(keys)
    pool = single_bpe_surname_pool()
    records = []
    counts = {'heldout_unique_games': len(keys), 'alias_and_bracket': 0,
              'aligned': 0}
    for key in keys:
        row = by_game[key]
        aliases, brackets = candidates(row), bracket_pairs(row['summary'])
        if not aliases or not brackets:
            continue
        counts['alias_and_bracket'] += 1
        choices = [name for name in pool if name not in row['summary']
                   and name not in (row['home_city'], row['vis_city'])]
        if len(choices) < 2:
            continue
        first, surname, source_pos, target_pos = rng.choice(aliases)
        choices = [name for name in choices if name != surname]
        if len(choices) < 2:
            continue
        name_a, name_b = rng.sample(choices, 2)
        alias = AliasPlan(key, tuple(row['summary']), first, surname,
                          source_pos, target_pos, name_a, name_b)
        opener, closer = rng.choice(brackets)
        plan = Plan(alias, opener, closer)
        panel = quad(plan, tokenizer, n_ctx)
        if panel is None:
            continue
        counts['aligned'] += 1
        records.append((key, panel))
        if n is not None and len(records) == n:
            break
    if n is not None and len(records) < n:
        raise RuntimeError(f'Only {len(records)} aligned {dataset} games; '
                           f'counts={counts}')
    return records, counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--n', type=int, default=80)
    p.add_argument('--seed', type=int, default=98351)
    p.add_argument('--source', type=Path, default=SOURCE)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    args = p.parse_args()
    states = [torch.load(path, map_location='cpu', weights_only=True)
              for path in args.checkpoints]
    configs = [state['config'] for state in states]
    fields = ('n_ctx', 'alias_prior', 'alias_exponent')
    if any(tuple(config[field] for field in fields) !=
           tuple(configs[0][field] for field in fields)
           for config in configs[1:]):
        raise ValueError('Incompatible DGP configurations')
    if configs[0]['alias_prior'] != 'roster' or configs[0]['alias_exponent'] != 0:
        raise ValueError('This transfer sampler matches the uniform roster prior')
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    records, counts = draw(args.n, args.seed, tokenizer,
                           configs[0]['n_ctx'], args.source)
    batch = stack(records)
    results = []
    for path, state, cfg in zip(args.checkpoints, states, configs):
        model = DeepModel(50257, cfg['d_model'], cfg['heads'],
                          SPECS[f"block{cfg['depth']}"], cfg['n_ctx'],
                          norm='rms', residual='lerp',
                          attention='bilinear').to(args.device)
        model.load_state_dict(state['state_dict'])
        results.append({'checkpoint_sha256': hashlib.sha256(
            path.read_bytes()).hexdigest(),
            'metrics': evaluate(model, batch, args.device)})
        del model
    result = {'panel': {'source': 'SBNation train archive, held-out game '
                                 'partition, 2×2 alias/bracket factorial',
                        'n': args.n, 'seed': args.seed,
                        'counts': counts,
                        'source_sha256': hashlib.sha256(
                            args.source.read_bytes()).hexdigest(),
                        'unique_game_sha256': [hashlib.sha256(
                            repr(key).encode()).hexdigest()
                            for key, _ in records]},
              'results': results,
              'caveat': ('RotoWire-trained models are evaluated on different '
                         'human reports and game content as well as a new '
                         'writing style. This is a domain-transfer test, '
                         'not a pure style intervention. Coreference '
                         'candidates use conservative surface heuristics '
                         'and require manual spot checks. Human prose has '
                         'no full next-token oracle.')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('Aligned panel:', counts, flush=True)
    for i, row in enumerate(results):
        m = row['metrics']
        print(i, 'alias', m['alias']['paired_top1'],
              'bracket', m['bracket']['paired_top1'], flush=True)
    print('Saved', args.output)


if __name__ == '__main__':
    main()
