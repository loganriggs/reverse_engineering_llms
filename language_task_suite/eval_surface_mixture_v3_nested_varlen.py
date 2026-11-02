"""Held-out exact-oracle KL and causal edits for the joint three-task model."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

from eval_surface_mixture_edits import score
from eval_surface_mixture_v2_edits import build_pairs as build_verb_pairs
from eval_surface_mixture_v2_nested_edits import (
    build_pairs as build_bracket_pairs, probabilities, summarize)
from eval_surface_mixture_v3_edits import build_object_pairs
from surface_mixture_v3_nested_varlen import SurfaceMixtureV3NestedVarlen
from train_surface_mixture import evaluate, make_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--nstreams', type=int, default=256)
    parser.add_argument('--length', type=int, default=128)
    parser.add_argument('--seed', type=int, default=20261105)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    metadata = json.loads((args.model_dir/'results.json').read_text())
    assert metadata['config']['generator'] == 'v3_nested_varlen'
    kwargs = metadata['generator_kwargs']
    sys.path.insert(0, args.source)
    from model import BilinearGPT, Config
    record = next(row for row in metadata['records'] if row['depth'] == args.depth)
    model = BilinearGPT(Config(**record['model_config']))
    model.load_state_dict(torch.load(args.model_dir/f'model_depth{args.depth}.pt',
                                     map_location='cpu', weights_only=True))
    model = model.to(args.device).eval()
    test = make_eval(SurfaceMixtureV3NestedVarlen(seed=args.seed, **kwargs),
                     args.nstreams, args.length, torch.device(args.device))
    with torch.no_grad():
        oracle_scores = evaluate(model, test)
    pair_sets = {}
    for name, builder, offset, length in (
            ('verb', build_verb_pairs, 1, args.length-1),
            ('object', build_object_pairs, 2, args.length-1),
            ('bracket', build_bracket_pairs, 3, args.length)):
        gen = SurfaceMixtureV3NestedVarlen(seed=args.seed+offset, **kwargs)
        pair_sets[name] = (gen, builder(gen, args.nstreams, length))
    edits = {}
    for name, (gen, pairs) in pair_sets.items():
        before = probabilities(model, [p['original'] for p in pairs], args.device)
        after = probabilities(model, [p['edited'] for p in pairs], args.device)
        edits[name] = (summarize(pairs, before, after) if name == 'bracket'
                       else score(pairs, before, after, gen))
    result = dict(config={**vars(args), 'model_dir': str(args.model_dir),
                          'output': str(args.output)},
                  oracle_scores=oracle_scores,
                  pair_counts={name: dict(Counter(p['kind'] for p in pairs))
                               for name, (_, pairs) in pair_sets.items()},
                  edits=edits,
                  caveat='Panels are independent of model training and of '
                         'the earlier generator audit. Exact-oracle KL and '
                         'paired causal effects test competence on these '
                         'known synthetic tasks, not new-task discovery.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('KL', {key: round(value['kl'], 4) for key, value in
                 oracle_scores.items() if key in ('all', 'verb', 'object',
                    'nested_close_inner', 'nested_close_outer')})
    print('pairs', result['pair_counts'])
    print('bracket', {key: {'effect_l1': round(value['effect_l1'], 4),
                            'after_correct': round(value['after_correct'], 4)}
                      for key, value in edits['bracket'].items()
                      if '@' not in key})


if __name__ == '__main__':
    main()
