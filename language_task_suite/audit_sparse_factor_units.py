"""Probe first-layer sparse projection coordinates with causal rule edits.

For each high-norm intermediate coordinate, zero its activation at every
position, then measure changes in the model's response to valid source edits.
The flat sparse model's projection output coordinates are a reference, not a
strictly equivalent parameterization of the factor units.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from distill_sparse_depth_vs_factor import build_students, generator_for_teacher
from eval_surface_mixture_edits import predict, score
from eval_surface_mixture_v2_edits import build_pairs as build_verb_pairs
from eval_surface_mixture_v3_edits import build_object_pairs
from surface_mixture_v3 import SurfaceMixtureV3


PROJECTIONS = ('q1', 'k1', 'q2', 'k2', 'v', 'out')


def ablate_coordinate(module, coordinate):
    def hook(_, __, output):
        result = output.clone()
        result[..., coordinate] = 0.
        return result
    return module.register_forward_hook(hook)


@torch.no_grad()
def token_responses(model, module):
    tokens = torch.arange(model.config.vocab_size)
    vectors = model.blocks[0].norm_attn(model.embed(tokens))
    return module(vectors).float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--teacher-dir', type=Path,
                   default=Path('language_task_suite/surface_mixture_v2_varied_soft_models'))
    p.add_argument('--factor-dir', type=Path,
                   default=Path('language_task_suite/sparse_factor_weight_then_distill_pilot'))
    p.add_argument('--flat-dir', type=Path,
                   default=Path('language_task_suite/sparse_architecture_recovery_pilot'))
    p.add_argument('--edge-budget', type=int, default=29491)
    p.add_argument('--top-per-projection', type=int, default=8)
    p.add_argument('--arms', default='flat,factor')
    p.add_argument('--unit-specs', default='',
                   help='Optional projection:index list, e.g. k2:1,out:0; '
                        'overrides top-norm selection')
    p.add_argument('--nstreams', type=int, default=32)
    p.add_argument('--length', type=int, default=72)
    p.add_argument('--seed', type=int, default=95634)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    record_file = json.loads((a.teacher_dir/'results.json').read_text())
    record = next(r for r in record_file['records'] if r['depth'] == 2)
    teacher = BilinearGPT(Config(**record['model_config']))
    teacher.load_state_dict(torch.load(a.teacher_dir/'model_depth2.pt',
                                       weights_only=True, map_location='cpu'))
    teacher.eval()
    generator_class, kwargs = generator_for_teacher(record_file)
    gen = generator_class(seed=a.seed, **kwargs)
    pairs = build_verb_pairs(gen, a.nstreams, a.length)
    if generator_class is SurfaceMixtureV3:
        pairs += build_object_pairs(generator_class(seed=a.seed, **kwargs),
                                    a.nstreams, a.length)
    original = [row['original'] for row in pairs]
    edited = [row['edited'] for row in pairs]
    kinds = sorted({row['kind'] for row in pairs})
    indices = {kind: np.array([i for i, row in enumerate(pairs)
                               if row['kind'] == kind]) for kind in kinds}
    output = {}
    candidates = {'flat': (a.flat_dir, 49152),
                  'factor': (a.factor_dir, 98304)}
    explicit = {}
    if a.unit_specs:
        for specification in a.unit_specs.split(','):
            projection, coordinate = specification.split(':')
            if projection not in PROJECTIONS or not 0 <= int(coordinate) < 64:
                raise ValueError(specification)
            explicit.setdefault(projection, []).append(int(coordinate))
    for arm in a.arms.split(','):
        directory, raw = candidates[arm]
        model = build_students(teacher, BilinearGPT, .01)[arm]
        saved = torch.load(directory/f'{arm}_{a.edge_budget/raw:.4f}.pt',
                           weights_only=False, map_location='cpu')
        model.load_state_dict(saved['model'])
        model.eval()
        base_before = predict(model, original, len(gen.vocab))
        base_after = predict(model, edited, len(gen.vocab))
        base_effect = base_after-base_before
        base_score = score(pairs, base_before, base_after, gen)
        units = []
        for projection in PROJECTIONS:
            linear = getattr(model.blocks[0].attn, projection)
            if arm == 'factor':
                module = linear.first
                strengths = (linear.first.weight.detach().norm(dim=1) *
                             linear.second.weight.detach().norm(dim=0))
                mask_first = saved['masks'][
                    f'blocks.0.attn.{projection}.first.weight']
                mask_second = saved['masks'][
                    f'blocks.0.attn.{projection}.second.weight']
            else:
                module = linear
                strengths = linear.weight.detach().norm(dim=1)
                mask_first = saved['masks'][
                    f'blocks.0.attn.{projection}.weight']
                mask_second = None
            responses = (token_responses(model, module).numpy()
                         if projection != 'out' else None)
            coordinates = (explicit.get(projection, []) if a.unit_specs else
                           torch.argsort(strengths, descending=True)[
                               :a.top_per_projection].tolist())
            for coordinate in coordinates:
                hook = ablate_coordinate(module, coordinate)
                try:
                    before = predict(model, original, len(gen.vocab))
                    after = predict(model, edited, len(gen.vocab))
                finally:
                    hook.remove()
                ablated_score = score(pairs, before, after, gen)
                effect = after-before
                impact = {
                    kind: float(np.abs(base_effect[ids]-effect[ids]).sum(-1).mean())
                    for kind, ids in indices.items()}
                error_delta = {
                    kind: ablated_score[kind]['effect_l1_error']-
                          base_score[kind]['effect_l1_error']
                    for kind in kinds}
                top = (np.argsort(responses[:, coordinate])
                       if responses is not None else None)
                units.append(dict(projection=projection,
                                  coordinate=coordinate,
                                  strength=float(strengths[coordinate]),
                                  fan_in=int(mask_first[coordinate].sum()),
                                  fan_out=(int(mask_second[:, coordinate].sum())
                                           if mask_second is not None else None),
                                  top_positive=([gen.id2token[int(i)]
                                                 for i in top[-6:][::-1]]
                                                if top is not None else None),
                                  top_negative=([gen.id2token[int(i)]
                                                 for i in top[:6]]
                                                if top is not None else None),
                                  effect_impact=impact,
                                  error_delta=error_delta))
            print(arm, projection, 'done', flush=True)
        output[arm] = dict(base_score=base_score, units=units)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(config={**vars(a),
        'teacher_dir': str(a.teacher_dir), 'factor_dir': str(a.factor_dir),
        'flat_dir': str(a.flat_dir), 'output': str(a.output)},
        pair_count=len(pairs), results=output), indent=2)+'\n')


if __name__ == '__main__':
    main()
