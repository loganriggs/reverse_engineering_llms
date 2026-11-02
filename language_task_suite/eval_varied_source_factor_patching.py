"""At an edited source position, separate QK routing from V content."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval_surface_mixture_v2_edits import build_pairs
from eval_varied_second_layer_sources import source_contributions
from surface_mixture_v2_varied import SurfaceMixtureV2Varied


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--n', type=int, default=128)
    p.add_argument('--length', type=int, default=72)
    p.add_argument('--seed', type=int, default=20261205)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    metadata = json.loads((a.model_dir/'results.json').read_text())
    assert metadata['config']['generator'] == 'v2_varied'
    gen = SurfaceMixtureV2Varied(seed=a.seed, **metadata['generator_kwargs'])
    pairs = build_pairs(gen, a.n, a.length)
    record = next(row for row in metadata['records'] if row['depth'] == 2)
    model = BilinearGPT(Config(**record['model_config']))
    model.load_state_dict(torch.load(a.model_dir/'model_depth2.pt',
                                     weights_only=True, map_location='cpu'))
    model.eval()
    h = model.config.n_head
    d = model.config.n_embd//h
    out_weight = model.blocks[1].attn.out.weight.reshape(
        model.config.n_embd, h, d)
    groups = defaultdict(list)
    for index, pair in enumerate(pairs):
        groups[len(pair['original'])].append(index)
    records = [None]*len(pairs)
    max_factor_replay_error = 0.
    for indices in groups.values():
        for start in range(0, len(indices), 32):
            batch = indices[start:start+32]
            x0 = torch.tensor([pairs[i]['original'] for i in batch])
            x1 = torch.tensor([pairs[i]['edited'] for i in batch])
            differences = x0 != x1
            if not bool((differences.sum(-1) == 1).all()):
                raise AssertionError('Expected exactly one edited token')
            edit = differences.int().argmax(-1)
            arange = torch.arange(len(batch))
            c0, r0, logits0, factors0 = source_contributions(model, x0)
            c1, _, logits1, factors1 = source_contributions(model, x1)
            scores0 = {name: factors0[name][arange, :, edit]
                       for name in ('score1', 'score2')}
            scores1 = {name: factors1[name][arange, :, edit]
                       for name in ('score1', 'score2')}
            v0 = factors0['value'][arange, :, edit]
            v1 = factors1['value'][arange, :, edit]
            base = c0[arange, :, edit]
            full_source = c1[arange, :, edit]
            s0 = scores0['score1']*scores0['score2']
            s1 = scores1['score1']*scores1['score2']
            max_factor_replay_error = max(max_factor_replay_error,
                float((base-s0[..., None]*v0).abs().max()),
                float((full_source-s1[..., None]*v1).abs().max()))
            variants = dict(
                full=s1[..., None]*v1,
                routing_only=s1[..., None]*v0,
                value_only=s0[..., None]*v1,
                qk1_only=(scores1['score1']*scores0['score2'])[..., None]*v0,
                qk2_only=(scores0['score1']*scores1['score2'])[..., None]*v0)
            p0 = F.softmax(logits0, -1)
            full_effect = F.softmax(logits1, -1)-p0
            denominator = full_effect.square().sum(-1).clamp_min(1e-20)
            per_variant = {}
            for name, candidate in variants.items():
                delta = candidate-base
                residual_change = torch.einsum('bhd,chd->bhc', delta,
                                                out_weight)
                counterfactual = model.lm_head(model.final_norm(
                    r0[:, None, :]+residual_change))
                effect = F.softmax(counterfactual, -1)-p0[:, None, :]
                per_variant[name] = dict(
                    aligned=((effect*full_effect[:, None, :]).sum(-1)/
                             denominator[:, None]).numpy(),
                    residual_l1=(full_effect[:, None, :]-effect).abs()
                    .sum(-1).numpy())
            for j, index in enumerate(batch):
                records[index] = dict(kind=pairs[index]['kind'],
                    source_lag=len(pairs[index]['original'])-1-int(edit[j]),
                    variants={name: dict(aligned=row['aligned'][j].tolist(),
                                         residual_l1=row['residual_l1'][j].tolist())
                              for name, row in per_variant.items()})
    if max_factor_replay_error > 1e-5:
        raise AssertionError(f'QK×V factor replay: {max_factor_replay_error}')
    by_kind = defaultdict(list)
    for record in records:
        by_kind[record['kind']].append(record)
    summary = {}
    for kind, rows in by_kind.items():
        head_rows = {}
        for head in range(h):
            variant_rows = {}
            for name in ('full', 'routing_only', 'value_only',
                         'qk1_only', 'qk2_only'):
                variant_rows[name] = dict(
                    mean_aligned_full_fraction=float(np.mean([
                        row['variants'][name]['aligned'][head] for row in rows])),
                    mean_residual_to_full_l1=float(np.mean([
                        row['variants'][name]['residual_l1'][head]
                        for row in rows])))
            head_rows[str(head)] = variant_rows
        source_lags = [row['source_lag'] for row in rows]
        summary[kind] = dict(n=len(rows),
                             source_lag_median=float(np.median(source_lags)),
                             source_lag_gt24=sum(lag > 24 for lag in source_lags),
                             source_lag_gt48=sum(lag > 48 for lag in source_lags),
                             heads=head_rows)
    result = dict(config={**vars(a), 'model_dir': str(a.model_dir),
                          'output': str(a.output)},
                  n_pairs=len(pairs),
                  max_factor_replay_error=max_factor_replay_error,
                  summary=summary,
                  caveat='At only the edited source position of the final '
                         'second-layer query, each counterfactual changes QK1, '
                         'QK2, their product, or V while holding the remaining '
                         'factors at their original-run values. The edited '
                         'factors themselves come from a changed full context. '
                         'Probability effects are nonlinear and variants are '
                         'not additive causal shares.')
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print('pairs', len(pairs), 'factor replay', max_factor_replay_error)
    for kind in ('lexical', 'induction', 'place_skip', 'cue'):
        if kind not in summary:
            continue
        print(kind, {head: {
            key: round(summary[kind]['heads'][str(head)][key]
                       ['mean_aligned_full_fraction'], 3)
            for key in ('full', 'routing_only', 'value_only')}
            for head in range(h)}, flush=True)


if __name__ == '__main__':
    main()
