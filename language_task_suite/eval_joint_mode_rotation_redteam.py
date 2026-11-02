"""Compare selected three-task basis axes with random subspace rotations."""

import argparse
import json
import sys
from pathlib import Path

import torch

from eval_joint_basis_mode_ablations import mode_system
from eval_joint_shared_token_basis import edit_scores, prediction_scores
from eval_multi_genre_token_tensor import score_tables
from eval_nested_basis_mode_ablations import mode_contribution
from eval_nested_shared_token_basis import shared_bases, token_marginal
from eval_surface_mixture_v2_nested_edits import build_pairs as bracket_pairs
from surface_mixture_v3_nested_varlen import SurfaceMixtureV3NestedVarlen
from train_surface_mixture import make_eval


def metric(label, prediction, bracket):
    if label == 'outer':
        return bracket['outer_relevant']['effect_l1']
    return prediction[label]['oracle_kl']


def random_rotation(rank, rng, device):
    matrix = torch.randn((rank, rank), generator=rng).to(device)
    q, r = torch.linalg.qr(matrix)
    return q*torch.sign(torch.diag(r)).unsqueeze(0)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--nrotations', type=int, default=16)
    parser.add_argument('--nstreams', type=int, default=64)
    parser.add_argument('--panel-seed', type=int, default=20261140)
    parser.add_argument('--rotation-seed', type=int, default=20261141)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    selected = json.loads(args.selection.read_text())
    original = selected['config']
    model_dir = Path(original['model_dir'])
    depth, rank, length = (original[key] for key in ('depth', 'rank', 'length'))
    metadata = json.loads((model_dir/'results.json').read_text())
    kwargs = metadata['generator_kwargs']
    sys.path.insert(0, args.source)
    from model import BilinearGPT, Config
    torch.set_num_threads(4)
    device = torch.device(args.device)
    record = next(row for row in metadata['records'] if row['depth'] == depth)
    model = BilinearGPT(Config(**record['model_config']))
    model.load_state_dict(torch.load(model_dir/f'model_depth{depth}.pt',
                                     map_location='cpu', weights_only=True))
    model = model.to(device).eval()
    model.blocks[0].attn.rotary.inv_freq = (
        model.blocks[0].attn.rotary.inv_freq.to(device))
    fit = SurfaceMixtureV3NestedVarlen(seed=original['fit_seed'], **kwargs)
    probability, fit_stats = token_marginal(
        fit, original['nfit'], length, fit.vocab,
        original['smoothing'], device)
    tables, downstream = score_tables(model, length)
    baseline, _, replay_error = mode_system(tables, probability, rank)
    query, key = shared_bases(tables, probability)
    u, v = query[:, :, :rank], key[:, :, :rank]
    scale = probability.sqrt()
    weighted = (tables*scale[None, None, :, None]*
                scale[None, None, None, :])
    core = torch.einsum('hqi,hlqk,hkj->hlij', u, weighted, v)
    data = make_eval(SurfaceMixtureV3NestedVarlen(
        seed=args.panel_seed, **kwargs), args.nstreams, length, device)
    bracket_gen = SurfaceMixtureV3NestedVarlen(
        seed=args.panel_seed+1, **kwargs)
    brackets = bracket_pairs(bracket_gen, args.nstreams, length)
    baseline_prediction, _ = prediction_scores(model, data, baseline,
                                                downstream)
    baseline_bracket = edit_scores(model, bracket_gen, brackets,
                                   baseline, downstream, device)
    rng = torch.Generator().manual_seed(args.rotation_seed)
    result = dict(config={**vars(args), 'selection':str(args.selection),
                          'output':str(args.output)},
                  selected_modes=[row for row in selected['selected']
                                  if row['label'] != 'control'],
                  fit=fit_stats, component_replay_max_abs=replay_error,
                  baseline_prediction=baseline_prediction,
                  baseline_bracket=baseline_bracket,
                  rows=[],
                  caveat='Random orthogonal rotations preserve the complete '
                         'rank-restricted score table. Each trial compares '
                         'its best ablated axis with the development-selected '
                         'canonical axis on an independent panel. A finite '
                         'rotation comparison is not a uniqueness proof.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    for choice in result['selected_modes']:
        label = choice['label']
        head, side, index = choice['mode']
        baseline_value = metric(label, baseline_prediction, baseline_bracket)
        def axis_damage(uu, vv, cc, axis):
            altered = baseline.clone()
            altered[head] -= mode_contribution(
                cc, uu, vv, scale, head, side, axis)
            if label == 'outer':
                bracket = edit_scores(model, bracket_gen, brackets,
                                      altered, downstream, device)
                return metric(label, None, bracket)-baseline_value
            prediction, _ = prediction_scores(model, data, altered,
                                              downstream)
            return metric(label, prediction, None)-baseline_value
        canonical = axis_damage(u, v, core, index)
        trials = []
        for rotation_index in range(args.nrotations):
            rotation = random_rotation(rank, rng, device)
            uu, vv, cc = u.clone(), v.clone(), core.clone()
            if side == 'query':
                uu[head] = u[head]@rotation
                cc[head] = torch.einsum('ji,ljk->lik',
                                        rotation, core[head])
            else:
                vv[head] = v[head]@rotation
                cc[head] = torch.einsum('lij,jk->lik',
                                        core[head], rotation)
            replay = sum(mode_contribution(cc, uu, vv, scale, head,
                side, i) for i in range(rank))
            assert float((replay-baseline[head]).abs().max()) < 1e-3
            damages = [axis_damage(uu, vv, cc, i) for i in range(rank)]
            trials.append(dict(rotation=rotation_index,
                               max_damage=max(damages),
                               mean_damage=sum(damages)/rank,
                               axis_damages=damages))
        result['rows'].append(dict(label=label, mode=choice['mode'],
                                   canonical_damage=canonical,
                                   random_trials=trials,
                                   random_max_max=max(row['max_damage']
                                                      for row in trials),
                                   fraction_random_max_below_canonical=sum(
                                       row['max_damage'] < canonical
                                       for row in trials)/len(trials)))
        save()
        print(label, choice['mode'], 'canonical', round(canonical,4),
              'random maxima max', round(result['rows'][-1]['random_max_max'],4),
              'fraction below',
              result['rows'][-1]['fraction_random_max_below_canonical'],
              flush=True)


if __name__ == '__main__':
    main()
