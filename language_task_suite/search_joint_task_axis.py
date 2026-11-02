"""Search within fixed rank-8 token subspaces for task-specific causal axes.

The intact projected score table never changes. Only the direction removed
for a functional intervention changes. Fit, development, and holdout streams
are disjoint; the head and query/key side were frozen by a prior scan.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from eval_joint_basis_mode_ablations import (
    damage, mode_system, pairs_for, selected_edit_scores)
from eval_joint_shared_token_basis import edit_scores, prediction_scores
from eval_multi_genre_token_tensor import score_tables
from eval_nested_basis_mode_ablations import mode_contribution
from eval_nested_shared_token_basis import shared_bases, token_marginal
from eval_surface_mixture_v2_nested_edits import build_pairs as bracket_pairs
from surface_mixture_v3_nested_varlen import SurfaceMixtureV3NestedVarlen
from train_surface_mixture import make_eval


@torch.no_grad()
def directional_component(core, u, v, scale, head, side, direction):
    if side == 'query':
        token_axis = u[head]@direction
        row = torch.einsum('i,lij,kj->lk',
                           direction, core[head], v[head])
        weighted = token_axis[None, :, None]*row[:, None, :]
    else:
        token_axis = v[head]@direction
        column = torch.einsum('qi,lij,j->lq',
                              u[head], core[head], direction)
        weighted = column[:, :, None]*token_axis[None, None, :]
    return weighted/scale[None, :, None]/scale[None, None, :]


def objective(label, changes, penalty):
    target = {'outer':'outer_effect_l1',
              'verb':'verb_kl','object':'object_kl'}[label]
    others = [name for name in ('outer_effect_l1','verb_kl','object_kl')
              if name != target]
    return changes[target]-penalty*sum(max(changes[name], 0.)
                                       for name in others)


def random_unit(rank, rng, device):
    vector = torch.randn(rank, generator=rng, device='cpu').to(device)
    return vector/vector.norm()


@torch.no_grad()
def directional_rotation_check(device):
    """An arbitrary direction equals axis zero after rotating the basis."""
    rng = torch.Generator().manual_seed(3317)
    heads, lags, vocab, rank = 2, 5, 23, 8
    u = torch.linalg.qr(torch.randn((heads, vocab, rank),
                        generator=rng, device='cpu').to(device))[0]
    v = torch.linalg.qr(torch.randn((heads, vocab, rank),
                        generator=rng, device='cpu').to(device))[0]
    core = torch.randn((heads, lags, rank, rank),
                       generator=rng, device='cpu').to(device)
    scale = torch.rand(vocab, generator=rng, device='cpu').to(device)+.3
    direction = random_unit(rank, rng, device)
    proposal = torch.column_stack((direction, torch.randn(
        (rank, rank-1), generator=rng, device='cpu').to(device)))
    rotation = torch.linalg.qr(proposal)[0]
    if rotation[:, 0]@direction < 0:
        rotation[:, 0] *= -1
    errors = {}
    for side in ('query', 'key'):
        uu, vv, cc = u.clone(), v.clone(), core.clone()
        if side == 'query':
            uu[0] = u[0]@rotation
            cc[0] = torch.einsum('ji,ljk->lik', rotation, core[0])
        else:
            vv[0] = v[0]@rotation
            cc[0] = torch.einsum('lij,jk->lik', core[0], rotation)
        direct = directional_component(core, u, v, scale, 0, side, direction)
        rotated = mode_contribution(cc, uu, vv, scale, 0, side, 0)
        errors[side] = float((direct-rotated).abs().max())
    assert max(errors.values()) < 1e-4, errors
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--random-candidates', type=int, default=64)
    parser.add_argument('--local-candidates', type=int, default=32)
    parser.add_argument('--local-scale', type=float, default=.25)
    parser.add_argument('--offtask-penalty', type=float, default=1.)
    parser.add_argument('--nstreams-dev', type=int, default=64)
    parser.add_argument('--nstreams-holdout', type=int, default=128)
    parser.add_argument('--dev-seed', type=int, default=20261150)
    parser.add_argument('--holdout-seed', type=int, default=20261160)
    parser.add_argument('--search-seed', type=int, default=20261151)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    rotation_check = directional_rotation_check(device)
    selected = json.loads(args.selection.read_text())
    original = selected['config']
    model_dir = Path(original['model_dir'])
    depth, rank, length = (original[name] for name in
                           ('depth', 'rank', 'length'))
    metadata = json.loads((model_dir/'results.json').read_text())
    kwargs = metadata['generator_kwargs']
    sys.path.insert(0, args.source)
    from model import BilinearGPT, Config
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
    for choice in selected['selected']:
        if choice['label'] == 'control':
            continue
        head, side, index = choice['mode']
        basis_axis = torch.eye(rank, device=device)[index]
        actual = directional_component(core, u, v, scale, head, side,
                                       basis_axis)
        expected = mode_contribution(core, u, v, scale, head, side, index)
        assert float((actual-expected).abs().max()) < 1e-4
    dev = make_eval(SurfaceMixtureV3NestedVarlen(seed=args.dev_seed, **kwargs),
                    args.nstreams_dev, length, device)
    dev_bracket_gen = SurfaceMixtureV3NestedVarlen(
        seed=args.dev_seed+1, **kwargs)
    dev_brackets = bracket_pairs(dev_bracket_gen, args.nstreams_dev, length)
    baseline_dev, _ = prediction_scores(model, dev, baseline, downstream)
    baseline_dev_bracket = edit_scores(model, dev_bracket_gen, dev_brackets,
                                      baseline, downstream, device)
    rng = torch.Generator().manual_seed(args.search_seed)
    result = dict(config={**vars(args), 'selection':str(args.selection),
                          'output':str(args.output)},
                  fit=fit_stats, component_replay_max_abs=replay_error,
                  directional_rotation_check=rotation_check,
                  search_head_side='Frozen by the prior development scan',
                  dev_baseline=dict(prediction=baseline_dev,
                                    bracket=baseline_dev_bracket),
                  searches=[],holdout=None,
                  caveat='A unit direction in a fixed rank-restricted '
                         'query/key token subspace is searched to maximize '
                         'task damage after removal, with positive off-task '
                         'damage penalized. This changes the interpretive '
                         'axis, not the intact score function. The axis '
                         'search is heuristic and the model saw all tasks.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    def evaluate_direction(label, head, side, direction, panel, bracket_gen,
                           bracket_edits, base_pred, base_bracket):
        altered = baseline.clone()
        altered[head] -= directional_component(core, u, v, scale, head,
                                               side, direction)
        prediction, _ = prediction_scores(model, panel, altered, downstream)
        bracket = edit_scores(model, bracket_gen, bracket_edits,
                              altered, downstream, device)
        changes = damage(prediction, bracket, base_pred, base_bracket)
        return dict(prediction=prediction, bracket=bracket, damage=changes,
                    objective=objective(label, changes,
                                        args.offtask_penalty))
    for choice in selected['selected']:
        label = choice['label']
        if label == 'control':
            continue
        head, side, index = choice['mode']
        canonical_axis = torch.eye(rank, device=device)[index]
        canonical_metrics = evaluate_direction(
            label, head, side, canonical_axis, dev, dev_bracket_gen,
            dev_brackets, baseline_dev, baseline_dev_bracket)
        best_axis = canonical_axis.clone()
        best_metrics = canonical_metrics
        records = []
        for phase, count in (('random', args.random_candidates),
                             ('local', args.local_candidates)):
            anchor = best_axis.clone()
            for _ in range(count):
                proposal = random_unit(rank, rng, device)
                if phase == 'local':
                    proposal = anchor+args.local_scale*proposal
                    proposal = proposal/proposal.norm()
                metrics = evaluate_direction(
                    label, head, side, proposal, dev, dev_bracket_gen,
                    dev_brackets, baseline_dev, baseline_dev_bracket)
                records.append(dict(phase=phase, objective=metrics['objective'],
                                    damage=metrics['damage']))
                if metrics['objective'] > best_metrics['objective']:
                    best_axis = proposal.clone()
                    best_metrics = metrics
        result['searches'].append(dict(label=label, head=head, side=side,
            canonical_index=index, canonical_dev=canonical_metrics,
            selected_direction=best_axis.cpu().tolist(),
            selected_dev=best_metrics, candidates=records))
        save()
        print(label, 'canonical objective',
              round(canonical_metrics['objective'],4), 'searched',
              round(best_metrics['objective'],4), flush=True)
    save()  # Directions are fixed before holdout construction.
    holdout = make_eval(SurfaceMixtureV3NestedVarlen(
        seed=args.holdout_seed, **kwargs), args.nstreams_holdout,
        length, device)
    holdout_bracket_gen = SurfaceMixtureV3NestedVarlen(
        seed=args.holdout_seed+1, **kwargs)
    holdout_brackets = bracket_pairs(holdout_bracket_gen,
                                    args.nstreams_holdout, length)
    holdout_pairs = pairs_for(args.holdout_seed+2,
                              args.nstreams_holdout, length, kwargs)
    baseline_holdout, _ = prediction_scores(model, holdout, baseline,
                                            downstream)
    baseline_holdout_bracket = edit_scores(
        model, holdout_bracket_gen, holdout_brackets,
        baseline, downstream, device)
    baseline_holdout_edits = selected_edit_scores(
        model, holdout_pairs, baseline, downstream, device)
    result['holdout'] = dict(baseline_prediction=baseline_holdout,
        baseline_bracket=baseline_holdout_bracket,
        baseline_edits=baseline_holdout_edits, rows=[])
    for search in result['searches']:
        label, head, side, index = (search[name] for name in
                                   ('label','head','side','canonical_index'))
        directions = dict(canonical=torch.eye(rank, device=device)[index],
                          searched=torch.tensor(search['selected_direction'],
                                                device=device))
        row = dict(label=label,head=head,side=side,axes={})
        for name, direction in directions.items():
            metrics = evaluate_direction(
                label, head, side, direction, holdout,
                holdout_bracket_gen, holdout_brackets,
                baseline_holdout, baseline_holdout_bracket)
            altered = baseline.clone()
            altered[head] -= directional_component(core, u, v, scale,
                                                   head, side, direction)
            edits = selected_edit_scores(model, holdout_pairs,
                                         altered, downstream, device)
            row['axes'][name] = dict(**metrics, edits=edits)
        result['holdout']['rows'].append(row)
        save()
        print(label, 'holdout canonical/searched',
              round(row['axes']['canonical']['objective'],4),
              round(row['axes']['searched']['objective'],4), flush=True)


if __name__ == '__main__':
    main()
