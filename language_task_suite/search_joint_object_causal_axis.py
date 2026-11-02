"""Select a score-subspace direction by object-rule edit damage, not KL."""

import argparse
import json
import sys
from pathlib import Path

import torch

from eval_joint_basis_mode_ablations import (
    mode_system, pairs_for, selected_edit_scores)
from eval_joint_shared_token_basis import edit_scores, prediction_scores
from eval_multi_genre_token_tensor import score_tables
from eval_nested_shared_token_basis import shared_bases, token_marginal
from eval_surface_mixture_v2_nested_edits import build_pairs as bracket_pairs
from eval_surface_mixture_v3_edits import build_object_pairs
from search_joint_task_axis import directional_component, random_unit
from surface_mixture_v3_nested_varlen import SurfaceMixtureV3NestedVarlen
from train_surface_mixture import make_eval


RELEVANT = ('object_adjective', 'object_cue',
            'object_induction', 'object_event_skip')


def causal_objective(edits, baseline_edits, verb_kl, baseline_verb_kl,
                     outer_l1, baseline_outer_l1, penalty):
    relevant = sum(edits[k]['effect_l1_error']-
                   baseline_edits[k]['effect_l1_error']
                   for k in RELEVANT)/len(RELEVANT)
    nuisance = max(0., edits['object_nuisance_place']['effect_l1_error']-
                   baseline_edits['object_nuisance_place']['effect_l1_error'])
    off_task = max(0., verb_kl-baseline_verb_kl)+max(
        0., outer_l1-baseline_outer_l1)
    return dict(relevant_damage=relevant, nuisance_damage=nuisance,
                off_task_damage=off_task,
                score=relevant-penalty*(nuisance+off_task))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--kl-search', type=Path, required=True)
    parser.add_argument('--random-candidates', type=int, default=32)
    parser.add_argument('--local-candidates', type=int, default=16)
    parser.add_argument('--local-scale', type=float, default=.25)
    parser.add_argument('--offtask-penalty', type=float, default=1.)
    parser.add_argument('--dev-streams', type=int, default=32)
    parser.add_argument('--holdout-streams', type=int, default=128)
    parser.add_argument('--dev-seed', type=int, default=20261170)
    parser.add_argument('--holdout-seed', type=int, default=20261180)
    parser.add_argument('--search-seed', type=int, default=20261171)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    previous = json.loads(args.selection.read_text())
    prior_search = json.loads(args.kl_search.read_text())
    original = previous['config']
    model_dir = Path(original['model_dir'])
    depth, rank, length = (original[k] for k in ('depth', 'rank', 'length'))
    object_choice = next(row for row in previous['selected']
                         if row['label'] == 'object')
    head, side, index = object_choice['mode']
    assert side in ('query', 'key')
    kl_direction = next(row['selected_direction']
                        for row in prior_search['searches']
                        if row['label'] == 'object')
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
    canonical = torch.eye(rank, device=device)[index]
    kl_axis = torch.tensor(kl_direction, device=device)
    kl_axis = kl_axis/kl_axis.norm()
    rng = torch.Generator().manual_seed(args.search_seed)
    dev = make_eval(SurfaceMixtureV3NestedVarlen(seed=args.dev_seed,
                    **kwargs), args.dev_streams, length, device)
    dev_object_gen = SurfaceMixtureV3NestedVarlen(
        seed=args.dev_seed+1, **kwargs)
    dev_objects = build_object_pairs(dev_object_gen, args.dev_streams,
                                     length-1)
    dev_bracket_gen = SurfaceMixtureV3NestedVarlen(
        seed=args.dev_seed+2, **kwargs)
    dev_brackets = bracket_pairs(dev_bracket_gen, args.dev_streams, length)
    dev_base_prediction, _ = prediction_scores(model, dev, baseline,
                                                downstream)
    dev_base_object = edit_scores(model, dev_object_gen, dev_objects,
                                  baseline, downstream, device)
    dev_base_bracket = edit_scores(model, dev_bracket_gen, dev_brackets,
                                   baseline, downstream, device)
    result = dict(config={**vars(args), 'selection':str(args.selection),
                          'kl_search':str(args.kl_search),
                          'output':str(args.output)},
                  object_mode=dict(head=head,side=side,canonical_index=index),
                  fit=fit_stats,component_replay_max_abs=replay_error,
                  dev_baseline=dict(prediction=dev_base_prediction,
                                    object_edits=dev_base_object,
                                    bracket_edits=dev_base_bracket),
                  dev_candidates=[],selected_direction=None,holdout=None,
                  caveat='Select by mean increase in four relevant '
                         'object-rule effect-L1 errors, penalizing nuisance '
                         'object-place, verb KL, and outer-bracket edit '
                         'damage. A single axis may affect several rules. '
                         'All directions preserve the intact rank-8 score '
                         'function; only interventions differ.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    @torch.no_grad()
    def measure(direction, panel, object_gen, objects, bracket_gen, brackets,
                base_prediction, base_object, base_bracket):
        altered = baseline.clone()
        altered[head] -= directional_component(core, u, v, scale,
                                               head, side, direction)
        prediction, _ = prediction_scores(model, panel, altered, downstream)
        object_edits = edit_scores(model, object_gen, objects,
                                   altered, downstream, device)
        bracket_edits = edit_scores(model, bracket_gen, brackets,
                                    altered, downstream, device)
        scores = causal_objective(
            object_edits, base_object,
            prediction['verb']['oracle_kl'],
            base_prediction['verb']['oracle_kl'],
            bracket_edits['outer_relevant']['effect_l1'],
            base_bracket['outer_relevant']['effect_l1'],
            args.offtask_penalty)
        return dict(objective=scores,prediction=prediction,
                    object_edits=object_edits,bracket_edits=bracket_edits)
    baselines = (dev,dev_object_gen,dev_objects,dev_bracket_gen,dev_brackets,
                 dev_base_prediction,dev_base_object,dev_base_bracket)
    initial = [('canonical',canonical),('kl_searched',kl_axis)]
    for name, direction in initial:
        measured = measure(direction,*baselines)
        result['dev_candidates'].append(dict(name=name,
            direction=direction.cpu().tolist(),metrics=measured))
    best = max(result['dev_candidates'],
               key=lambda row:row['metrics']['objective']['score'])
    best_axis = torch.tensor(best['direction'],device=device)
    for phase, count in (('random',args.random_candidates),
                         ('local',args.local_candidates)):
        anchor = best_axis.clone()
        for i in range(count):
            proposal = random_unit(rank,rng,device)
            if phase == 'local':
                proposal = anchor+args.local_scale*proposal
                proposal = proposal/proposal.norm()
            measured = measure(proposal,*baselines)
            row = dict(name=f'{phase}_{i}',
                       direction=proposal.cpu().tolist(),metrics=measured)
            result['dev_candidates'].append(row)
            if measured['objective']['score'] > best['metrics']['objective']['score']:
                best, best_axis = row, proposal.clone()
        save()
        print(phase,'best score',
              round(best['metrics']['objective']['score'],4),flush=True)
    result['selected_direction'] = best_axis.cpu().tolist()
    result['selected_name'] = best['name']
    save()  # Selection fixed before holdout is sampled.
    holdout = make_eval(SurfaceMixtureV3NestedVarlen(
        seed=args.holdout_seed, **kwargs), args.holdout_streams,
        length, device)
    holdout_object_gen = SurfaceMixtureV3NestedVarlen(
        seed=args.holdout_seed+1, **kwargs)
    holdout_objects = build_object_pairs(holdout_object_gen,
                                        args.holdout_streams,length-1)
    holdout_bracket_gen = SurfaceMixtureV3NestedVarlen(
        seed=args.holdout_seed+2, **kwargs)
    holdout_brackets = bracket_pairs(holdout_bracket_gen,
                                    args.holdout_streams,length)
    holdout_pairs = pairs_for(args.holdout_seed+3,
                              args.holdout_streams,length,kwargs)
    holdout_base_prediction, _ = prediction_scores(model,holdout,
                                                    baseline,downstream)
    holdout_base_object = edit_scores(model,holdout_object_gen,
                                      holdout_objects,baseline,downstream,device)
    holdout_base_bracket = edit_scores(model,holdout_bracket_gen,
                                       holdout_brackets,baseline,downstream,device)
    holdout_base_edits = selected_edit_scores(model,holdout_pairs,
                                              baseline,downstream,device)
    result['holdout'] = dict(base_prediction=holdout_base_prediction,
        base_object_edits=holdout_base_object,
        base_bracket_edits=holdout_base_bracket,
        base_all_edits=holdout_base_edits,axes={})
    holdout_args = (holdout,holdout_object_gen,holdout_objects,
        holdout_bracket_gen,holdout_brackets,holdout_base_prediction,
        holdout_base_object,holdout_base_bracket)
    for name, direction in [('canonical',canonical),
                            ('kl_searched',kl_axis),
                            ('causal_searched',best_axis)]:
        measured = measure(direction,*holdout_args)
        altered = baseline.clone()
        altered[head] -= directional_component(core,u,v,scale,
                                               head,side,direction)
        measured['all_edits'] = selected_edit_scores(model,holdout_pairs,
                                                     altered,downstream,device)
        result['holdout']['axes'][name] = measured
        save()
        print(name,'holdout score',
              round(measured['objective']['score'],4),
              'object KL',round(measured['prediction']['object']['oracle_kl'],4),
              flush=True)


if __name__ == '__main__':
    main()
