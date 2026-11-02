"""Development-selected functional score modes for three joint task sites.

Select modes on separate development prediction and bracket-edit panels.
Only then evaluate the chosen modes on fresh exact-oracle predictions and
grammar-valid verb, object, and bracket edits.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

from eval_joint_shared_token_basis import edit_scores, prediction_scores
from eval_multi_genre_token_tensor import score_tables
from eval_nested_basis_mode_ablations import mode_contribution
from eval_nested_shared_token_basis import project_tables, shared_bases, token_marginal
from eval_surface_mixture_v2_edits import build_pairs as verb_pairs
from eval_surface_mixture_v2_nested_edits import build_pairs as bracket_pairs
from eval_surface_mixture_v3_edits import build_object_pairs
from surface_mixture_v3_nested_varlen import SurfaceMixtureV3NestedVarlen
from train_surface_mixture import make_eval


@torch.no_grad()
def mode_system(tables, probability, rank):
    query, key = shared_bases(tables, probability)
    u, v = query[:, :, :rank], key[:, :, :rank]
    scale = probability.sqrt()
    weighted = (tables * scale[None, None, :, None] *
                scale[None, None, None, :])
    core = torch.einsum('hqi,hlqk,hkj->hlij', u, weighted, v)
    baseline = project_tables(tables, probability, (query, key), rank)
    def remove(head, side, index):
        contribution = mode_contribution(core, u, v, scale, head, side, index)
        altered = baseline.clone()
        altered[head] -= contribution
        return altered
    # Every mode in either side sums back to the same projected table.
    error = max(float((sum(mode_contribution(core, u, v, scale, head,
        side, i) for i in range(rank)) - baseline[head]).abs().max())
        for head in range(tables.shape[0]) for side in ('query', 'key'))
    assert error < 1e-3, error
    return baseline, remove, error


def pairs_for(seed, nstreams, length, kwargs):
    result = {}
    for name, builder, offset, panel_length in (
            ('verb', verb_pairs, 0, length-1),
            ('object', build_object_pairs, 1, length-1),
            ('bracket', bracket_pairs, 2, length)):
        gen = SurfaceMixtureV3NestedVarlen(seed=seed+offset, **kwargs)
        pairs = builder(gen, nstreams, panel_length)
        assert pairs
        result[name] = (gen, pairs)
    return result


def selected_edit_scores(model, pairs, tables, downstream, device):
    return {name: edit_scores(model, gen, rows, tables, downstream, device)
            for name, (gen, rows) in pairs.items()}


def damage(prediction, bracket, baseline_prediction, baseline_bracket):
    return dict(
        verb_kl=prediction['verb']['oracle_kl']-
                baseline_prediction['verb']['oracle_kl'],
        object_kl=prediction['object']['oracle_kl']-
                  baseline_prediction['object']['oracle_kl'],
        outer_effect_l1=bracket['outer_relevant']['effect_l1']-
                        baseline_bracket['outer_relevant']['effect_l1'],
        inner_effect_l1=bracket['inner_relevant']['effect_l1']-
                        baseline_bracket['inner_relevant']['effect_l1'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--length', type=int, default=128)
    parser.add_argument('--nfit', type=int, default=2048)
    parser.add_argument('--dev-streams', type=int, default=64)
    parser.add_argument('--holdout-streams', type=int, default=128)
    parser.add_argument('--fit-seed', type=int, default=20261130)
    parser.add_argument('--dev-seed', type=int, default=20261131)
    parser.add_argument('--holdout-seed', type=int, default=20261133)
    parser.add_argument('--smoothing', type=float, default=16.)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    metadata = json.loads((args.model_dir/'results.json').read_text())
    assert metadata['config']['generator'] == 'v3_nested_varlen'
    kwargs = metadata['generator_kwargs']
    sys.path.insert(0, args.source)
    from model import BilinearGPT, Config
    record = next(row for row in metadata['records'] if row['depth'] == args.depth)
    model = BilinearGPT(Config(**record['model_config']))
    model.load_state_dict(torch.load(args.model_dir/f'model_depth{args.depth}.pt',
                                     map_location='cpu', weights_only=True))
    model = model.to(device).eval()
    model.blocks[0].attn.rotary.inv_freq = (
        model.blocks[0].attn.rotary.inv_freq.to(device))
    fit = SurfaceMixtureV3NestedVarlen(seed=args.fit_seed, **kwargs)
    probability, fit_stats = token_marginal(
        fit, args.nfit, args.length, fit.vocab, args.smoothing, device)
    tables, downstream = score_tables(model, args.length)
    baseline, remove, replay_error = mode_system(tables, probability, args.rank)
    dev = make_eval(SurfaceMixtureV3NestedVarlen(seed=args.dev_seed, **kwargs),
                    args.dev_streams, args.length, device)
    native_dev, native_logp = prediction_scores(model, dev)
    full_dev, _ = prediction_scores(model, dev, tables, downstream, native_logp)
    assert full_dev['all']['teacher_kl'] < 1e-4
    baseline_dev, _ = prediction_scores(model, dev, baseline, downstream,
                                        native_logp)
    dev_bracket_gen = SurfaceMixtureV3NestedVarlen(seed=args.dev_seed+1,
                                                  **kwargs)
    dev_bracket_pairs = bracket_pairs(dev_bracket_gen, args.dev_streams,
                                     args.length)
    dev_bracket_base = edit_scores(model, dev_bracket_gen, dev_bracket_pairs,
                                   baseline, downstream, device)
    result = dict(config={**vars(args), 'model_dir': str(args.model_dir),
                          'output': str(args.output)},
                  fit=fit_stats, component_replay_max_abs=replay_error,
                  dev=dict(native=native_dev, full_table=full_dev,
                           baseline=baseline_dev,
                           bracket_edit_counts=dict(Counter(p['kind']
                               for p in dev_bracket_pairs)),
                           baseline_bracket=dev_bracket_base, modes=[]),
                  selection_rule='Maximum development outer-opener edit '
                      'L1 increase, maximum verb oracle-KL increase, maximum '
                      'object oracle-KL increase, and minimum sum of absolute '
                      'changes across those three scores; deduplicate.',
                  holdout=None,
                  caveat='Ablation removes one functional first-layer '
                      'token-score mode across all relative lags, not a '
                      'native weight unit. The basis is fitted on separate '
                      'unlabeled streams containing all task families. '
                      'Only chosen modes are evaluated on the holdout.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    save()
    for head in range(tables.shape[0]):
        for side in ('query', 'key'):
            for index in range(args.rank):
                altered = remove(head, side, index)
                prediction, _ = prediction_scores(model, dev, altered,
                                                  downstream, native_logp)
                bracket = edit_scores(model, dev_bracket_gen,
                                      dev_bracket_pairs, altered, downstream,
                                      device)
                row = dict(head=head, side=side, index=index,
                           prediction=prediction, bracket=bracket,
                           damage=damage(prediction, bracket, baseline_dev,
                                         dev_bracket_base))
                result['dev']['modes'].append(row)
                if len(result['dev']['modes']) % 8 == 0:
                    save()
                    print('scanned', len(result['dev']['modes']), flush=True)
    rows = result['dev']['modes']
    top_outer = max(rows, key=lambda row: row['damage']['outer_effect_l1'])
    top_verb = max(rows, key=lambda row: row['damage']['verb_kl'])
    top_object = max(rows, key=lambda row: row['damage']['object_kl'])
    low_impact = min(rows, key=lambda row: sum(abs(row['damage'][key])
                     for key in ('outer_effect_l1', 'verb_kl', 'object_kl')))
    selected = []
    for label, row in (('outer', top_outer), ('verb', top_verb),
                       ('object', top_object), ('control', low_impact)):
        key = (row['head'], row['side'], row['index'])
        if key not in [tuple(item['mode']) for item in selected]:
            selected.append(dict(label=label, mode=key,
                                 development_damage=row['damage']))
    result['selected'] = selected
    save()  # Freeze selection before the holdout is generated or scored.
    holdout = make_eval(SurfaceMixtureV3NestedVarlen(
        seed=args.holdout_seed, **kwargs), args.holdout_streams,
        args.length, device)
    holdout_pairs = pairs_for(args.holdout_seed+1,
                              args.holdout_streams, args.length, kwargs)
    native_holdout, native_logp = prediction_scores(model, holdout)
    baseline_holdout, _ = prediction_scores(model, holdout, baseline,
                                            downstream, native_logp)
    baseline_edits = selected_edit_scores(model, holdout_pairs, baseline,
                                          downstream, device)
    result['holdout'] = dict(native=native_holdout,
        baseline=baseline_holdout, baseline_edits=baseline_edits,
        pair_counts={name:dict(Counter(pair['kind'] for pair in pairs))
                     for name,(_,pairs) in holdout_pairs.items()}, selected=[])
    for chosen in selected:
        head, side, index = chosen['mode']
        altered = remove(head, side, index)
        prediction, _ = prediction_scores(model, holdout, altered,
                                          downstream, native_logp)
        edits = selected_edit_scores(model, holdout_pairs, altered,
                                     downstream, device)
        row = dict(**chosen, prediction=prediction, edits=edits,
                   damage=damage(prediction, edits['bracket'],
                                 baseline_holdout, baseline_edits['bracket']))
        result['holdout']['selected'].append(row)
        save()
        print(chosen['label'], chosen['mode'],
              {k:round(v,4) for k,v in row['damage'].items()}, flush=True)


if __name__ == '__main__':
    main()
