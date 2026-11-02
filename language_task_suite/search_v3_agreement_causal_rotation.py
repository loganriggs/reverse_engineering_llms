"""Search a weight-SVD subspace for a selective causal ablation axis.

Any unit vector in the fixed subspace can be completed to an orthogonal
rotation. Candidate selection uses development data only; holdout results
compare the selected rotation with the best original SVD axis.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from eval_multi_genre_v3_agreement_edits import build_pairs
from eval_surface_mixture_edits import predict, score
from multi_genre_v3_agreement import GENRES_V3, MultiGenreV3Agreement
from train_multi_genre import TASK_LABELS, evaluate, make_eval


AGREEMENT_LABELS = ('agreement_main_match', 'agreement_main_conflict',
                    'agreement_embedded')
NO_EFFECT = ('preposition_attractor_flip_invariant',
             'relative_attractor_flip_invariant',
             'preposition_swap_invariant',
             'embedded_subject_flip_invariant')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', required=True)
    p.add_argument('--depth', type=int, default=3)
    p.add_argument('--layer', type=int, default=2)
    p.add_argument('--subspace', type=int, default=8)
    p.add_argument('--random-candidates', type=int, default=64)
    p.add_argument('--ndev', type=int, default=64)
    p.add_argument('--nholdout', type=int, default=128)
    p.add_argument('--nedit-dev', type=int, default=16)
    p.add_argument('--nedit-holdout', type=int, default=64)
    p.add_argument('--seed', type=int, default=14551)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    torch.set_num_threads(4)
    saved = json.loads((Path(a.model_dir)/'results.json').read_text())
    assert saved['config']['language'] == 'v3'
    record = next(r for r in saved['records'] if r['depth'] == a.depth)
    model = BilinearGPT(Config(**record['model_config']))
    model.load_state_dict(torch.load(
        Path(a.model_dir)/f'model_depth{a.depth}.pt',
        weights_only=True, map_location='cpu'))
    model.eval()
    bond = model.blocks[a.layer].attn.out
    _, _, vh = torch.linalg.svd(bond.weight.detach().double())
    basis = vh.T[:, :a.subspace].float()
    length = saved['config']['length']
    vocab = len(saved['vocabulary'])

    def panel(seed, n, nedit):
        def data(genre, offset):
            weights = tuple(float(g == genre) for g in GENRES_V3)
            return make_eval(MultiGenreV3Agreement(
                seed=seed+offset, genre_weights=weights), n, length, 'cpu',
                TASK_LABELS + AGREEMENT_LABELS, GENRES_V3)
        edit_gen = MultiGenreV3Agreement(
            seed=seed+3, genre_weights=(0, 0, 0, 0, 0, 1))
        return dict(agreement=data('Agreement', 0),
                    faq=data('FAQ', 1), policy=data('Policy', 2),
                    pairs=build_pairs(edit_gen, nedit, length),
                    edit_gen=edit_gen)

    dev = panel(a.seed, a.ndev, a.nedit_dev)
    holdout = panel(a.seed+1009, a.nholdout, a.nedit_holdout)
    assert dev['pairs'] and holdout['pairs']

    def measure(pan, direction=None):
        if direction is not None:
            u = direction
            def remove(_, inputs):
                x = inputs[0]
                return (x-(x@u).unsqueeze(-1)*u,)
            hook = bond.register_forward_pre_hook(remove)
        out = {}
        for name in ('agreement', 'faq', 'policy'):
            out[name] = evaluate(model, pan[name])
        pairs = pan['pairs']
        before = predict(model, [pair['original'] for pair in pairs], vocab)
        after = predict(model, [pair['edited'] for pair in pairs], vocab)
        out['edits'] = score(pairs, before, after, pan['edit_gen'])
        if direction is not None:
            hook.remove()
        return out

    def objective(result, native):
        main = (result['edits']['main_subject_flip']['effect_l1_error'] -
                native['edits']['main_subject_flip']['effect_l1_error'])
        faq = (result['faq']['faq_answer_copy']['kl'] -
               native['faq']['faq_answer_copy']['kl'])
        embedded = (result['agreement']['agreement_embedded']['kl'] -
                    native['agreement']['agreement_embedded']['kl'])
        policy = (result['policy']['policy_mixed_action']['kl'] -
                  native['policy']['policy_mixed_action']['kl'])
        nuisance = sum(
            result['edits'][name]['effect_l1_error'] -
            native['edits'][name]['effect_l1_error'] for name in NO_EFFECT)
        return dict(score=main-.5*max(0., faq)-.5*max(0., embedded)-
                          .25*max(0., policy)-nuisance,
                    main_edit_damage=main, faq_copy_kl_damage=faq,
                    embedded_kl_damage=embedded, policy_kl_damage=policy,
                    nuisance_edit_damage=nuisance)

    native_dev = measure(dev)
    native_holdout = measure(holdout)
    rng = np.random.default_rng(a.seed+29)
    candidates = []
    for j in range(a.subspace):
        coefficients = torch.zeros(a.subspace)
        coefficients[j] = 1.
        candidates.append((f'svd_axis_{j}', coefficients))
    for j in range(a.random_candidates):
        coefficients = torch.tensor(rng.normal(size=a.subspace),
                                    dtype=torch.float32)
        coefficients /= coefficients.norm()
        candidates.append((f'rotation_{j}', coefficients))
    dev_rows = []
    for name, coefficients in candidates:
        direction = basis @ coefficients
        result = measure(dev, direction)
        dev_rows.append(dict(name=name, coefficients=coefficients.tolist(),
                             objective=objective(result, native_dev)))
    selected = {}
    holdout_rows = []
    for family, prefix in (('svd', 'svd_axis_'), ('rotation', 'rotation_')):
        winner = max((row for row in dev_rows if row['name'].startswith(prefix)),
                     key=lambda row: row['objective']['score'])
        selected[family] = winner['name']
        direction = basis @ torch.tensor(winner['coefficients'])
        result = measure(holdout, direction)
        holdout_rows.append(dict(name=winner['name'],
                                 objective=objective(result, native_holdout),
                                 metrics=result))
        print(family, winner['name'], 'dev score',
              round(winner['objective']['score'], 4), 'holdout',
              {k: round(v, 4) for k, v in
               holdout_rows[-1]['objective'].items()}, flush=True)
    output = dict(config=vars(a), n_dev_pairs=len(dev['pairs']),
                  n_holdout_pairs=len(holdout['pairs']),
                  objective='main_edit_damage - 0.5 positive FAQ-copy KL '
                            '- 0.5 positive embedded-verb KL - 0.25 positive '
                            'policy-action KL - summed no-effect edit damage',
                  native_dev=native_dev, native_holdout=native_holdout,
                  dev_candidates=dev_rows, selected=selected,
                  holdout=holdout_rows,
                  caveat='Supervised search over ablation directions, not '
                         'unlabeled discovery. High damage may still reflect '
                         'a destructive direction rather than a sufficient '
                         'or interpretable computation.')
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
