"""Fit local bond contrast bases on one mechanism, test on other mechanisms."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from eval_multi_genre_edits import build_pairs
from eval_surface_edit_bases import infer
from multi_genre_language import MultiGenreLanguage
from surface_mixture_language import OBJECTS


KINDS = ('policy_condition', 'policy_induction', 'policy_nuisance_person',
         'faq_answer_copy', 'faq_place_recall')


def pca(z):
    z = np.asarray(z, dtype=np.float64)
    _, u = np.linalg.eigh(z.T@z)
    return torch.tensor(np.flip(u, 1).copy(), dtype=torch.float32)


def fit_bases(pairs, before, after, weight):
    if isinstance(before, list):
        labels = np.concatenate([[pair['kind']]*len(z)
                                 for pair, z in zip(pairs, before)])
        delta = np.concatenate([a-b for a, b in zip(after, before)])
        before = np.concatenate(before)
    else:
        labels = np.array([pair['kind'] for pair in pairs])
        delta = after-before
    d = before.shape[-1]
    _, _, vh = np.linalg.svd(weight.detach().cpu().double().numpy())
    bases = dict(physical=torch.eye(d),
                 activation_pca=pca(before),
                 weight_svd=torch.tensor(vh.T.copy(), dtype=torch.float32))
    spectra = {}
    for kind in KINDS:
        subset = delta[labels == kind]
        if not len(subset):
            continue
        bases[kind+'_contrast'] = pca(subset)
        singular = np.linalg.svd(subset, compute_uv=False)**2
        spectra[kind] = {str(k): float(singular[:k].sum()/singular.sum())
                         for k in (1, 2, 4, 8, 16, 32)}
    return bases, spectra


def score(pairs, base_p, donor_p, patched_p, full_p, vocab):
    result = {}
    for kind in KINDS:
        indices = np.array([i for i, pair in enumerate(pairs)
                            if pair['kind'] == kind], dtype=int)
        if not len(indices):
            continue
        model_effect = donor_p[indices]-base_p[indices]
        patch_effect = patched_p[indices]-base_p[indices]
        full_effect = full_p[indices]-base_p[indices]
        oracle_effect = np.zeros_like(model_effect)
        for j, index in enumerate(indices):
            pair = pairs[index]
            for token, mass in pair['q_after'].items():
                oracle_effect[j, vocab[token]] += mass
            for token, mass in pair['q_before'].items():
                oracle_effect[j, vocab[token]] -= mass
        result[kind] = dict(n=len(indices),
            model_effect_l1=float(np.abs(model_effect).sum(-1).mean()),
            model_oracle_effect_l1=float(np.abs(model_effect-oracle_effect).sum(-1).mean()),
            full_bond_residual_l1=float(np.abs(full_effect-model_effect).sum(-1).mean()),
            patch_to_full_l1=float(np.abs(patch_effect-full_effect).sum(-1).mean()),
            patch_to_model_l1=float(np.abs(patch_effect-model_effect).sum(-1).mean()),
            patch_to_oracle_l1=float(np.abs(patch_effect-oracle_effect).sum(-1).mean()))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', required=True)
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--layer', type=int, default=2)
    p.add_argument('--nfit', type=int, default=256)
    p.add_argument('--ntest', type=int, default=256)
    p.add_argument('--length', type=int, default=71)
    p.add_argument('--holdout-objects', action='store_true',
                   help='Fit on first half of objects; test on disjoint second half')
    p.add_argument('--patch-scope', choices=('last', 'all'), default='last')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    torch.set_num_threads(4)
    saved = json.loads((Path(a.model_dir)/'results.json').read_text())
    record = next(row for row in saved['records'] if row['depth'] == a.depth)
    model = BilinearGPT(Config(**record['model_config']))
    model.load_state_dict(torch.load(Path(a.model_dir)/f'model_depth{a.depth}.pt',
                                     weights_only=True, map_location='cpu'))
    model.eval()
    bond = model.blocks[a.layer].attn.out
    fit_pairs = build_pairs(MultiGenreLanguage(seed=9801), a.nfit, a.length)
    test_pairs = build_pairs(MultiGenreLanguage(seed=9802), a.ntest, a.length)
    if a.holdout_objects:
        first = set(OBJECTS[:len(OBJECTS)//2])
        fit_pairs = [pair for pair in fit_pairs if pair['context_object'] in first]
        test_pairs = [pair for pair in test_pairs if pair['context_object'] not in first]
    assert fit_pairs and test_pairs
    fit_base = [pair['original'] for pair in fit_pairs]
    fit_edit = [pair['edited'] for pair in fit_pairs]
    test_base = [pair['original'] for pair in test_pairs]
    test_edit = [pair['edited'] for pair in test_pairs]
    full = a.patch_scope == 'all'
    _, fit_before = infer(model, bond, fit_base, capture_full=full)
    _, fit_after = infer(model, bond, fit_edit, capture_full=full)
    bases, spectra = fit_bases(fit_pairs, fit_before, fit_after, bond.weight)
    base_p, _ = infer(model, bond, test_base)
    donor_p, donor_acts = infer(model, bond, test_edit, capture_full=full)
    full_p, _ = infer(model, bond, test_base, donor_acts,
                      torch.eye(model.config.n_embd), model.config.n_embd,
                      capture_full=full)
    rows = []
    for name, basis in bases.items():
        assert torch.linalg.matrix_norm(basis.T@basis-torch.eye(basis.shape[1])) < 1e-4
        for width in (4, 8, 16, 32):
            patched_p, _ = infer(model, bond, test_base, donor_acts, basis, width,
                                 capture_full=full)
            rows.append(dict(basis=name, width=width,
                             groups=score(test_pairs, base_p, donor_p,
                                          patched_p, full_p,
                                          MultiGenreLanguage().token2id)))
    result = dict(config=vars(a), n_fit_pairs=len(fit_pairs),
                  n_test_pairs=len(test_pairs), contrast_spectra=spectra,
                  full_bond_groups=score(test_pairs, base_p, donor_p,
                                         full_p, full_p,
                                         MultiGenreLanguage().token2id),
                  rows=rows,
                  caveat='At one attention output bond, the selected position '
                         'scope is patched. Full-width mediation is a '
                         'local ceiling; other layers, positions, and residual '
                         'paths can carry the effect. Fit and test streams '
                         'have disjoint seeds but share task families.')
    Path(a.output).write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(Path(a.output).with_suffix('.npz'),
                        **{name: basis.cpu().numpy() for name, basis in bases.items()})
    for row in rows:
        if row['width'] == 16:
            print(row['basis'],
                  {kind: round(value['patch_to_full_l1'], 3)
                   for kind, value in row['groups'].items()}, flush=True)


if __name__ == '__main__':
    main()
