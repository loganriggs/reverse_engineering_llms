"""Task-blind sparse bond gauges with whole-genre featurizer holdouts.

Fit activation-based coordinates on Reports and Policy only. Test FAQ and
Guide, which the language model saw during training but the basis fit did not.
Every full-width orthogonal basis is an exact local gauge; top-|coordinate|
gating is a lossy intervention. Report model fidelity and oracle fidelity
separately, including supported FAQ causal edits.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval_multi_genre_bond_gauges import group_scores
from eval_multi_genre_edits import build_pairs
from eval_surface_mixture_edits import predict, score
from multi_genre_language import GENRES, MultiGenreLanguage
from train_multi_genre import make_batch, make_eval


def varimax(matrix, max_iter=200, tolerance=1e-9):
    n, k = matrix.shape
    rotation = np.eye(k)
    previous = -np.inf
    for _ in range(max_iter):
        z = matrix @ rotation
        gradient = matrix.T @ (z**3-z @ np.diag(np.sum(z*z, axis=0)/n))
        u, singular, vh = np.linalg.svd(gradient, full_matrices=False)
        rotation = u@vh
        value = float(singular.sum())
        if value-previous < max(tolerance, abs(value)*tolerance):
            break
        previous = value
    return rotation


@torch.no_grad()
def fit_bases(model, bond, nfit, length, seed, fit_genres):
    assert set(GENRES) == {'Reports', 'Guide', 'FAQ', 'Policy'}
    assert fit_genres and fit_genres.issubset(GENRES)
    weights = tuple(float(name in fit_genres) for name in GENRES)
    xfit, _ = make_batch(MultiGenreLanguage(seed=seed,
                                            genre_weights=weights), nfit,
                         length, 'cpu')
    captured = []
    def collect(_, inputs):
        captured.append(inputs[0].detach().cpu())
    hook = bond.register_forward_pre_hook(collect)
    try:
        for batch in xfit.split(32):
            model(batch)
    finally:
        hook.remove()
    token_names = MultiGenreLanguage().id2token
    assert all(token_names[int(token)] in fit_genres for token in xfit[:, 0])
    z = torch.cat(captured).reshape(-1, model.config.n_embd).double().numpy()
    weight = bond.weight.detach().cpu().double().numpy()
    _, _, activation_vh = np.linalg.svd(z, full_matrices=False)
    _, _, weight_vh = np.linalg.svd(weight, full_matrices=False)
    activation = activation_vh.T
    weight_basis = weight_vh.T
    rng = np.random.default_rng(seed+700)
    bases = dict(physical=np.eye(z.shape[1]),
                 activation_pca=activation,
                 activation_varimax=activation@varimax(z@activation),
                 weight_svd=weight_basis,
                 weight_varimax=weight_basis@varimax(weight@weight_basis))
    for index in range(3):
        q, r = np.linalg.qr(rng.normal(size=(z.shape[1], z.shape[1])))
        bases[f'random_{index}'] = q*np.sign(np.diag(r))[None, :]
    for name, basis in bases.items():
        assert np.max(np.abs(basis.T@basis-np.eye(len(basis)))) < 1e-8, name
    return {name: torch.tensor(basis, dtype=torch.float32)
            for name, basis in bases.items()}, dict(
                nfit=nfit, length=length, n_bond_vectors=len(z),
                fit_genres=sorted(fit_genres),
                heldout_genres=sorted(set(GENRES)-fit_genres))


def gate_hook(basis, keep, stats=None):
    """Return a pre-output hook implementing one sparse orthogonal gauge."""
    def gate(_, inputs):
        z = inputs[0]
        coordinates = z@basis
        if keep < basis.shape[1]:
            selected = coordinates.abs().topk(keep, dim=-1).indices
            mask = torch.zeros_like(coordinates).scatter_(-1, selected, 1.)
            coordinates = coordinates*mask
        reconstructed = coordinates@basis.T
        if stats is not None:
            stats['error'] += float((z-reconstructed).square().sum().detach())
            stats['energy'] += float(z.square().sum().detach())
        return (reconstructed,)
    return gate


@torch.no_grad()
def evaluate_distribution(model, bond, basis, keep, test, reference_logp):
    stats = dict(error=0., energy=0.)
    hook = bond.register_forward_pre_hook(gate_hook(basis, keep, stats))
    try:
        logits = torch.cat([model(batch)[0].float()
                            for batch in test['x'].split(32)])
    finally:
        hook.remove()
    if keep == basis.shape[1]:
        assert (logits-reference_logp['logits']).abs().max() < 3e-4
    return dict(groups=group_scores(logits, test, reference_logp['logp']),
                relative_bond_mse=stats['error']/max(stats['energy'], 1e-30))


@torch.no_grad()
def evaluate_edits(model, bond, basis, keep, pairs, vocab_size, gen):
    originals = [pair['original'] for pair in pairs]
    edited = [pair['edited'] for pair in pairs]
    hook = bond.register_forward_pre_hook(gate_hook(basis, keep))
    try:
        before = predict(model, originals, vocab_size)
        after = predict(model, edited, vocab_size)
    finally:
        hook.remove()
    return score(pairs, before, after, gen)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--nfit', type=int, default=256)
    parser.add_argument('--ntest', type=int, default=512)
    parser.add_argument('--nedit-streams', type=int, default=256)
    parser.add_argument('--length', type=int, default=71)
    parser.add_argument('--fit-genres', default='Reports,Policy')
    parser.add_argument('--keeps', default='8,16,32,64')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.source)
    from model import BilinearGPT, Config

    torch.set_num_threads(4)
    saved = json.loads((Path(args.model_dir)/'results.json').read_text())
    record = next(row for row in saved['records'] if row['depth'] == args.depth)
    model = BilinearGPT(Config(**record['model_config']))
    model.load_state_dict(torch.load(Path(args.model_dir)/f'model_depth{args.depth}.pt',
                                     map_location='cpu', weights_only=True))
    model.eval()
    assert 0 <= args.layer < args.depth
    bond = model.blocks[args.layer].attn.out
    fit_genres = set(args.fit_genres.split(','))
    bases, fit = fit_bases(model, bond, args.nfit, args.length, 9551,
                           fit_genres)
    test = make_eval(MultiGenreLanguage(seed=9552), args.ntest,
                     args.length, 'cpu')
    with torch.no_grad():
        reference_logits = torch.cat([model(batch)[0].float()
                                      for batch in test['x'].split(32)])
    reference = dict(logits=reference_logits,
                     logp=F.log_softmax(reference_logits, -1))
    gen = MultiGenreLanguage(seed=9553)
    all_pairs = build_pairs(gen, args.nedit_streams, args.length)
    # FAQ is a whole-genre featurizer holdout. Keep policy-condition as a
    # within-fit task comparator and omit sparse, noisy induction pairs.
    kinds = {'faq_answer_copy', 'faq_place_recall',
             'policy_condition', 'policy_nuisance_person'}
    pairs = [pair for pair in all_pairs if pair['kind'] in kinds]
    assert any(pair['kind'] == 'faq_place_recall' for pair in pairs)
    keeps = sorted(set(int(x) for x in args.keeps.split(',')))
    assert keeps[-1] == model.config.n_embd
    rows = []
    for name, basis in bases.items():
        for keep in keeps:
            distribution = evaluate_distribution(model, bond, basis, keep,
                                                 test, reference)
            edits = evaluate_edits(model, bond, basis, keep, pairs,
                                   len(gen.vocab), gen)
            rows.append(dict(basis=name, keep=keep,
                             distribution=distribution, edits=edits))
            print(name, keep,
                  'FAQ recall model KL', round(distribution['groups'][
                      'faq_place_recall']['model_to_truncated_kl'], 5),
                  'FAQ recall effect L1', round(edits[
                      'faq_place_recall']['effect_l1_error'], 4), flush=True)
    result = dict(config={**vars(args), 'output': str(args.output)},
                  fit=fit, model_config=record['model_config'],
                  n_edit_pairs=len(pairs),
                  edit_pair_counts=dict(Counter(pair['kind'] for pair in pairs)),
                  rows=rows,
                  caveat='Orthogonal full-width gauges are exact; sparse top-absolute '
                         'coordinate gates are interventions. The fit-genres '
                         'field specifies which genres the basis saw; all '
                         'genres were seen during model training. '
                         'Task-wise oracle KL, model-to-gated KL, bond MSE, and '
                         'causal effects are reported separately. The final '
                         'bond is local and does not uniquely specify a full circuit.')
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
