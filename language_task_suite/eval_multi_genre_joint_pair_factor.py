"""Low-rank first-layer token-score fits under actual token-pair frequencies.

The fit sees Reports/Policy only; FAQ/Guide remain whole-genre factor-fit
holdouts. All methods have the same rank per head/lag and replay the native
model suffix. The joint-pair method minimizes a smoothed empirical pair MSE
by alternating weighted least squares, initialized from weight-only SVD.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from eval_multi_genre_bond_gauges import group_scores
from eval_multi_genre_edits import build_pairs
from eval_multi_genre_token_tensor import forward_from_table, score_tables
from eval_multi_genre_token_tensor_truncation import (evaluate, fit_marginals,
                                                       truncated_scores)
from eval_surface_mixture_edits import predict, score
from multi_genre_language import GENRES, MultiGenreLanguage
from train_multi_genre import make_batch, make_eval


@torch.no_grad()
def fit_joint_pairs(nfit, length, seed, vocab, fit_genres, uniform_mass):
    weights = tuple(float(name in fit_genres) for name in GENRES)
    gen = MultiGenreLanguage(seed=seed, genre_weights=weights)
    x, _ = make_batch(gen, nfit, length, 'cpu')
    assert all(gen.id2token[int(token)] in fit_genres for token in x[:, 0])
    counts = []
    for lag in range(length):
        q, k = x[:, lag:], x[:, :length-lag]
        flat = (q*vocab+k).reshape(-1)
        count = torch.bincount(flat, minlength=vocab*vocab).reshape(vocab, vocab)
        counts.append(count)
    counts = torch.stack(counts)
    probability = ((1-uniform_mass)*counts.double()/counts.sum((-2, -1))[:, None, None]
                   +uniform_mass/(vocab*vocab))
    assert torch.allclose(probability.sum((-2, -1)), torch.ones(length, dtype=torch.float64))
    return probability.float(), dict(
        nfit=nfit, length=length, fit_genres=sorted(fit_genres),
        uniform_mass=uniform_mass, observed_pair_cells=int((counts>0).sum()),
        possible_pair_cells=int(counts.numel()),
        min_lag_samples=int(counts.sum((-2, -1)).min()),
        max_lag_samples=int(counts.sum((-2, -1)).max()))


def weighted_error(target, approximation, weights):
    return float((weights[None]*(target-approximation).square()).sum((-2, -1)).mean())


@torch.no_grad()
def joint_pair_als(tables, probability, rank, iterations=8, ridge=1e-5,
                   device='cuda'):
    target = tables.to(device=device, dtype=torch.float32)
    vocab = target.shape[-1]
    # Scale weights to mean one for numerically comparable ridge across lags.
    weights = probability.to(device=device)*vocab*vocab
    u0, s0, vh0 = torch.linalg.svd(target, full_matrices=False)
    root = s0[..., :rank].sqrt()
    left = u0[..., :rank]*root[..., None, :]
    right = vh0[..., :rank, :].transpose(-2, -1)*root[..., None, :]
    eye = torch.eye(rank, device=device, dtype=target.dtype)
    initial = left@right.transpose(-2, -1)
    history = [weighted_error(target, initial, weights)]
    for _ in range(iterations):
        gram = torch.einsum('lij,hljr,hljs->hlirs',
                            weights, right, right)
        rhs = torch.einsum('lij,hljr,hlij->hlir', weights, right, target)
        left = torch.linalg.solve(gram+ridge*eye, rhs.unsqueeze(-1)).squeeze(-1)
        gram = torch.einsum('lij,hlir,hlis->hljrs',
                            weights, left, left)
        rhs = torch.einsum('lij,hlir,hlij->hljr', weights, left, target)
        right = torch.linalg.solve(gram+ridge*eye, rhs.unsqueeze(-1)).squeeze(-1)
        history.append(weighted_error(target, left@right.transpose(-2, -1), weights))
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise RuntimeError('Nonfinite joint-pair factors')
    assert history[-1] <= history[0]*(1+1e-5), history
    return (left@right.transpose(-2, -1)).cpu(), history


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dirs', nargs='+', required=True)
    p.add_argument('--depth', type=int, default=3)
    p.add_argument('--length', type=int, default=71)
    p.add_argument('--nfit', type=int, default=512)
    p.add_argument('--ntest', type=int, default=1024)
    p.add_argument('--nedit-streams', type=int, default=512)
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--iterations', type=int, default=8)
    p.add_argument('--uniform-mass', type=float, default=.1)
    p.add_argument('--fit-genres', default='Reports,Policy')
    p.add_argument('--fit-seed', type=int, default=10551)
    p.add_argument('--test-seed', type=int, default=10772)
    p.add_argument('--edit-seed', type=int, default=10773)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert 0 < a.uniform_mass < 1 and 0 < a.rank
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config
    torch.set_num_threads(4)
    fit_genres = set(a.fit_genres.split(','))
    assert fit_genres and fit_genres.issubset(GENRES)
    gen = MultiGenreLanguage(seed=a.edit_seed)
    probability, fit_info = fit_joint_pairs(a.nfit, a.length, a.fit_seed,
                                              len(gen.vocab), fit_genres,
                                              a.uniform_mass)
    left, right = fit_marginals(a.nfit, a.length, a.fit_seed,
                                len(gen.vocab), fit_genres)
    test = make_eval(MultiGenreLanguage(seed=a.test_seed), a.ntest,
                     a.length, 'cpu')
    pairs = [pair for pair in build_pairs(gen, a.nedit_streams, a.length)
             if pair['kind'] in ('faq_answer_copy', 'faq_place_recall',
                                 'policy_condition', 'policy_nuisance_person')]
    assert any(pair['kind'] == 'faq_place_recall' for pair in pairs)
    results = {}
    for directory in a.model_dirs:
        folder = Path(directory)
        saved = json.loads((folder/'results.json').read_text())
        record = next(row for row in saved['records'] if row['depth'] == a.depth)
        model = BilinearGPT(Config(**record['model_config']))
        model.load_state_dict(torch.load(folder/f'model_depth{a.depth}.pt',
                                         map_location='cpu', weights_only=True))
        model.eval()
        with torch.no_grad():
            native = torch.cat([model(batch)[0].float()
                                for batch in test['x'].split(32)])
            reference = dict(logp=F.log_softmax(native, -1))
            native_edits = score(pairs,
                predict(model, [pair['original'] for pair in pairs], len(gen.vocab)),
                predict(model, [pair['edited'] for pair in pairs], len(gen.vocab)), gen)
        tables, downstream = score_tables(model, a.length)
        assert a.rank < tables.shape[-1]
        weight = truncated_scores(tables, a.rank)
        marginal = truncated_scores(tables, a.rank, left, right)
        joint, history = joint_pair_als(tables, probability, a.rank,
                                        a.iterations, device=a.device)
        methods = {'full_table': tables, 'weight_svd': weight,
                   'fit_marginal_svd': marginal, 'fit_joint_pair_als': joint}
        rows = []
        for name, compressed in methods.items():
            metrics = evaluate(model, compressed, downstream, test,
                               reference, pairs, gen)
            rows.append(dict(method=name,
                             joint_weighted_score_mse=weighted_error(
                                 tables, compressed, probability), **metrics))
            print(folder.name, name,
                  'FAQ model KL', round(metrics['groups']['faq_place_recall'][
                      'model_to_truncated_kl'], 5),
                  'FAQ edit L1', round(metrics['edits']['faq_place_recall'][
                      'effect_l1_error'], 4), flush=True)
        results[folder.name] = dict(model_config=record['model_config'],
                                    native_edits=native_edits,
                                    joint_fit_objective_history=history,
                                    rows=rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(
        config={**vars(a), 'output': str(a.output)}, fit=fit_info,
        test=dict(seed=a.test_seed, ntest=a.ntest,
                  heldout_from_tensor_fit=sorted(set(GENRES)-fit_genres)),
        edits=dict(seed=a.edit_seed, n_streams=a.nedit_streams,
                   counts=dict(Counter(pair['kind'] for pair in pairs))),
        results=results,
        caveat='Joint pair weights are empirical fit-genre frequencies with a '
               'uniform floor. ALS solves a nonconvex weighted rank fit; the '
               'initialization and objective history are recorded. Rank is per '
               'head and lag, not a circuit size. The model saw all genres '
               'during training; FAQ/Guide are held out only from factor fit.'),
        indent=2)+'\n')


if __name__ == '__main__':
    main()
