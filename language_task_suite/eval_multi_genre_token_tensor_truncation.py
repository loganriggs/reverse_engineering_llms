"""Weight-only versus task-blind data-aware SVD of layer-0 token scores.

Fit data-aware token marginals on Reports/Policy only; evaluate FAQ/Guide as
whole-genre tensor-fit holdouts. Both decompositions approximate each head's
token-pair score matrix at each relative lag, then run the native model suffix.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval_multi_genre_bond_gauges import group_scores
from eval_multi_genre_edits import build_pairs
from eval_multi_genre_token_tensor import forward_from_table, score_tables
from eval_surface_mixture_edits import predict, score
from multi_genre_language import GENRES, MultiGenreLanguage
from train_multi_genre import make_batch, make_eval


@torch.no_grad()
def fit_marginals(nfit, length, seed, vocab_size, fit_genres, smoothing=1.):
    weights = tuple(float(name in fit_genres) for name in GENRES)
    gen = MultiGenreLanguage(seed=seed, genre_weights=weights)
    x, _ = make_batch(gen, nfit, length, 'cpu')
    assert all(gen.id2token[int(token)] in fit_genres for token in x[:, 0])
    left, right = [], []
    for lag in range(length):
        q = x[:, lag:].reshape(-1)
        k = x[:, :length-lag].reshape(-1)
        q_count = torch.bincount(q, minlength=vocab_size).double()+smoothing
        k_count = torch.bincount(k, minlength=vocab_size).double()+smoothing
        left.append((q_count/q_count.sum()).sqrt())
        right.append((k_count/k_count.sum()).sqrt())
    return torch.stack(left), torch.stack(right)


@torch.no_grad()
def truncated_scores(tables, rank, left=None, right=None):
    h, length, vocab, _ = tables.shape
    if left is None:
        matrix = tables.double()
    else:
        matrix = tables.double()*left[None, :, :, None]*right[None, :, None, :]
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    approx = (u[..., :rank]*s[..., None, :rank])@vh[..., :rank, :]
    if left is not None:
        approx = approx/left[None, :, :, None]/right[None, :, None, :]
    return approx.float()


@torch.no_grad()
def predict_table(model, prefixes, tables, downstream, vocab_size):
    output = np.zeros((len(prefixes), vocab_size), dtype=np.float32)
    groups = defaultdict(list)
    for i, prefix in enumerate(prefixes):
        groups[len(prefix)].append(i)
    for indices in groups.values():
        for start in range(0, len(indices), 64):
            batch_ids = indices[start:start+64]
            x = torch.tensor([prefixes[i] for i in batch_ids], dtype=torch.long)
            p = F.softmax(forward_from_table(model, x, tables, downstream)[:, -1].float(), -1)
            output[batch_ids] = p.numpy()
    return output


@torch.no_grad()
def evaluate(model, tables, downstream, test, reference, pairs, gen):
    logits = torch.cat([forward_from_table(model, batch, tables, downstream).float()
                        for batch in test['x'].split(32)])
    groups = group_scores(logits, test, reference['logp'])
    originals = [pair['original'] for pair in pairs]
    edited = [pair['edited'] for pair in pairs]
    before = predict_table(model, originals, tables, downstream, len(gen.vocab))
    after = predict_table(model, edited, tables, downstream, len(gen.vocab))
    return dict(groups=groups, edits=score(pairs, before, after, gen))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dirs', nargs='+', required=True)
    p.add_argument('--depth', type=int, default=3)
    p.add_argument('--length', type=int, default=71)
    p.add_argument('--nfit', type=int, default=512)
    p.add_argument('--ntest', type=int, default=1024)
    p.add_argument('--nedit-streams', type=int, default=512)
    p.add_argument('--ranks', default='8,16,32')
    p.add_argument('--fit-genres', default='Reports,Policy')
    p.add_argument('--fit-seed', type=int, default=9551)
    p.add_argument('--test-seed', type=int, default=9552)
    p.add_argument('--edit-seed', type=int, default=9553)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config
    torch.set_num_threads(4)
    ranks = [int(x) for x in a.ranks.split(',')]
    gen = MultiGenreLanguage(seed=a.edit_seed)
    pairs = [pair for pair in build_pairs(gen, a.nedit_streams, a.length)
             if pair['kind'] in ('faq_answer_copy', 'faq_place_recall',
                                 'policy_condition', 'policy_nuisance_person')]
    assert any(pair['kind'] == 'faq_place_recall' for pair in pairs)
    test = make_eval(MultiGenreLanguage(seed=a.test_seed), a.ntest, a.length, 'cpu')
    fit_genres = set(a.fit_genres.split(','))
    assert fit_genres and fit_genres.issubset(GENRES)
    left, right = fit_marginals(a.nfit, a.length, a.fit_seed,
                                len(gen.vocab), fit_genres)
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
        native_before = predict(model, [pair['original'] for pair in pairs],
                                len(gen.vocab))
        native_after = predict(model, [pair['edited'] for pair in pairs],
                               len(gen.vocab))
        native_edits = score(pairs, native_before, native_after, gen)
        tables, downstream = score_tables(model, a.length)
        rows = []
        for name in ('full_table', 'weight_svd', 'fit_marginal_svd'):
            for rank in ([len(gen.vocab)] if name == 'full_table' else ranks):
                compressed = (tables if name == 'full_table' else
                              truncated_scores(tables, rank,
                                  left if name == 'fit_marginal_svd' else None,
                                  right if name == 'fit_marginal_svd' else None))
                metrics = evaluate(model, compressed, downstream, test,
                                   reference, pairs, gen)
                rows.append(dict(method=name, rank=rank, **metrics))
                faq = metrics['groups']['faq_place_recall']
                edit = metrics['edits']['faq_place_recall']
                print(folder.name, name, rank, 'FAQ model KL',
                      faq['model_to_truncated_kl'], 'edit L1',
                      edit['effect_l1_error'], flush=True)
        results[folder.name] = dict(model_config=record['model_config'],
                                    native_edits=native_edits, rows=rows)
    output = dict(config={**vars(a), 'output': str(a.output)},
                  fit=dict(genres=sorted(fit_genres), seed=a.fit_seed,
                           nfit=a.nfit, smoothing=1.,
                           type='product of lag-conditional token marginals'),
                  test=dict(seed=a.test_seed, ntest=a.ntest,
                            heldout_from_tensor_fit=sorted(set(GENRES)-fit_genres)),
                  edits=dict(seed=a.edit_seed, n_streams=a.nedit_streams,
                             counts=dict(Counter(pair['kind'] for pair in pairs))),
                  results=results,
                  caveat='Relative-lag table approximates bfloat16 RoPE; '
                         'full-table baseline quantifies its error. Rank is '
                         'per head and lag, not a global circuit count. '
                         'Data-aware fit uses only listed fit genres; the '
                         'native model was trained on all genres.')
    a.output.write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
