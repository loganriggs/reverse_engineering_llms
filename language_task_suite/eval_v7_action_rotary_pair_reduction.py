"""Reduce the causally localized V7 head's QK paths by rotary pairs.

Compare a weight-only pair ranking, task-blind other-kind activation energy,
and a supervised action-edit ranking. Selection is on disjoint data from
held-out narrative evaluation. These are physical RoPE pair subsets, not a
canonical basis for the full attention tensor.
"""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import random

import torch

from document_mixture_v7_cross import sample_document
from eval_document_mixture_v7_action_causal import collect, score
from eval_v7_action_attention_edge_patch import load_model
from train_document_mixture_v5 import encode, evaluate


def pair_terms(layer, x, branch, head):
    h = layer.norm(x)
    b, t, _ = h.shape
    def project(name):
        z = getattr(layer, name)(h).reshape(b, t, layer.n_head, layer.d_head)
        return layer.rotary(z)[:, :, head, :]
    q, k = project('q'+branch), project('k'+branch)
    half = layer.d_head//2
    return q[:, :, :half], q[:, :, half:], k[:, :, :half], k[:, :, half:]


def pair_scores(terms, selected):
    q0, q1, k0, k1 = terms
    if not selected:
        return q0.new_zeros((len(q0), q0.shape[1], k0.shape[1]))
    idx = list(selected)
    return (torch.einsum('bqp,bkp->bqk', q0[:, :, idx], k0[:, :, idx])+
            torch.einsum('bqp,bkp->bqk', q1[:, :, idx], k1[:, :, idx]))


@contextmanager
def masked_qk(layer, selections, head):
    original = layer.pattern
    def replacement(x):
        all_heads = original(x).clone()
        one = pair_scores(pair_terms(layer, x, '1', head), selections['1'])
        two = pair_scores(pair_terms(layer, x, '2', head), selections['2'])
        length = x.shape[1]
        all_heads[:, head] = (one*two/layer.d_head**2)*layer.mask[:length, :length]
        return all_heads
    layer.pattern = replacement
    try:
        yield
    finally:
        layer.pattern = original


def weight_ranking(layer, head):
    half = layer.d_head//2
    offset = head*layer.d_head
    rankings = {}
    for branch in ('1', '2'):
        q = getattr(layer, 'q'+branch).weight.detach()
        k = getattr(layer, 'k'+branch).weight.detach()
        scores = []
        for pair in range(half):
            rows = [offset+pair, offset+half+pair]
            effective = q[rows].T @ k[rows]
            scores.append(effective.norm().item())
        rankings[branch] = scores
    return rankings


@torch.no_grad()
def data_ranking(model, layer_index, docs, vocab, n_ctx, device,
                 seed, batch_size, head):
    layer = model.layers[layer_index]
    rng = random.Random(seed)
    scores = {branch: torch.zeros(layer.d_head//2, device=device)
              for branch in ('1', '2')}
    count = 0
    for start in range(0, len(docs), batch_size):
        part = docs[start:start+batch_size]
        x, _ = encode(part, vocab, n_ctx)
        stream = model.embed(x.to(device))
        for earlier in model.layers[:layer_index]:
            stream = earlier(stream)
        terms = {branch: pair_terms(layer, stream, branch, head)
                 for branch in ('1', '2')}
        for i, doc in enumerate(part):
            length = len(doc.tokens)-1
            for _ in range(8):
                query = rng.randrange(1, length)
                key = rng.randrange(query+1)
                for branch, (q0, q1, k0, k1) in terms.items():
                    contribution = (q0[i, query]*k0[i, key] +
                                    q1[i, query]*k1[i, key])
                    scores[branch] += contribution.square()
                count += 1
    return {branch: (values/count).cpu().tolist()
            for branch, values in scores.items()}, count


@torch.no_grad()
def task_ranking(model, layer_index, cases, vocab, n_ctx, device, batch_size,
                 head):
    layer = model.layers[layer_index]
    half = layer.d_head//2
    native = score(model, cases, vocab, n_ctx, device, batch_size)[
        'mean_model_response']
    rankings = {}
    for branch in ('1', '2'):
        values = []
        for pair in range(half):
            selections = {'1': tuple(range(half)),
                          '2': tuple(range(half))}
            selections[branch] = tuple(i for i in range(half) if i != pair)
            with masked_qk(layer, selections, head):
                measured = score(model, cases, vocab, n_ctx, device,
                                 batch_size)['mean_model_response']
            values.append(native-measured)
        rankings[branch] = values
    return rankings


def select(scores, rank):
    return {branch: tuple(sorted(range(len(values)),
                                 key=lambda i: -values[i])[:rank])
            for branch, values in scores.items()}


def run(a):
    torch.set_num_threads(4)
    model, vocab, n_ctx = load_model(a.checkpoint, a.device)
    layer_index = 2
    layer = model.layers[layer_index]
    assert layer.n_head == 4 and layer.d_head == 16
    assert 0 <= a.head < layer.n_head
    rng = random.Random(a.fit_seed)
    fit = []
    while len(fit) < a.fit_docs:
        doc = sample_document(rng)
        if doc.kind != 'narrative' and len(doc.tokens)-1 <= n_ctx:
            fit.append(doc)
    rng = random.Random(a.test_seed)
    test = []
    while len(test) < a.test_docs:
        doc = sample_document(rng)
        if doc.kind == 'narrative' and len(doc.tokens)-1 <= n_ctx:
            test.append(doc)
    development, _ = collect(a.fit_seed+1000, 200, 192, a.dev_cases)
    cases, available = collect(a.test_seed+1000, 400, 192, a.test_cases)
    assert len(development) == a.dev_cases and len(cases) == a.test_cases
    native = evaluate(model, test, encode(test, vocab, n_ctx), vocab, a.device,
                      a.batch_size)
    native_edit = score(model, cases, vocab, n_ctx, a.device, a.batch_size)
    weight = weight_ranking(layer, a.head)
    data, n_pairs = data_ranking(model, layer_index, fit, vocab, n_ctx,
                                 a.device, a.fit_seed+17, a.batch_size,
                                 a.head)
    task = task_ranking(model, layer_index, development, vocab, n_ctx,
                        a.device, a.batch_size, a.head)
    control_rng = random.Random(a.fit_seed+812)
    random_ranking = {branch: [control_rng.random() for _ in values]
                      for branch, values in weight.items()}
    rankings = {'weight_only': weight, 'unlabeled_other_kinds': data,
                'supervised_action_development': task,
                'random_control': random_ranking,
                'low_weight_control': {branch: [-v for v in values]
                                       for branch, values in weight.items()}}
    ranks = (1, 2, 4, 6, 8)
    rows = []
    for name, ranking in rankings.items():
        for rank in ranks:
            chosen = select(ranking, rank)
            with masked_qk(layer, chosen, a.head):
                lm = evaluate(model, test, encode(test, vocab, n_ctx),
                              vocab, a.device, a.batch_size)
                edit = score(model, cases, vocab, n_ctx, a.device,
                             a.batch_size)
            if rank == 8:
                assert abs(lm['token_nll']-native['token_nll']) < 1e-5
                assert abs(edit['mean_model_response']-
                           native_edit['mean_model_response']) < 1e-5
            rows.append(dict(method=name, pairs_per_branch=rank,
                             selected_pairs={k:list(v) for k,v in chosen.items()},
                             token_nll=lm['token_nll'],
                             action_accuracy=lm['by_site'][
                                 'conditional_action'][
                                     'deterministic_full_vocab_accuracy'],
                             edit_response=edit['mean_model_response']))
            print(name, rank, round(lm['token_nll'],3),
                  round(edit['mean_model_response'],3), flush=True)
    return dict(config={**vars(a), 'checkpoint':str(a.checkpoint),
                        'output':str(a.output)},
                fit=dict(non_narrative_documents=len(fit),
                         sampled_causal_token_pairs=n_pairs,
                         supervised_development_edits=len(development)),
                test=dict(narrative_documents=len(test),
                          supported_action_edits=len(cases),
                          available_action_edits=available),
                native=dict(token_nll=native['token_nll'],
                            action_accuracy=native['by_site'][
                                'conditional_action'][
                                    'deterministic_full_vocab_accuracy'],
                            edit_response=native_edit['mean_model_response']),
                rankings=rankings, rows=rows,
                caveat='Only physical rotary-pair coordinates of one layer-2 '
                       'attention head QK branches are removed. Data ranking fits '
                       'other V7 kinds, not narrative; supervised ranking '
                       'uses disjoint action edits. This is not a unique '
                       'canonical tensor decomposition or model-seed replication.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--head', type=int, default=3)
    p.add_argument('--fit-seed', type=int, default=35151)
    p.add_argument('--test-seed', type=int, default=35152)
    p.add_argument('--fit-docs', type=int, default=300)
    p.add_argument('--test-docs', type=int, default=200)
    p.add_argument('--dev-cases', type=int, default=32)
    p.add_argument('--test-cases', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = run(a)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
