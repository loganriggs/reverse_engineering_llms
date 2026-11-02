"""Test whether a model matches exact DGP changes under valid prefix edits."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from surface_mixture_language import (ADJECTIVES, LEADS, PEOPLE, PLACES,
                                      VERBS, SurfaceMixtureLanguage,
                                      _history_and_clause)
from eval_surface_bond_gauges import bases_for
from train_surface_mixture import make_batch


def id_prefix(gen, ids):
    return [gen.id2token[i] for i in ids]


def replacement(gen, prefix, index, allowed, distinct=None):
    q = gen.next_distribution(prefix[:index])
    candidates = [tok for tok in allowed if tok != prefix[index] and q.get(tok, 0) > 0]
    if distinct is not None:
        candidates = [tok for tok in candidates if distinct(tok) != distinct(prefix[index])]
    return max(candidates, key=lambda tok: q[tok]) if candidates else None


def edited_pair(gen, prefix, kind):
    history, clause = _history_and_clause(prefix)
    assert len(clause) == 3
    edited = list(prefix)
    start = sum(len(c) for c in history)
    if kind == 'lead':
        old = gen.candidates(prefix)['lead']
        choices = [lead for lead in LEADS
                   if lead != clause[0] and gen.candidates(
                       prefix[:start]+[lead]+prefix[start+1:])['lead'] != old]
        if not choices:
            return None
        edited[start] = choices[0]
    elif kind == 'induction':
        subject = clause[2]
        prior = [(i, c) for i, c in enumerate(history) if c[2] == subject]
        if not prior:
            return None
        i, earlier = prior[-1]
        index = sum(len(c) for c in history[:i])+3
        new = replacement(gen, prefix, index, VERBS)
        if new is None:
            return None
        edited[index] = new
    elif kind == 'place_skip':
        if not history:
            return None
        index = start-len(history[-1])+9
        old_place = prefix[index]
        old_target = VERBS[(PLACES.index(old_place)*7+2)%len(VERBS)]
        new = replacement(gen, prefix, index, PLACES,
                          distinct=lambda place: VERBS[(PLACES.index(place)*7+2)%len(VERBS)])
        if new is None:
            return None
        edited[index] = new
        assert VERBS[(PLACES.index(new)*7+2)%len(VERBS)] != old_target
    elif kind == 'nuisance_adjective':
        if not history:
            return None
        index = start-len(history[-1])+5
        new = replacement(gen, prefix, index, ADJECTIVES)
        if new is None:
            return None
        edited[index] = new
    else:
        raise ValueError(kind)
    q_before = gen.next_distribution(prefix)
    q_after = gen.next_distribution(edited)
    if kind == 'nuisance_adjective':
        assert q_before == q_after
    else:
        assert q_before != q_after, (kind, prefix, edited)
    return edited, q_before, q_after


def build_pairs(gen, n, length):
    rows, labels, _ = gen.sample_batch(n, length+1)
    pairs = []
    for row, labs in zip(rows, labels):
        for t in range(1, length+1):
            if labs[t] != 'mixed_verb':
                continue
            prefix = id_prefix(gen, row[:t])
            for kind in ('lead', 'induction', 'place_skip', 'nuisance_adjective'):
                result = edited_pair(gen, prefix, kind)
                if result is None:
                    continue
                edited, q_before, q_after = result
                pairs.append(dict(kind=kind,
                                  original=[gen.token2id[tok] for tok in prefix],
                                  edited=[gen.token2id[tok] for tok in edited],
                                  q_before=q_before, q_after=q_after))
    return pairs


@torch.no_grad()
def predict(model, prefixes, vocab_size):
    output = np.zeros((len(prefixes), vocab_size), dtype=np.float32)
    groups = defaultdict(list)
    for i, prefix in enumerate(prefixes):
        groups[len(prefix)].append(i)
    for _, indices in groups.items():
        for start in range(0, len(indices), 64):
            batch_ids = indices[start:start+64]
            x = torch.tensor([prefixes[i] for i in batch_ids], dtype=torch.long)
            p = F.softmax(model(x)[0][:, -1].float(), -1).cpu().numpy()
            output[batch_ids] = p
    return output


def score(pairs, before, after, gen):
    grouped = defaultdict(list)
    for i, pair in enumerate(pairs):
        q0 = np.zeros(len(gen.vocab))
        q1 = np.zeros(len(gen.vocab))
        for token, p in pair['q_before'].items():
            q0[gen.token2id[token]] = p
        for token, p in pair['q_after'].items():
            q1[gen.token2id[token]] = p
        target = q1-q0
        predicted = after[i]-before[i]
        target_tv = .5*np.abs(target).sum()
        predicted_tv = .5*np.abs(predicted).sum()
        l1_error = np.abs(target-predicted).sum()
        cosine = (float(np.dot(target, predicted)/(
                    np.linalg.norm(target)*np.linalg.norm(predicted)))
                  if target_tv > 1e-10 and predicted_tv > 1e-10 else 0.0)
        grouped[pair['kind']].append((target_tv, predicted_tv, l1_error, cosine))
    return {name: dict(n=len(rows), oracle_tv=float(np.mean([r[0] for r in rows])),
                       model_tv=float(np.mean([r[1] for r in rows])),
                       effect_l1_error=float(np.mean([r[2] for r in rows])),
                       effect_cosine=float(np.mean([r[3] for r in rows])))
            for name, rows in grouped.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', default='language_task_suite/surface_mixture_models_v1')
    p.add_argument('--n', type=int, default=256)
    p.add_argument('--length', type=int, default=70)
    p.add_argument('--seed', type=int, default=8701)
    p.add_argument('--truncate-layer', type=int, default=None)
    p.add_argument('--basis', choices=('physical', 'random', 'activation_pca',
                                       'weight_svd'), default='weight_svd')
    p.add_argument('--width', type=int, default=32)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    torch.set_num_threads(4)
    gen = SurfaceMixtureLanguage(seed=a.seed)
    pairs = build_pairs(gen, a.n, a.length)
    saved = json.loads((Path(a.model_dir)/'results.json').read_text())
    results = {}
    originals = [pair['original'] for pair in pairs]
    edits = [pair['edited'] for pair in pairs]
    for record in saved['records']:
        depth = record['depth']
        model = BilinearGPT(Config(**record['model_config']))
        model.load_state_dict(torch.load(Path(a.model_dir)/f'model_depth{depth}.pt',
                                         weights_only=True, map_location='cpu'))
        model.eval()
        hook = None
        if a.truncate_layer is not None:
            assert 0 <= a.truncate_layer < depth
            bond = model.blocks[a.truncate_layer].attn.out
            captured = []
            def collect(_, inputs):
                captured.append(inputs[0].detach())
            hook = bond.register_forward_pre_hook(collect)
            xfit, _ = make_batch(SurfaceMixtureLanguage(seed=8501), 256,
                                 a.length, 'cpu')
            with torch.no_grad():
                for batch in xfit.split(32):
                    model(batch)
            hook.remove()
            basis = bases_for(torch.cat(captured), bond.weight.detach())[a.basis]
            uk = basis[:, :a.width]
            def truncate(_, inputs):
                x = inputs[0]
                return ((x@uk)@uk.T,)
            hook = bond.register_forward_pre_hook(truncate)
        before = predict(model, originals, len(gen.vocab))
        after = predict(model, edits, len(gen.vocab))
        if hook is not None:
            hook.remove()
        results[str(depth)] = score(pairs, before, after, gen)
        print('depth', depth, {k: round(v['effect_l1_error'], 3)
                               for k, v in results[str(depth)].items()}, flush=True)
    output = dict(config=vars(a), n_pairs=len(pairs), results=results,
                  caveat='Edits are grammar-valid and oracle-scored but may change '
                         'the probability of the edited prefix. Effects are on '
                         'the same visible current prefix apart from one token.')
    Path(a.output).write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
