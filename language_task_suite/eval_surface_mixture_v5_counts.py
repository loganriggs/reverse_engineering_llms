"""Evaluate subject-count prediction and supported causal edits in v5."""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from eval_surface_mixture_edits import predict, replacement, score
from eval_surface_mixture_v4_edits import full_support
from surface_mixture_v2 import PEOPLE, PLACES, _frame, _indices, _split
from surface_mixture_v4 import _content
from surface_mixture_v5 import SurfaceMixtureV5


def history_slots(history):
    out = []
    offset = 0
    for raw in history:
        clause = _content(raw)
        if clause and clause[0] not in ('Where', 'How'):
            frame = _frame(clause)
            content_start = offset+len(raw)-len(clause)
            out.append(dict(subject=clause[_indices(frame)[0]],
                            subject_index=content_start+_indices(frame)[0],
                            place_index=content_start+_indices(frame)[2]))
        offset += len(raw)
    return out


def build(gen, n, length):
    ids, labels, _ = gen.sample_batch(n, length+1)
    questions = []
    pairs = []
    attempted = Counter()
    for row, labs in zip(ids, labels):
        tokens = [gen.id2token[i] for i in row]
        for t in range(1, length+1):
            if labs[t] != 'count_answer':
                continue
            prefix = tokens[:t]
            history, raw = _split(prefix)
            content = _content(raw)
            name = content[4]
            slots = history_slots(history)
            counts = Counter(slot['subject'] for slot in slots)
            target = str(counts[name])
            assert gen.next_distribution(prefix) == {target: 1.0}
            multiplicities = Counter(counts.values())
            prior = [mass/len(counts) for mass in multiplicities.values()]
            questions.append(dict(prefix=row[:t], target=gen.token2id[target],
                                  count=counts[name],
                                  key_blind_bayes=max(prior),
                                  key_blind_entropy=-sum(p*math.log(p) for p in prior)))
            name_index = t-7
            for other_name in counts:
                if counts[other_name] == counts[name]:
                    continue
                edited = prefix.copy()
                edited[name_index] = other_name
                assert full_support(gen, edited)
                q1 = gen.next_distribution(edited)
                assert q1 == {str(counts[other_name]): 1.0}
                pairs.append(dict(kind='query_person',
                                  original=[gen.token2id[x] for x in prefix],
                                  edited=[gen.token2id[x] for x in edited],
                                  q_before={target: 1.0}, q_after=q1))
                break
            attempted['historical_subject'] += counts[name] >= 2
            if counts[name] >= 2:
                for slot in reversed(slots):
                    if slot['subject'] != name:
                        continue
                    index = slot['subject_index']
                    for other_name in PEOPLE:
                        if other_name == name:
                            continue
                        if gen.next_distribution(prefix[:index]).get(other_name, 0) == 0:
                            continue
                        edited = prefix.copy()
                        edited[index] = other_name
                        if not full_support(gen, edited):
                            continue
                        q1 = gen.next_distribution(edited)
                        if q1 == {target: 1.0}:
                            continue
                        pairs.append(dict(kind='historical_subject',
                                          original=[gen.token2id[x] for x in prefix],
                                          edited=[gen.token2id[x] for x in edited],
                                          q_before={target: 1.0}, q_after=q1))
                        break
                    else:
                        continue
                    break
            if slots:
                attempted['nuisance_place'] += 1
                index = slots[-1]['place_index']
                new = replacement(gen, prefix, index, PLACES)
                if new is not None:
                    edited = prefix.copy()
                    edited[index] = new
                    if full_support(gen, edited):
                        q1 = gen.next_distribution(edited)
                        assert q1 == {target: 1.0}
                        pairs.append(dict(kind='nuisance_place',
                                          original=[gen.token2id[x] for x in prefix],
                                          edited=[gen.token2id[x] for x in edited],
                                          q_before={target: 1.0}, q_after=q1))
    return questions, pairs, attempted


def predict_scores(items, probs):
    target = np.array([item['target'] for item in items])
    p = probs[np.arange(len(items)), target]
    return dict(n=len(items), accuracy=float(np.mean(probs.argmax(-1) == target)),
                kl=float(np.mean(-np.log(np.maximum(p, 1e-30)))),
                correct_token_probability=float(np.mean(p)),
                key_blind_bayes_accuracy=float(np.mean(
                    [item['key_blind_bayes'] for item in items])),
                key_blind_cross_entropy=float(np.mean(
                    [item['key_blind_entropy'] for item in items])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dirs', nargs='+', required=True)
    p.add_argument('--n', type=int, default=3000)
    p.add_argument('--length', type=int, default=72)
    p.add_argument('--seed', type=int, default=9101)
    p.add_argument('--question-rate', type=float, default=.05)
    p.add_argument('--count-rate', type=float, default=.20)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    gen = SurfaceMixtureV5(seed=a.seed, question_rate=a.question_rate,
                           count_rate=a.count_rate,
                           counterbalanced_queries=True)
    questions, pairs, attempted = build(gen, a.n, a.length)
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config
    results = {}
    for directory in a.model_dirs:
        folder = Path(directory)
        saved = json.loads((folder/'results.json').read_text())
        for record in saved['records']:
            depth = record['depth']
            model = BilinearGPT(Config(**record['model_config']))
            model.load_state_dict(torch.load(folder/f'model_depth{depth}.pt',
                                             map_location='cpu', weights_only=True))
            model.eval()
            probs = predict(model, [q['prefix'] for q in questions], len(gen.vocab))
            before = predict(model, [pair['original'] for pair in pairs], len(gen.vocab))
            after = predict(model, [pair['edited'] for pair in pairs], len(gen.vocab))
            name = f'{folder.name}_depth{depth}'
            results[name] = dict(prediction={
                'all': predict_scores(questions, probs),
                'count_1': predict_scores([q for q in questions if q['count'] == 1],
                                          probs[[i for i, q in enumerate(questions) if q['count'] == 1]]),
                'count_2plus': predict_scores([q for q in questions if q['count'] >= 2],
                                              probs[[i for i, q in enumerate(questions) if q['count'] >= 2]])},
                edits=score(pairs, before, after, gen))
            print(name, results[name], flush=True)
    output = dict(config=vars(a), n_questions=len(questions),
                  attempted=dict(attempted),
                  pair_counts=dict(Counter(pair['kind'] for pair in pairs)),
                  results=results,
                  caveat='Every edit is a supported full prefix. Query-person and '
                         'historical-subject changes alter the exact count answer; '
                         'nuisance-place edits do not. Prediction and causal effect '
                         'scores are separate.')
    Path(a.output).write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
