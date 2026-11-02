"""Question accuracy and KL, split by intervening narrative distractors."""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from eval_surface_mixture_edits import predict
from surface_mixture_v2 import _frame, _indices, _split
from surface_mixture_v3 import OBJECT_INDEX
from surface_mixture_v4 import SurfaceMixtureV4, _content


def items(gen, n, length, seed, hard_query_rate=None,
          counterbalanced_queries=False, question_rate=.05):
    rows, labels, _ = SurfaceMixtureV4(
        seed=seed, hard_query_rate=hard_query_rate,
        counterbalanced_queries=counterbalanced_queries,
        question_rate=question_rate).sample_batch(n, length+1)
    out = []
    for row, labs in zip(rows, labels):
        tokens = [gen.id2token[i] for i in row]
        for t in range(1, length+1):
            if labs[t] != 'question_answer':
                continue
            prefix = tokens[:t]
            history, raw = _split(prefix)
            obj = _content(raw)[3]
            narratives = [_content(clause) for clause in history
                          if _content(clause) and _content(clause)[0] != 'Where']
            last_match = max(i for i, clause in enumerate(narratives)
                             if clause[OBJECT_INDEX[_frame(clause)]] == obj)
            clause = narratives[last_match]
            answer = clause[_indices(_frame(clause))[2]]
            latest = narratives[-1]
            last_place = latest[_indices(_frame(latest))[2]]
            previous_place = (narratives[-2][
                _indices(_frame(narratives[-2]))[2]] if len(narratives) >= 2
                else last_place)
            key_pos = t-5
            key_q = gen.next_distribution(tokens[:key_pos])
            place_q = defaultdict(float)
            for key, mass in key_q.items():
                alternative = prefix.copy()
                alternative[key_pos] = key
                place_q[gen.question_answer(alternative)] += mass
            key_blind_bayes = max(place_q.values())
            key_blind_entropy = -sum(mass*math.log(mass)
                                     for mass in place_q.values() if mass > 0)
            out.append(dict(prefix=row[:t], target=gen.token2id[answer],
                            intervening=len(narratives)-1-last_match,
                            last_place_correct=last_place == answer,
                            second_last_place_correct=previous_place == answer,
                            key_blind_bayes=key_blind_bayes,
                            key_blind_entropy=key_blind_entropy))
    return out


def summarize(rows, probs):
    answer = np.array([row['target'] for row in rows])
    p = probs[np.arange(len(rows)), answer]
    return dict(n=len(rows), accuracy=float(np.mean(probs.argmax(-1) == answer)),
                kl=float(np.mean(-np.log(np.maximum(p, 1e-30)))),
                correct_token_probability=float(np.mean(p)),
                last_place_baseline_accuracy=float(np.mean(
                    [row['last_place_correct'] for row in rows])),
                second_last_place_baseline_accuracy=float(np.mean(
                    [row['second_last_place_correct'] for row in rows])),
                key_blind_bayes_accuracy=float(np.mean(
                    [row['key_blind_bayes'] for row in rows])),
                key_blind_cross_entropy=float(np.mean(
                    [row['key_blind_entropy'] for row in rows])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dirs', nargs='+', required=True)
    p.add_argument('--n', type=int, default=1024)
    p.add_argument('--length', type=int, default=72)
    p.add_argument('--seed', type=int, default=8101)
    p.add_argument('--hard-query-rate', type=float, default=None)
    p.add_argument('--counterbalanced-queries', action='store_true')
    p.add_argument('--question-rate', type=float, default=.05)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    gen = SurfaceMixtureV4(hard_query_rate=a.hard_query_rate,
                           counterbalanced_queries=a.counterbalanced_queries,
                           question_rate=a.question_rate)
    records = items(gen, a.n, a.length, a.seed, a.hard_query_rate,
                    a.counterbalanced_queries, a.question_rate)
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
            probs = predict(model, [row['prefix'] for row in records], len(gen.vocab))
            name = f'{folder.name}_depth{depth}'
            results[name] = {}
            for kind, indices in (('all', list(range(len(records)))),
                                  ('intervening', [i for i, r in enumerate(records)
                                                   if r['intervening'] >= 1]),
                                  ('intervening_2plus', [i for i, r in enumerate(records)
                                                         if r['intervening'] >= 2])):
                if indices:
                    results[name][kind] = summarize([records[i] for i in indices],
                                                    probs[indices])
            print(name, results[name], flush=True)
    output = dict(config=vars(a), results=results,
                  caveat='Question answers are deterministic, so oracle KL is '
                         'negative log probability of the correct place. This '
                         'is a prediction measure, not causal fidelity.')
    Path(a.output).write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
