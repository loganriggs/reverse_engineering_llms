"""Test object-to-place retrieval under supported historical place edits."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

from eval_surface_mixture_edits import predict, replacement, score
from surface_mixture_v2 import PLACES, _frame, _indices, _split
from surface_mixture_v3 import OBJECT_INDEX
from surface_mixture_v4 import SurfaceMixtureV4, _content


def full_support(gen, tokens):
    return all(gen.next_distribution(tokens[:i]).get(token, 0) > 0
               for i, token in enumerate(tokens))


def build_pairs(gen, n, length):
    ids, labels, _ = gen.sample_batch(n, length+1)
    pairs = []
    attempted = Counter()
    for row, labs in zip(ids, labels):
        tokens = [gen.id2token[i] for i in row]
        for t in range(1, length+1):
            if labs[t] != 'question_answer':
                continue
            prefix = tokens[:t]
            history, raw = _split(prefix)
            obj = _content(raw)[3]
            prior = []
            offset = 0
            for clause in history:
                content = _content(clause)
                if content and content[0] != 'Where':
                    frame = _frame(content)
                    object_name = content[OBJECT_INDEX[frame]]
                    place_index = offset+len(clause)-len(content)+_indices(frame)[2]
                    prior.append((object_name, place_index))
                offset += len(clause)
            matches = [item for item in prior if item[0] == obj]
            others = [item for item in prior if item[0] != obj]
            intervening = (len(prior)-1-max(i for i, item in enumerate(prior)
                                            if item[0] == obj))
            current_answer = gen.question_answer(prefix)
            attempted['question_object'] += 1
            query_index = offset+len(raw)-len(_content(raw))+3
            for other_obj, _ in reversed(others):
                changed = prefix.copy()
                changed[query_index] = other_obj
                if not full_support(gen, changed):
                    continue
                q1 = gen.next_distribution(changed)
                if q1 == {current_answer: 1.0}:
                    continue
                pairs.append(dict(kind='question_object',
                                  intervening_narratives=intervening,
                                  original=[gen.token2id[x] for x in prefix],
                                  edited=[gen.token2id[x] for x in changed],
                                  q_before={current_answer: 1.0}, q_after=q1))
                break
            for kind, candidate in (('answer_place', matches[-1] if matches else None),
                                    ('other_object_place', others[-1] if others else None)):
                attempted[kind] += 1
                if candidate is None:
                    continue
                _, index = candidate
                new = replacement(gen, prefix, index, PLACES)
                if new is None:
                    continue
                edited = prefix.copy()
                edited[index] = new
                if not full_support(gen, edited):
                    attempted[kind+'_unsupported'] += 1
                    continue
                q0, q1 = gen.next_distribution(prefix), gen.next_distribution(edited)
                assert (q0 != q1) == (kind == 'answer_place')
                pairs.append(dict(kind=kind,
                                  intervening_narratives=intervening,
                                  original=[gen.token2id[x] for x in prefix],
                                  edited=[gen.token2id[x] for x in edited],
                                  q_before=q0, q_after=q1))
    return pairs, attempted


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dirs', nargs='+', required=True)
    p.add_argument('--n', type=int, default=1024)
    p.add_argument('--length', type=int, default=72)
    p.add_argument('--seed', type=int, default=8701)
    p.add_argument('--hard-query-rate', type=float, default=None)
    p.add_argument('--counterbalanced-queries', action='store_true')
    p.add_argument('--question-rate', type=float, default=.05)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    gen = SurfaceMixtureV4(seed=a.seed, hard_query_rate=a.hard_query_rate,
                           counterbalanced_queries=a.counterbalanced_queries,
                           question_rate=a.question_rate)
    pairs, attempted = build_pairs(gen, a.n, a.length)
    if not pairs:
        raise RuntimeError('No supported question edit pairs')
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
            before = predict(model, [pair['original'] for pair in pairs], len(gen.vocab))
            after = predict(model, [pair['edited'] for pair in pairs], len(gen.vocab))
            name = f'{folder.name}_depth{depth}'
            results[name] = score(pairs, before, after, gen)
            hard_indices = [i for i, pair in enumerate(pairs)
                            if pair['kind'] == 'answer_place'
                            and pair['intervening_narratives'] >= 1]
            if hard_indices:
                hard = score([pairs[i] for i in hard_indices],
                             before[hard_indices], after[hard_indices], gen)
                results[name]['answer_place_with_intervening_narrative'] = hard['answer_place']
            print(name, {kind: round(value['effect_l1_error'], 3)
                         for kind, value in results[name].items()}, flush=True)
    output = dict(config=vars(a), attempted=dict(attempted),
                  pair_counts=dict(Counter(pair['kind'] for pair in pairs)),
                  results=results,
                  caveat='Every edited full prefix retains positive oracle support. '
                         'The answer edit changes the last mentioned place for the '
                         'queried object; the control changes another object place.')
    Path(a.output).write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
