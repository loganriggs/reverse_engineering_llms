"""Causal next-token checks for grammar-valid multi-genre prefix edits."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from multi_genre_language import (CONDITIONS, MultiGenreLanguage,
                                  _split_sentences)
from surface_mixture_language import PEOPLE, PLACES, VERBS
from eval_surface_mixture_edits import predict, score


def supported_replacement(gen, prefix, index, allowed, target_change=None):
    q, _ = gen.next_distribution(prefix[:index])
    choices = [token for token in allowed
               if token != prefix[index] and q.get(token, 0) > 0 and
               (target_change is None or target_change(token))]
    return max(choices, key=lambda token: q[token]) if choices else None


def edit_pair(gen, prefix, kind):
    # The first two tokens are the genre header. All edits are inside a
    # generated sentence and have positive probability under the DGP.
    genre = prefix[0]
    body = prefix[2:]
    history, clause = _split_sentences(body)
    start = 2 + sum(map(len, history))
    edited = list(prefix)
    if kind == 'policy_condition':
        assert genre == 'Policy' and len(clause) == 8
        before = gen._policy_candidates(history, clause)['condition']
        candidates = [condition for condition in CONDITIONS
                      if condition != clause[4] and
                      gen._policy_candidates(history,
                          clause[:4]+[condition]+clause[5:])['condition'] != before]
        index = start+4
        new = supported_replacement(gen, prefix, index, candidates)
    elif kind == 'policy_induction':
        assert genre == 'Policy' and len(clause) == 8
        matches = [(i, old) for i, old in enumerate(history)
                   if old[2] == clause[2]]
        if not matches:
            return None
        i, old = matches[-1]
        index = 2 + sum(map(len, history[:i]))+8
        new = supported_replacement(gen, prefix, index, VERBS)
    elif kind == 'policy_nuisance_person':
        assert genre == 'Policy' and len(clause) == 8
        index = start+7
        new = supported_replacement(gen, prefix, index, PEOPLE)
    elif kind == 'faq_answer_copy':
        assert genre == 'FAQ' and len(clause) == 10
        index = start+5
        from surface_mixture_language import OBJECTS
        new = supported_replacement(gen, prefix, index, OBJECTS)
    elif kind == 'faq_place_recall':
        assert genre == 'FAQ' and len(clause) == 14
        matches = [(i, old) for i, old in enumerate(history)
                   if old[5] == clause[5]]
        if len(matches) != 1:
            return None
        i, _ = matches[0]
        index = 2 + sum(map(len, history[:i]))+14
        new = supported_replacement(gen, prefix, index, PLACES)
    else:
        raise ValueError(kind)
    if new is None:
        return None
    edited[index] = new
    q0, _ = gen.next_distribution(prefix)
    q1, _ = gen.next_distribution(edited)
    if kind == 'policy_nuisance_person':
        assert q0 == q1
    elif q0 == q1:
        return None
    # Replay the edited token's own conditional to guard against invalid
    # counterfactual histories; subsequent deterministic frames are unchanged.
    assert gen.next_distribution(edited[:index])[0].get(new, 0) > 0
    return dict(kind=kind,
                context_object=clause[5] if genre == 'FAQ' else clause[2],
                original=[gen.token2id[token] for token in prefix],
                edited=[gen.token2id[token] for token in edited],
                q_before=q0, q_after=q1)


def build_pairs(gen, n, length):
    ids, labels, _ = gen.sample_batch(n, length+1)
    pairs = []
    targets = {'policy_mixed_action': ('policy_condition',
                                      'policy_induction',
                                      'policy_nuisance_person'),
               'faq_answer_copy': ('faq_answer_copy',),
               'faq_place_recall': ('faq_place_recall',)}
    for row, labs in zip(ids, labels):
        for t in range(1, length+1):
            if labs[t] not in targets:
                continue
            prefix = [gen.id2token[i] for i in row[:t]]
            for kind in targets[labs[t]]:
                pair = edit_pair(gen, prefix, kind)
                if pair is not None:
                    pairs.append(pair)
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', required=True)
    p.add_argument('--n', type=int, default=256)
    p.add_argument('--length', type=int, default=71)
    p.add_argument('--seed', type=int, default=9701)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    torch.set_num_threads(4)
    gen = MultiGenreLanguage(seed=a.seed)
    pairs = build_pairs(gen, a.n, a.length)
    assert pairs
    saved = json.loads((Path(a.model_dir)/'results.json').read_text())
    originals = [pair['original'] for pair in pairs]
    edits = [pair['edited'] for pair in pairs]
    results = {}
    for record in saved['records']:
        depth = record['depth']
        model = BilinearGPT(Config(**record['model_config']))
        model.load_state_dict(torch.load(Path(a.model_dir)/f'model_depth{depth}.pt',
                                         weights_only=True, map_location='cpu'))
        model.eval()
        before = predict(model, originals, len(gen.vocab))
        after = predict(model, edits, len(gen.vocab))
        results[str(depth)] = score(pairs, before, after, gen)
        print('depth', depth,
              {kind: round(row['effect_l1_error'], 3)
               for kind, row in results[str(depth)].items()}, flush=True)
    output = dict(config=vars(a), n_pairs=len(pairs), results=results,
                  caveat='Valid, supported single-token edits. The edit may '
                         'change probability of its later prefix; effects '
                         'condition on the edited visible prefix.')
    Path(a.output).write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
