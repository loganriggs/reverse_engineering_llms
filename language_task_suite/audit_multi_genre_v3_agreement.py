"""Exact-prefix and grammar-preserving causal audit for v3 agreement."""

import argparse
import collections
import json
from pathlib import Path

from multi_genre_v3_agreement import (MultiGenreV3Agreement, NOUN_NUMBER,
                                      NOUN_OTHER_NUMBER)


def answer(gen, words):
    q, label = gen.next_distribution(words)
    assert abs(sum(q.values()) - 1) < 1e-10
    return next(iter(q)) if len(q) == 1 else None, label


def audit(seed=0, streams=100, length=129):
    gen = MultiGenreV3Agreement(seed=seed,
                                genre_weights=(0, 0, 0, 0, 0, 1))
    counts = collections.Counter()
    examples = {}
    for _ in range(streams):
        ids, labels, distributions = gen.sample_sequence(length)
        words = [gen.id2token[i] for i in ids]
        for t, (word, label, q) in enumerate(zip(words, labels,
                                                  distributions)):
            replay, replay_label = gen.next_distribution(words[:t])
            assert label == replay_label and q == replay
            assert word in q and abs(sum(q.values()) - 1) < 1e-10
            counts[label] += 1
            if label not in ('agreement_main_match',
                             'agreement_main_conflict',
                             'agreement_embedded'):
                continue
            clause = words[2:t]
            dot = max((i for i, token in enumerate(clause) if token == '.'),
                      default=-1)
            prefix = clause[dot+1:]
            assert prefix[:1] == ['The']
            subject, connector, attractor = prefix[1], prefix[2], prefix[4]
            main = 'are' if NOUN_NUMBER[subject] else 'is'
            embedded = 'help' if NOUN_NUMBER[attractor] else 'helps'
            expected = embedded if label == 'agreement_embedded' else main
            assert answer(gen, words[:t])[0] == expected
            counts['relative' if connector == 'that' else 'prepositional'] += 1
            if label.startswith('agreement_main'):
                assert (label == 'agreement_main_conflict') == (
                    NOUN_NUMBER[subject] != NOUN_NUMBER[attractor])
                flipped_subject = prefix.copy()
                flipped_subject[1] = NOUN_OTHER_NUMBER[subject]
                changed, changed_label = answer(
                    gen, words[:t-len(prefix)] + flipped_subject)
                assert changed != main and changed_label == label.replace(
                    'match', 'TMP').replace('conflict', 'match').replace(
                    'TMP', 'conflict')
                counts['main_subject_flip_changes'] += 1
                flipped_attractor = prefix.copy()
                flipped_attractor[4] = NOUN_OTHER_NUMBER[attractor]
                if connector == 'that':
                    flipped_attractor[5] = ('help' if
                        NOUN_NUMBER[flipped_attractor[4]] else 'helps')
                unchanged, _ = answer(
                    gen, words[:t-len(prefix)] + flipped_attractor)
                assert unchanged == main
                counts['attractor_flip_main_invariant'] += 1
                if connector != 'that':
                    other_connector = ('beside' if connector == 'near'
                                       else 'near')
                    edited = prefix.copy()
                    edited[2] = other_connector
                    assert answer(gen, words[:t-len(prefix)] + edited)[0] == main
                    counts['preposition_swap_main_invariant'] += 1
            else:
                edited = prefix.copy()
                edited[4] = NOUN_OTHER_NUMBER[attractor]
                changed, _ = answer(gen, words[:t-len(prefix)] + edited)
                assert changed != embedded
                counts['embedded_attractor_flip_changes'] += 1
            examples.setdefault(label, ' '.join(prefix) + ' → ' + expected)
    mixed = MultiGenreV3Agreement(seed=seed+1)
    mixed_genres = collections.Counter()
    for _ in range(streams):
        ids, labels, distributions = mixed.sample_sequence(length)
        mixed_genres[mixed.id2token[ids[0]]] += 1
        for t, (token, label, q) in enumerate(zip(ids, labels,
                                                   distributions)):
            replay, replay_label = mixed.next_distribution(ids[:t])
            assert replay == q and replay_label == label
            assert mixed.id2token[token] in replay
    return dict(config=dict(seed=seed, streams=streams, length=length),
                counts=dict(counts), examples=examples,
                mixed_genres_replayed=dict(mixed_genres))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--streams', type=int, default=100)
    p.add_argument('--length', type=int, default=129)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    result = audit(args.seed, args.streams, args.length)
    payload = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload)
