"""Verify the multi-genre oracle, task invariants, and surface gaps."""

import argparse
import json
from collections import Counter
from pathlib import Path

from compare_realism import summary
from multi_genre_language import (GENRES, GUIDE_ACTIONS, MultiGenreLanguage,
                                  _split_sentences, render)
from surface_mixture_language import (ADJECTIVES, EVENTS, OBJECTS, PEOPLE,
                                      PLACES, PLACE_PREPS)


def check_guide(tokens):
    history, _ = _split_sentences(tokens[2:])
    for i, clause in enumerate(history):
        assert clause[0] == str((i%9)+1)
        assert (clause[1] == ')' and clause[2] in GUIDE_ACTIONS and
                clause[3] == 'the' and clause[4] in ADJECTIVES and
                clause[5] in OBJECTS and clause[6] in PLACE_PREPS and
                clause[7] == 'the' and clause[8] in PLACES and
                clause[9] == 'before' and clause[10] == 'the' and
                clause[11] in EVENTS and clause[12] == '.')


def check_faq(tokens):
    history, _ = _split_sentences(tokens[2:])
    places = {}
    for clause in history:
        assert (len(clause) == 16 and clause[:5] == [
            'Question', ':', 'Where', 'is', 'the'] and
            clause[5] in OBJECTS and clause[6:10] == [
            '?', 'Answer', ':', 'The'] and clause[10] == clause[5] and
            clause[11:14] == ['is', 'near', 'the'] and
            clause[14] in PLACES and clause[15] == '.')
        if clause[5] in places:
            assert clause[14] == places[clause[5]]
        places[clause[5]] = clause[14]


def check_policy(tokens):
    history, _ = _split_sentences(tokens[2:])
    for clause in history:
        assert (len(clause) == 14 and clause[:2] == ['If', 'the'] and
                clause[2] in OBJECTS and clause[3] == 'is' and
                clause[5:7] == [',', 'then'] and clause[7] in PEOPLE and
                clause[9:12] == ['it', 'before', 'the'] and
                clause[12] in EVENTS and clause[13] == '.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=4000)
    p.add_argument('--length', type=int, default=72)
    p.add_argument('--output', default='language_task_suite/multi_genre_audit.json')
    a = p.parse_args()
    gen = MultiGenreLanguage(seed=5101)
    ids, labels, distributions = gen.sample_batch(a.n, a.length)
    genre_counts, label_counts = Counter(), Counter()
    texts_by_genre = {genre: [] for genre in GENRES}
    for row, labs, qs in zip(ids, labels, distributions):
        words = [gen.id2token[i] for i in row]
        genre = words[0]
        assert genre in GENRES and words[1] == ':'
        genre_counts[genre] += 1
        texts_by_genre[genre].append(render(row))
        label_counts.update(labs)
        if genre == 'Guide':
            check_guide(words)
        elif genre == 'FAQ':
            check_faq(words)
        elif genre == 'Policy':
            check_policy(words)
        for t, (token, q, label) in enumerate(zip(words, qs, labs)):
            assert abs(sum(q.values())-1) < 1e-12
            assert q.get(token, 0) > 0
            replay, replay_label = gen.next_distribution(row[:t])
            assert label == replay_label and q == replay
    fineweb = json.loads(Path(__file__).with_name(
        'world_v2_surface_comparison.json').read_text())['fineweb_513_token_windows']
    result = dict(config=vars(a), n_vocab=len(gen.vocab),
                  genres=dict(genre_counts), labels=dict(label_counts),
                  overall_surface=summary([render(row) for row in ids]),
                  genre_surfaces={genre: summary(texts)
                                  for genre, texts in texts_by_genre.items()},
                  fineweb_surface=fineweb,
                  examples={genre: texts[:2] for genre, texts in texts_by_genre.items()},
                  caveat='Exact oracle and task invariants are verified on generated '
                         'streams. This does not establish naturalness or corpus '
                         'matching; the FineWeb panel consists of fixed token windows.')
    Path(a.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in
                      ('n_vocab', 'genres', 'labels', 'overall_surface',
                       'genre_surfaces')}, indent=2))


if __name__ == '__main__':
    main()
