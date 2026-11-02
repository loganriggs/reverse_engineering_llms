"""Two-record score comparison with an exact visible-prefix language oracle.

Each block independently draws two people and distinct scores. The answer
depends on binding each person to a number and comparing those numbers.
Variable wording and question order prevent a fixed positional answer rule.
This is a controlled numeric task, not a model of sports/news frequencies.
"""

import random

from multi_genre_language import VOCAB as BASE_VOCAB
from surface_mixture_language import PEOPLE


PEOPLE_SCORE = PEOPLE[:12]
SCORES = tuple(str(i) for i in range(1, 10))
GENRES = ('Scores',)
EXTRA = ('Scores', 'scored', 'earned', 'had', 'point', 'points', 'more', 'fewer',
         'Who', 'or', 'did', '\n') + SCORES
VOCAB = tuple(dict.fromkeys(BASE_VOCAB + EXTRA))
TOKEN2ID = {word: i for i, word in enumerate(VOCAB)}
ID2TOKEN = {i: word for word, i in TOKEN2ID.items()}
TASK_LABELS = ('score_record_person', 'score_value', 'score_query_person',
               'score_comparison_answer')


def lines_state(body):
    completed, current = [], []
    for token in body:
        if token == '\n':
            if current:
                completed.append(current)
            current = []
        else:
            current.append(token)
    return completed, current


def record_state(completed):
    assert len(completed) >= 2
    first, second = completed[-2:]
    assert (len(first) == len(second) == 5 and
            first[0] != second[0] and first[2] != second[2])
    assert all(line[0] in PEOPLE_SCORE and line[1] in ('scored', 'earned')
               and line[2] in SCORES and
               line[3:] == [('point' if line[2] == '1' else 'points'), '.']
               for line in (first, second))
    return {first[0]: int(first[2]), second[0]: int(second[2])}


class NumericReportLanguage:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.vocab = VOCAB
        self.token2id = TOKEN2ID
        self.id2token = ID2TOKEN

    def next_distribution(self, prefix):
        words = [ID2TOKEN[i] if isinstance(i, int) else i for i in prefix]
        if not words:
            return {'Scores': 1.}, 'genre'
        if len(words) == 1:
            return {':': 1.}, 'header'
        assert words[:2] == ['Scores', ':']
        if len(words) == 2:
            return {'\n': 1.}, 'frame'
        assert words[2] == '\n'
        completed, line = lines_state(words[3:])
        stage, pos = len(completed) % 3, len(line)
        if stage in (0, 1):
            prior = completed[-1] if stage == 1 else None
            if pos == 0:
                options = [person for person in PEOPLE_SCORE
                           if prior is None or person != prior[0]]
                return {name: 1/len(options) for name in options}, 'score_record_person'
            if pos == 1:
                return {'scored': .5, 'earned': .5}, 'frame'
            if pos == 2:
                options = [number for number in SCORES
                           if prior is None or number != prior[2]]
                return {number: 1/len(options) for number in options}, 'score_value'
            fixed = {3: ('point' if line[2] == '1' else 'points'),
                     4: '.', 5: '\n'}
            if pos in fixed:
                return {fixed[pos]: 1.}, 'frame'
        else:
            facts = record_state(completed)
            people = list(facts)
            fixed = {0: 'Who', 3: 'points', 4: ',', 6: 'or', 8: '?',
                     10: 'did', 11: '.', 12: '\n'}
            if pos == 1:
                return {'scored': .5, 'had': .5}, 'frame'
            if pos == 2:
                return {'more': .5, 'fewer': .5}, 'frame'
            if pos == 5:
                return {name: .5 for name in people}, 'score_query_person'
            if pos == 7:
                return {next(name for name in people if name != line[5]): 1.}, 'frame'
            if pos == 9:
                winner = (max if line[2] == 'more' else min)(
                    people, key=lambda name: facts[name])
                return {winner: 1.}, 'score_comparison_answer'
            if pos in fixed:
                return {fixed[pos]: 1.}, 'frame'
        raise ValueError(f'Bad score-report stage {stage}, position {pos}')

    def sample_sequence(self, length=64):
        words, labels, distributions = [], [], []
        for _ in range(length):
            q, label = self.next_distribution(words)
            word = self.rng.choices(list(q), weights=list(q.values()))[0]
            words.append(word)
            labels.append(label)
            distributions.append(q)
        return [TOKEN2ID[word] for word in words], labels, distributions

    def sample_batch(self, n, length=64):
        rows = [self.sample_sequence(length) for _ in range(n)]
        return ([row[0] for row in rows], [row[1] for row in rows],
                [row[2] for row in rows])


def render(ids):
    words = [ID2TOKEN[i] if isinstance(i, int) else i for i in ids]
    return (' '.join(words).replace(' \n ', '\n').replace(' ,', ',')
            .replace(' ?', '?').replace(' .', '.').replace(' :', ':'))
