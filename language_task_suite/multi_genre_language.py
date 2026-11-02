"""Multi-genre autoregressive language with an exact visible-prefix oracle.

Genres are selected by the first visible header token. Reports reuse the
competing-rule event language; guides test list increments; FAQ entries test
answer copying and object-place recall; policies test conditional action rules.
This is a controlled task mixture, not a distributional match to web text.
"""

import random
from collections import defaultdict

from surface_mixture_language import (ADJECTIVES, EVENTS, LEADS, OBJECTS,
                                      PEOPLE, PLACES, PLACE_PREPS, VERBS,
                                      SurfaceMixtureLanguage, uniform)


GENRES = ('Reports', 'Guide', 'FAQ', 'Policy')
GENRE_WEIGHTS = (.4, .2, .2, .2)
GUIDE_ACTIONS = ('Check', 'Move', 'Keep', 'Find', 'Take', 'Use', 'Store', 'Carry')
CONDITIONS = ('broken', 'dusty', 'valuable', 'clean', 'empty', 'heavy',
              'rare', 'old', 'new', 'small')
EXTRA = (GENRES + GUIDE_ACTIONS + CONDITIONS +
         ('1', '2', '3', '4', '5', '6', '7', '8', '9',
          ':', ')', 'Question', 'Where', 'is', '?', 'Answer', 'The',
          'If', 'then', 'it'))
from surface_mixture_language import VOCAB as EVENT_VOCAB
VOCAB = tuple(dict.fromkeys(EVENT_VOCAB + EXTRA))
TOKEN2ID = {token: i for i, token in enumerate(VOCAB)}
ID2TOKEN = {i: token for token, i in TOKEN2ID.items()}


def _split_sentences(tokens):
    history, clause = [], []
    for token in tokens:
        clause.append(token)
        if token == '.':
            history.append(clause)
            clause = []
    return history, clause


def _choose_by_weight(candidates, weights):
    total = sum(weights[name] for name in candidates)
    if total <= 0:
        raise ValueError('No active rule has positive weight')
    out = defaultdict(float)
    for name, dist in candidates.items():
        for token, p in dist.items():
            out[token] += weights[name]*p/total
    return dict(out)


class MultiGenreLanguage:
    def __init__(self, seed=0, genre_weights=GENRE_WEIGHTS,
                 report_weights=None, policy_weights=None):
        self.rng = random.Random(seed)
        self.genre_weights = tuple(genre_weights)
        assert len(self.genre_weights) == len(GENRES)
        assert min(self.genre_weights) >= 0 and sum(self.genre_weights) > 0
        self.report = SurfaceMixtureLanguage(seed=seed+31,
                                              weights=report_weights)
        self.policy_weights = dict(policy_weights or
                                   {'condition': .45, 'object': .30,
                                    'induction': .25})
        assert min(self.policy_weights.values()) >= 0
        assert self.policy_weights['condition']+self.policy_weights['object'] > 0
        self.vocab = VOCAB
        self.token2id = TOKEN2ID
        self.id2token = ID2TOKEN

    def _faq(self, body):
        history, clause = _split_sentences(body)
        pos = len(clause)
        first = {0: {'Question': 1.0}, 1: {':': 1.0},
                 2: {'Where': 1.0}, 3: {'is': 1.0},
                 4: {'the': 1.0}}
        if pos in first:
            return first[pos], 'faq_frame'
        if pos == 5:
            out = uniform(OBJECTS)
            if history:
                out = {token: .6*p for token, p in out.items()}
                out[history[-1][5]] += .4
            return out, 'faq_question_object'
        middle = {6: {'?': 1.0}, 7: {'Answer': 1.0},
                  8: {':': 1.0}, 9: {'The': 1.0}}
        if pos in middle:
            return middle[pos], 'faq_frame'
        if pos == 10:
            return {clause[5]: 1.0}, 'faq_answer_copy'
        final = {11: {'is': 1.0}, 12: {'near': 1.0},
                 13: {'the': 1.0}, 15: {'.': 1.0}}
        if pos in final:
            return final[pos], 'faq_frame'
        if pos == 14:
            matches = [old for old in history if old[5] == clause[5]]
            if matches:
                return {matches[-1][14]: 1.0}, 'faq_place_recall'
            return uniform(PLACES), 'faq_new_place'
        raise ValueError(f'Bad FAQ prefix position {pos}')

    def _guide(self, body):
        history, clause = _split_sentences(body)
        pos = len(clause)
        if pos == 0:
            return {str((len(history) % 9)+1): 1.0}, 'list_number'
        slots = {1: {')': 1.0}, 2: uniform(GUIDE_ACTIONS),
                 3: {'the': 1.0}, 4: uniform(ADJECTIVES),
                 5: uniform(OBJECTS), 6: uniform(PLACE_PREPS),
                 7: {'the': 1.0}, 9: {'before': 1.0},
                 10: {'the': 1.0}, 11: uniform(EVENTS),
                 12: {'.': 1.0}}
        if pos == 8:
            from surface_mixture_language import INTERIOR_PLACES
            return uniform(INTERIOR_PLACES if clause[6] == 'inside' else PLACES), 'guide_place'
        if pos in slots:
            return slots[pos], 'guide_frame'
        raise ValueError(f'Bad Guide prefix position {pos}')

    def _policy_candidates(self, history, clause):
        obj, condition, person = clause[2], clause[4], clause[7]
        condition_verb = VERBS[(CONDITIONS.index(condition)*5+1)%len(VERBS)]
        object_verb = VERBS[(OBJECTS.index(obj)*3+2)%len(VERBS)]
        out = {'condition': {condition_verb: 1.0},
               'object': {object_verb: 1.0}}
        matches = [prior for prior in history if prior[2] == obj]
        if matches:
            out['induction'] = {matches[-1][8]: 1.0}
        return out

    def _policy(self, body):
        history, clause = _split_sentences(body)
        pos = len(clause)
        slots = {0: {'If': 1.0}, 1: {'the': 1.0},
                 2: uniform(OBJECTS), 3: {'is': 1.0},
                 4: uniform(CONDITIONS), 5: {',': 1.0},
                 6: {'then': 1.0}, 7: uniform(PEOPLE),
                 9: {'it': 1.0}, 10: {'before': 1.0},
                 11: {'the': 1.0}, 12: uniform(EVENTS),
                 13: {'.': 1.0}}
        if pos == 2 and history:
            out = {token: .6*p for token, p in uniform(OBJECTS).items()}
            out[history[-1][2]] += .4
            return out, 'policy_object'
        if pos == 8:
            return _choose_by_weight(self._policy_candidates(history, clause),
                                     self.policy_weights), 'policy_mixed_action'
        if pos in slots:
            return slots[pos], 'policy_frame'
        raise ValueError(f'Bad Policy prefix position {pos}')

    def next_distribution(self, prefix):
        tokens = [ID2TOKEN[t] if isinstance(t, int) else t for t in prefix]
        if not tokens:
            z = sum(self.genre_weights)
            return {genre: weight/z for genre, weight in
                    zip(GENRES, self.genre_weights)}, 'genre'
        genre = tokens[0]
        if genre not in GENRES:
            raise ValueError(f'Unknown genre token {genre}')
        if len(tokens) == 1:
            return {':': 1.0}, 'header'
        assert tokens[1] == ':'
        body = tokens[2:]
        if genre == 'Reports':
            dist = self.report.next_distribution(body)
            pos = len(_split_sentences(body)[1])
            return dist, ('report_mixed_verb' if pos == 3 else 'report_frame')
        if genre == 'Guide':
            return self._guide(body)
        if genre == 'FAQ':
            return self._faq(body)
        return self._policy(body)

    def sample_sequence(self, length=72):
        tokens, labels, distributions = [], [], []
        for _ in range(length):
            q, label = self.next_distribution(tokens)
            token = self.rng.choices(list(q), weights=list(q.values()))[0]
            tokens.append(token)
            labels.append(label)
            distributions.append(q)
        return [TOKEN2ID[t] for t in tokens], labels, distributions

    def sample_batch(self, n, length=72):
        rows = [self.sample_sequence(length) for _ in range(n)]
        return ([r[0] for r in rows], [r[1] for r in rows],
                [r[2] for r in rows])


def render(ids):
    tokens = [ID2TOKEN[i] if isinstance(i, int) else i for i in ids]
    text = ' '.join(tokens)
    for punctuation in (',', '.', ':', '?', ')'):
        text = text.replace(' '+punctuation, punctuation)
    return text
