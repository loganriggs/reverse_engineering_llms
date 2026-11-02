"""Readable autoregressive toy language with an exact mixed-rule oracle.

Every sentence has a visible frame. Four candidate mechanisms compete
at the verb slot: subject lexical bigram, prior-subject induction, discourse
lead, and previous-place carryover. Other slots have explicit distributions.
The narrow grammar is intentional: this is a stepping stone toward realistic
text, not a claim that the resulting corpus matches FineWeb.
"""

import random
from collections import defaultdict


LEADS = ('Today', 'Meanwhile', 'Later', 'Afterward', 'Lately', 'Often',
         'Sometimes', 'Usually')
PEOPLE = ('Alice', 'Bob', 'Carol', 'Dave', 'Eve', 'Frank', 'Grace', 'Henry',
          'Iris', 'Jack', 'Kate', 'Leo', 'Mia', 'Nick', 'Olive', 'Paul')
VERBS = ('finds', 'keeps', 'takes', 'moves', 'checks', 'holds', 'saves', 'uses',
         'brings', 'picks', 'stores', 'carries')
ADJECTIVES = ('small', 'old', 'new', 'red', 'blue', 'green', 'dark', 'bright',
              'warm', 'cool', 'soft', 'heavy', 'light', 'clean', 'dusty', 'rare',
              'large', 'round', 'wooden', 'silver', 'broken', 'useful', 'empty',
              'valuable')
OBJECTS = ('book', 'key', 'photo', 'cup', 'ticket', 'map', 'note', 'badge',
           'box', 'bag', 'letter', 'phone', 'camera', 'bottle', 'folder', 'coat',
           'watch', 'card', 'tool', 'toy', 'painting', 'chair', 'basket', 'lamp')
PLACES = ('park', 'home', 'store', 'school', 'beach', 'office', 'garden', 'lake',
          'tower', 'farm', 'hill', 'bridge', 'plaza', 'court', 'field', 'grove',
          'harbor', 'island', 'lodge', 'market', 'pier', 'ranch', 'station', 'library')
EVENTS = ('festival', 'meeting', 'storm', 'morning', 'cleanup', 'game', 'visit',
          'sale', 'journey', 'parade', 'exhibit', 'lesson', 'project', 'season',
          'holiday', 'workshop')
PLACE_PREPS = ('near', 'beside', 'outside', 'inside')
INTERIOR_PLACES = ('home', 'store', 'school', 'office', 'tower', 'lodge',
                   'market', 'station', 'library')
TIME_PREPS = ('during', 'after', 'before')
TAIL_TYPES = ('short', 'while', 'according', 'as')
VOCAB = (LEADS + PEOPLE + VERBS + ADJECTIVES + OBJECTS + PLACES + EVENTS +
         PLACE_PREPS + TIME_PREPS + (',', 'the', '.', 'while', 'watches',
                                     'according', 'to', 'as', 'looks', 'on'))
assert len(VOCAB) == len(set(VOCAB))
TOKEN2ID = {token: i for i, token in enumerate(VOCAB)}
ID2TOKEN = {i: token for token, i in TOKEN2ID.items()}
DEFAULT_WEIGHTS = dict(lexical=.40, induction=.25, lead=.20, place_skip=.15)


def uniform(tokens):
    return {token: 1/len(tokens) for token in tokens}


def _history_and_clause(tokens):
    history, clause = [], []
    for token in tokens:
        clause.append(token)
        if token == '.':
            history.append(clause)
            clause = []
    return history, clause


class SurfaceMixtureLanguage:
    def __init__(self, seed=0, weights=None):
        self.rng = random.Random(seed)
        self.weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
        assert all(self.weights[k] >= 0 for k in DEFAULT_WEIGHTS)
        assert sum(self.weights.values()) > 0
        assert self.weights['lexical']+self.weights['lead'] > 0, (
            'At least one first-sentence verb rule must have positive weight')
        self.vocab = VOCAB
        self.token2id = TOKEN2ID
        self.id2token = ID2TOKEN

    def candidates(self, prefix):
        """Named next-verb candidate distributions from a visible prefix."""
        tokens = [ID2TOKEN[t] if isinstance(t, int) else t for t in prefix]
        history, clause = _history_and_clause(tokens)
        if len(clause) != 3:
            raise ValueError('Candidates exist only at the verb slot')
        lead, comma, subject = clause
        assert comma == ',' and lead in LEADS and subject in PEOPLE
        subject_i = PEOPLE.index(subject)
        verb_i = (subject_i*3) % len(VERBS)
        lexical = {VERBS[(verb_i+j)%len(VERBS)]: p
                   for j, p in enumerate((.7, .2, .1))}
        out = {'lexical': lexical,
               'lead': {VERBS[(LEADS.index(lead)*5+1)%len(VERBS)]: 1.0}}
        earlier = [clause for clause in history if clause[2] == subject]
        if earlier:
            out['induction'] = {earlier[-1][3]: 1.0}
        if history:
            place = history[-1][9]
            out['place_skip'] = {VERBS[(PLACES.index(place)*7+2)%len(VERBS)]: 1.0}
        return out

    def next_distribution(self, prefix):
        tokens = [ID2TOKEN[t] if isinstance(t, int) else t for t in prefix]
        history, clause = _history_and_clause(tokens)
        position = len(clause)
        if position == 0:
            return uniform(LEADS)
        if position == 1:
            return {',': 1.0}
        if position == 2:
            out = uniform(PEOPLE)
            if history:
                out = {name: .6*p for name, p in out.items()}
                out[history[-1][2]] += .4
            return out
        if position == 3:
            candidates = self.candidates(tokens)
            total = sum(self.weights[name] for name in candidates)
            out = defaultdict(float)
            for name, dist in candidates.items():
                for token, p in dist.items():
                    out[token] += self.weights[name]*p/total
            return dict(out)
        if position == 9:
            return uniform(INTERIOR_PLACES if clause[7] == 'inside' else PLACES)
        slots = {4: {'the': 1.0}, 5: uniform(ADJECTIVES),
                 6: uniform(OBJECTS), 7: uniform(PLACE_PREPS),
                 8: {'the': 1.0},
                 10: uniform(TIME_PREPS), 11: {'the': 1.0},
                 12: uniform(EVENTS)}
        if position in slots:
            return slots[position]
        tail = TAIL_TYPES[LEADS.index(clause[0]) % len(TAIL_TYPES)]
        tails = {
            'short': [{'.': 1.0}],
            'while': [{'while': 1.0}, uniform(PEOPLE),
                      {'watches': 1.0}, {'.': 1.0}],
            'according': [{',': 1.0}, {'according': 1.0},
                          {'to': 1.0}, uniform(PEOPLE), {'.': 1.0}],
            'as': [{'as': 1.0}, uniform(PEOPLE), {'looks': 1.0},
                   {'on': 1.0}, {'.': 1.0}],
        }
        return tails[tail][position-13]

    def sample_sequence(self, length=70):
        tokens, labels, distributions = [], [], []
        for _ in range(length):
            dist = self.next_distribution(tokens)
            token = self.rng.choices(list(dist), weights=list(dist.values()))[0]
            _, clause = _history_and_clause(tokens)
            position = len(clause)
            labels.append('mixed_verb' if position == 3 else f'frame_{position}')
            distributions.append(dist)
            tokens.append(token)
        return [TOKEN2ID[t] for t in tokens], labels, distributions

    def sample_batch(self, n, length=70):
        rows = [self.sample_sequence(length) for _ in range(n)]
        return ([r[0] for r in rows], [r[1] for r in rows],
                [r[2] for r in rows])


def render(ids):
    words = [ID2TOKEN[i] if isinstance(i, int) else i for i in ids]
    text = ' '.join(words).replace(' ,', ',').replace(' .', '.')
    return text
