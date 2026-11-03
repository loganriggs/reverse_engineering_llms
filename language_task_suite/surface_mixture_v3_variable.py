"""A variable-length realization of the original joint verb/object language.

The short clause ``Kate moves a clean watch near the park after the game.``
keeps the same visible subject, verb, adjective, object, place, and event as
the longer V3 frames. Completed short clauses are canonically replayed as
V3 ``At`` clauses for subsequent induction/skip rules. No sampled hidden
event is inserted into the conditioning history.

This is a *core-rule form pilot*, not yet an extension of the later V14
source, fact, and question stream.
"""

from surface_mixture_v2 import (ADJECTIVES, EVENTS, FRAME_WEIGHTS, PEOPLE,
                                PLACES, VERBS, _frame, _indices, _split,
                                uniform)
from surface_mixture_v3 import (OBJECT_INDEX, SurfaceMixtureV3)


EXTRA = ('an',)
VOCAB_VARIABLE = tuple(dict.fromkeys(SurfaceMixtureV3().vocab + EXTRA))
TOKEN2ID_VARIABLE = {word: i for i, word in enumerate(VOCAB_VARIABLE)}
ID2TOKEN_VARIABLE = {i: word for word, i in TOKEN2ID_VARIABLE.items()}
VOWELS = frozenset('aeiou')


def _article(adjective, obj):
    if obj == 'headphones':
        return 'the'
    return 'an' if adjective[0].lower() in VOWELS else 'a'


def _is_short(clause):
    return (len(clause) >= 2 and clause[0] in PEOPLE and
            clause[1] in VERBS)


def _canonical_short(clause):
    """Map one complete short clause to the V3 At-frame event record."""
    assert len(clause) == 12 and clause[-1] == '.' and _is_short(clause)
    assert clause[2] == _article(clause[3], clause[4])
    assert clause[5:7] == ['near', 'the']
    assert clause[8:10] == ['after', 'the']
    return ['At', 'the', clause[7], ',', clause[0], clause[1], 'the',
            clause[3], clause[4], 'before', 'the', clause[10], '.']


class SurfaceMixtureV3Variable(SurfaceMixtureV3):
    def __init__(self, seed=0, short_mass=.55, **kwargs):
        super().__init__(seed=seed, **kwargs)
        if not 0 <= short_mass <= 1:
            raise ValueError('short_mass must lie in [0, 1]')
        self.short_mass = short_mass
        self.vocab = VOCAB_VARIABLE
        self.token2id = TOKEN2ID_VARIABLE
        self.id2token = ID2TOKEN_VARIABLE

    def _tokens(self, prefix):
        return [self.id2token[word] if isinstance(word, int) else word
                for word in prefix]

    def _normalized(self, prefix):
        history, current = _split(self._tokens(prefix))
        out = []
        for clause in history:
            out.extend(_canonical_short(clause) if _is_short(clause)
                       else clause)
        out.extend(current)
        return out

    def _short_query(self, prefix, clause):
        """Use the existing mixture at the actual verb/object decision sites."""
        history, _ = _split(self._normalized(prefix))
        canonical = [word for sentence in history for word in sentence]
        person = clause[0]
        place = PLACES[0]  # Formal placeholder, absent from these rule inputs.
        if len(clause) == 1:
            query = ['At', 'the', place, ',', person]
        else:
            query = ['At', 'the', place, ',', person, clause[1], 'the',
                     clause[3]]
        return super().next_distribution(canonical + query)

    def _article_masses(self, prefix, person, verb, adjective):
        q = self._short_query(prefix, [person, verb, 'a', adjective])
        return {article: sum(mass for obj, mass in q.items()
                             if _article(adjective, obj) == article)
                for article in ('a', 'an', 'the')}

    def next_distribution(self, prefix):
        words = self._tokens(prefix)
        history, clause = _split(words)
        if not clause:
            base = super().next_distribution(self._normalized(words))
            out = {word: (1-self.short_mass)*mass
                   for word, mass in base.items()}
            canonical_history, _ = _split(self._normalized(words))
            subjects = self._subject_distribution(canonical_history)
            for person, mass in subjects.items():
                out[person] = out.get(person, 0.) + self.short_mass*mass
            return {word: mass for word, mass in out.items() if mass > 0}
        if len(clause) == 1 and clause[0] in PEOPLE:
            verbs = self._short_query(words, clause)
            long_subject_mass = (1-self.short_mass)*FRAME_WEIGHTS['subject']
            short_given_person = self.short_mass/(
                self.short_mass+long_subject_mass)
            out = {',': 1-short_given_person}
            out.update({word: short_given_person*mass
                        for word, mass in verbs.items()})
            return {word: mass for word, mass in out.items() if mass > 0}
        if _is_short(clause):
            pos = len(clause)
            if pos == 2:
                out = {article: 0. for article in ('a', 'an', 'the')}
                for adjective in ADJECTIVES:
                    for article, mass in self._article_masses(
                            words, clause[0], clause[1], adjective).items():
                        out[article] += mass/len(ADJECTIVES)
                return {article: mass for article, mass in out.items() if mass}
            if pos == 3:
                masses = {adjective: self._article_masses(
                    words, clause[0], clause[1], adjective)[clause[2]]
                          for adjective in ADJECTIVES}
                total = sum(masses.values())
                return {adjective: mass/total for adjective, mass in
                        masses.items() if mass}
            if pos == 4:
                base = self._short_query(words, clause)
                out = {obj: mass for obj, mass in base.items()
                       if _article(clause[3], obj) == clause[2]}
                total = sum(out.values())
                return {obj: mass/total for obj, mass in out.items()}
            slots = {5: {'near': 1.}, 6: {'the': 1.},
                     7: uniform(PLACES), 8: {'after': 1.},
                     9: {'the': 1.}, 10: uniform(EVENTS), 11: {'.': 1.}}
            return slots[pos]
        return super().next_distribution(self._normalized(words))

    def candidates(self, prefix):
        return super().candidates(self._normalized(prefix))

    def object_candidates(self, prefix):
        return super().object_candidates(self._normalized(prefix))

    def sample_sequence(self, length=72):
        tokens, labels, distributions = [], [], []
        for _ in range(length):
            q = self.next_distribution(tokens)
            _, clause = _split(tokens)
            if len(clause) == 1 and clause[0] in PEOPLE:
                label = 'short_form_or_verb'
            elif _is_short(clause):
                label = 'mixed_object' if len(clause) == 4 else 'short_frame'
            elif clause:
                frame = _frame(clause)
                if len(clause) == _indices(frame)[1]:
                    label = 'mixed_verb'
                elif len(clause) == OBJECT_INDEX[frame]:
                    label = 'mixed_object'
                else:
                    label = 'frame'
            else:
                label = 'frame'
            token = self.rng.choices(list(q), weights=list(q.values()))[0]
            assert q[token] > 0 and abs(sum(q.values())-1) < 1e-10
            tokens.append(token)
            labels.append(label)
            distributions.append(q)
        return [self.token2id[word] for word in tokens], labels, distributions


def render(ids):
    words = [ID2TOKEN_VARIABLE[word] if isinstance(word, int) else word
             for word in ids]
    return ' '.join(words).replace(' ,', ',').replace(' .', '.')
