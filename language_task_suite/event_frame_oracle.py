"""Variable-frame event grammar with a marginalized visible-prefix oracle.

This is an integration fixture, not a realistic language distribution. A
document has a latent genre, but next_distribution marginalizes it. Completed
sentences reveal semantic object facts; later actions use the latest visible
attribute of their object. Several sentence frames express the same event in
different orders and lengths. All surfaces have unique semantic parses.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import random

from surface_mixture_v14_conditional import ACTION


ATTRIBUTES = tuple(ACTION)
PASSIVE = {'clean': 'used', 'dusty': 'cleaned',
           'broken': 'repaired', 'valuable': 'protected',
           'rare': 'inspected', 'heavy': 'moved'}
assert set(PASSIVE) == set(ACTION)
GENRES = ('brief', 'report')
EOS = '<eos>'


@dataclass(frozen=True)
class Candidate:
    tokens: tuple[str, ...]
    probability: float
    kind: str
    obj: str
    attribute: str
    frame: int
    person: str | None = None
    action_position: int | None = None


@dataclass(frozen=True)
class ActionSite:
    position: int
    obj: str
    attribute: str
    answer: str


@dataclass(frozen=True)
class Sample:
    tokens: tuple[str, ...]
    genre: str
    actions: tuple[ActionSite, ...]
    kinds: tuple[str, ...]
    events: tuple[tuple[str, str, str, str | None], ...]


class EventFrameOracle:
    def __init__(self, seed=0, n_objects=16, n_people=8,
                 frequency_exponent=.5, n_sentences=8,
                 variable_frames=True):
        if not 1 <= n_objects <= 146 or not 1 <= n_people <= 97:
            raise ValueError('Invalid typed lexicon size')
        if frequency_exponent < 0 or n_sentences < 2:
            raise ValueError('Invalid frequency or document length')
        bank = json.loads(Path(__file__).with_name(
            'typed_lexicon_v17.json').read_text())['categories']
        self.objects = tuple(row['word'] for row in bank['objects'][:n_objects])
        self.people = tuple(row['word'] for row in bank['people'][:n_people])
        self.object_prob = self._prior(bank['objects'][:n_objects],
                                       frequency_exponent)
        self.person_prob = self._prior(bank['people'][:n_people],
                                       frequency_exponent)
        self.n_sentences = n_sentences
        self.variable_frames = variable_frames
        self.rng = random.Random(seed)
        self.vocab = tuple(dict.fromkeys(
            self.objects + self.people + ATTRIBUTES + tuple(ACTION.values()) +
            tuple(PASSIVE.values()) +
            ('The', 'is', '.', 'A', ',', 'the', 'as', 'for', 'As',
             'was', 'noted', 'sees', 'and', 'it', 'by',
             'After', 'examining', EOS)))
        self.token2id = {token: i for i, token in enumerate(self.vocab)}
        self.id2token = dict(enumerate(self.vocab))

    @staticmethod
    def _prior(rows, exponent):
        weights = [(row['dev_pos_count'] + 1)**exponent for row in rows]
        total = sum(weights)
        return tuple(weight/total for weight in weights)

    def _frames(self, genre, kind):
        if not self.variable_frames:
            return (1., 0., 0.)
        if kind == 'fact':
            return ((.7, .1, .2) if genre == 'brief' else (.2, .2, .6))
        return ((.7, .2, .1) if genre == 'brief' else (.1, .4, .5))

    @lru_cache(maxsize=2048)
    def candidates(self, state: tuple[tuple[str, str], ...], genre: str):
        if genre not in GENRES:
            raise ValueError(genre)
        facts = dict(state)
        p_fact = 1. if not facts else (.6 if genre == 'brief' else .5)
        rows = []
        for obj, p_obj in zip(self.objects, self.object_prob):
            for attr in ATTRIBUTES:
                for frame, p_frame in enumerate(self._frames(genre, 'fact')):
                    if not p_frame:
                        continue
                    surfaces = (
                        ('The', obj, 'is', attr, '.'),
                        ('A', attr, obj, 'was', 'noted', '.'),
                        ('As', 'for', 'the', obj, ',', 'it', 'is', attr, '.'),
                    )
                    rows.append(Candidate(surfaces[frame],
                        p_fact*p_obj/len(ATTRIBUTES)*p_frame,
                        'fact', obj, attr, frame))
        if facts:
            obj_total = sum(self.object_prob[self.objects.index(obj)]
                            for obj in facts)
            for obj, attr in facts.items():
                p_obj = self.object_prob[self.objects.index(obj)]/obj_total
                action = ACTION[attr]
                passive = PASSIVE[attr]
                for person, p_person in zip(self.people, self.person_prob):
                    for frame, p_frame in enumerate(self._frames(
                            genre, 'action')):
                        if not p_frame:
                            continue
                        surfaces = (
                            (person, 'sees', 'the', obj, 'and', action,
                             'it', '.'),
                            ('The', obj, 'is', passive, 'by', person, '.'),
                            ('After', 'examining', 'the', obj, ',', person,
                             action, 'it', '.'),
                        )
                        tokens = surfaces[frame]
                        answer = passive if frame == 1 else action
                        rows.append(Candidate(tokens,
                            (1.-p_fact)*p_obj*p_person*p_frame,
                            'action', obj, attr, frame, person,
                            tokens.index(answer)))
        assert abs(sum(row.probability for row in rows)-1.) < 1e-10
        assert len({row.tokens for row in rows}) == len(rows)
        return tuple(rows)

    @staticmethod
    def _split(prefix):
        sentences, current = [], []
        for token in prefix:
            if token == EOS:
                if current:
                    raise ValueError('EOS inside sentence')
                return tuple(sentences), (), True
            current.append(token)
            if token == '.':
                sentences.append(tuple(current))
                current = []
        return tuple(sentences), tuple(current), False

    def _replay(self, sentences, genre):
        state, likelihood = {}, 1.
        for sentence in sentences:
            options = [row for row in self.candidates(
                tuple(sorted(state.items())), genre) if row.tokens == sentence]
            if len(options) != 1:
                return None, 0.
            row = options[0]
            likelihood *= row.probability
            if row.kind == 'fact':
                state[row.obj] = row.attribute
        return tuple(sorted(state.items())), likelihood

    def next_distribution(self, prefix):
        prefix = tuple(self.id2token[x] if isinstance(x, int) else x
                       for x in prefix)
        sentences, current, ended = self._split(prefix)
        if ended:
            return {}
        if len(sentences) == self.n_sentences:
            if current:
                raise ValueError('Tokens after last sentence')
            return {EOS: 1.}
        out = {}
        for genre in GENRES:
            state, history_likelihood = self._replay(sentences, genre)
            if state is None:
                continue
            for row in self.candidates(state, genre):
                if row.tokens[:len(current)] != current:
                    continue
                if len(current) >= len(row.tokens):
                    continue
                next_token = row.tokens[len(current)]
                out[next_token] = out.get(next_token, 0.) + (
                    .5*history_likelihood*row.probability)
        total = sum(out.values())
        if total <= 0:
            raise ValueError('Prefix outside generator support')
        return {token: mass/total for token, mass in out.items()}

    def sample_document(self):
        genre = self.rng.choice(GENRES)
        state, tokens, actions, kinds, events = {}, [], [], [], []
        for _ in range(self.n_sentences):
            p_fact = 1. if not state else (.6 if genre == 'brief' else .5)
            kind = 'fact' if not state or self.rng.random() < p_fact else 'action'
            if kind == 'fact':
                obj = self.rng.choices(self.objects,
                                       weights=self.object_prob)[0]
                attr = self.rng.choice(ATTRIBUTES)
                person = None
            else:
                known = tuple(state)
                known_weights = [self.object_prob[self.objects.index(obj)]
                                 for obj in known]
                obj = self.rng.choices(known, weights=known_weights)[0]
                attr = state[obj]
                person = self.rng.choices(self.people,
                                          weights=self.person_prob)[0]
            frame = self.rng.choices(range(3),
                    weights=self._frames(genre, kind))[0]
            rows = self.candidates(tuple(sorted(state.items())), genre)
            matched = [row for row in rows if
                       (row.kind, row.obj, row.attribute, row.frame,
                        row.person) == (kind, obj, attr, frame, person)]
            assert len(matched) == 1
            row = matched[0]
            if row.kind == 'action':
                actions.append(ActionSite(len(tokens)+row.action_position,
                                          row.obj, row.attribute,
                                          row.tokens[row.action_position]))
            tokens.extend(row.tokens)
            kinds.append(row.kind)
            events.append((kind, obj, attr, person))
            if row.kind == 'fact':
                state[row.obj] = row.attribute
        tokens.append(EOS)
        return Sample(tuple(tokens), genre, tuple(actions), tuple(kinds),
                      tuple(events))

    def direct_document_probability(self, tokens):
        sentences, current, ended = self._split(tokens)
        if not ended or current or len(sentences) != self.n_sentences:
            return 0.
        return sum(.5*self._replay(sentences, genre)[1] for genre in GENRES)

    def render(self, tokens):
        text = ''
        for token in tokens:
            if token == EOS:
                break
            if token in ('.', ','):
                text = text.rstrip() + token
            else:
                text += ('' if not text or text[-1].isspace() else ' ') + token
        return text
