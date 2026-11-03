"""Task-rich document mixture with natural-text-inspired contrastive sites.

This keeps V14's exact-oracle local-rule stream and mixes it with visible
event documents, colliding records, and score updates. It is a *task-rich
pilot*, not a claim to match FineWeb form or task prevalence. Every scored
semantic target is recoverable from its visible prefix. Only V14 narrative
sites currently have an exact full-prefix next-token distribution.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from document_world_v4_interleaved import (answer_from_visible_prefix as
                                           event_answer, draw_interleaved,
                                           realize as realize_event)
from surface_mixture_v14_conditional import SurfaceMixtureV14Conditional

# Frozen copy of train_v13_attributes.COMMON/DIAGNOSTIC. Importing that
# training script pulls in an unrelated `model` module and collides with the
# original DeepModel loader used by the V5 full-LM trainer.
COMMON = dict(counterbalanced_queries=True, question_rate=.05,
              count_rate=0., quote_rate=.08, source_rate=.15,
              section_rate=.06, detail_rate=.8, scope_question_rate=.08,
              source_object_mass=.15, source_subject_mass=.15,
              bracket_rate=.06, linked_subject_mass=.65,
              linked_object_mass=.65, linked_question_mass=.30,
              linked_query_key_mass=.65)
DIAGNOSTIC = dict(price_fact_rate=.12, price_question_rate=.24,
                  price_object_mass=.65, attribute_fact_rate=.20,
                  attribute_adjective_mass=.80, attribute_object_mass=.80)


SHIPS = ('Aurora', 'Horizon', 'Mariner', 'Compass', 'Voyager', 'Odyssey')
ROOMS = ('6085', '5032', '7140', '8291')
CATEGORIES = ('11B', '9C', '7A', '6D', '4B', '3C')
TEAMS = ('Falcons', 'Bears', 'Tigers', 'Eagles', 'Wolves', 'Sharks')
KINDS = ('narrative', 'event_report', 'record_listing', 'score_report')
TASK_RICH_WEIGHTS = (.30, .35, .20, .15)
NARRATIVE_LABELS = frozenset((
    'mixed_verb', 'mixed_object', 'bracket_close', 'conditional_action',
    'attribute_object', 'attribute_adjective', 'source_answer',
    'scope_answer', 'price_answer', 'question_answer', 'priced_item_object'))


@dataclass(frozen=True)
class Site:
    index: int
    family: str
    target: str
    distribution: dict[str, float]


@dataclass(frozen=True)
class Document:
    tokens: tuple[str, ...]
    sites: tuple[Site, ...]
    kind: str
    latent: dict

    def text(self):
        return (' '.join(self.tokens).replace(' .', '.')
                .replace(' ,', ',').replace(' :', ':'))


def _point_word(n):
    return 'point' if n == 1 else 'points'


def draw_record(rng):
    left, right = rng.sample(SHIPS, 2)
    value_left, value_right = rng.sample(CATEGORIES, 2)
    return dict(ships=(left, right), categories=(value_left, value_right),
                room=rng.choice(ROOMS), query_ship=rng.choice((left, right)),
                order=tuple(rng.sample((0, 1), 2)))


def realize_record(world):
    tokens = []
    for i in world['order']:
        tokens.extend(('The', world['ships'][i], 'listed', 'room', world['room'],
                       'as', 'category', world['categories'][i], '.'))
    ship = world['query_ship']
    tokens.extend(('For', 'the', ship, ',', 'room', world['room'],
                   'was', 'category'))
    site_index = len(tokens)
    answer = world['categories'][world['ships'].index(ship)]
    tokens.extend((answer, '.'))
    return Document(tuple(tokens), (Site(site_index, 'scoped_record', answer,
                                         {answer: 1.}),), 'record_listing', world)


def record_answer(prefix):
    complete, current = [], []
    for token in prefix:
        current.append(token)
        if token == '.':
            complete.append(current[:-1])
            current = []
    facts = {}
    for sentence in complete:
        assert len(sentence) == 8 and sentence[:1] == ['The']
        assert sentence[2:4] == ['listed', 'room']
        assert sentence[5:7] == ['as', 'category']
        facts[(sentence[1], sentence[4])] = sentence[7]
    assert len(current) == 8 and current[:2] == ['For', 'the']
    assert current[3:5] == [',', 'room'] and current[6:8] == ['was', 'category']
    return facts[(current[2], current[5])]


def draw_score(rng):
    left, right = rng.sample(TEAMS, 2)
    a, b = rng.randint(1, 5), rng.randint(1, 5)
    add = rng.randint(1, 3)
    return dict(teams=(left, right), initial=(a, b), increment=add,
                query_team=left, headline_leak=rng.random() < .5)


def realize_score(world):
    left, right = world['teams']
    a, b = world['initial']
    final = a + world['increment']
    tokens = []
    if world['headline_leak']:
        tokens.extend(('Final', 'score', ':', left, str(final), ',',
                       right, str(b), '.'))
    tokens.extend(('The', left, 'had', str(a), _point_word(a), 'and', 'the', right,
                   'had', str(b), _point_word(b), '.'))
    tokens.extend(('Later', ',', 'the', left, 'scored',
                   str(world['increment']), _point_word(world['increment']), '.'))
    tokens.extend(('At', 'the', 'end', ',', 'the', world['query_team'], 'had'))
    site_index = len(tokens)
    answer = str(final if world['query_team'] == left else b)
    tokens.extend((answer, _point_word(int(answer)), '.'))
    return Document(tuple(tokens), (Site(site_index, 'score_update', answer,
                                         {answer: 1.}),), 'score_report', world)


def score_answer(prefix):
    complete, current = [], []
    for token in prefix:
        current.append(token)
        if token == '.':
            complete.append(current[:-1])
            current = []
    state = {}
    for sentence in complete:
        if sentence[0] == 'Final':
            continue  # Headline is a potential copy shortcut, not the oracle.
        if sentence[0] == 'The':
            assert sentence[2] == 'had' and sentence[4] in ('point', 'points')
            assert sentence[5:7] == ['and', 'the']
            assert sentence[8] == 'had' and sentence[10] in ('point', 'points')
            assert sentence[4] == _point_word(int(sentence[3]))
            assert sentence[10] == _point_word(int(sentence[9]))
            state[sentence[1]] = int(sentence[3])
            state[sentence[7]] = int(sentence[9])
        else:
            assert sentence[:3] == ['Later', ',', 'the']
            assert sentence[4] == 'scored'
            assert sentence[6] == _point_word(int(sentence[5]))
            state[sentence[3]] += int(sentence[5])
    assert current[:5] == ['At', 'the', 'end', ',', 'the']
    assert current[6] == 'had'
    return str(state[current[5]])


def draw_narrative(rng, length=192):
    seed = rng.randrange(2**32)
    generator = SurfaceMixtureV14Conditional(
        seed=seed, conditional_continuation_rate=.35, **COMMON, **DIAGNOSTIC)
    ids, labels, distributions = generator.sample_sequence(length)
    words = [generator.id2token[i] for i in ids]
    end = len(words) - 1 - words[::-1].index('.')
    words = words[:end + 1]
    sites = [Site(i, label, words[i], distributions[i])
             for i, label in enumerate(labels[:end + 1])
             if label in NARRATIVE_LABELS]
    return Document(tuple(words), tuple(sites), 'narrative',
                    dict(seed=seed, generator='V14 diagnostic', length=len(words)))


def draw_event(rng):
    world = draw_interleaved(rng)
    original = realize_event(world)
    sites = [Site(site.index, site.family, site.answer,
                  {site.answer: 1.}) for site in original.sites]
    return Document(original.tokens, tuple(sites), 'event_report', world)


def sample_document(rng, kind=None):
    kind = kind or rng.choices(KINDS, TASK_RICH_WEIGHTS)[0]
    if kind == 'narrative':
        return draw_narrative(rng)
    if kind == 'event_report':
        return draw_event(rng)
    if kind == 'record_listing':
        return realize_record(draw_record(rng))
    if kind == 'score_report':
        return realize_score(draw_score(rng))
    raise ValueError(kind)


def site_oracle(document, site):
    prefix = document.tokens[:site.index]
    if document.kind == 'narrative':
        generator = SurfaceMixtureV14Conditional(
            seed=0, conditional_continuation_rate=.35, **COMMON, **DIAGNOSTIC)
        return generator.next_distribution(prefix)
    if document.kind == 'event_report':
        assert event_answer(prefix) == site.target
    elif document.kind == 'record_listing':
        assert record_answer(prefix) == site.target
    elif document.kind == 'score_report':
        assert score_answer(prefix) == site.target
    else:
        raise ValueError(document.kind)
    return {site.target: 1.}
