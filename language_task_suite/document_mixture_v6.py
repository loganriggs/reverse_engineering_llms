"""V5 task mixture with bidirectional score queries and composition holdouts.

This version changes only record/score sampling. Narrative and event-report
arms retain their V5 generators. Score initial-total x increment pairs and
ship x category assignments are partitioned before documents are sampled,
so the composition split has no exact overlap in these task families.
"""

from __future__ import annotations

import random

from document_mixture_v5 import (CATEGORIES, KINDS, SHIPS, TASK_RICH_WEIGHTS,
                                 TEAMS, draw_event, draw_narrative,
                                 draw_record, realize_record, realize_score,
                                 site_oracle)


def score_holdout(initial_left, increment):
    assert 1 <= initial_left <= 5 and 1 <= increment <= 4
    return (initial_left + 2 * increment) % 5 == 0


def record_holdout(ship, category):
    return (SHIPS.index(ship) * 3 + CATEGORIES.index(category)) % 5 == 0


def draw_record_v6(rng, split='train'):
    assert split in ('train', 'iid', 'composition')
    for _ in range(10000):
        world = draw_record(rng)
        held = [record_holdout(ship, value) for ship, value in
                zip(world['ships'], world['categories'], strict=True)]
        query_held = held[world['ships'].index(world['query_ship'])]
        if split == 'composition':
            if query_held:
                return world
        elif not any(held):
            return world
    raise RuntimeError('record split rejection exhausted')


def draw_score_v6(rng, split='train'):
    assert split in ('train', 'iid', 'composition')
    for _ in range(10000):
        left, right = rng.sample(TEAMS, 2)
        a, b = rng.randint(1, 5), rng.randint(1, 5)
        increment = rng.randint(1, 4)
        query_team = rng.choice((left, right))
        world = dict(teams=(left, right), initial=(a, b),
                     increment=increment, query_team=query_team,
                     headline_leak=rng.random() < .5)
        if score_holdout(a, increment) == (split == 'composition'):
            return world
    raise RuntimeError('score split rejection exhausted')


def sample_document(rng: random.Random, kind=None, split='train'):
    kind = kind or rng.choices(KINDS, TASK_RICH_WEIGHTS)[0]
    if kind == 'narrative':
        return draw_narrative(rng)
    if kind == 'event_report':
        return draw_event(rng)
    if kind == 'record_listing':
        return realize_record(draw_record_v6(rng, split))
    if kind == 'score_report':
        return realize_score(draw_score_v6(rng, split))
    raise ValueError(kind)
