"""Variable entity/event graph over the V10 typed document semantics.

This is a data-law intervention. The number of entities, event acts, and
revisits can vary; the V10 renderer and visible-prefix rule checker remain
unchanged. It is a form/semantic candidate, not an accepted language DGP.
"""

from __future__ import annotations

import random

from document_world import ATTRS, OBJECTS, VALUES, _new_value, replay
from document_world_v3_multisite import ALIASES, GENRES, TOPICS
from document_world_v9_typed_lexicon import draw_lexicon
from document_world_v10_fact_acts import plan_acts, realize as realize_v10


def draw_graph(rng: random.Random, n_objects=(2, 6), extra_events=(2, 10),
               revisit_rate=.55, source_rate=.5, world_rate=.18,
               alias_rate=.65, source_delay_rate=.5, act_rate=.8,
               lexical_exponent=0.):
    assert 2 <= n_objects[0] <= n_objects[1] <= len(OBJECTS)
    assert 0 <= extra_events[0] <= extra_events[1]
    assert all(0 <= x <= 1 for x in (revisit_rate, source_rate,
                                    world_rate, alias_rate,
                                    source_delay_rate, act_rate))
    count = rng.randint(*n_objects)
    objects = rng.sample(OBJECTS, count)
    initial = {obj: {attr: rng.choice(VALUES[attr]) for attr in ATTRS}
               for obj in objects}
    query_object = rng.choice(objects)
    query_attribute = rng.choice(ATTRS)
    current = replay(initial, [])
    events = []

    def add_event(obj, attr, exclude=()):
        value = _new_value(rng, attr, current[obj][attr], exclude)
        events.append(dict(object=obj, attribute=attr, value=value))
        current[obj][attr] = value

    # A relevant transition, variable revisits, and a final other-object
    # distractor preserve the core source/state challenge across graph sizes.
    add_event(query_object, query_attribute)
    for _ in range(rng.randint(*extra_events)):
        if rng.random() < revisit_rate:
            obj = rng.choice(tuple(event['object'] for event in events))
        else:
            obj = rng.choice(objects)
        add_event(obj, rng.choice(ATTRS))
    other = rng.choice([obj for obj in objects if obj != query_object])
    add_event(other, query_attribute,
              exclude=(current[query_object][query_attribute],))

    world = dict(initial=initial, events=events,
                 query_object=query_object,
                 query_attribute=query_attribute)
    world['aliases'] = dict(zip(objects, rng.sample(ALIASES, count)))
    world['topic'] = rng.choice(TOPICS)
    world['genre'] = rng.choice(GENRES)
    world['alias_query_object'] = rng.choice(objects)
    sources, use_alias = [], []
    state = replay(initial, [])
    for event in events:
        from document_world import NAMES
        actor = state[event['object']]['owner']
        sources.append(rng.choice([name for name in NAMES if name != actor]))
        use_alias.append(rng.random() < .5)
        state[event['object']][event['attribute']] = event['value']
    use_alias[0] = True
    world['sources'], world['use_alias'] = sources, use_alias

    plans = []
    if rng.random() < alias_rate:
        plans.append((-1, 'alias_identity', rng.choice(objects), None))
    for j, event in enumerate(events):
        if rng.random() < world_rate:
            plans.append((j, 'world_'+rng.choice(ATTRS),
                          rng.choice(objects), None))
        if rng.random() < source_rate:
            after = j
            if rng.random() < source_delay_rate:
                possible = [k for k in range(j+1, len(events))
                            if (events[k]['object'], events[k]['attribute'])
                            != (event['object'], event['attribute'])]
                if possible:
                    after = rng.choice(possible)
            plans.append((after, 'source_attribution', event['object'],
                          event['attribute']))
    if not plans:
        plans.append((len(events)-1, 'world_'+rng.choice(ATTRS),
                      rng.choice(objects), None))
    world['recap_plan'] = sorted(set(plans))
    world['realization_seed'] = rng.randrange(2**63)
    world['fact_act_plan'] = plan_acts(world, act_rate)
    world['surface_lexicon'] = draw_lexicon(
        random.Random(world['realization_seed'] ^ 0xC019), lexical_exponent)
    world['graph_controls'] = dict(n_objects=count,
                                  extra_events=len(events)-2,
                                  revisit_rate=revisit_rate,
                                  source_rate=source_rate,
                                  act_rate=act_rate)
    return world


def draw_varied(rng, **kwargs):
    world = draw_graph(rng, **kwargs)
    return realize_v10(world, lexical_enabled=True,
                       syntax_entropy=1., aside_rate=.3,
                       context_rate=.35)
