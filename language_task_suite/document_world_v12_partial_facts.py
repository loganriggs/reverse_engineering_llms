"""Sample a subset of initially visible facts, retaining supported queries.

Only facts needed by an event or a prediction site are mandatory. Other
initial attributes are sampled independently. The visible-prefix checker
fills unobserved slots with inert canonical placeholders for compatibility
with the earlier event parser; it never scores those placeholders as answers.
No latent world or fact-selection plan is read by the checker.
"""

from __future__ import annotations

import random

from document_world_v10_fact_acts import draw_varied as draw_v10
from document_world_v11_initial_facts import (
    initial_sentences, parse_initial_fact, plan_initial,
    realize as realize_v11)
from document_world_v10_fact_acts import answer_from_visible_prefix as answer_v10
from document_world_v3_multisite import article
from document_world_v5_clause_realizations import _sentences


_SELECTION_SALT = 0xC1A5


def required_initial_facts(world):
    required = set()
    for event, use_alias in zip(world['events'], world['use_alias'],
                                strict=True):
        obj = event['object']
        required.add((obj, 'owner'))
        if use_alias:
            required.add((obj, 'alias'))
    for after, family, obj, attr in world['recap_plan']:
        if family == 'alias_identity':
            # Keep all candidate bindings visible. This also makes an
            # upstream alias swap a supported edit rather than a hidden
            # change that the reader could not infer.
            required.update((other, 'alias') for other in world['initial'])
        elif family.startswith('world_'):
            target = family.removeprefix('world_')
            if not any(event['object'] == obj and
                       event['attribute'] == target
                       for event in world['events'][:after+1]):
                required.add((obj, target))
    return required


def plan_partial(world, optional_fact_rate=.35,
                 order_entropy=1., form_entropy=1.):
    assert 0 <= optional_fact_rate <= 1
    required = required_initial_facts(world)
    rng = random.Random(world['realization_seed'] ^ _SELECTION_SALT)
    return [item for item in plan_initial(world, order_entropy,
                                          form_entropy)
            if (item['object'], item['attribute']) in required or
            rng.random() < optional_fact_rate]


def realize(world, lexical_enabled=False, **kwargs):
    kwargs.setdefault('aside_rate', 0.)
    kwargs.setdefault('context_rate', 0.)
    if kwargs['aside_rate'] or kwargs['context_rate']:
        raise ValueError('Partial-fact V12 requires zero aside/context rates; '
                         'those V8 clauses can reveal omitted values')
    return realize_v11(world, lexical_enabled=lexical_enabled,
                       act_enabled=False, **kwargs)


def draw_varied(rng, optional_fact_rate=.35, lexical_exponent=None,
                **kwargs):
    kwargs.setdefault('aside_rate', 0.)
    kwargs.setdefault('context_rate', 0.)
    if kwargs['aside_rate'] or kwargs['context_rate']:
        raise ValueError('Partial-fact V12 requires zero aside/context rates')
    base = draw_v10(rng, act_rate=0., lexical_exponent=lexical_exponent,
                    **kwargs)
    world = base.latent
    world['initial_fact_plan'] = plan_partial(
        world, optional_fact_rate)
    render_keys = ('syntax_entropy', 'aside_rate', 'context_rate',
                   'paragraph_rate', 'header_rate')
    return realize(world, lexical_enabled=lexical_exponent is not None,
                   **{key: kwargs[key] for key in render_keys if key in kwargs})


def answer_from_visible_prefix(prefix, lexical_enabled=False):
    complete, raw = _sentences([word for word in prefix if word != '\n'])
    retained, facts = [], {}
    flushed = False

    def flush():
        nonlocal flushed
        if flushed:
            return
        for index, (obj, slot) in enumerate(facts.items()):
            owner = slot.get('owner', ('Maya', None))[0]
            color = slot.get('color', ('red', None))[0]
            place, prep = slot.get('place', ('box', 'in'))
            alias = slot.get('alias', (f'UNSEENLABEL{index}', None))[0]
            retained.extend([owner, 'kept', article(color), color, obj,
                             prep, 'the', place, ',', 'labeled', alias, '.'])
        flushed = True

    for sentence in complete:
        parsed = parse_initial_fact(sentence)
        if parsed is not None:
            if flushed:
                raise ValueError('Initial fact after event or query')
            obj, attr, value, prep = parsed
            slot = facts.setdefault(obj, {})
            if attr in slot:
                raise ValueError(f'Duplicate {attr} for {obj}')
            slot[attr] = (value, prep)
            continue
        if sentence[:1] not in (['Story'], ['Report'], ['File']):
            flush()
        retained.extend(sentence + ['.'])
    flush()
    retained.extend(raw)
    return answer_v10(retained, lexical_enabled=lexical_enabled)
