"""Sampled, truth-conditional discourse acts over the V8 event world.

V10 adds observations, contrasts, comparisons, and source-history sentences
after events. Their claims are calculated from the same sampled world state;
they do not change state. The visible-prefix answer checker removes only
these known redundant acts before running V8/V9's independent parser.
This is a controlled document-form experiment, not a natural-text oracle.
"""

from __future__ import annotations

import random

from document_world import _preposition, replay
from document_world_v5_clause_realizations import _sentences
from document_world_v8_discourse import (
    answer_from_visible_prefix as answer_v8,
    draw_varied as draw_v8,
    realize as realize_v8,
)
from document_world_v9_typed_lexicon import (
    answer_from_visible_prefix as answer_v9,
    draw_lexicon, lexicalize_document,
)


_ACT_SALT = 0xE71A
_LEX_SALT = 0xC019


def plan_acts(world, rate=.7):
    assert 0 <= rate <= 1
    rng = random.Random(world['realization_seed'] ^ _ACT_SALT)
    sources = {}
    plan = []
    objects = tuple(world['initial'])
    for j, (event, source) in enumerate(zip(
            world['events'], world['sources'], strict=True)):
        key = (event['object'], event['attribute'])
        previous_source = sources.get(key)
        if rng.random() < rate:
            kinds = ('state_fact', 'contrast', 'relation') + (
                ('source_history',) if previous_source is not None else ())
            kind = rng.choice(kinds)
            if kind == 'state_fact':
                obj, attr = rng.choice(objects), rng.choice(
                    ('owner', 'color', 'place'))
                plan.append(dict(after=j, kind=kind, object=obj,
                                 attribute=attr))
            elif kind == 'relation':
                obj, other = rng.sample(objects, 2)
                plan.append(dict(after=j, kind=kind, object=obj,
                                 other=other,
                                 attribute=rng.choice(
                                     ('owner', 'color', 'place'))))
            else:
                plan.append(dict(after=j, kind=kind,
                                 object=event['object'],
                                 attribute=event['attribute']))
        sources[key] = source
    return plan


def _predicate(attr, value):
    if attr == 'owner':
        return ['belonged', 'to', value]
    if attr == 'color':
        return ['was', value]
    return ['was', _preposition(value), 'the', value]


def _render_act(act, before, after, old_sources, new_source, rng,
                form_entropy):
    kind, obj, attr = (act[k] for k in ('kind', 'object', 'attribute'))
    form = rng.randrange(3) if rng.random() < form_entropy else 0
    value = after[obj][attr]
    if kind == 'state_fact':
        if form == 0:
            return ['Meanwhile', ',', 'the', obj] + _predicate(attr, value) + ['.']
        if form == 1:
            return ['The', 'ledger', 'listed', 'the', attr, 'of', obj,
                    'as', value, '.']
        return ['As', 'of', 'that', 'report', ',', obj, 'had', value,
                'as', 'its', attr, '.']
    if kind == 'contrast':
        old = before[obj][attr]
        if form == 0:
            return ['Before', new_source + "'s", 'report', ',', 'the',
                    obj, 'had', old, 'as', 'its', attr, ';', 'afterward',
                    'it', 'had', value, '.']
        if form == 1:
            return ['That', 'change', 'replaced', old, 'with', value,
                    'for', 'the', attr, 'of', obj, '.']
        return ['In', 'the', 'revision', ',', 'the', attr, 'of', obj,
                'went', 'from', old, 'to', value, '.']
    if kind == 'relation':
        other = act['other']
        other_value = after[other][attr]
        if form == 0:
            return ['For', 'comparison', ',', obj, 'had', value,
                    'while', other, 'had', other_value, 'for', attr, '.']
        if form == 1:
            return ['The', 'two', 'entries', 'listed', obj, 'as', value,
                    'and', other, 'as', other_value, 'for', attr, '.']
        return ['Separately', ',', 'the', attr, 'values', 'were',
                obj, ':', value, ';', other, ':', other_value, '.']
    assert kind == 'source_history'
    old_source = old_sources[(obj, attr)]
    if form == 0:
        return ['A', 'prior', attr, 'report', 'on', obj, 'came', 'from',
                old_source, ';', 'this', 'one', 'came', 'from', new_source, '.']
    if form == 1:
        return ['The', 'report', 'from', new_source, 'followed',
                old_source + "'s", 'earlier', attr, 'report', 'on', obj, '.']
    return ['After', old_source + "'s", 'account', ',', new_source,
            'filed', 'the', 'new', attr, 'report', 'on', obj, '.']


def extra_sentences(world, form_entropy=1.):
    assert 0 <= form_entropy <= 1
    rng = random.Random(world['realization_seed'] ^ (_ACT_SALT + 1))
    current = replay(world['initial'], [])
    sources = {}
    by_event = {}
    plans = {act['after']: act for act in world['fact_act_plan']}
    for j, (event, source) in enumerate(zip(
            world['events'], world['sources'], strict=True)):
        before = replay(current, [])
        old_sources = dict(sources)
        current[event['object']][event['attribute']] = event['value']
        sources[(event['object'], event['attribute'])] = source
        if j in plans:
            by_event[j] = [_render_act(plans[j], before, current,
                                       old_sources, source, rng,
                                       form_entropy)]
    return by_event


def realize(world, act_enabled=True, form_entropy=1.,
            lexical_enabled=False, **render_kwargs):
    extra = extra_sentences(world, form_entropy) if act_enabled else None
    canonical = realize_v8(world, extra_event_sentences=extra,
                           **render_kwargs)
    if lexical_enabled:
        return lexicalize_document(canonical, world)
    return canonical


def draw_varied(rng, act_rate=.7, form_entropy=1.,
                lexical_exponent=None, **kwargs):
    base = draw_v8(rng, **kwargs)
    world = base.latent
    world['fact_act_plan'] = plan_acts(world, act_rate)
    if lexical_exponent is not None:
        world['surface_lexicon'] = draw_lexicon(
            random.Random(world['realization_seed'] ^ _LEX_SALT),
            lexical_exponent)
    render_keys = ('syntax_entropy', 'aside_rate', 'context_rate',
                   'paragraph_rate', 'header_rate', 'intro_interleave_rate')
    return realize(world, form_entropy=form_entropy,
                   lexical_enabled=lexical_exponent is not None,
                   **{key: kwargs[key] for key in render_keys if key in kwargs})


def _is_act(sentence):
    return (sentence[:2] in (['Meanwhile', ','], ['The', 'ledger'],
                             ['For', 'comparison'], ['The', 'two'],
                             ['Separately', ','], ['A', 'prior']) or
            sentence[:3] in (['As', 'of', 'that'],
                              ['That', 'change', 'replaced'],
                              ['In', 'the', 'revision'],
                              ['The', 'report', 'from']) or
            (sentence[:1] == ['Before'] and sentence[2:4] ==
             ['report', ',']) or
            (sentence[:1] == ['After'] and sentence[2:4] ==
             ['account', ',']))


def answer_from_visible_prefix(prefix, lexical_enabled=False):
    complete, raw = _sentences([word for word in prefix if word != '\n'])
    retained = [word for sentence in complete if not _is_act(sentence)
                for word in sentence + ['.']]
    retained.extend(raw)
    return (answer_v9(retained) if lexical_enabled
            else answer_v8(retained))
