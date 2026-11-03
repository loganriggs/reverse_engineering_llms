"""Interleaved partial initial facts over the sampled V10 document world.

The opening no longer requires one complete introduction per object. It
samples the order and form of four independently meaningful facts (owner,
color, place, label) for each object. A visible-prefix checker accumulates
those facts and constructs a canonical introduction only after all four have
appeared. This checker receives no sampled world state or lexical map.
"""

from __future__ import annotations

import random

from document_world import _preposition
from document_world_v3_multisite import article
from document_world_v5_clause_realizations import _sentences
from document_world_v8_discourse import realize as realize_v8
from document_world_v9_typed_lexicon import lexicalize_document
from document_world_v10_fact_acts import (
    answer_from_visible_prefix as answer_v10,
    draw_varied as draw_v10,
    extra_sentences,
)


_INITIAL_SALT = 0x911F


def plan_initial(world, order_entropy=1., form_entropy=1.):
    assert 0 <= order_entropy <= 1 and 0 <= form_entropy <= 1
    rng = random.Random(world['realization_seed'] ^ _INITIAL_SALT)
    plan = [(obj, attr) for obj in world['initial']
            for attr in ('owner', 'color', 'place', 'alias')]
    if rng.random() < order_entropy:
        rng.shuffle(plan)
    return [dict(object=obj, attribute=attr,
                 form=(rng.randrange(3) if rng.random() < form_entropy
                       else 0)) for obj, attr in plan]


def _render_fact(world, item):
    obj, attr, form = item['object'], item['attribute'], item['form']
    value = (world['aliases'][obj] if attr == 'alias'
             else world['initial'][obj][attr])
    if attr == 'owner':
        if form == 0:
            return [obj, 'belonged', 'to', value, '.']
        if form == 1:
            return [value, 'was', 'listed', 'as', 'the', 'owner',
                    'of', obj, '.']
        return ['The', 'owner', 'recorded', 'for', obj, 'was', value, '.']
    if attr == 'color':
        if form == 0:
            return [obj, 'was', value, '.']
        if form == 1:
            return ['The', 'color', 'recorded', 'for', obj, 'was', value, '.']
        return [value, 'was', 'the', 'listed', 'color', 'of', obj, '.']
    if attr == 'place':
        prep = _preposition(value)
        if form == 0:
            return [obj, 'was', prep, 'the', value, '.']
        if form == 1:
            return ['The', 'location', 'recorded', 'for', obj, 'was',
                    prep, 'the', value, '.']
        return ['The', obj, 'appeared', prep, 'the', value, '.']
    assert attr == 'alias'
    if form == 0:
        return [obj, 'carried', 'the', 'label', value, '.']
    if form == 1:
        return [value, 'was', 'the', 'recorded', 'label', 'for', obj, '.']
    return ['The', 'label', 'on', obj, 'read', value, '.']


def initial_sentences(world):
    return [_render_fact(world, item)
            for item in world['initial_fact_plan']]


def realize(world, initial_enabled=True, act_enabled=True,
            lexical_enabled=False, act_form_entropy=1., **render_kwargs):
    initial = initial_sentences(world) if initial_enabled else None
    extra = extra_sentences(world, act_form_entropy) if act_enabled else None
    canonical = realize_v8(world, initial_sentence_plan=initial,
                           extra_event_sentences=extra, **render_kwargs)
    return (lexicalize_document(canonical, world) if lexical_enabled
            else canonical)


def draw_varied(rng, order_entropy=1., initial_form_entropy=1.,
                act_rate=.8, lexical_exponent=None, **kwargs):
    base = draw_v10(rng, act_rate=act_rate,
                    lexical_exponent=lexical_exponent, **kwargs)
    world = base.latent
    world['initial_fact_plan'] = plan_initial(
        world, order_entropy, initial_form_entropy)
    render_keys = ('syntax_entropy', 'aside_rate', 'context_rate',
                   'paragraph_rate', 'header_rate')
    return realize(world, lexical_enabled=lexical_exponent is not None,
                   **{key: kwargs[key] for key in render_keys if key in kwargs})


def parse_initial_fact(sentence):
    """Return (object, attribute, value, preposition) or None."""
    s = list(sentence)
    if len(s) == 4 and s[1] == 'belonged' and s[2] == 'to':
        return s[0], 'owner', s[3], None
    if len(s) == 8 and s[1:7] == [
            'was', 'listed', 'as', 'the', 'owner', 'of']:
        return s[7], 'owner', s[0], None
    if s[:4] == ['The', 'owner', 'recorded', 'for'] and len(s) == 7:
        assert s[5] == 'was'
        return s[4], 'owner', s[6], None
    if len(s) == 3 and s[1] == 'was':
        return s[0], 'color', s[2], None
    if s[:4] == ['The', 'color', 'recorded', 'for'] and len(s) == 7:
        assert s[5] == 'was'
        return s[4], 'color', s[6], None
    if len(s) == 7 and s[1:6] == [
            'was', 'the', 'listed', 'color', 'of']:
        return s[6], 'color', s[0], None
    if len(s) == 5 and s[1] == 'was' and s[3] == 'the':
        assert s[2] in ('in', 'on')
        return s[0], 'place', s[4], s[2]
    if s[:4] == ['The', 'location', 'recorded', 'for'] and len(s) == 9:
        assert s[5] == 'was' and s[6] in ('in', 'on') and s[7] == 'the'
        return s[4], 'place', s[8], s[6]
    if s[:1] == ['The'] and len(s) == 6 and s[2] == 'appeared':
        assert s[3] in ('in', 'on') and s[4] == 'the'
        return s[1], 'place', s[5], s[3]
    if len(s) == 5 and s[1:4] == ['carried', 'the', 'label']:
        return s[0], 'alias', s[4], None
    if len(s) == 7 and s[1:6] == [
            'was', 'the', 'recorded', 'label', 'for']:
        return s[6], 'alias', s[0], None
    if s[:3] == ['The', 'label', 'on'] and len(s) == 6:
        assert s[4] == 'read'
        return s[3], 'alias', s[5], None
    return None


def answer_from_visible_prefix(prefix, lexical_enabled=False):
    complete, raw = _sentences([word for word in prefix if word != '\n'])
    facts = {}
    retained = []
    for sentence in complete:
        parsed = parse_initial_fact(sentence)
        if parsed is None:
            retained.extend(sentence + ['.'])
            continue
        obj, attr, value, prep = parsed
        slot = facts.setdefault(obj, {})
        if attr in slot:
            raise ValueError(f'Duplicate initial {attr} for {obj}')
        slot[attr] = (value, prep)
        if len(slot) == 4:
            owner = slot['owner'][0]
            color = slot['color'][0]
            place, place_prep = slot['place']
            alias = slot['alias'][0]
            retained.extend([owner, 'kept', article(color), color, obj,
                             place_prep, 'the', place, ',', 'labeled',
                             alias, '.'])
    retained.extend(raw)
    return answer_v10(retained, lexical_enabled=lexical_enabled)
