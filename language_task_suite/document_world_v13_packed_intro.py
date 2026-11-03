"""Pack a typed initial proposition graph into coordinated clauses.

The sampled V10 event world and answer sites stay fixed. The introductory
facts for several objects can share a sentence, and four clause orders are
composed with two coordinators. The visible-prefix checker parses these
clauses back into facts before invoking the existing event/recap checker.
This is a candidate structural intervention, not an accepted DGP.
"""

from __future__ import annotations

import random

from document_world import _preposition
from document_world_v3_multisite import article
from document_world_v5_clause_realizations import _sentences
from document_world_v8_discourse import realize as realize_v8
from document_world_v9_typed_lexicon import ON_WORDS, lexicalize_document
from document_world_v10_fact_acts import (
    answer_from_visible_prefix as answer_v10,
    draw_varied as draw_v10,
    extra_sentences,
)


_PLAN_SALT = 0xD138
_COORDINATORS = ((';',), (',', 'and'))


def _clause(obj, state, alias, form):
    owner, color, place = (state[k] for k in ('owner', 'color', 'place'))
    prep = _preposition(place)
    if form == 0:
        return [owner, 'owned', article(color), color, obj, prep,
                'the', place, 'under', 'the', 'label', alias]
    if form == 1:
        return ['The', color, obj, prep, 'the', place, ',',
                'labeled', alias, ',', 'belonged', 'to', owner]
    if form == 2:
        return ['At', 'the', place, ',', 'the', color, obj,
                'carried', 'the', 'label', alias, 'and', 'belonged', 'to', owner]
    assert form == 3
    return [alias, 'marked', 'the', color, obj, 'that', owner,
            'kept', prep, 'the', place]


def intro_plan(world, pack_rate=.75):
    """Sample an ordered, grouped proposition plan independent of values."""
    assert 0 <= pack_rate <= 1
    rng = random.Random(world['realization_seed'] ^ _PLAN_SALT)
    objects = list(world['initial'])
    rng.shuffle(objects)
    groups, current = [], []
    for obj in objects:
        if current and rng.random() >= pack_rate:
            groups.append(current)
            current = []
        current.append((obj, rng.randrange(4)))
    if current:
        groups.append(current)
    return [(group, [rng.choice(_COORDINATORS)
                     for _ in range(len(group)-1)]) for group in groups]


def intro_sentences(world, pack_rate=.75):
    sentences = []
    for group, joins in intro_plan(world, pack_rate):
        sentence = []
        for i, (obj, form) in enumerate(group):
            if i:
                sentence.extend(joins[i-1])
            clause = _clause(obj, world['initial'][obj],
                             world['aliases'][obj], form)
            if i and clause[0] in ('The', 'At'):
                clause[0] = clause[0].lower()
            sentence.extend(clause)
        sentence.append('.')
        sentences.append(sentence)
    return sentences


def _parse_clause(clause):
    """Read a complete surface clause without a latent world or plan."""
    s = list(clause)
    if len(s) == 12 and s[1] == 'owned' and s[8:11] == [
            'under', 'the', 'label']:
        owner, color, obj, prep, place, alias = (
            s[0], s[3], s[4], s[5], s[7], s[11])
        assert s[2] == article(color) and s[6] == 'the'
    elif len(s) == 13 and s[0] in ('The', 'the') and s[6:8] == [',', 'labeled']:
        color, obj, prep, place, alias, owner = (
            s[1], s[2], s[3], s[5], s[8], s[12])
        assert s[4] == 'the' and s[9:12] == [',', 'belonged', 'to']
    elif len(s) == 15 and s[0] in ('At', 'at') and s[1] == 'the':
        place, color, obj, alias, owner = s[2], s[5], s[6], s[10], s[14]
        prep = 'on' if place in ON_WORDS else _preposition(place)
        assert s[3:5] == [',', 'the'] and s[7:9] == [
            'carried', 'the'] and s[9] == 'label' and s[11:14] == [
            'and', 'belonged', 'to']
    elif len(s) == 11 and s[1:3] == ['marked', 'the']:
        alias, color, obj, owner, prep, place = (
            s[0], s[3], s[4], s[6], s[8], s[10])
        assert s[5] == 'that' and s[7] == 'kept' and s[9] == 'the'
    else:
        raise ValueError(f'Unrecognized introductory clause: {s}')
    assert prep == ('on' if place in ON_WORDS else _preposition(place))
    return obj, dict(owner=owner, color=color, place=place), alias, prep


def _split_intro_sentence(sentence):
    clauses, current = [], []
    i = 0
    while i < len(sentence):
        if sentence[i] == ';':
            clauses.append(current); current = []; i += 1
        elif sentence[i:i+2] == [',', 'and']:
            clauses.append(current); current = []; i += 2
        else:
            current.append(sentence[i]); i += 1
    clauses.append(current)
    return [_parse_clause(clause) for clause in clauses]


def realize(world, pack_rate=.75, lexical_enabled=False, **render_kwargs):
    initial = intro_sentences(world, pack_rate)
    extra = extra_sentences(world) if world.get('fact_act_plan') else None
    canonical = realize_v8(world, initial_sentence_plan=initial,
                           extra_event_sentences=extra, **render_kwargs)
    return lexicalize_document(canonical, world) if lexical_enabled else canonical


def draw_varied(rng, pack_rate=.75, lexical_exponent=None, **kwargs):
    base = draw_v10(rng, lexical_exponent=lexical_exponent, **kwargs)
    world = base.latent
    render_keys = ('syntax_entropy', 'aside_rate', 'context_rate',
                   'paragraph_rate', 'header_rate')
    return realize(world, pack_rate=pack_rate,
                   lexical_enabled=lexical_exponent is not None,
                   **{k: kwargs[k] for k in render_keys if k in kwargs})


def answer_from_visible_prefix(prefix, lexical_enabled=False):
    complete, raw = _sentences([word for word in prefix if word != '\n'])
    normal = []
    seen_non_intro = False
    for sentence in complete:
        if not seen_non_intro:
            # A V8 heading, if present, precedes every introduction.
            if sentence[:1] in (['Story'], ['Report'], ['File']):
                normal.extend(sentence+['.'])
                continue
            try:
                rows = _split_intro_sentence(sentence)
            except (ValueError, AssertionError):
                seen_non_intro = True
            else:
                for obj, state, alias, prep in rows:
                    normal.extend([state['owner'], 'kept',
                                   article(state['color']), state['color'],
                                   obj, prep, 'the', state['place'], ',',
                                   'labeled', alias, '.'])
                continue
        normal.extend(sentence+['.'])
    normal.extend(raw)
    return answer_v10(normal, lexical_enabled=lexical_enabled)
