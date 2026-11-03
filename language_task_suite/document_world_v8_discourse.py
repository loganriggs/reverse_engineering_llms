"""Fact-bearing clause variation over the V5 sampled event world.

The world, question plan, and answers come from V5. New intro/report forms
change the prose carrying the facts; a visible-text transducer normalizes
them before the independent V5 answer parser runs. This is a DGP experiment,
not a claim of web-text realism or a full-token next-word oracle.
"""

from __future__ import annotations

import random

from document_world import _preposition, replay
from document_world_v4_interleaved import draw_interleaved
from document_world_v5_clause_realizations import (
    AnswerSite, VariedDocument, _aside, _choose_form, _event,
    _intro, _recap_prefix, _sentences,
    canonical_visible_prefix as canonical_v5,
)
from document_world_v4_interleaved import (
    answer_from_visible_prefix as canonical_answer,
)
from document_world_v3_multisite import article


INTRO_FORMS = 5
EVENT_FORMS = 8
RECAP_FORMS = 4


def _extended_intro(obj, state, alias, form):
    if form < 3:
        return _intro(obj, state, alias, form)
    owner, color, place = (state[key] for key in ('owner', 'color', 'place'))
    prep = _preposition(place)
    if form == 3:
        return [owner, 'registered', 'the', color, obj, 'under', 'label',
                alias, 'and', 'stored', 'it', prep, 'the', place, '.']
    assert form == 4
    return ['At', 'the', place, ',', owner, 'had', article(color), color,
            obj, ',', 'known', 'as', alias, '.']


def _extended_event(event, previous, source, ref, form):
    if form < 4:
        return _event(event, previous, source, ref, form)
    # Reuse the independently checked action renderer, changing only the
    # attribution clause and its position in the sentence.
    action = _event(event, previous, source, ref, 0)[3:-1]
    if form == 4:
        return action + [',', 'according', 'to', source, '.']
    if form == 5:
        return [source, 'said', ':'] + action + ['.']
    if form == 6:
        return ['A', 'notice', 'from', source, 'stated', 'that'] + action + ['.']
    assert form == 7
    return ['The', 'update', 'came', 'from', source, ':'] + action + ['.']


def _context_act(obj, attr, previous_source):
    assert previous_source is not None
    return ['The', 'new', attr, 'report', 'about', 'the', obj,
            'superseded', previous_source + "'s", 'earlier', 'account', '.']


def realize(world, syntax_entropy=.85, aside_rate=.3, context_rate=.35,
            paragraph_rate=.22, header_rate=.5, intro_interleave_rate=0.,
            extra_event_sentences=None, initial_sentence_plan=None):
    assert all(0 <= rate <= 1 for rate in (
        syntax_entropy, aside_rate, context_rate, paragraph_rate,
        header_rate, intro_interleave_rate))
    rng = random.Random(world['realization_seed'])
    genre = world['genre']
    tokens, sites = [], []
    current = replay(world['initial'], [])
    sources = {}

    def emit(sentence, layout_rng=None):
        layout_rng = rng if layout_rng is None else layout_rng
        if tokens and layout_rng.random() < paragraph_rate:
            tokens.append('\n')
        tokens.extend(sentence)

    extra_layout_rng = (random.Random(world['realization_seed'] ^ 0x5A17)
                        if extra_event_sentences else None)

    if rng.random() < header_rate:
        kind = {'story': 'Story', 'summary': 'Report', 'log': 'File'}[genre]
        emit([kind] + list(str(rng.randrange(100, 1000))) +
             [':', world['topic'], 'notes', '.'])

    def emit_intro(obj):
        state = world['initial'][obj]
        form = _choose_form(rng, genre, INTRO_FORMS, syntax_entropy)
        emit(_extended_intro(obj, state, world['aliases'][obj], form))
        if rng.random() < aside_rate:
            emit(_aside(rng, obj, state['place'], world['topic'], genre))

    # Each object may enter the account at a different point. Introduce it
    # before its first event or question, so the visible prose still supplies
    # every initial fact needed by the independent answer parser. The zero
    # setting retains the previous renderer's random stream exactly.
    introduced = set()
    if initial_sentence_plan is not None:
        assert intro_interleave_rate == 0
        for sentence in initial_sentence_plan:
            emit(sentence)
        introduced.update(world['initial'])
        intro_slots = {obj: -1 for obj in world['initial']}
    elif intro_interleave_rate:
        last_slot = len(world['events']) - 1
        intro_slots = {
            obj: (rng.randrange(last_slot + 1)
                  if rng.random() < intro_interleave_rate else -1)
            for obj in world['initial']}
    else:
        intro_slots = {obj: -1 for obj in world['initial']}

    def introduce_due(slot, required=None):
        due = [obj for obj in world['initial'] if obj not in introduced and
               (intro_slots[obj] <= slot or obj == required)]
        if intro_interleave_rate:
            rng.shuffle(due)
        for obj in due:
            emit_intro(obj)
            introduced.add(obj)

    introduce_due(-1)

    def append_recap(family, obj, attr):
        form = _choose_form(rng, genre, RECAP_FORMS, syntax_entropy)
        prefix, answer, candidates = _recap_prefix(
            family, obj, attr, current, world['aliases'], form)
        if family == 'source_attribution':
            answer = sources[(obj, attr)]
        if tokens and rng.random() < paragraph_rate:
            tokens.append('\n')
        tokens.extend(prefix)
        index = len(tokens)
        tokens.extend((answer, '.'))
        sites.append(AnswerSite(index, family, answer,
                                tuple(candidates), obj))

    for after, family, obj, attr in world['recap_plan']:
        if after == -1:
            introduce_due(-1, obj)
            append_recap(family, obj, attr)
    for j, (event, source, use_alias) in enumerate(zip(
            world['events'], world['sources'], world['use_alias'],
            strict=True)):
        obj, attr = event['object'], event['attribute']
        introduce_due(j, obj)
        ref = world['aliases'][obj] if use_alias else obj
        old_source = sources.get((obj, attr))
        form = _choose_form(rng, genre, EVENT_FORMS, syntax_entropy)
        emit(_extended_event(event, current, source, ref, form))
        current[obj][attr] = event['value']
        sources[(obj, attr)] = source
        if extra_event_sentences:
            for sentence in extra_event_sentences.get(j, ()):
                emit(sentence, extra_layout_rng)
        if old_source is not None and rng.random() < context_rate:
            emit(_context_act(obj, attr, old_source))
        if rng.random() < aside_rate:
            emit(_aside(rng, obj, current[obj]['place'], world['topic'], genre))
        for after, family, recap_obj, recap_attr in world['recap_plan']:
            if after == j:
                introduce_due(j, recap_obj)
                append_recap(family, recap_obj, recap_attr)
    assert len(introduced) == len(world['initial'])
    return VariedDocument(tuple(tokens), tuple(sites), genre, world)


def draw_varied(rng, genre=None, source_delay_rate=1.,
                extra_source_recap_rate=0., **render_kwargs):
    world = draw_interleaved(rng, genre)
    assert 0 <= source_delay_rate <= 1
    assert 0 <= extra_source_recap_rate <= 1
    if extra_source_recap_rate:
        plans = list(world['recap_plan'])
        for j, event in enumerate(world['events']):
            key = (j, 'source_attribution', event['object'],
                   event['attribute'])
            if key not in plans and rng.random() < extra_source_recap_rate:
                plans.append(key)
        world['recap_plan'] = sorted(plans, key=lambda row: row[0])
    delayed = []
    for after, family, obj, attr in world['recap_plan']:
        if (family == 'source_attribution' and source_delay_rate and
                rng.random() < source_delay_rate):
            candidates = [k for k in range(after+1, len(world['events']))
                          if (world['events'][k]['object'],
                              world['events'][k]['attribute']) != (obj, attr)]
            if candidates:
                after = rng.choice(candidates)
        delayed.append((after, family, obj, attr))
    world['recap_plan'] = sorted(delayed, key=lambda row: row[0])
    world['realization_seed'] = rng.randrange(2**63)
    return realize(world, **render_kwargs)


def _normalize_complete(sentence):
    if len(sentence) == 14 and sentence[1:3] == ['registered', 'the']:
        (owner, _, _, color, obj, under, label, alias, and_, stored,
         it, prep, the, place) = sentence
        assert (under, label, and_, stored, it, the) == (
            'under', 'label', 'and', 'stored', 'it', 'the')
        assert prep == _preposition(place)
        return _intro(obj, dict(owner=owner, color=color, place=place),
                      alias, 0)[:-1]
    if len(sentence) == 13 and sentence[:2] == ['At', 'the']:
        (_, _, place, comma, owner, had, det, color, obj, comma2,
         known, as_, alias) = sentence
        assert (comma, had, comma2, known, as_) == (
            ',', 'had', ',', 'known', 'as')
        assert det == article(color)
        return _intro(obj, dict(owner=owner, color=color, place=place),
                      alias, 0)[:-1]
    if sentence[:2] == ['The', 'new']:
        assert (len(sentence) == 11 and sentence[3:6] ==
                ['report', 'about', 'the'] and sentence[7] == 'superseded'
                and sentence[8].endswith("'s") and
                sentence[9:11] == ['earlier', 'account'])
        return None
    if sentence[:2] == ['A', 'notice']:
        assert sentence[2] == 'from' and sentence[4:6] == [
            'stated', 'that']
        return [sentence[3], 'reported', 'that'] + sentence[6:]
    if sentence[:4] == ['The', 'update', 'came', 'from']:
        assert sentence[5] == ':'
        return [sentence[4], 'reported', 'that'] + sentence[6:]
    if len(sentence) > 2 and sentence[1:3] == ['said', ':']:
        return [sentence[0], 'reported', 'that'] + sentence[3:]
    if len(sentence) > 4 and sentence[-4:-2] == [',', 'according']:
        assert sentence[-2] == 'to'
        return [sentence[-1], 'reported', 'that'] + sentence[:-4]
    return sentence


def canonical_visible_prefix(prefix):
    """Use visible tokens only; reject malformed new fact clauses."""
    complete, raw = _sentences([word for word in prefix if word != '\n'])
    normalized = []
    for sentence in complete:
        row = _normalize_complete(sentence)
        if row is not None:
            normalized.extend(row + ['.'])
    normalized.extend(raw)
    return canonical_v5(normalized)


def answer_from_visible_prefix(prefix):
    return canonical_answer(canonical_visible_prefix(prefix))
