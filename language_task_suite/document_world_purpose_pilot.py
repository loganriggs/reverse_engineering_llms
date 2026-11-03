"""Document-purpose renderers over one sampled, auditable event world.

This is a structural DGP pilot. Purpose changes ordering, layout, and the
form in which initial facts and updates are presented. It does not provide
an exact full-token oracle or claim to match natural web text.
"""

from __future__ import annotations

import random

from document_world import NAMES, VALUES, _preposition, replay
from document_world_v4_interleaved import answer_from_visible_prefix as canonical_answer
from document_world_v5_clause_realizations import (
    AnswerSite, VariedDocument, _canonical_query, _recap_prefix, _sentences,
)
from document_world_v3_multisite import ALIASES, article, parse_visible
from document_world_v8_discourse import draw_varied as draw_v8


PURPOSES = ('report', 'correspondence', 'register')


def _intro(obj, state, alias, purpose, rng):
    owner, color, place = (state[k] for k in ('owner', 'color', 'place'))
    if purpose == 'report':
        forms = (
            [owner, 'held', 'the', color, obj, 'at', place, 'under', alias, '.'],
            ['The', obj, 'marked', alias, 'was', color, 'and', 'stood', 'at',
             place, 'with', owner, '.'],
        )
    elif purpose == 'correspondence':
        forms = (
            ['About', 'the', obj, ':', owner, 'had', 'it', 'at', place,
             ';', 'its', 'color', 'was', color, ';', 'we', 'called', 'it',
             alias, '.'],
            ['I', 'found', 'the', color, obj, 'at', place, ';',
             owner, 'had', 'it', 'and', 'used', 'the', 'code', alias, '.'],
        )
    else:
        forms = (
            ['Item', obj, ':', 'holder', owner, ';', 'shade', color,
             ';', 'site', place, ';', 'code', alias, '.'],
            ['Record', alias, ':', obj, '|', owner, '|', color, '|', place, '.'],
        )
    return rng.choice(forms)


def _event(event, source, ref, purpose, rng):
    obj, attr, value = (event[k] for k in ('object', 'attribute', 'value'))
    if purpose == 'report':
        forms = (
            [source, 'reported', 'a', attr, 'change', 'for', ref, ':',
             value, '.'],
            ['The', attr, 'of', ref, 'became', value, ',', 'according',
             'to', source, '.'],
        )
    elif purpose == 'correspondence':
        forms = (
            ['From', source, ':', 'I', 'now', 'have', ref, 'listed', 'as',
             value, 'for', attr, '.'],
            [source, 'writes', ':', 'Please', 'change', 'the', attr, 'of',
             ref, 'to', value, '.'],
        )
    else:
        forms = (
            ['Amendment', source, ':', ref, '/', attr, '=', value, '.'],
            ['Revision', 'for', ref, ':', attr, value, ';', 'filed', 'by',
             source, '.'],
        )
    return rng.choice(forms)


def realize(world, purpose: str):
    if purpose not in PURPOSES:
        raise ValueError(purpose)
    rng = random.Random(world['realization_seed'] ^ 0xD0C0)
    tokens, sites = [], []
    current, sources = replay(world['initial'], []), {}

    def emit(words, break_before=False):
        if tokens and break_before:
            tokens.append('\n')
        tokens.extend(words)

    objs = list(world['initial'])
    rng.shuffle(objs)
    if purpose == 'report':
        emit(['Case', world['topic'], ':', 'opening', 'account', '.'])
    elif purpose == 'correspondence':
        emit(['To', 'the', world['topic'], 'team', ':', 'here', 'is',
              'what', 'we', 'know', '.'])
    else:
        emit(['Register', world['topic'], ':', 'current', 'entries', '.'])
    for obj in objs:
        emit(_intro(obj, world['initial'][obj], world['aliases'][obj],
                    purpose, rng), purpose == 'register')

    def append_recap(family, obj, attr):
        form = rng.randrange(4)
        prefix, answer, candidates = _recap_prefix(
            family, obj, attr, current, world['aliases'], form)
        if family == 'source_attribution':
            answer = sources[(obj, attr)]
        emit(prefix, purpose != 'report')
        index = len(tokens)
        tokens.extend((answer, '.'))
        sites.append(AnswerSite(index, family, answer,
                                tuple(candidates), obj))

    for after, family, obj, attr in world['recap_plan']:
        if after == -1:
            append_recap(family, obj, attr)
    for j, (event, source, use_alias) in enumerate(zip(
            world['events'], world['sources'], world['use_alias'],
            strict=True)):
        ref = (world['aliases'][event['object']] if use_alias
               else event['object'])
        emit(_event(event, source, ref, purpose, rng),
             purpose != 'report' or rng.random() < .35)
        current[event['object']][event['attribute']] = event['value']
        sources[(event['object'], event['attribute'])] = source
        for after, family, obj, attr in world['recap_plan']:
            if after == j:
                append_recap(family, obj, attr)
    return VariedDocument(tuple(tokens), tuple(sites), purpose, world)


def draw_purpose(rng, purpose=None, **v8_draw_kwargs):
    if purpose is None:
        purpose = rng.choice(PURPOSES)
    base = draw_v8(rng, **v8_draw_kwargs)
    return realize(base.latent, purpose)


def _canonical_intro(row):
    if row[0] in NAMES and row[1:3] == ['held', 'the']:
        owner, _, _, color, obj, at, place, under, alias = row
        assert (at, under) == ('at', 'under')
    elif row[:1] == ['The'] and row[2] == 'marked':
        _, obj, _, alias, was, color, and_, stood, at, place, with_, owner = row
        assert (was, and_, stood, at, with_) == (
            'was', 'and', 'stood', 'at', 'with')
    elif row[:2] == ['About', 'the']:
        (_, _, obj, colon, owner, had, it, at, place, semi, its,
         colorword, was, color, semi2, we, called, it2, alias) = row
        assert (colon, had, it, at, semi, its, colorword, was, semi2,
                we, called, it2) == (':', 'had', 'it', 'at', ';', 'its',
                                     'color', 'was', ';', 'we', 'called', 'it')
    elif row[:2] == ['I', 'found']:
        assert row[2] == 'the' and row[5] == 'at'
        color, obj, place = row[3], row[4], row[6]
        assert row[7] == ';' and row[9] == 'had'
        owner, alias = row[8], row[-1]
        assert row[9:] == ['had', 'it', 'and', 'used', 'the', 'code', alias]
    elif row[0] == 'Item':
        (_, obj, colon, holder, owner, semi, shade, color, semi2, site,
         place, semi3, code, alias) = row
        assert (colon, holder, semi, shade, semi2, site, semi3, code) == (
            ':', 'holder', ';', 'shade', ';', 'site', ';', 'code')
    else:
        assert row[0] == 'Record' and row[2] == ':'
        _, alias, _, obj, bar, owner, bar2, color, bar3, place = row
        assert (bar, bar2, bar3) == ('|', '|', '|')
    assert owner in VALUES['owner'] and color in VALUES['color']
    assert place in VALUES['place']
    return ([owner, 'kept', article(color), color, obj,
             _preposition(place), 'the', place, '.'],
            ['The', obj, 'was', 'labeled', alias, '.'])


def _canonical_event(row):
    if row[0] in NAMES and row[1:3] == ['reported', 'a']:
        source, _, _, attr, change, for_, ref, colon, value = row
        assert (change, for_, colon) == ('change', 'for', ':')
    elif row[:1] == ['The'] and row[2] == 'of':
        _, attr, _, ref, became, value, comma, according, to, source = row
        assert (became, comma, according, to) == (
            'became', ',', 'according', 'to')
    elif row[:1] == ['From']:
        (_, source, colon, i, now, have, ref, listed, as_, value,
         for_, attr) = row
        assert (colon, i, now, have, listed, as_, for_) == (
            ':', 'I', 'now', 'have', 'listed', 'as', 'for')
    elif row[0] in NAMES and row[1:3] == ['writes', ':']:
        source, _, _, please, change, the, attr, of, ref, to, value = row
        assert (please, change, the, of, to) == (
            'Please', 'change', 'the', 'of', 'to')
    elif row[:1] == ['Amendment']:
        _, source, colon, ref, slash, attr, equals, value = row
        assert (colon, slash, equals) == (':', '/', '=')
    else:
        assert row[0] == 'Revision'
        (_, for_, ref, colon, attr, value, semi, filed, by, source) = row
        assert (for_, colon, semi, filed, by) == (
            'for', ':', ';', 'filed', 'by')
    assert source in NAMES and attr in VALUES and value in VALUES[attr]
    return source, ref, attr, value


def answer_from_visible_prefix(prefix):
    """Read only the emitted prefix, with no access to the sampled world."""
    complete, raw = _sentences([word for word in prefix if word != '\n'])
    canonical = []
    for row in complete:
        if not row or row[0] in ('Case', 'To', 'Register'):
            continue
        if (row[0] in ('About', 'I', 'Item', 'Record') or
                (row[0] in NAMES and row[1:3] == ['held', 'the']) or
                (row[0] == 'The' and len(row) > 3 and row[2] == 'marked')):
            for sentence in _canonical_intro(row):
                canonical.extend(sentence)
            continue
        if (row[0] in ('From', 'Amendment', 'Revision') or
                (row[0] in NAMES and row[1] in ('reported', 'writes')) or
                (row[0] == 'The' and len(row) == 10 and
                 row[2] == 'of' and row[4] == 'became')):
            source, ref, attr, value = _canonical_event(row)
            state, aliases, _, _ = parse_visible(canonical)
            obj = aliases.get(ref, ref)
            actor = state[obj]['owner']
            action = {'owner': [actor, 'handed', 'the', ref, 'to', value],
                      'color': [actor, 'painted', 'the', ref, value],
                      'place': [actor, 'moved', 'the', ref, 'to', 'the',
                                value]}[attr]
            canonical.extend([source, 'reported', 'that'] + action + ['.'])
            continue
        # Completed recaps do not modify state. Reject unknown scaffolds.
        assert (row[0] in ('The', 'For', 'By', 'At', 'Current', 'Which',
                           'Who', 'What', 'Where') or row[0] in NAMES or
                row[0] in ALIASES), row
    return canonical_answer(canonical + _canonical_query(raw, canonical))
