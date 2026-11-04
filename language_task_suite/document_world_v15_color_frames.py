"""Two sampled color-update frames over V14's known proposition graph.

The extra frames preserve event/source semantics. A visible-text transducer
normalizes completed frames before the existing V10 answer parser runs.
This is a form experiment, not an exact next-token oracle.
"""

from __future__ import annotations

import random

from document_world import replay
from document_world_v10_fact_acts import (
    answer_from_visible_prefix as answer_v10, realize as realize_v10)
from document_world_v14_variable_graph import draw_graph
from document_world_v8_discourse import _extended_event
from document_world_v5_clause_realizations import _sentences


def _color_event(event, previous, source, ref, form):
    actor = previous[event['object']]['owner']
    value = event['value']
    if form == 0:
        return [source, 'described', 'the', ref, 'as', value, 'after',
                actor, 'changed', 'its', 'color', '.']
    assert form == 1
    return ['After', actor, 'changed', 'the', ref, ',', source,
            'reported', 'its', 'new', 'color', 'as', value, '.']


def event_plan(world, novel_frame_rate=1.):
    assert 0 <= novel_frame_rate <= 1
    rng = random.Random(world['realization_seed'] ^ 0xC010F)
    previous = replay(world['initial'], [])
    sentences = []
    for event, source, use_alias in zip(world['events'], world['sources'],
                                        world['use_alias'], strict=True):
        ref = (world['aliases'][event['object']] if use_alias
               else event['object'])
        if event['attribute'] == 'color' and rng.random() < novel_frame_rate:
            sentence = _color_event(event, previous, source, ref,
                                    rng.randrange(2))
        else:
            sentence = _extended_event(event, previous, source, ref,
                                       rng.randrange(8))
        sentences.append(sentence)
        previous[event['object']][event['attribute']] = event['value']
    return sentences


def realize(world, novel_frame_rate=1., lexical_enabled=True,
            **render_kwargs):
    return realize_v10(world, lexical_enabled=lexical_enabled,
                       event_sentence_plan=event_plan(world, novel_frame_rate),
                       **render_kwargs)


def draw_varied(rng, **kwargs):
    frame_rate = kwargs.pop('novel_frame_rate', 1.)
    lexical_enabled = kwargs.pop('lexical_enabled', True)
    world = draw_graph(rng, **{key: value for key, value in kwargs.items()
                               if key in ('n_objects', 'extra_events',
                                          'revisit_rate', 'source_rate',
                                          'world_rate', 'alias_rate',
                                          'source_delay_rate', 'act_rate',
                                          'lexical_exponent')})
    return realize(world, novel_frame_rate=frame_rate,
                   lexical_enabled=lexical_enabled,
                   syntax_entropy=1., aside_rate=.3, context_rate=.35)


def _normalize_color(sentence):
    if (len(sentence) == 11 and sentence[1:3] == ['described', 'the']
            and sentence[4] == 'as' and sentence[6] == 'after'
            and sentence[8:] == ['changed', 'its', 'color']):
        source, _, _, ref, _, value, _, actor, *_ = sentence
        return [source, 'reported', 'that', actor, 'painted', 'the', ref,
                value]
    if (len(sentence) == 13 and sentence[0] == 'After'
            and sentence[2:4] == ['changed', 'the']
            and sentence[5] == ',' and sentence[7:11] ==
            ['reported', 'its', 'new', 'color'] and sentence[11] == 'as'):
        _, actor, _, _, ref, _, source, *_, value = sentence
        return [source, 'reported', 'that', actor, 'painted', 'the', ref,
                value]
    return sentence


def answer_from_visible_prefix(prefix, lexical_enabled=True):
    complete, raw = _sentences([word for word in prefix if word != '\n'])
    normalized = [word for sentence in complete
                  for word in _normalize_color(sentence) + ['.']]
    normalized.extend(raw)
    return answer_v10(normalized, lexical_enabled=lexical_enabled)
