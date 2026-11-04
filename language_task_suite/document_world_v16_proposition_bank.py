"""Broader sampled discourse propositions over a fixed event/world graph.

One extra sentence is emitted at each V10 fact-act slot. The new bank
expresses persistence, conjunction, comparison, revision order, alias
identity, and source state. Claims are calculated from the same event ledger;
they do not update it. A prefix parser recognizes and removes complete
redundant acts before the existing independent answer parser runs.
"""

from __future__ import annotations

from collections import Counter
import random

from document_world import _preposition, replay
from document_world_v5_clause_realizations import _sentences
from document_world_v8_discourse import (
    answer_from_visible_prefix as answer_v8, realize as realize_v8)
from document_world_v9_typed_lexicon import (
    answer_from_visible_prefix as answer_v9, lexicalize_document)
from document_world_v14_variable_graph import draw_graph


KINDS = ('persistence', 'conjunction', 'comparison', 'revision_order',
         'alias_identity', 'source_state')


def _ordinal(n):
    words = ('first', 'second', 'third', 'fourth', 'fifth', 'sixth',
             'seventh', 'eighth', 'ninth', 'tenth', 'eleventh', 'twelfth')
    assert 1 <= n <= len(words)
    return words[n-1]


def _sentence(kind, event, source, before, after, world, counts, rng):
    obj, attr = event['object'], event['attribute']
    objects = tuple(world['initial'])
    if kind == 'persistence':
        other = rng.choice([item for item in objects if item != obj])
        other_attr = rng.choice(('owner', 'color', 'place'))
        assert before[other][other_attr] == after[other][other_attr]
        value = after[other][other_attr]
        return ['Throughout', 'the', 'revision', ',', 'the', other,
                'kept', value, 'as', 'its', other_attr, '.']
    if kind == 'conjunction':
        target = rng.choice(objects)
        state = after[target]
        return ['At', 'that', 'point', ',', state['owner'], 'held',
                'the', state['color'], target, _preposition(state['place']),
                'the', state['place'], '.']
    if kind == 'comparison':
        left, right = rng.sample(objects, 2)
        selected = rng.choice(('owner', 'color', 'place'))
        relation = ('same' if after[left][selected] == after[right][selected]
                    else 'different')
        return ['Compared', 'with', 'the', right, ',', 'the', left,
                'had', 'the', relation, selected, '.']
    if kind == 'revision_order':
        position = counts[(obj, attr)]
        return ['This', 'was', 'the', _ordinal(position), 'reported',
                'change', 'to', 'the', attr, 'of', 'the', obj, '.']
    if kind == 'alias_identity':
        target = rng.choice(objects)
        return ['Under', 'the', 'label', world['aliases'][target], ',',
                'the', 'record', 'still', 'referred', 'to', 'the', target, '.']
    assert kind == 'source_state'
    return ['For', 'the', attr, 'of', 'the', obj, ',', 'the',
            'latest', 'named', 'source', 'was', source, '.']


def extra_sentences(world):
    rng = random.Random(world['realization_seed'] ^ 0xB16A)
    current = replay(world['initial'], [])
    counts = Counter()
    slots = {act['after'] for act in world['fact_act_plan']}
    result = {}
    for j, (event, source) in enumerate(zip(
            world['events'], world['sources'], strict=True)):
        before = replay(current, [])
        key = event['object'], event['attribute']
        counts[key] += 1
        current[event['object']][event['attribute']] = event['value']
        if j in slots:
            kind = rng.choice(KINDS)
            result[j] = (kind, _sentence(kind, event, source, before,
                                         current, world, counts, rng))
    return result


def realize(world, lexical_enabled=True, **render_kwargs):
    extras = extra_sentences(world)
    canonical = realize_v8(
        world, extra_event_sentences={j: [sentence]
                                      for j, (_, sentence) in extras.items()},
        **render_kwargs)
    return lexicalize_document(canonical, world) if lexical_enabled else canonical


def draw_varied(rng, **kwargs):
    world = draw_graph(rng, **kwargs)
    return realize(world, syntax_entropy=1., aside_rate=.3,
                   context_rate=.35)


def _is_bank_act(sentence):
    return (sentence[:3] == ['Throughout', 'the', 'revision'] or
            sentence[:3] == ['At', 'that', 'point'] or
            sentence[:2] == ['Compared', 'with'] or
            sentence[:3] == ['This', 'was', 'the'] or
            sentence[:3] == ['Under', 'the', 'label'] or
            (sentence[:2] == ['For', 'the'] and
             sentence[-4:-1] == ['named', 'source', 'was']))


def answer_from_visible_prefix(prefix, lexical_enabled=True):
    complete, raw = _sentences([word for word in prefix if word != '\n'])
    retained = [word for sentence in complete if not _is_bank_act(sentence)
                for word in sentence + ['.']]
    retained.extend(raw)
    return (answer_v9(retained) if lexical_enabled
            else answer_v8(retained))
