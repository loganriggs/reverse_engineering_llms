"""Typed semantic and discourse plans for a future sampled language DGP.

This module deliberately emits *plans*, not prose. It makes document form,
rule coupling, and visible proposition dependencies explicit so a later
renderer can be audited against them. Form weights are experimental knobs;
the 32-document FineWeb opening panel is too small to calibrate priors.
"""

from __future__ import annotations

import random
import copy
from collections import Counter

from document_world import NAMES, OBJECTS, VALUES
from document_world_v3_multisite import ALIASES


FORMS = ('report', 'personal', 'reference', 'quoted_feature',
         'procedure', 'collection', 'commercial')
ATTRS = tuple(VALUES)
STEP_ACTIONS = ('inspect', 'move', 'combine', 'record', 'check')
# Counts describe the document's external record, so typed lexical
# substitution can change an object (map -> motorcycle, for example)
# without creating an impossible physical-unit claim.
COUNT_UNITS = ('copies', 'reports', 'entries')

# These are qualitative coverage settings, not rates inferred from 32 docs.
MOVE_WEIGHTS = {
    'report': dict(update=4, attribute=3, compare=1, revisit=1,
                   measure=1, question=.5, intro=1),
    'personal': dict(update=2, attribute=1, compare=1, revisit=2,
                     measure=.5, question=1, intro=1),
    'reference': dict(update=.5, attribute=.5, compare=3, revisit=2,
                      measure=1, question=1, intro=1),
    'quoted_feature': dict(update=2, attribute=4, compare=1,
                           revisit=2, question=1, intro=1),
    'procedure': dict(step=5, measure=3, revisit=1, question=.5,
                      intro=1),
    'collection': dict(scope=3, record=3, revisit=1, question=1,
                       intro=1),
    'commercial': dict(record=3, measure=3, compare=1, revisit=1,
                       question=1, intro=1),
}
REQUIRED = {
    'report': 'update', 'personal': 'update',
    'reference': 'compare', 'quoted_feature': 'attribute',
    'procedure': 'step', 'collection': 'scope',
    'commercial': 'measure',
}
OPENINGS = {
    'report': ('headline', 'dateline', 'lead_claim', 'change', 'profile'),
    'personal': ('question', 'personal_scene', 'lead_claim',
                 'reflection', 'headline'),
    'reference': ('question', 'definition', 'enumeration',
                  'measure', 'headline'),
    'quoted_feature': ('series_context', 'byline', 'attribution'),
    'procedure': ('item_list', 'headline', 'ordered_step'),
    'collection': ('scene', 'lead_claim', 'item'),
    'commercial': ('headline', 'specification', 'review'),
}


def sample_plan(rng: random.Random, form=None, form_weights=None,
                n_moves=None, coupling_rate=.5, question_rate=1.,
                shared_attribute_rate=.6):
    """Sample a truth-conditional graph and a legal visible move order."""
    if form_weights is None:
        form_weights = {name: 1. for name in FORMS}
    assert set(form_weights) == set(FORMS)
    assert all(weight >= 0 for weight in form_weights.values())
    assert sum(form_weights.values()) > 0
    assert (0 <= coupling_rate <= 1 and question_rate >= 0 and
            0 <= shared_attribute_rate <= 1)
    if form is None:
        form = rng.choices(FORMS, [form_weights[k] for k in FORMS])[0]
    if form not in FORMS:
        raise ValueError(form)
    if n_moves is None:
        n_moves = rng.randint(8, 14)
    assert n_moves >= 3
    opening_role = rng.choice(OPENINGS[form])

    entities = rng.sample(OBJECTS, rng.randint(3, 5))
    aliases = dict(zip(entities, rng.sample(ALIASES, len(entities))))
    initial = {}
    for entity in entities:
        initial[entity] = {}
        for attr in ATTRS:
            earlier_entities = tuple(e for e in initial if e != entity)
            if earlier_entities and rng.random() < shared_attribute_rate:
                earlier = rng.choice(earlier_entities)
                initial[entity][attr] = initial[earlier][attr]
            else:
                initial[entity][attr] = rng.choice(VALUES[attr])
    nodes, moves = [], []
    first_fact = {}

    def node(kind, **fields):
        result = dict(id=len(nodes), kind=kind, **fields)
        assert all(dep < result['id'] for dep in result.get('deps', ()))
        nodes.append(result)
        return result['id']

    for entity in entities:
        for attr in ATTRS:
            first_fact[(entity, attr)] = node(
                'fact', entity=entity, attr=attr,
                value=initial[entity][attr], deps=())
    current = dict(first_fact)
    introduced = set()
    shown = set()
    updates, claims, measures, steps, records = [], [], [], [], []
    latest_claim, latest_measure, latest_record = {}, {}, {}
    scope = 0

    def emit(kind, refs=(), **fields):
        refs = tuple(refs)
        assert all(dep in shown for ref in refs
                   for dep in nodes[ref].get('deps', ()))
        moves.append(dict(kind=kind, refs=refs, scope=scope, **fields))
        shown.update(refs)

    def introduce(entity):
        assert entity not in introduced
        refs = tuple(first_fact[(entity, attr)] for attr in ATTRS)
        introduced.add(entity)
        emit('intro', refs, entity=entity, alias=aliases[entity])

    def update():
        entity = rng.choice(tuple(sorted(introduced)))
        attr = rng.choice(ATTRS)
        previous = current[(entity, attr)]
        old = nodes[previous]['value']
        new = rng.choice([v for v in VALUES[attr] if v != old])
        idx = node('update', entity=entity, attr=attr, old=old,
                   value=new, ordinal=len(updates)+1,
                   deps=(previous,))
        current[(entity, attr)] = idx
        updates.append(idx)
        emit('update', (idx,))

    def attribute():
        event = rng.choice(updates)
        source = rng.choice(NAMES)
        idx = node('source_claim', value=source, deps=(event,))
        claims.append(idx)
        latest_claim[event] = idx
        emit('attribute', (idx,))

    def compare():
        entity, other = rng.sample(sorted(introduced), 2)
        attr = rng.choice(ATTRS)
        left, right = current[(entity, attr)], current[(other, attr)]
        relation = ('same' if nodes[left]['value'] == nodes[right]['value']
                    else 'different')
        idx = node('comparison', entity=entity, other=other, attr=attr,
                   value=relation, deps=(left, right))
        emit('compare', (idx,))

    def measure():
        entity = rng.choice(tuple(sorted(introduced)))
        unit = rng.choice(COUNT_UNITS)
        idx = node('measure', entity=entity, unit=unit,
                   value=rng.randint(1, 30),
                   ordinal=len(measures)+1,
                   deps=(current[(entity, rng.choice(ATTRS))],))
        measures.append(idx)
        latest_measure[(entity, unit)] = idx
        emit('measure', (idx,))

    def step():
        entity = rng.choice(tuple(sorted(introduced)))
        dependencies = (steps[-1],) if steps else (
            current[(entity, rng.choice(ATTRS))],)
        idx = node('step', entity=entity, value=rng.choice(STEP_ACTIONS),
                   ordinal=len(steps)+1, deps=dependencies)
        steps.append(idx)
        emit('step', (idx,))

    def shift_scope():
        nonlocal scope
        scope += 1
        emit('scope', (), scope_name=f'part_{scope}')

    def record():
        entity = rng.choice(tuple(sorted(introduced)))
        key = rng.choice(('status', 'category', 'code'))
        candidates = {'status': ('active', 'closed', 'pending'),
                      'category': ('standard', 'special'),
                      'code': tuple(aliases.values())}[key]
        value = rng.choice(candidates)
        idx = node('record', entity=entity, key=key, value=value,
                   scope=scope, ordinal=len(records)+1,
                   deps=(current[(entity, 'owner')],))
        records.append(idx)
        latest_record[(scope, entity, key)] = idx
        emit('record', (idx,))

    def revisit():
        current_ids = (set(current.values()) |
                       set(latest_claim.values()) |
                       set(latest_measure.values()) |
                       set(latest_record.values()) |
                       set(steps[-1:]))
        options = [idx for idx in shown if idx in current_ids]
        emit('revisit', (rng.choice(options),))

    def question():
        current_ids = (set(current.values()) |
                       set(latest_claim.values()) |
                       set(latest_measure.values()) |
                       set(latest_record.values()) |
                       set(steps[-1:]))
        options = [idx for idx in shown if idx in current_ids]
        target = rng.choice(options)
        emit('question', (target,), answer=nodes[target]['value'])

    operations = dict(intro=lambda: introduce(
                          rng.choice([e for e in entities
                                      if e not in introduced])),
                      update=update, attribute=attribute,
                      compare=compare, measure=measure, step=step,
                      scope=shift_scope, record=record,
                      revisit=revisit, question=question)

    introduce(rng.choice(entities))
    if form in ('reference', 'collection', 'commercial'):
        introduce(rng.choice([e for e in entities if e not in introduced]))
    required = REQUIRED[form]
    if required == 'attribute':
        update()
    operations[required]()

    while len(moves) < n_moves:
        valid = []
        for kind, weight in MOVE_WEIGHTS[form].items():
            if kind == 'intro' and len(introduced) == len(entities):
                continue
            if kind == 'attribute' and not updates:
                continue
            if kind == 'compare' and len(introduced) < 2:
                continue
            if kind == 'question' and rng.random() > question_rate:
                continue
            if kind in ('revisit', 'question') and not shown:
                continue
            # Coupling controls reuse of existing semantic nodes. Low
            # coupling favors fresh facts/events over revisit/attribution.
            if kind in ('revisit', 'attribute', 'question'):
                weight *= coupling_rate
            valid.append((kind, weight))
        if not valid or sum(weight for _, weight in valid) == 0:
            kind = 'update'
        else:
            kind = rng.choices([kind for kind, _ in valid],
                               weights=[weight for _, weight in valid])[0]
        operations[kind]()

    result = dict(form=form, entities=entities, aliases=aliases,
                  initial=initial, nodes=nodes, moves=moves,
                  opening_role=opening_role,
                  controls=dict(n_moves=n_moves,
                                coupling_rate=coupling_rate,
                                question_rate=question_rate,
                                shared_attribute_rate=shared_attribute_rate,
                                form_weights=form_weights))
    validate(result)
    return result


def validate(plan):
    """Replay visibility and semantic dependencies without latent shortcuts."""
    nodes = plan['nodes']
    assert plan['opening_role'] in OPENINGS[plan['form']]
    assert [node['id'] for node in nodes] == list(range(len(nodes)))
    for node in nodes:
        assert all(dep < node['id'] for dep in node.get('deps', ()))
    visible, current = set(), {}
    scopes = {0}
    prior_steps = []
    for move in plan['moves']:
        kind, refs = move['kind'], move['refs']
        if kind == 'intro':
            entity = move['entity']
            assert entity in plan['initial']
            assert len(refs) == len(ATTRS)
            for ref in refs:
                node = nodes[ref]
                assert node['kind'] == 'fact' and node['entity'] == entity
                assert node['value'] == plan['initial'][entity][node['attr']]
                current[(entity, node['attr'])] = ref
                visible.add(ref)
            continue
        assert all(ref in range(len(nodes)) for ref in refs)
        if kind == 'scope':
            assert move['scope'] not in scopes
            scopes.add(move['scope'])
            continue
        for ref in refs:
            node = nodes[ref]
            assert all(dep in visible for dep in node.get('deps', ()))
            if kind == 'update':
                assert node['kind'] == 'update'
                key = (node['entity'], node['attr'])
                assert current[key] == node['deps'][0]
                assert nodes[current[key]]['value'] == node['old']
                current[key] = ref
            elif kind == 'compare':
                assert node['kind'] == 'comparison'
                left, right = node['deps']
                assert current[(node['entity'], node['attr'])] == left
                assert current[(node['other'], node['attr'])] == right
                assert node['value'] == (
                    'same' if nodes[left]['value'] == nodes[right]['value']
                    else 'different')
            elif kind == 'attribute':
                assert node['kind'] == 'source_claim'
                assert nodes[node['deps'][0]]['kind'] == 'update'
            elif kind == 'step':
                assert node['kind'] == 'step'
                assert node['ordinal'] == len(prior_steps)+1
                if prior_steps:
                    assert node['deps'] == (prior_steps[-1],)
                prior_steps.append(ref)
            elif kind == 'record':
                assert node['kind'] == 'record'
                assert node['scope'] in scopes
            elif kind == 'measure':
                assert node['kind'] == 'measure'
            elif kind in ('question', 'revisit'):
                assert ref in visible
                if kind == 'question':
                    assert move['answer'] == node['value']
            else:
                raise AssertionError(kind)
            visible.add(ref)
    assert any(move['kind'] == REQUIRED[plan['form']]
               for move in plan['moves'])
    return True


def edit_update(plan, node_id, new_value):
    """Intervene on one event value and repair downstream derived claims."""
    result = copy.deepcopy(plan)
    nodes = result['nodes']
    target = nodes[node_id]
    assert target['kind'] == 'update'
    assert new_value in VALUES[target['attr']]
    assert new_value != target['value']
    target['value'] = new_value
    current = {}
    for move in result['moves']:
        kind = move['kind']
        if kind == 'intro':
            for ref in move['refs']:
                fact = nodes[ref]
                current[(fact['entity'], fact['attr'])] = ref
        elif kind == 'update':
            update = nodes[move['refs'][0]]
            key = (update['entity'], update['attr'])
            prior = current[key]
            update['old'] = nodes[prior]['value']
            assert update['old'] != update['value']
            current[key] = update['id']
        elif kind == 'compare':
            relation = nodes[move['refs'][0]]
            left, right = relation['deps']
            relation['value'] = (
                'same' if nodes[left]['value'] == nodes[right]['value']
                else 'different')
        elif kind == 'question':
            move['answer'] = nodes[move['refs'][0]]['value']
    validate(result)
    return result


def edit_initial_fact(plan, entity, attr, new_value):
    """Intervene on an opening fact and repair dependent world claims."""
    assert attr in ATTRS and new_value in VALUES[attr]
    result = copy.deepcopy(plan)
    assert new_value != result['initial'][entity][attr]
    result['initial'][entity][attr] = new_value
    first = next(node for node in result['nodes']
                 if node['kind'] == 'fact' and
                 (node['entity'], node['attr']) == (entity, attr))
    first['value'] = new_value
    nodes, current = result['nodes'], {}
    for move in result['moves']:
        kind = move['kind']
        if kind == 'intro':
            for ref in move['refs']:
                fact = nodes[ref]
                current[(fact['entity'], fact['attr'])] = ref
        elif kind == 'update':
            update = nodes[move['refs'][0]]
            key = (update['entity'], update['attr'])
            prior = current[key]
            update['old'] = nodes[prior]['value']
            assert update['old'] != update['value']
            current[key] = update['id']
        elif kind == 'compare':
            relation = nodes[move['refs'][0]]
            left, right = relation['deps']
            relation['value'] = (
                'same' if nodes[left]['value'] == nodes[right]['value']
                else 'different')
        elif kind == 'question':
            move['answer'] = nodes[move['refs'][0]]['value']
    validate(result)
    return result


def audit(seed=8817, per_form=200):
    rng = random.Random(seed)
    counts, unique = Counter(), set()
    for form in FORMS:
        for _ in range(per_form):
            plan = sample_plan(rng, form=form,
                               coupling_rate=rng.choice((0., .25, .5, 1.)),
                               question_rate=rng.choice((0., .5, 1.)))
            validate(plan)
            counts.update(move['kind'] for move in plan['moves'])
            unique.add((form, tuple(move['kind'] for move in plan['moves'])))
    return dict(seed=seed, per_form=per_form,
                plans=per_form*len(FORMS),
                unique_move_sequences=len(unique),
                move_counts=dict(counts))
