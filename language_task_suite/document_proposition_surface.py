"""First visible-text realization of typed document proposition plans.

The text grammar is intentionally explicit and independently parseable. It
is a semantic integration fixture; natural-form audits decide whether it is
usable as a training distribution. The parser reads text only, never a plan.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from document_world import NAMES, VALUES
from document_world_intro_paragraphs import _clause, _is_intro, _parse_clause
from document_world_v5_clause_realizations import _sentences
from document_world_v9_typed_lexicon import (
    _CANONICAL_POOLS, _KIND, _surface, draw_lexicon,
)
from document_proposition_plan import COUNT_UNITS, FORMS, STEP_ACTIONS


@dataclass(frozen=True)
class AnswerSite:
    index: int
    family: str
    answer: str
    candidates: tuple[str, ...]
    entity: str | None


@dataclass(frozen=True)
class RenderedDocument:
    tokens: tuple[str, ...]
    sites: tuple[AnswerSite, ...]
    form: str
    plan: dict

    def text(self):
        return (' '.join(self.tokens).replace(' .', '.')
                .replace(' ,', ',').replace(' :', ':')
                .replace(' ;', ';').replace(' ?', '?')
                .replace(' \n ', '\n'))


def _opening(plan):
    entity = plan['moves'][0]['entity']
    role = plan['opening_role']
    forms = {
        'headline': ['A', 'closer', 'look', 'at', 'the', entity, '.'],
        'dateline': ['Today', ',', 'an', 'account', 'of', 'the', entity, '.'],
        'lead_claim': ['Here', 'is', 'what', 'happened', 'to', 'the', entity, '.'],
        'change': ['An', 'update', 'concerning', 'the', entity, '.'],
        'profile': ['A', 'profile', 'of', 'the', entity, '.'],
        'question': ['What', 'should', 'we', 'know', 'about', 'the', entity,
                     '?', 'Read', 'on', '.'],
        'personal_scene': ['I', 'had', 'been', 'thinking', 'about', 'the', entity, '.'],
        'reflection': ['The', entity, 'came', 'up', 'again', 'today', '.'],
        'definition': ['What', 'is', 'the', entity, '?', 'The', 'entry',
                       'follows', '.'],
        'enumeration': ['Several', 'points', 'about', 'the', entity, '.'],
        'measure': ['The', 'numbers', 'behind', 'the', entity, '.'],
        'series_context': ['Another', 'entry', 'in', 'the', entity, 'series', '.'],
        'byline': ['Notes', 'on', 'the', entity, 'from', 'the', 'interview', '.'],
        'attribution': ['An', 'account', 'of', 'the', entity, 'from', 'several',
                        'speakers', '.'],
        'item_list': ['Items', 'needed', 'for', 'the', entity, '.'],
        'ordered_step': ['A', 'short', 'procedure', 'for', 'the', entity, '.'],
        'scene': ['From', 'around', 'the', 'room', ':', 'the', entity, '.'],
        'item': ['First', 'item', ':', 'the', entity, '.'],
        'specification': ['Details', 'for', 'the', entity, '.'],
        'review': ['A', 'review', 'of', 'the', entity, '.'],
    }
    assert role in forms
    return forms[role]


def _query(node, nodes, question):
    kind = node['kind']
    if kind in ('fact', 'update'):
        obj, attr = node['entity'], node['attr']
        if question:
            return ['What', 'was', 'the', attr, 'of', 'the', obj,
                    '?', 'Answer', ':']
        return ['At', 'this', 'point', ',', 'the', attr, 'of', 'the',
                obj, 'was']
    if kind == 'source_claim':
        ordinal = nodes[node['deps'][0]]['ordinal']
        if question:
            return ['Who', 'reported', 'update', str(ordinal),
                    '?', 'Answer', ':']
        return ['The', 'latest', 'source', 'for', 'update',
                str(ordinal), 'was']
    if kind == 'measure':
        ordinal = node['ordinal']
        if question:
            return ['What', 'was', 'count', str(ordinal),
                    '?', 'Answer', ':']
        return ['Count', str(ordinal), 'had', 'value']
    if kind == 'step':
        ordinal = node['ordinal']
        if question:
            return ['What', 'was', 'the', 'action', 'at', 'step',
                    str(ordinal), '?', 'Answer', ':']
        return ['The', 'action', 'at', 'step', str(ordinal), 'was']
    assert kind == 'record'
    obj, key, scope = node['entity'], node['key'], node['scope']
    if question:
        return ['What', 'was', key, 'for', 'the', obj, 'in',
                'part', str(scope), '?', 'Answer', ':']
    return ['In', 'part', str(scope), ',', 'the', key,
            'of', 'the', obj, 'was']


def realize(plan, seed=0, paragraph_rate=.35, lexical_exponent=None):
    assert plan['form'] in FORMS and 0 <= paragraph_rate <= 1
    rng = random.Random(seed)
    tokens, sites = [], []
    nodes = plan['nodes']

    def emit(sentence, force_break=False):
        if tokens and (force_break or rng.random() < paragraph_rate):
            tokens.append('\n')
        tokens.extend(sentence)

    emit(_opening(plan))
    for move in plan['moves']:
        kind = move['kind']
        node = nodes[move['refs'][0]] if move['refs'] else None
        form = rng.randrange(2)
        if kind == 'intro':
            obj = move['entity']
            sentence = _clause(obj, plan['initial'][obj],
                               move['alias'], rng.randrange(5)) + ['.']
        elif kind == 'update':
            obj, attr, old, new = (node[k] for k in (
                'entity', 'attr', 'old', 'value'))
            if form == 0:
                sentence = ['The', attr, 'of', 'the', obj, 'changed',
                            'from', old, 'to', new, '.']
            else:
                sentence = ['Now', 'the', obj, 'had', new, 'as', 'its', attr,
                            ',', 'up', 'from', old, '.']
        elif kind == 'attribute':
            ordinal = nodes[node['deps'][0]]['ordinal']
            source = node['value']
            sentence = ([source, 'reported', 'update', str(ordinal), '.']
                        if form == 0 else
                        ['Update', str(ordinal), 'was', 'reported',
                         'by', source, '.'])
        elif kind == 'compare':
            obj, other, attr, relation = (node[k] for k in (
                'entity', 'other', 'attr', 'value'))
            lead = (['The', attr, 'values', 'for', obj, 'and', other,
                     'were'] if form == 0 else
                    ['For', obj, 'and', other, ',', attr, 'values',
                     'were'])
            if tokens and rng.random() < paragraph_rate:
                tokens.append('\n')
            tokens.extend(lead)
            index = len(tokens)
            tokens.extend((relation, '.'))
            sites.append(AnswerSite(index, 'comparison', relation,
                                    ('same', 'different'), obj))
            continue
        elif kind == 'measure':
            obj, value, unit, ordinal = (node[k] for k in (
                'entity', 'value', 'unit', 'ordinal'))
            sentence = (['The', 'file', 'lists', str(value), unit,
                         'for', 'the', obj, '.']
                        if form == 0 else
                        ['Count', str(ordinal), ':', obj, ',',
                         str(value), unit, '.'])
        elif kind == 'step':
            obj, action, ordinal = (node[k] for k in (
                'entity', 'value', 'ordinal'))
            lead = ['Step'] if form == 0 else ['For', 'step']
            if tokens:
                tokens.append('\n')
            tokens.extend(lead)
            index = len(tokens)
            tokens.append(str(ordinal))
            tokens.extend(([':', action, 'the', obj, '.'] if form == 0 else
                           [',', action, 'the', obj, '.']))
            sites.append(AnswerSite(index, 'step_number', str(ordinal),
                                    tuple(str(j) for j in range(1, 32)),
                                    obj))
            continue
        elif kind == 'scope':
            scope = str(move['scope'])
            sentence = (['Part', scope, ':', 'new', 'listing', '.']
                        if form == 0 else
                        ['The', 'next', 'section', 'is', 'part', scope, '.'])
        elif kind == 'record':
            obj, key, value, scope, ordinal = (node[k] for k in (
                'entity', 'key', 'value', 'scope', 'ordinal'))
            sentence = (['In', 'part', str(scope), ',', obj, 'had',
                         key, value, '.'] if form == 0 else
                        ['Record', str(ordinal), ':', obj, ',', key,
                         value, ',', 'part', str(scope), '.'])
        else:
            assert kind in ('revisit', 'question')
            question = kind == 'question'
            lead = _query(node, nodes, question)
            answer = str(node['value'])
            if tokens and rng.random() < paragraph_rate:
                tokens.append('\n')
            tokens.extend(lead)
            index = len(tokens)
            tokens.extend((answer, '.'))
            if node['kind'] in ('fact', 'update'):
                candidates = tuple(VALUES[node['attr']])
            elif node['kind'] == 'source_claim':
                candidates = tuple(NAMES)
            elif node['kind'] == 'measure':
                candidates = tuple(str(i) for i in range(1, 31))
            elif node['kind'] == 'step':
                candidates = STEP_ACTIONS
            else:
                candidates = {'status': ('active', 'closed', 'pending'),
                              'category': ('standard', 'special'),
                              'code': tuple(plan['aliases'].values())}[
                                  node['key']]
            sites.append(AnswerSite(index, node['kind'], answer,
                                    candidates, node.get('entity')))
            continue
        emit(sentence, force_break=kind in ('scope', 'step') or
             (plan['form'] in ('collection', 'procedure') and
              kind in ('intro', 'record')))
    if lexical_exponent is not None:
        mapping = draw_lexicon(random.Random(seed ^ 0xC019),
                               lexical_exponent)
        tokens = [_surface(token, mapping) for token in tokens]
        sites = [AnswerSite(site.index, site.family,
                            mapping.get(site.answer, site.answer),
                            tuple(mapping.get(value, value)
                                  for value in site.candidates),
                            mapping.get(site.entity, site.entity))
                 for site in sites]
    return RenderedDocument(tuple(tokens), tuple(sites), plan['form'], plan)


class _VisibleState:
    def __init__(self):
        self.facts = {}
        self.aliases = {}
        self.updates = []
        self.sources = {}
        self.measures = []
        self.records = {}
        self.steps = []
        self.scope = 0

    def query(self, raw):
        if raw == ['Step'] or raw == ['For', 'step']:
            return str(len(self.steps)+1)
        if raw[:4] == ['The', 'action', 'at', 'step']:
            assert raw[5:] == ['was']
            return self.steps[int(raw[4])-1][0]
        if raw[:6] == ['What', 'was', 'the', 'action', 'at', 'step']:
            assert raw[7:] == ['?', 'Answer', ':']
            return self.steps[int(raw[6])-1][0]
        if (len(raw) == 8 and raw[:3] ==
                ['The', raw[1], 'values'] and raw[3] == 'for'):
            _, attr, _, _, obj, and_, other, were = raw
            assert (and_, were) == ('and', 'were')
            return ('same' if self.facts[(obj, attr)] ==
                    self.facts[(other, attr)] else 'different')
        if len(raw) == 8 and raw[0] == 'For' and raw[2] == 'and':
            _, obj, _, other, comma, attr, values, were = raw
            assert (comma, values, were) == (',', 'values', 'were')
            return ('same' if self.facts[(obj, attr)] ==
                    self.facts[(other, attr)] else 'different')
        if raw[:4] == ['At', 'this', 'point', ',']:
            _, _, _, _, the, attr, of, the2, obj, was = raw
            assert (the, of, the2, was) == ('the', 'of', 'the', 'was')
            return self.facts[(obj, attr)]
        if raw[:3] == ['What', 'was', 'the']:
            _, _, _, attr, of, the, obj, mark, answer, colon = raw
            assert (of, the, mark, answer, colon) == (
                'of', 'the', '?', 'Answer', ':')
            return self.facts[(obj, attr)]
        if raw[:3] == ['The', 'latest', 'source']:
            assert raw[3:5] == ['for', 'update'] and raw[6] == 'was'
            return self.sources[int(raw[5])]
        if raw[:2] == ['Who', 'reported']:
            assert raw[2] == 'update' and raw[4:] == [
                '?', 'Answer', ':']
            return self.sources[int(raw[3])]
        if raw[:1] == ['Count'] and raw[2:] == ['had', 'value']:
            return self.measures[int(raw[1])-1]['value']
        if raw[:3] == ['What', 'was', 'count']:
            assert raw[4:] == ['?', 'Answer', ':']
            return self.measures[int(raw[3])-1]['value']
        if raw[:2] == ['In', 'part'] and raw[3:5] == [',', 'the']:
            _, _, scope, _, _, key, of, the, obj, was = raw
            assert (of, the, was) == ('of', 'the', 'was')
            return self.records[(int(scope), obj, key)]
        if (len(raw) == 12 and raw[:2] == ['What', 'was'] and
                raw[3:5] == ['for', 'the']):
            (_, _, key, for_, the, obj, in_, part, scope,
             mark, answer, colon) = raw
            assert (for_, the, in_, part, mark, answer, colon) == (
                'for', 'the', 'in', 'part', '?', 'Answer', ':')
            return self.records[(int(scope), obj, key)]
        return None

    def observe(self, row):
        expected = self.query(row[:-1]) if len(row) > 1 else None
        if expected is not None:
            assert row[-1] == expected, (row, expected)
            return
        if _is_intro(row):
            canonical = _parse_clause(row)
            owner, color, obj, place, alias = (
                canonical[i] for i in (0, 3, 4, 7, 10))
            assert obj not in self.aliases.values()
            self.aliases[alias] = obj
            self.facts[(obj, 'owner')] = owner
            self.facts[(obj, 'color')] = color
            self.facts[(obj, 'place')] = place
            return
        if row[:1] == ['The'] and len(row) == 10 and row[5] == 'changed':
            _, attr, of, the, obj, _, from_, old, to, new = row
            assert (of, the, from_, to) == ('of', 'the', 'from', 'to')
            assert self.facts[(obj, attr)] == old and new in VALUES[attr]
            self.facts[(obj, attr)] = new
            self.updates.append((obj, attr, new))
            return
        if len(row) == 12 and row[:2] == ['Now', 'the']:
            _, _, obj, had, new, as_, its, attr, comma, up, from_, old = row
            assert had == 'had'
            assert (as_, its, comma, up, from_) == (
                'as', 'its', ',', 'up', 'from')
            assert self.facts[(obj, attr)] == old and new in VALUES[attr]
            self.facts[(obj, attr)] = new
            self.updates.append((obj, attr, new))
            return
        if row[0] in NAMES and row[1:3] == ['reported', 'update']:
            ordinal = int(row[3])
            assert 1 <= ordinal <= len(self.updates)
            self.sources[ordinal] = row[0]
            return
        if row[:1] == ['Update'] and row[2:5] == [
                'was', 'reported', 'by']:
            ordinal = int(row[1])
            assert 1 <= ordinal <= len(self.updates)
            assert row[5] in NAMES
            self.sources[ordinal] = row[5]
            return
        if len(row) == 8 and row[0] == 'The' and row[2] == 'and':
            _, obj, _, other, had, relation, attr, values = row
            assert had == 'had'
            assert values == 'values'
            actual = ('same' if self.facts[(obj, attr)] ==
                      self.facts[(other, attr)] else 'different')
            assert relation == actual
            return
        if row[:1] == ['The'] and len(row) == 9 and row[2:4] == [
                'values', 'of']:
            _, attr, _, _, obj, and_, other, were, relation = row
            assert (and_, were) == ('and', 'were')
            actual = ('same' if self.facts[(obj, attr)] ==
                      self.facts[(other, attr)] else 'different')
            assert relation == actual
            return
        if len(row) == 8 and row[:3] == ['The', 'file', 'lists']:
            _, _, _, value, unit, for_, the, obj = row
            assert (for_, the) == ('for', 'the')
            assert (obj, 'owner') in self.facts
            assert unit in COUNT_UNITS and 1 <= int(value) <= 30
            self.measures.append(dict(entity=obj, value=value, unit=unit))
            return
        if row[:1] == ['Count'] and len(row) == 7:
            _, ordinal, colon, obj, comma, value, unit = row
            assert (int(ordinal), colon, comma) == (
                len(self.measures)+1, ':', ',')
            assert (obj, 'owner') in self.facts
            assert unit in COUNT_UNITS and 1 <= int(value) <= 30
            self.measures.append(dict(entity=obj, value=value, unit=unit))
            return
        if row[:1] == ['Step']:
            _, ordinal, colon, action, the, obj = row
            assert (int(ordinal), colon, the) == (
                len(self.steps)+1, ':', 'the')
            assert action in STEP_ACTIONS and (obj, 'owner') in self.facts
            self.steps.append((action, obj))
            return
        if row[:2] == ['For', 'step']:
            _, _, ordinal, comma, action, the, obj = row
            assert (int(ordinal), comma, the) == (
                len(self.steps)+1, ',', 'the')
            assert action in STEP_ACTIONS and (obj, 'owner') in self.facts
            self.steps.append((action, obj))
            return
        if row[:1] == ['Part']:
            _, scope, colon, new, listing = row
            assert (int(scope), colon, new, listing) == (
                self.scope+1, ':', 'new', 'listing')
            self.scope += 1
            return
        if row[:3] == ['The', 'next', 'section']:
            assert row[3:5] == ['is', 'part']
            assert int(row[5]) == self.scope+1
            self.scope += 1
            return
        if row[:2] == ['In', 'part']:
            _, _, scope, comma, obj, had, key, value = row
            assert (comma, had) == (',', 'had')
            assert int(scope) <= self.scope
            self.records[(int(scope), obj, key)] = value
            return
        if row[:1] == ['Record']:
            (_, ordinal, colon, obj, comma, key, value, comma2,
             part, scope) = row
            assert (colon, comma, comma2, part) == (':', ',', ',', 'part')
            assert int(scope) <= self.scope
            self.records[(int(scope), obj, key)] = value
            return
        raise AssertionError(f'Unrecognized semantic sentence: {row}')


def _normalize_lexical(prefix):
    encountered = {kind: {} for kind in _CANONICAL_POOLS}
    reverse, normalized = {}, []
    for token in prefix:
        possessive = token.endswith("'s")
        base = token[:-2] if possessive else token
        kind = _KIND.get(base)
        if kind is None:
            normalized.append(token)
            continue
        assigned = encountered[kind]
        if base not in assigned:
            pool = _CANONICAL_POOLS[kind]
            if len(assigned) >= len(pool):
                raise ValueError(f'Too many visible {kind} symbols')
            assigned[base] = pool[len(assigned)]
            reverse[assigned[base]] = base
        canonical = assigned[base]
        normalized.append(canonical + "'s" if possessive else canonical)
    return normalized, reverse


def answer_from_visible_prefix(prefix, lexical_enabled=False):
    reverse = None
    if lexical_enabled:
        prefix, reverse = _normalize_lexical(prefix)
    complete, raw = _sentences([word for word in prefix if word != '\n'])
    assert complete
    # The opening is descriptive layout; it asserts no scored world fact.
    state = _VisibleState()
    for row in complete[1:]:
        state.observe(row)
    answer = state.query(raw)
    assert answer is not None, raw
    if reverse is not None and answer in reverse:
        return reverse[answer]
    return answer


def parse_visible_document(tokens, lexical_enabled=False):
    """Parse and validate every complete sentence from emitted text only."""
    if lexical_enabled:
        tokens, _ = _normalize_lexical(tokens)
    complete, raw = _sentences([word for word in tokens if word != '\n'])
    assert complete and not raw
    state = _VisibleState()
    for row in complete[1:]:
        state.observe(row)
    return state
