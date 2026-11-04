"""Jointly sampled semantic events and compositional sentence derivations.

This is a bounded vertical slice, not a realism claim. The document world
supplies object states and attributed changes. A syntax tree combines change,
attribution, relative fact, adjunct, and coordination operators; its visible
text is checked independently by replaying the emitted assertions.
"""

from __future__ import annotations

from dataclasses import dataclass
import random

from document_world import COLORS, NAMES, OBJECTS
from document_world_v9_typed_lexicon import draw_lexicon


@dataclass(frozen=True)
class Event:
    entity: str
    old: str
    new: str
    source: str


@dataclass(frozen=True)
class Syntax:
    op: str
    children: tuple['Syntax', ...] = ()
    payload: tuple = ()


@dataclass(frozen=True)
class Site:
    index: int
    family: str
    answer: str
    entity: str


@dataclass(frozen=True)
class Document:
    tokens: tuple[str, ...]
    sites: tuple[Site, ...]
    world: dict
    trees: tuple[Syntax, ...]

    def text(self):
        return (' '.join(self.tokens).replace(' .', '.')
                .replace(' ,', ',').replace(' ;', ';')
                .replace(' \\n ', '\n'))


ADJUNCTS = (
    ('Later', ','), ('After', 'the', 'review', ','),
    ('In', 'the', 'account', ','), ('By', 'then', ','),
)


def draw_world(rng: random.Random, n_entities=3, n_events=5,
               source_coupling=.5, lexical_exponent=1.):
    if not (2 <= n_entities <= len(OBJECTS) and
            n_events >= n_entities and 0 <= source_coupling <= 1):
        raise ValueError('Invalid world configuration')
    mapping = draw_lexicon(rng, lexical_exponent)
    canonical = rng.sample(OBJECTS, n_entities)
    entities = tuple(mapping[obj] for obj in canonical)
    initial = {obj: mapping[rng.choice(COLORS)] for obj in entities}
    current = dict(initial)
    last_source = {}
    events = []
    schedule = list(entities)
    schedule.extend(rng.choice(entities) for _ in range(n_events-n_entities))
    rng.shuffle(schedule)
    for obj in schedule:
        old = current[obj]
        new = rng.choice([mapping[color] for color in COLORS
                          if mapping[color] != old])
        previous = last_source.get(obj)
        source = (previous if previous and rng.random() < source_coupling
                  else mapping[rng.choice(NAMES)])
        events.append(Event(obj, old, new, source))
        current[obj] = new
        last_source[obj] = source
    return {'entities': entities, 'initial': initial, 'events': tuple(events),
            'final': current, 'last_source': last_source,
            'controls': {'n_entities': n_entities, 'n_events': n_events,
                         'source_coupling': source_coupling,
                         'lexical_exponent': lexical_exponent}}


def _event_tree(event: Event, rng: random.Random, adjunct_rate,
                relative_rate):
    node = Syntax('change', payload=(event.entity, event.old, event.new))
    if rng.random() < relative_rate:
        node = Syntax('relative', (node,), (event.old,))
    node = Syntax('attribute', (node,),
                  (event.source, rng.randrange(3)))
    if rng.random() < adjunct_rate:
        node = Syntax('adjunct', (node,), rng.choice(ADJUNCTS))
    return node


def _render(node: Syntax):
    if node.op == 'intro':
        obj, color, form = node.payload
        if form == 0:
            return ['Initially', ',', 'the', obj, 'was', color]
        if form == 1:
            return ['The', obj, 'was', color, 'at', 'the', 'outset']
        return ['At', 'first', ',', 'the', obj, 'looked', color]
    if node.op == 'change':
        obj, old, new = node.payload
        return ['the', obj, 'changed', 'from', old, 'to', new]
    if node.op == 'relative':
        words = _render(node.children[0])
        assert words[:1] == ['the']
        assert words[2:4] == ['changed', 'from']
        return words[:2] + [',', 'which', 'had', 'been',
                            node.payload[0], ',', 'turned', words[-1]]
    if node.op == 'attribute':
        source, mode = node.payload
        words = _render(node.children[0])
        if mode == 0:
            return [source, 'reported', 'that'] + words
        if mode == 1:
            return ['According', 'to', source, ','] + words
        return words + [',', source, 'reported']
    if node.op == 'adjunct':
        return list(node.payload) + _render(node.children[0])
    if node.op == 'coordinate':
        return _render(node.children[0]) + [';'] + _render(node.children[1])
    raise ValueError(node.op)


def realize(world, seed=0, coordination_rate=.4, adjunct_rate=.25,
            relative_rate=.18, paragraph_rate=.2,
            intro_interleave_rate=.6, recap_interleave_rate=.6):
    if not all(0 <= v <= 1 for v in (coordination_rate, adjunct_rate,
                                    relative_rate, paragraph_rate,
                                    intro_interleave_rate,
                                    recap_interleave_rate)):
        raise ValueError('Rates must lie in [0,1]')
    rng = random.Random(seed)
    tokens, sites, trees = [], [], []

    def emit(words, tree=None):
        words = list(words)
        if words and words[0] == 'the':
            words[0] = 'The'
        for i, token in enumerate(words[:-1]):
            if token in (',', ';') and words[i+1] in ('According', 'The'):
                words[i+1] = words[i+1].lower()
            if token == ';' and words[i+1] in ('Later', 'After', 'In', 'By'):
                words[i+1] = words[i+1].lower()
        if tokens and rng.random() < paragraph_rate:
            tokens.append('\n')
        tokens.extend(words)
        tokens.append('.')
        if tree is not None:
            trees.append(tree)

    chosen = rng.sample(world['entities'], 2)
    last_event = {obj: max(i for i, event in enumerate(world['events'])
                           if event.entity == obj)
                  for obj in world['entities']}
    first_event = {obj: min(i for i, event in enumerate(world['events'])
                            if event.entity == obj)
                   for obj in world['entities']}
    intro_at = {obj: (rng.randrange(first_event[obj]+1)
                      if rng.random() < intro_interleave_rate else 0)
                for obj in world['entities']}
    recap_at = {obj: (rng.randrange(last_event[obj], len(world['events']))
                      if rng.random() < recap_interleave_rate else
                      len(world['events'])-1)
                for obj in chosen}
    introduced = set()

    def intro_due(index):
        due = [obj for obj in world['entities'] if obj not in introduced and
               intro_at[obj] <= index]
        rng.shuffle(due)
        for obj in due:
            tree = Syntax('intro', payload=(obj, world['initial'][obj],
                                            rng.randrange(3)))
            emit(_render(tree), tree)
            introduced.add(obj)

    def recap(obj, family):
        color, source = world['final'][obj], world['last_source'][obj]
        if family == 'current_color':
            prefix = ['For', 'the', obj, ',', source, 'recorded', 'the',
                      'final', 'color', 'as']
            answer = color
        else:
            prefix = ['The', obj, 'was', color, 'by', 'the', 'end', ',']
            answer = source
        if tokens and rng.random() < paragraph_rate:
            tokens.append('\n')
        tokens.extend(prefix)
        sites.append(Site(len(tokens), family, answer, obj))
        tokens.append(answer)
        if family == 'latest_source':
            tokens.append('reported')
        tokens.append('.')
    index = 0
    while index < len(world['events']):
        intro_due(index)
        event = world['events'][index]
        tree = _event_tree(event, rng, adjunct_rate, relative_rate)
        end = index
        if (index+1 < len(world['events']) and
                rng.random() < coordination_rate and
                first_event[world['events'][index+1].entity] <= index):
            # A second event can only be folded in if its entity has already
            # been introduced; the next loop iteration would otherwise emit
            # its introduction between the two event clauses.
            following = _event_tree(world['events'][index+1], rng,
                                    adjunct_rate, relative_rate)
            tree = Syntax('coordinate', (tree, following))
            end = index+1
        emit(_render(tree), tree)
        for obj, family in zip(chosen,
                               ('current_color', 'latest_source'),
                               strict=True):
            if index <= recap_at[obj] <= end and last_event[obj] <= end:
                recap(obj, family)
        index = end+1
    doc = Document(tuple(tokens), tuple(sites), world, tuple(trees))
    check_document(doc)
    return doc


def _split_complete(tokens):
    fragments, current = [], []
    for token in tokens:
        if token == '.':
            if current:
                fragments.append(tuple(current))
                current = []
        elif token != '\n':
            current.append(token)
    return fragments, tuple(current)


def _strip_adjunct(words):
    for adjunct in ADJUNCTS:
        if (len(words) >= len(adjunct) and
                words[0].lower() == adjunct[0].lower() and
                words[1:len(adjunct)] == adjunct[1:]):
            return words[len(adjunct):]
    return words


def _parse_intro(words):
    if (len(words) == 6 and words[:3] == ('Initially', ',', 'the')
            and words[4] == 'was'):
        return words[3], words[5]
    if (len(words) == 7 and words[:1] == ('The',)
            and words[2] == 'was'
            and words[4:] == ('at', 'the', 'outset')):
        return words[1], words[3]
    if (len(words) == 7 and words[:4] == ('At', 'first', ',', 'the')
            and words[5] == 'looked'):
        return words[4], words[6]
    return None


def _parse_change(words, state):
    words = list(words)
    if len(words) >= 10 and words[2:6] == [',', 'which', 'had', 'been']:
        assert words[0].lower() == 'the' and words[7] == ','
        obj, earlier = words[1], words[6]
        if state.get(obj) != earlier:
            raise ValueError('False relative state')
        if words[8] != 'turned' or len(words) != 10:
            raise ValueError('Malformed relative change')
        return obj, words[9]
    if len(words) != 7 or words[0].lower() != 'the' or words[2] != 'changed' \
            or words[3] != 'from' or words[5] != 'to':
        raise ValueError(('Unrecognized change', words))
    obj, old, new = words[1], words[4], words[6]
    if obj not in state or state[obj] != old or new == old:
        raise ValueError('Change contradicts visible state')
    return obj, new


def _parse_event(words, state, source_map):
    words = _strip_adjunct(tuple(words))
    if ';' in words:
        k = words.index(';')
        _parse_event(words[:k], state, source_map)
        _parse_event(words[k+1:], state, source_map)
        return
    if len(words) >= 4 and words[1:3] == ('reported', 'that'):
        source, change = words[0], words[3:]
    elif len(words) >= 5 and words[0].lower() == 'according' \
            and words[1] == 'to' \
            and words[3] == ',':
        source, change = words[2], words[4:]
    elif len(words) >= 3 and words[-1] == 'reported' and words[-3] == ',':
        source, change = words[-2], words[:-3]
    else:
        raise ValueError(('Unrecognized attribution', words))
    obj, new = _parse_change(change, state)
    state[obj], source_map[obj] = new, source


def replay_visible(tokens):
    """Independently parse complete visible sentences into current state."""
    complete, partial = _split_complete(tokens)
    state, sources = {}, {}
    for words in complete:
        intro = _parse_intro(words)
        if intro is not None:
            obj, color = intro
            if obj in state:
                raise ValueError('Duplicate introduction')
            state[obj] = color
            continue
        if _parse_recap(words, state, sources):
            continue
        _parse_event(words, state, sources)
    return state, sources, partial


def _parse_recap(words, state, sources):
    if (len(words) == 11 and words[:2] == ('For', 'the')
            and words[3] == ',' and words[5:10] ==
            ('recorded', 'the', 'final', 'color', 'as')):
        obj, source, color = words[2], words[4], words[10]
        if state.get(obj) != color or sources.get(obj) != source:
            raise ValueError('False final-color recap')
        return True
    if (len(words) == 10 and words[:1] == ('The',)
            and words[2] == 'was' and
            words[4:8] == ('by', 'the', 'end', ',')
            and words[9] == 'reported'):
        obj, color, source = words[1], words[3], words[8]
        if state.get(obj) != color or sources.get(obj) != source:
            raise ValueError('False latest-source recap')
        return True
    return False


def answer_from_visible_prefix(prefix):
    state, sources, partial = replay_visible(prefix)
    if (len(partial) == 10 and partial[:2] == ('For', 'the')
            and partial[3] == ',' and partial[5:] ==
            ('recorded', 'the', 'final', 'color', 'as')):
        obj, source = partial[2], partial[4]
        if sources.get(obj) != source:
            raise ValueError('Wrong source cue')
        return 'current_color', state[obj]
    if (len(partial) == 8 and partial[:1] == ('The',)
            and partial[2] == 'was' and
            partial[4:] == ('by', 'the', 'end', ',')):
        obj, color = partial[1], partial[3]
        if state.get(obj) != color:
            raise ValueError('Wrong color cue')
        return 'latest_source', sources[obj]
    raise ValueError(('Not a scored prefix', partial))


def check_document(doc: Document):
    state, sources, partial = replay_visible(doc.tokens)
    if partial:
        raise ValueError('Incomplete document')
    for site in doc.sites:
        family, expected = answer_from_visible_prefix(doc.tokens[:site.index])
        if (family != site.family or expected != site.answer or
                doc.tokens[site.index] != expected):
            raise AssertionError((site, family, expected))
    if state != doc.world['final'] or sources != doc.world['last_source']:
        raise AssertionError('Visible replay disagrees with sampled world')
    return True


def sample(seed, **controls):
    rng = random.Random(seed)
    world_controls = {k: v for k, v in controls.items()
                      if k in ('n_entities', 'n_events',
                               'source_coupling', 'lexical_exponent')}
    if 'n_entities' not in world_controls:
        world_controls['n_entities'] = rng.choice((2, 3, 4))
    if 'n_events' not in world_controls:
        world_controls['n_events'] = rng.randint(
            world_controls['n_entities']+1,
            world_controls['n_entities']+4)
    world = draw_world(rng, **world_controls)
    world['render_seed'] = rng.randrange(2**63)
    return realize(world, seed=world['render_seed'],
                   **{k: v for k, v in controls.items()
                      if k in ('coordination_rate', 'adjunct_rate',
                               'relative_rate', 'paragraph_rate',
                               'intro_interleave_rate',
                               'recap_interleave_rate')})


def edit_last_event(world, entity, *, new_color=None, new_source=None):
    """Return a world with one supported last-event intervention."""
    if (new_color is None) == (new_source is None):
        raise ValueError('Change exactly one field')
    if entity not in world['entities']:
        raise ValueError(entity)
    events = list(world['events'])
    index = max(i for i, event in enumerate(events)
                if event.entity == entity)
    old = events[index]
    color = old.new if new_color is None else new_color
    source = old.source if new_source is None else new_source
    if color == old.old or (new_color is not None and color == old.new) or \
            (new_source is not None and source == old.source):
        raise ValueError('No-op or invalid event edit')
    events[index] = Event(entity, old.old, color, source)
    edited = dict(world)
    edited['events'] = tuple(events)
    edited['final'] = dict(world['final'], **{entity: color})
    edited['last_source'] = dict(world['last_source'], **{entity: source})
    return edited
