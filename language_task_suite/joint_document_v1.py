"""A visible-prefix document DGP joining local rules and semantic facts.

Every token is sampled from next_distribution(prefix). Completed sentences
encode all state needed for the next distribution. This is a structural
pilot, not a natural-language match or a claim that every task is learned.
"""

from __future__ import annotations

import random

from surface_mixture_v2 import (EVENTS as BASE_EVENTS,
                                OBJECTS as BASE_OBJECTS, PEOPLE,
                                PLACES as BASE_PLACES, VERBS as BASE_VERBS)


NAMES = tuple(dict.fromkeys(PEOPLE + (
    'Amina', 'Boris', 'Celia', 'Dario', 'Esther', 'Farah', 'Gideon', 'Hector',
    'Iris', 'Jonah', 'Keira', 'Leon', 'Marta', 'Nora', 'Omar', 'Petra',
    'Ravi', 'Selma', 'Tobias', 'Ursula', 'Vera', 'Willa', 'Yara', 'Zane')))
VERBS = tuple(dict.fromkeys(BASE_VERBS + (
    'repairs', 'inspects', 'cleans', 'hides', 'opens', 'closes', 'records',
    'describes', 'examines', 'delivers', 'collects', 'displays', 'borrows',
    'returns', 'measures', 'packs', 'labels', 'lifts', 'drops', 'paints',
    'photographs', 'copies', 'files', 'sells', 'buys', 'tests', 'builds',
    'draws', 'wraps', 'sorts', 'shares', 'presents', 'studies', 'restores',
    'reviews', 'prepares', 'transports', 'observes')))
OBJECTS = tuple(dict.fromkeys(BASE_OBJECTS + (
    'parcel', 'wallet', 'notebook', 'tablet', 'laptop', 'umbrella', 'bicycle',
    'suitcase', 'photograph', 'sculpture', 'invoice', 'receipt', 'passport',
    'journal', 'newspaper', 'calendar', 'blanket', 'radio', 'lantern',
    'compass', 'medal', 'poster', 'microphone', 'headphones', 'drawing',
    'statue', 'trophy', 'certificate', 'envelope', 'manuscript', 'sample',
    'specimen', 'container', 'device', 'instrument', 'document', 'package',
    'report', 'record', 'painting', 'vase', 'mirror', 'clock', 'fabric',
    'jacket', 'glove', 'helmet', 'backpack', 'magazine', 'newspaper',
    'brochure', 'manual', 'tablet', 'keyboard', 'screen', 'cable',
    'battery', 'charger', 'sensor', 'camera', 'lens', 'tripod', 'speaker',
    'microphone', 'projector', 'display', 'printer', 'scanner', 'archive',
    'drawing', 'model', 'prototype', 'contract', 'proposal', 'chart',
    'diagram', 'schedule', 'inventory', 'label', 'stamp', 'coin',
    'badge', 'uniform', 'banner', 'flag', 'toolkit', 'brush', 'frame',
    'plaque', 'ticket', 'map', 'sign', 'booklet', 'catalog', 'form',
    'permit', 'license', 'folder', 'file', 'case', 'crate', 'box')))
PLACES = tuple(dict.fromkeys(BASE_PLACES + (
    'museum', 'gallery', 'clinic', 'hospital', 'village', 'town', 'city',
    'studio', 'warehouse', 'platform', 'airport', 'campus', 'kitchen',
    'hallway', 'classroom', 'theater', 'cafe', 'restaurant', 'courthouse',
    'balcony', 'basement', 'apartment', 'lobby', 'garage', 'temple',
    'mountain', 'valley', 'river', 'forest', 'desert', 'coast', 'depot',
    'workshop', 'laboratory', 'factory', 'port', 'vessel', 'stadium',
    'arena', 'pavilion', 'square', 'avenue', 'road', 'trail', 'museum')))
EVENTS = tuple(dict.fromkeys(BASE_EVENTS + (
    'conference', 'lecture', 'concert', 'dinner', 'ceremony', 'campaign',
    'election', 'opening', 'inspection', 'interview', 'launch', 'delivery',
    'rehearsal', 'repair', 'protest', 'contest', 'auction', 'celebration',
    'exhibition', 'presentation', 'discussion', 'appointment', 'trip',
    'tour', 'training', 'competition', 'survey', 'meeting', 'storm',
    'inspection', 'renovation', 'performance', 'practice', 'search')))
ROLES = ('reporter', 'engineer', 'curator', 'teacher', 'player', 'pilot',
         'designer', 'researcher', 'visitor', 'manager', 'coach', 'artist')
ATTRIBUTES = ('old', 'new', 'missing', 'damaged', 'rare', 'valuable',
              'heavy', 'fragile', 'clean', 'broken', 'local', 'official')
ASIDES = ('briefly', 'earlier', 'again', 'quietly', 'later', 'nearby')
TOPICS = PLACES
GENRES = ('Report', 'Guide')
MARKS = ('.', ',', ':', '?', '(', ')', '[', ']')
FUNCTION = ('Section', 'Part', 'According', 'As', 'said', 'to', 'the',
            'this', 'that', 'is', 'a', 'an', 'as', 'Who', 'described',
            'Answer', 'At', 'After', 'near')
DETERMINERS = ('the', 'this', 'that', 'a', 'an')
VOCAB = tuple(dict.fromkeys(NAMES + ROLES + ATTRIBUTES + ASIDES + TOPICS +
                            EVENTS + OBJECTS + VERBS + GENRES + MARKS + FUNCTION +
                            tuple(str(i) for i in range(1, 100))))
TOKEN2ID = {w: i for i, w in enumerate(VOCAB)}
ID2TOKEN = {i: w for w, i in TOKEN2ID.items()}


def _normalize(weights):
    assert weights and min(weights.values()) >= 0
    total = sum(weights.values())
    assert total > 0
    return {w: value/total for w, value in weights.items() if value > 0}


def _zipf(words, exponent=.62):
    return _normalize({word: (i+1)**(-exponent)
                       for i, word in enumerate(words)})


def _boost(base, chosen, mass):
    assert 0 <= mass <= 1 and chosen in base
    out = {word: (1-mass)*p for word, p in base.items()}
    out[chosen] += mass
    return out


def _determiners(nouns):
    vowel = sum(p for word, p in nouns.items() if word[0] in 'aeiou')
    return _normalize({'the': .48, 'this': .22, 'that': .16,
                       'a': .14*(1-vowel), 'an': .14*vowel})


def _after_determiner(nouns, article):
    if article == 'a':
        return _normalize({word: p for word, p in nouns.items()
                           if word[0] not in 'aeiou'})
    if article == 'an':
        return _normalize({word: p for word, p in nouns.items()
                           if word[0] in 'aeiou'})
    assert article in DETERMINERS
    return nouns


def _split(prefix):
    history, raw = [], []
    for word in prefix:
        raw.append(word)
        if word == '.':
            history.append(tuple(raw))
            raw = []
    return history, tuple(raw)


def _event(sentence):
    if sentence[0] == 'At':
        return dict(name=sentence[4], verb=sentence[5], item=sentence[7],
                    place=sentence[2])
    if sentence[0] == 'After':
        return dict(name=sentence[4], verb=sentence[5], item=sentence[7],
                    place=sentence[9])
    if sentence[0] in NAMES and sentence[1] in VERBS:
        return dict(name=sentence[0], verb=sentence[1], item=sentence[3],
                    place=None)
    return None


def visible_state(history):
    """Parse only completed, visible sentences; no hidden document plan."""
    state = dict(genre=None, title_topic=None, section_topic=None,
                 section_number=0, profiles={}, claims={}, events=[])
    for sentence in history:
        s = sentence[:-1]
        if not s:
            continue
        if s[0] in GENRES:
            assert state['genre'] is None and s[1] == ':' and s[2] in TOPICS
            state['genre'], state['title_topic'] = s[0], s[2]
        elif s[0] in ('Section', 'Part'):
            assert int(s[1]) == state['section_number']+1
            assert s[2] == ':' and s[3] in TOPICS
            state['section_number'] += 1
            state['section_topic'] = s[3]
        elif s[0] in NAMES and s[1] == 'is':
            assert s[2] == ('an' if s[3][0] in 'aeiou' else 'a')
            assert s[3] in ROLES
            state['profiles'][s[0]] = s[3]
        elif s[0] in ('According', 'As'):
            speaker = s[2] if s[0] == 'According' else s[1]
            assert ((s[1] == 'to' if s[0] == 'According' else s[2] == 'said')
                    and speaker in state['profiles'] and
                    s[3] == ',' and s[4] in DETERMINERS and
                    s[5] in OBJECTS and
                    s[6] == 'is' and s[7] in ATTRIBUTES)
            state['claims'][(s[5], s[7])] = speaker
        elif s[0] == 'Who':
            assert (s[1] == 'described' and s[2] in DETERMINERS and
                    s[4] == 'as' and
                    s[6:9] == ('?', 'Answer', ':') and
                    state['claims'][(s[3], s[5])] == s[9])
        else:
            event = _event(s)
            assert event is not None and event['name'] in state['profiles']
            assert event['verb'] in VERBS and event['item'] in OBJECTS
            state['events'].append(event)
    return state


def _start(state):
    if state['genre'] is None:
        return {'Report': .6, 'Guide': .4}
    known = tuple(state['profiles'])
    new = tuple(name for name in NAMES if name not in state['profiles'])
    if not known:
        return _zipf(new, .25)
    weights = {}
    if new:
        for name, p in _zipf(new, .25).items():
            weights[name] = .13*p
    for name, p in _zipf(known, .35).items():
        weights[name] = weights.get(name, 0) + .27*p
    weights['Section'] = .055
    weights['Part'] = .035
    weights['According'] = .09
    weights['As'] = .05
    weights['At'] = .18
    weights['After'] = .16
    if state['claims']:
        weights['Who'] = .10
    return _normalize(weights)


def _name_distribution(state):
    known = tuple(state['profiles'])
    assert known
    out = _zipf(known, .35)
    if state['claims']:
        speaker = next(reversed(state['claims'].values()))
        out = _boost(out, speaker, .30)
    return out


def _place_distribution(state):
    out = _zipf(PLACES)
    topic = state['section_topic'] or state['title_topic']
    if topic in out:
        out = _boost(out, topic, .30)
    return out


def verb_components(state, name):
    """Named exact proposal masses at a narrative verb site."""
    lexical = VERBS[(NAMES.index(name)*3) % len(VERBS)]
    parts = {'name_bigram': (.50, {lexical: 1.})}
    prior_same = [e for e in state['events'] if e['name'] == name]
    if prior_same:
        parts['induction'] = (.25, {prior_same[-1]['verb']: 1.})
    topic = state['section_topic'] or state['title_topic']
    if topic:
        chosen = VERBS[(TOPICS.index(topic)*5+1) % len(VERBS)]
        parts['topic_skip'] = (.15, {chosen: 1.})
    prior_place = [e['place'] for e in state['events'] if e['place']]
    if prior_place:
        chosen = VERBS[(PLACES.index(prior_place[-1])*7+2) % len(VERBS)]
        parts['place_skip'] = (.10, {chosen: 1.})
    z = sum(mass for mass, _ in parts.values())
    return {kind: (mass/z, dist) for kind, (mass, dist) in parts.items()}


def _verb_distribution(state, name):
    out = {verb: 0. for verb in VERBS}
    for mass, dist in verb_components(state, name).values():
        for verb, p in dist.items():
            out[verb] += mass*p
    return _normalize(out)


def _item_distribution(state, name):
    base = _zipf(OBJECTS)
    matches = [(item, speaker) for (item, _), speaker in state['claims'].items()
               if speaker == name]
    if matches:
        base = _boost(base, matches[-1][0], .45)
    return base


def _tail(raw, item_index):
    suffix = raw[item_index+1:]
    if not suffix:
        return {'.': .74, '(': .13, '[': .13}
    opener = suffix[0]
    assert opener in ('(', '[')
    if len(suffix) == 1:
        return _zipf(ASIDES)
    if len(suffix) == 2:
        return {')' if opener == '(' else ']': 1.}
    assert len(suffix) == 3
    return {'.': 1.}


def next_distribution(prefix):
    if any(word not in TOKEN2ID for word in prefix):
        raise ValueError('prefix contains an unknown token')
    history, raw = _split(prefix)
    state = visible_state(history)
    if not raw:
        return _start(state)
    first, pos = raw[0], len(raw)
    if first in GENRES:
        assert not history
        slots = ({':': 1.}, _zipf(TOPICS), {'.': 1.})
        return slots[pos-1]
    if first in ('Section', 'Part'):
        slots = ({str(state['section_number']+1): 1.}, {':': 1.},
                 _zipf(TOPICS), {'.': 1.})
        return slots[pos-1]
    if first in NAMES and first not in state['profiles']:
        if pos == 1:
            return {'is': 1.}
        roles = _zipf(ROLES)
        if pos == 2:
            return _normalize({'an': sum(p for role, p in roles.items()
                                          if role[0] in 'aeiou'),
                               'a': sum(p for role, p in roles.items()
                                        if role[0] not in 'aeiou')})
        if pos == 3:
            return _normalize({role: p for role, p in roles.items()
                               if (role[0] in 'aeiou') == (raw[2] == 'an')})
        assert pos == 4
        return {'.': 1.}
    if first in ('According', 'As'):
        if pos == 4:
            return _determiners(_zipf(OBJECTS))
        if pos == 5:
            return _after_determiner(_zipf(OBJECTS), raw[4])
        slots = (({'to': 1.} if first == 'According' else
                  _name_distribution(state)),
                 (_name_distribution(state) if first == 'According' else
                  {'said': 1.}),
                 {',': 1.}, None, None, {'is': 1.},
                 _zipf(ATTRIBUTES), {'.': 1.})
        return slots[pos-1]
    if first == 'Who':
        claims = state['claims']
        assert claims
        items = tuple(dict.fromkeys(item for item, _ in claims))
        if pos == 1:
            return {'described': 1.}
        if pos == 2:
            items = _zipf(tuple(dict.fromkeys(item for item, _ in claims)))
            return _determiners(items)
        if pos == 3:
            return _after_determiner(_zipf(items), raw[2])
        if pos == 4:
            return {'as': 1.}
        item = raw[3]
        if pos == 5:
            attrs = tuple(attr for (key, attr) in claims if key == item)
            return _zipf(attrs)
        fixed = ({'?': 1.}, {'Answer': 1.}, {':': 1.},
                 {claims[(item, raw[5])]: 1.}, {'.': 1.})
        return fixed[pos-6]
    if first == 'At':
        slots = (_determiners(_place_distribution(state)),
                 _after_determiner(_place_distribution(state), raw[1])
                 if pos > 1 else None,
                 {',': 1.}, _name_distribution(state))
        if pos <= 4:
            return slots[pos-1]
        name = raw[4]
        if pos == 5:
            return _verb_distribution(state, name)
        if pos == 6:
            return _determiners(_item_distribution(state, name))
        if pos == 7:
            return _after_determiner(_item_distribution(state, name), raw[6])
        return _tail(raw, 7)
    if first == 'After':
        slots = (_determiners(_zipf(EVENTS)),
                 _after_determiner(_zipf(EVENTS), raw[1])
                 if pos > 1 else None,
                 {',': 1.}, _name_distribution(state))
        if pos <= 4:
            return slots[pos-1]
        name = raw[4]
        if pos == 5:
            return _verb_distribution(state, name)
        if pos == 6:
            return _determiners(_item_distribution(state, name))
        if pos == 7:
            return _after_determiner(_item_distribution(state, name), raw[6])
        if pos == 8:
            return {'near': 1.}
        if pos == 9:
            return _place_distribution(state)
        return _tail(raw, 7 if pos < 10 else 9)
    assert first in state['profiles']
    if pos == 1:
        return _verb_distribution(state, first)
    if pos == 2:
        return _determiners(_item_distribution(state, first))
    if pos == 3:
        return _after_determiner(_item_distribution(state, first), raw[2])
    return _tail(raw, 3)


def site_label(prefix):
    history, raw = _split(prefix)
    if not raw:
        return 'sentence_start'
    state = visible_state(history)
    first, pos = raw[0], len(raw)
    if first in ('Section', 'Part') and pos == 1:
        return 'section_number'
    if first == 'Who' and pos == 9:
        return 'source_answer'
    if first in ('At', 'After') and pos == 5:
        return 'mixed_verb'
    if first in ('At', 'After') and pos == 7:
        return 'claim_linked_item'
    if first in state['profiles'] and pos == 1:
        return 'mixed_verb'
    if first in state['profiles'] and pos == 3:
        return 'claim_linked_item'
    if raw[-1] in ASIDES and any(x in raw for x in ('(', '[')):
        return 'bracket_close'
    return 'other'


def sample_sequence(seed, length=256):
    rng = random.Random(seed)
    words, labels, distributions = [], [], []
    for _ in range(length):
        q = next_distribution(words)
        label = site_label(words)
        word = rng.choices(tuple(q), weights=tuple(q.values()))[0]
        assert q[word] > 0 and abs(sum(q.values())-1) < 1e-10
        words.append(word)
        labels.append(label)
        distributions.append(q)
    return words, labels, distributions
