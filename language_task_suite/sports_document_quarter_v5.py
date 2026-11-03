"""Quarter-event prose and distributed rule sites over the V4 world.

Uses V4's sampled world and edits, but spreads quarter evidence through the
base report and expresses it with several typed sentence structures. Only
the marked sites have a visible-prefix oracle.
"""

from dataclasses import replace
import random
import re

from sports_document_dgp import Document, Site, answer_from_visible_prefix as base_answer
from sports_document_dgp import realize as base_realize
from sports_document_quarter_v4 import (
    _visible_score, edit, sample_plan, Plan,
)


ORDINALS = ('first', 'second', 'third', 'fourth')
QUARTER_FORMS = (
    'In the {ordinal} quarter, {home} scored {h} and {away} scored {a}.',
    '{home} scored {h} in the {ordinal} quarter; {away} scored {a}.',
    'The {ordinal} quarter ended {h} for {home} and {a} for {away}.',
    'During the {ordinal} quarter, {away} put up {a} while {home} had {h}.',
    '{home} added {h} in the {ordinal} quarter, compared with {a} for {away}.',
    'For the {ordinal} quarter, {away} recorded {a} and {home} recorded {h}.',
)
QUARTER_PATTERNS = (
    re.compile(r'In the (first|second|third|fourth) quarter, (.+?) scored (\d+) and (.+?) scored (\d+)\.'),
    re.compile(r'(.+?) scored (\d+) in the (first|second|third|fourth) quarter; (.+?) scored (\d+)\.'),
    re.compile(r'The (first|second|third|fourth) quarter ended (\d+) for (.+?) and (\d+) for (.+?)\.'),
    re.compile(r'During the (first|second|third|fourth) quarter, (.+?) put up (\d+) while (.+?) had (\d+)\.'),
    re.compile(r'(.+?) added (\d+) in the (first|second|third|fourth) quarter, compared with (\d+) for (.+?)\.'),
    re.compile(r'For the (first|second|third|fourth) quarter, (.+?) recorded (\d+) and (.+?) recorded (\d+)\.'),
)
HALF_FORMS = (
    'At the break, HALFTIME_ANSWER led.',
    'By halftime, HALFTIME_ANSWER had the edge.',
    'The first-half lead belonged to HALFTIME_ANSWER.',
    'It was HALFTIME_ANSWER in front at the interval.',
    'After two quarters, HALFTIME_ANSWER was in front.',
    'HALFTIME_ANSWER carried a lead into the second half.',
    'The side ahead at halftime was HALFTIME_ANSWER.',
    'HALFTIME_ANSWER finished the opening half ahead.',
)
STATE_FORMS = (
    'The eventual winner had a halftime STATE_ANSWER.',
    'For the winner, halftime brought a STATE_ANSWER.',
    'The winning side entered the break with a STATE_ANSWER.',
    'At halftime, the team that eventually won held a STATE_ANSWER.',
    'The final winner went into halftime with a STATE_ANSWER.',
    'The winner\'s position at the interval was a STATE_ANSWER.',
    'The victorious side had a STATE_ANSWER at the break.',
)


def _base_sentences(doc):
    sentences = []
    for match in re.finditer(r'[^.!?]+[.!?]', doc.text):
        raw = match.group()
        left_trim = len(raw)-len(raw.lstrip())
        text = raw.strip()
        entries = []
        for site in doc.sites:
            if match.start() <= site.start < match.end():
                entries.append((site, site.start-match.start()-left_trim))
        sentences.append((text, entries))
    if len([site for _, entries in sentences for site, _ in entries]) != len(doc.sites):
        raise ValueError('A base scored site was not assigned to a sentence')
    return sentences


def _quarter_sentence(plan, quarter, rng):
    i = quarter-1
    return rng.choice(QUARTER_FORMS).format(
        ordinal=ORDINALS[i], home=plan.base.home, away=plan.base.away,
        h=plan.home_quarters[i], a=plan.away_quarters[i])


def realize(plan: Plan, act_spread=1., **surface_controls):
    if not 0 <= act_spread <= 1:
        raise ValueError('act_spread must be in [0,1]')
    # V4 validates the numerical world; preserve that invariant here.
    if sum(plan.home_quarters) != (
            plan.base.high_score if plan.base.home_win
            else plan.base.low_score) or sum(plan.away_quarters) != (
            plan.base.low_score if plan.base.home_win
            else plan.base.high_score):
        raise ValueError('Quarter totals disagree with final score')
    base = base_realize(plan.base, **surface_controls)
    skeleton = _base_sentences(base)
    slot_rng = random.Random(plan.base.surface_seed ^ 0x5A1D0C)
    form_rng = random.Random(plan.base.surface_seed ^ 0x5A1D0D)
    paragraph_rng = random.Random(plan.base.surface_seed ^ 0x5A1D0E)
    score_index = next(i for i, (sentence, _) in enumerate(skeleton)
                       if re.search(r'\b\d{2,3}\s*(?:-|–|to)\s*'
                                    r'\d{2,3}\b', sentence))
    available = list(range(score_index+1, len(skeleton)+1))
    if len(available) >= 6 and slot_rng.random() < act_spread:
        slots = sorted(slot_rng.sample(available, 6))
    else:
        slots = [available[len(available)//2]]*6
    quarter_events = [(_quarter_sentence(plan, i, form_rng), None, None)
                      for i in (1, 2, 3, 4)]
    half = (form_rng.choice(HALF_FORMS), 'halftime_leader',
            plan.halftime_leader)
    state = (form_rng.choice(STATE_FORMS), 'winner_halftime_state',
             'deficit' if plan.comeback == 'yes' else 'lead')
    events = (quarter_events[0], quarter_events[1], half,
              quarter_events[2], quarter_events[3], state)
    by_slot = {i: [] for i in range(len(skeleton)+1)}
    for slot, event in zip(slots, events):
        by_slot[slot].append(event)
    parts, sites = [], []
    length = 0
    def append(sentence, entries=()):
        nonlocal length
        if parts:
            separator = '\n\n' if paragraph_rng.random() < surface_controls.get(
                'paragraph_rate', .12) else ' '
            parts.append(separator)
            length += len(separator)
        for site, offset in entries:
            sites.append(replace(site, start=length+offset,
                                 end=length+offset+len(site.answer)))
        parts.append(sentence)
        length += len(sentence)
    for slot in range(len(skeleton)+1):
        for sentence, kind, answer in by_slot[slot]:
            if kind is None:
                append(sentence)
            else:
                marker = ('HALFTIME_ANSWER' if kind == 'halftime_leader'
                          else 'STATE_ANSWER')
                pre, post = sentence.split(marker)
                replacement = pre+answer+post
                candidates = ((plan.base.home, plan.base.away)
                              if kind == 'halftime_leader'
                              else ('deficit', 'lead'))
                append(replacement,
                       ((Site(kind, 0, 0, answer, candidates),
                         len(pre)),))
        if slot < len(skeleton):
            append(*skeleton[slot])
    text = ''.join(parts)
    # Correct the legacy plural-only rich-stat forms in this version.
    for match in reversed(list(re.finditer(r'\b1 (assists|rebounds)\b', text))):
        text = text[:match.start()]+'1 '+match.group(1)[:-1]+text[match.end():]
        sites = [replace(site, start=site.start-1, end=site.end-1)
                 if site.start >= match.end() else site for site in sites]
    return Document(text, tuple(sorted(sites, key=lambda s: s.start)), plan)


def _parse_quarter_sentence(sentence):
    for index, pattern in enumerate(QUARTER_PATTERNS):
        match = pattern.fullmatch(sentence)
        if match is None:
            continue
        g = match.groups()
        if index == 0:
            ordinal, home, h, away, a = g
        elif index == 1:
            home, h, ordinal, away, a = g
        elif index == 2:
            ordinal, h, home, a, away = g
        elif index == 3:
            ordinal, away, a, home, h = g
        elif index == 4:
            home, h, ordinal, a, away = g
        else:
            ordinal, away, a, home, h = g
        return ORDINALS.index(ordinal)+1, home, int(h), away, int(a)
    return None


def visible_quarters(prefix):
    rows = []
    for sentence in re.findall(r'[^.!?]+[.!?]', prefix):
        parsed = _parse_quarter_sentence(sentence.strip())
        if parsed is not None:
            rows.append(parsed)
    if [row[0] for row in rows] not in ([1, 2], [1, 2, 3, 4]):
        raise ValueError('Expected first two or all four ordered quarters')
    home, away = rows[0][1], rows[0][3]
    if any(row[1] != home or row[3] != away for row in rows):
        raise ValueError('Quarter team identities disagree')
    return rows


def answer_from_visible_prefix(prefix, kind):
    if kind in ('winner', 'margin'):
        high, low, winner = _visible_score(prefix)
        return winner if kind == 'winner' else str(high-low)
    if kind == 'top_player':
        return base_answer(prefix, kind)
    rows = visible_quarters(prefix)
    home, away = rows[0][1], rows[0][3]
    h = sum(row[2] for row in rows[:2])
    a = sum(row[4] for row in rows[:2])
    if h == a:
        raise ValueError('Halftime tie unsupported')
    leader = home if h > a else away
    if kind == 'halftime_leader':
        return leader
    if kind == 'winner_halftime_state':
        if len(rows) != 4:
            raise ValueError('Need four visible quarters for winner state')
        return 'deficit' if leader != _visible_score(prefix)[2] else 'lead'
    raise ValueError(kind)
