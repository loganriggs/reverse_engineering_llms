"""Purpose-driven, multi-rule project reports for DGP/form experiments.

This candidate is not accepted as realistic until independent form screens
pass. It samples one coherent repair/reopening event chain. Rule sites have
a text-only oracle; the renderer's hidden plan is not consulted by it.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, replace

from document_world_v9_typed_lexicon import PEOPLE


# Project, issue, plural work unit, local impact. These are semantic types;
# arbitrary object/unit substitutions would make implausible assertions.
DOMAINS = (
    ('bridge', 'damaged railings', 'railings',
     'Pedestrians used a nearby crossing while the damaged section was inspected.'),
    ('library', 'broken windows', 'windows',
     'Readers entered through a side door during the work.'),
    ('clinic', 'leaking pipes', 'pipes',
     'Appointments continued in another wing during the repair period.'),
    ('park', 'dark pathways', 'lamps',
     'The affected paths stayed closed after dusk.'),
    ('station', 'faulty signals', 'signals',
     'Trains used a reduced schedule while the equipment was checked.'),
    ('shelter', 'damaged doors', 'doors',
     'Visitors were directed to a temporary entrance.'),
    ('garden', 'worn benches', 'benches',
     'The public garden stayed open, but the seating area was fenced off.'),
    ('pool', 'leaking valves', 'valves',
     'Swimming sessions were moved to another facility.'),
    ('road', 'faded crossings', 'crossings',
     'Drivers were asked to slow down near the work zone.'),
    ('school', 'broken desks', 'desks',
     'Classes continued in other rooms while the furniture was replaced.'),
    ('theater', 'damaged seats', 'seats',
     'Several performances used a smaller audience area.'),
    ('market', 'faulty lights', 'lights',
     'Stall owners kept the affected aisle closed after sunset.'),
)
DISTRICTS = (
    'Riverside', 'Northgate', 'Westbrook', 'Hillview', 'Southfield',
    'Eastwood', 'Brookside', 'Lakeview', 'Midtown', 'Oakridge',
    'Fairview', 'Newtown', 'Woodside', 'Harborview', 'Kingsbridge',
    'Stonebridge', 'Greenfield', 'Parkside', 'Redwood', 'Cedarview',
)
DAYS = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday')
PURPOSES = ('report', 'memo', 'notice', 'newsletter')
MARKERS = ('TOTAL_ANSWER', 'STATUS_ANSWER', 'SOURCE_ANSWER')
WORD = re.compile(r"\b[A-Za-z]+(?:'[A-Za-z]+)?\b")


@dataclass(frozen=True)
class Plan:
    project: str
    issue: str
    unit: str
    impact: str
    district: str
    first_source: str
    second_source: str
    first_count: int
    second_count: int
    target: int
    decision: str
    coupled: bool
    purpose: str
    day1: str
    day2: str
    surface_seed: int

    @property
    def total(self):
        return self.first_count+self.second_count

    @property
    def status(self):
        return ('open' if self.total >= self.target else 'closed') \
            if self.coupled else self.decision


@dataclass(frozen=True)
class Site:
    kind: str
    start: int
    end: int
    answer: str
    candidates: tuple[str, ...]


@dataclass(frozen=True)
class Document:
    text: str
    sites: tuple[Site, ...]
    plan: Plan


def _draw(rng, rows, exponent):
    weights = [(index+1)**(-exponent)
               for index in range(len(rows))]
    return rng.choices(rows, weights)[0]


def sample_plan(rng: random.Random, coupling_rate=.5,
                frequency_exponent=.6, purpose_weights=None):
    """Sample a linked event chain; controls are DGP hyperparameters."""
    if not 0 <= coupling_rate <= 1 or frequency_exponent < 0:
        raise ValueError('Invalid DGP control')
    if purpose_weights is None:
        purpose_weights = (1.,)*len(PURPOSES)
    if len(purpose_weights) != len(PURPOSES) or \
            min(purpose_weights) < 0 or sum(purpose_weights) == 0:
        raise ValueError('Invalid purpose weights')
    project, issue, unit, impact = _draw(rng, DOMAINS,
                                       frequency_exponent)
    district = _draw(rng, DISTRICTS, frequency_exponent)
    first_source = _draw(rng, PEOPLE, frequency_exponent)
    second_source = _draw(rng,
                          tuple(name for name in PEOPLE
                                if name != first_source),
                          frequency_exponent)
    first_count = rng.randint(2, 12)
    second_count = rng.randint(2, 12)
    total = first_count+second_count
    target = total+rng.choice((-2, -1, 1, 2))
    assert target > 0
    coupled = rng.random() < coupling_rate
    decision = rng.choice(('open', 'closed'))
    purpose = rng.choices(PURPOSES, purpose_weights)[0]
    day1, day2 = sorted(rng.sample(DAYS, 2), key=DAYS.index)
    return Plan(project, issue, unit, impact, district, first_source,
                second_source, first_count, second_count, target,
                decision, coupled, purpose, day1, day2,
                rng.randrange(2**32))


TEMPLATES = {
    'opening': (
        'The {district} {project} is the subject of a new repair report.',
        'Repair work at the {district} {project} has prompted an update.',
        'Officials have issued a progress note on the {district} {project}.',
        'A new account of work at the {district} {project} was released this week.',
    ),
    'context': (
        'The project was put on the weekly agenda after residents asked for a repair schedule.',
        'A public timetable was requested when the site first closed for inspection.',
        'Local staff had asked the department to publish a timetable for the repairs.',
    ),
    'target': (
        'The site needed {target} {unit} repaired before it could reopen.',
        'The reopening target was {target} {unit} at the site.',
        'Officials set a target of {target} repaired {unit} for reopening.',
    ),
    'target_independent': (
        'The work schedule called for {target} {unit} to be repaired.',
        'Officials set a separate work target of {target} {unit}.',
        'The department expected to repair {target} {unit} under this schedule.',
    ),
    'issue': (
        'An inspection found {issue}, and the site was closed for repairs.',
        'The closure followed a review that found {issue} at the site.',
        'Inspectors found {issue}; work at the site could not continue as usual.',
    ),
    'impact': ('{impact}',),
    'first_work': (
        'On {day1}, {first_source} said crews had completed {first_count} {unit}.',
        'Crews finished {first_count} {unit} by {day1}, according to {first_source}.',
        '{first_source} reported {first_count} {unit} completed on {day1}.',
    ),
    'second_work': (
        'By {day2}, {second_source} reported another {second_count} {unit}.',
        'Another {second_count} {unit} were finished on {day2}, {second_source} said.',
        '{second_source} said that another {second_count} {unit} had been repaired by {day2}.',
    ),
    'scope': (
        'The figures cover finished repairs; work that is only planned is listed separately.',
        'Both reports count completed repairs rather than jobs still on the schedule.',
        'The department said these figures refer only to work already finished.',
    ),
    'total': (
        'That brought the total to TOTAL_ANSWER {unit}.',
        'The combined work covered TOTAL_ANSWER {unit}.',
        'Together, the two reports accounted for TOTAL_ANSWER {unit}.',
    ),
    'independent_decision': (
        'Separately, the safety office recommended that the site be {decision}.',
        'A separate review recommended that the site remain {decision}.',
    ),
    'status': (
        'Under the announced plan, the site would now be STATUS_ANSWER.',
        'The final notice therefore listed the site as STATUS_ANSWER.',
        'For the time being, its official status was STATUS_ANSWER.',
    ),
    'source_recap': (
        'The later figures came from SOURCE_ANSWER.',
        'The final memo credited SOURCE_ANSWER with the second count.',
        'The second report had been filed by SOURCE_ANSWER.',
    ),
    'closing': (
        'The department expects to publish another note after the final inspection.',
        'Officials said a further notice would follow once the next inspection was complete.',
        'The timetable may change again if inspectors find a new problem.',
    ),
}


def realize(plan: Plan, surface_entropy=1., paragraph_rate=.25):
    if not 0 <= surface_entropy <= 1 or not 0 <= paragraph_rate <= 1:
        raise ValueError('Invalid surface control')
    rng = random.Random(plan.surface_seed)
    clauses = []
    values = vars(plan)

    def add(kind):
        options = TEMPLATES[kind]
        template = (rng.choice(options) if rng.random() < surface_entropy
                    else options[0])
        clauses.append(template.format(**values))

    add('opening')
    add('context')
    add('target' if plan.coupled else 'target_independent')
    add('issue')
    add('impact')
    add('first_work')
    add('second_work')
    add('scope')
    add('total')
    if not plan.coupled:
        add('independent_decision')
    add('status')
    add('source_recap')
    add('closing')
    # A purpose controls layout without changing the fact order.
    if plan.purpose == 'memo':
        lead = 'Project memo. '
    elif plan.purpose == 'notice':
        lead = 'Public notice. '
    elif plan.purpose == 'newsletter':
        lead = 'Local update. '
    else:
        lead = ''
    text = lead
    for index, clause in enumerate(clauses):
        if index:
            text += ('\n\n' if rng.random() < paragraph_rate else ' ')
        text += clause
    replacements = {
        'TOTAL_ANSWER': ('total', str(plan.total),
                         tuple(str(i) for i in range(4, 28))),
        'STATUS_ANSWER': ('status', plan.status,
                          ('open', 'closed')),
        'SOURCE_ANSWER': ('source', plan.second_source,
                          tuple(PEOPLE)),
    }
    sites = []
    for marker in MARKERS:
        kind, answer, candidates = replacements[marker]
        start = text.index(marker)
        text = text.replace(marker, answer, 1)
        sites.append(Site(kind, start, start+len(answer),
                          answer, candidates))
    return Document(text, tuple(sites), plan)


TARGET = re.compile(r'(?:needed|target was|target of) (\d+) '
                    r'(?:repaired )?[a-z]+', re.I)
FIRST = re.compile(r'(?:completed|finished|reported) (\d+) '
                   r'(?:[a-z]+)', re.I)
SECOND = re.compile(r'\banother (\d+) [a-z]+', re.I)
SOURCE_A = re.compile(r'\b([A-Z][a-z]+) reported another \d+', re.I)
SOURCE_B = re.compile(r'\banother \d+ [a-z]+ .*?, ([A-Z][a-z]+) said', re.I)
SOURCE_C = re.compile(r'\b([A-Z][a-z]+) said that another \d+', re.I)
DECISION = re.compile(r'\brecommended that the site (?:be|remain) '
                      r'(open|closed)\b', re.I)


def answer_from_visible_prefix(prefix: str, kind: str):
    """Read only visible prose; do not access the sampled Plan."""
    if kind == 'total':
        first = FIRST.search(prefix)
        second = SECOND.search(prefix)
        if not first or not second:
            raise ValueError('Missing visible work counts')
        return str(int(first.group(1))+int(second.group(1)))
    if kind == 'status':
        decision = DECISION.search(prefix)
        if decision:
            return decision.group(1).lower()
        first = FIRST.search(prefix)
        second = SECOND.search(prefix)
        target = TARGET.search(prefix)
        if not first or not second or not target:
            raise ValueError('Missing visible threshold evidence')
        return ('open' if int(first.group(1))+int(second.group(1)) >=
                int(target.group(1)) else 'closed')
    if kind == 'source':
        for pattern in (SOURCE_A, SOURCE_B, SOURCE_C):
            match = pattern.search(prefix)
            if match:
                return match.group(1)
        raise ValueError('Missing visible later source')
    raise ValueError(kind)


def edit(plan: Plan, field: str, value):
    if field not in ('first_count', 'second_count', 'target',
                     'second_source', 'decision'):
        raise ValueError(field)
    if field == 'second_source' and value == plan.first_source:
        raise ValueError('Sources must differ')
    return replace(plan, **{field: value})
