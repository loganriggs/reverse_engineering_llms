"""Small, auditable language-like next-token tasks with explicit latent rules.

Each example supplies a natural-ish prefix, one rule-determined next token,
alternative tokens for a targeted choice test, and the latent state. Training
and evaluation can use the same generator without treating templates as proof
of natural-language validity.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Example:
    family: str
    prefix: tuple[str, ...]
    target: str
    candidates: tuple[str, ...]
    latent: tuple[tuple[str, str], ...]
    variant: int

    def text(self) -> str:
        out = ' '.join(self.prefix + (self.target,))
        for left, right in [(' .', '.'), (' ,', ','), (' ;', ';'),
                            (' )', ')'), (' ]', ']'), (' ( ', ' ('),
                            (' [ ', ' [')]:
            out = out.replace(left, right)
        return out.replace(' <NL> ', '\n')


NOUNS = (('pilot', 'pilots'), ('teacher', 'teachers'),
         ('artist', 'artists'), ('lawyer', 'lawyers'))
SUBJECTS = (('man', 'himself'), ('woman', 'herself'),
            ('men', 'themselves'), ('women', 'themselves'))
NAMES = ('Maya', 'Liam', 'Noah', 'Ava', 'Owen', 'Iris')
COLORS = ('red', 'blue', 'green', 'yellow')
OBJECTS = ('key', 'map', 'book', 'cup')
PLACES = ('box', 'drawer', 'shelf', 'cabinet')
WEEKDAYS = ('Monday', 'Tuesday', 'Wednesday', 'Thursday',
            'Friday', 'Saturday', 'Sunday')
ITEMS = ('apples', 'pears', 'books', 'maps')


def _example(family, prefix, target, candidates, variant, **latent):
    return Example(family, tuple(prefix), target, tuple(candidates),
                   tuple(sorted((k, str(v)) for k, v in latent.items())),
                   variant)


def agreement(rng: random.Random, variant=0):
    subj = rng.choice(NOUNS)
    distractor = rng.choice(NOUNS)
    number = rng.randrange(2)
    distractor_number = number if rng.random() < .3 else 1-number
    head, other = subj[number], distractor[distractor_number]
    if variant:
        prefix = ('The', head, 'who', 'the', other, 'praised')
    elif rng.random() < .2:
        prefix = ('The', head)
    else:
        prefix = ('The', head, 'near', 'the', other)
    return _example('agreement', prefix, ('is', 'are')[number],
        ('is', 'are'), variant, subject=head, distractor=other,
        subject_number=number)


def reflexive(rng: random.Random, variant=0):
    subject, target = rng.choice(SUBJECTS)
    distractor, _ = rng.choice([x for x in SUBJECTS if x[1] != target])
    if variant:
        prefix = ('The', subject, 'who', 'the', distractor,
                  'greeted', 'blamed')
    else:
        prefix = ('The', subject, 'beside', 'the', distractor, 'blamed')
    return _example('reflexive', prefix, target,
        ('himself', 'herself', 'themselves'), variant,
        subject=subject, distractor=distractor)


def attribute_binding(rng: random.Random, variant=0):
    a, b = rng.sample(NAMES, 2)
    ca, cb = rng.sample(COLORS, 2)
    query = rng.choice((a, b))
    events = [(a, ca), (b, cb)]
    rng.shuffle(events)
    prefix = []
    for name, color in events:
        prefix += [name, 'kept', 'the', color, 'folder', '.']
    if variant:
        prefix += ['Later', ',', 'the', 'folder', query, 'had', 'was']
    else:
        prefix += ['The', 'folder', 'belonging', 'to', query, 'was']
    return _example('attribute_binding', prefix, ca if query == a else cb,
        COLORS, variant, first=a, second=b, first_color=ca,
        second_color=cb, query=query)


def state_update(rng: random.Random, variant=0):
    obj, other = rng.sample(OBJECTS, 2)
    old, new, distractor = rng.sample(PLACES, 3)
    speaker = rng.choice(NAMES)
    prefix = [speaker, 'put', 'the', obj, 'in', 'the', old, '.']
    events = [([speaker, 'moved', 'the', obj, 'to', 'the', new, '.'], obj),
              ([speaker, 'put', 'the', other, 'in', 'the', distractor, '.'], other)]
    rng.shuffle(events)
    for event, _ in events:
        prefix += event
    if variant:
        prefix += ['After', 'those', 'changes', ',', 'the', obj,
                   'ended', 'up', 'in', 'the']
    else:
        prefix += ['At', 'the', 'end', ',', 'the', obj, 'was', 'in', 'the']
    return _example('state_update', prefix, new, PLACES, variant,
        object=obj, distractor_object=other, old=old, new=new)


def list_successor(rng: random.Random, variant=0):
    start = rng.randrange(1, 7)
    n_items = 3 if variant else 2
    items = [rng.choice(ITEMS) for _ in range(n_items)]
    prefix = []
    for offset, item in enumerate(items):
        prefix += [str(start+offset), '.', item, '<NL>']
    return _example('list_successor', prefix, str(start+n_items),
        tuple(str(x) for x in range(1, 10)), variant,
        start=start, n_items=n_items)


def weekday_successor(rng: random.Random, variant=0):
    start = rng.randrange(7)
    n_days = 3 if variant else 2
    observed = [WEEKDAYS[(start+i) % 7] for i in range(n_days)]
    prefix = ['The', 'meetings', 'were', 'on']
    for day in observed:
        prefix += [day, ',']
    prefix += ['and']
    return _example('weekday_successor', prefix,
        WEEKDAYS[(start+n_days) % 7], WEEKDAYS, variant,
        start=start, n_days=n_days)


def delimiter_stack(rng: random.Random, variant=0):
    pairs = (('(', ')'), ('[', ']'))
    depth = 3 if variant else rng.choice((1, 2))
    chosen = [rng.choice(pairs) for _ in range(depth)]
    prefix = ['The', 'report', 'included']
    for i, (opening, _) in enumerate(chosen):
        prefix += [opening, ('a', 'short', 'marked')[i]]
    for _, closing in reversed(chosen[1:]):
        prefix += [closing]
    prefix += ['item']
    return _example('delimiter_stack', prefix, chosen[0][1],
        (')', ']'), variant, outer=chosen[0][0], depth=depth)


def comparison_chain(rng: random.Random, variant=0):
    top, middle, bottom = rng.sample(NAMES, 3)
    facts = [(top, middle), (middle, bottom)]
    rng.shuffle(facts)
    prefix = []
    for taller, shorter in facts:
        prefix += [taller, 'is', 'taller', 'than', shorter, '.']
    ask_top = rng.random() < .5
    if variant:
        prefix += ['Among', 'the', 'three', ',', 'the',
                   ('highest' if ask_top else 'lowest'), 'person', 'was']
    else:
        prefix += ['The', ('tallest' if ask_top else 'shortest'), 'was']
    return _example('comparison_chain', prefix, top if ask_top else bottom,
        NAMES, variant, top=top, middle=middle, bottom=bottom,
        ask_top=ask_top)


def conditional_rule(rng: random.Random, variant=0):
    colors = rng.sample(COLORS, 2)
    states = rng.sample(('open', 'closed'), 2)
    rules = list(zip(colors, states))
    chosen_color = rng.choice(colors)
    rng.shuffle(rules)
    prefix = []
    for color, state in rules:
        if variant:
            prefix += [color, 'means', state, '.']
        else:
            prefix += ['If', 'the', 'light', 'is', color, ',',
                       'the', 'gate', 'is', state, '.']
    prefix += ['The', 'light', 'is', chosen_color, ',',
               'so', 'the', 'gate', 'is']
    answer = dict(rules)[chosen_color]
    return _example('conditional_rule', prefix, answer,
        ('open', 'closed'), variant, color=chosen_color,
        rules=';'.join(f'{c}:{s}' for c,s in rules))


def speaker_binding(rng: random.Random, variant=0):
    a, b = rng.sample(NAMES, 2)
    item = rng.choice(OBJECTS)
    events = [(a, 'found'), (b, 'lost')]
    rng.shuffle(events)
    prefix = []
    for speaker, action in events:
        prefix += [speaker, 'said', ',', 'I', action, 'the', item, '.']
    if variant:
        prefix += ['Who', 'found', 'the', item, '?', 'It', 'was']
    else:
        prefix += ['The', 'person', 'who', 'found', 'the', item, 'was']
    return _example('speaker_binding', prefix, a, NAMES, variant,
        finder=a, other=b, object=item)


def negation_correction(rng: random.Random, variant=0):
    obj, other = rng.sample(OBJECTS, 2)
    wrong, right, other_place = rng.sample(PLACES, 3)
    if variant:
        prefix = ['Although', 'the', obj, 'was', 'once', 'in', 'the', wrong,
                  ',', 'it', 'was', 'moved', 'to', 'the', right, '.',
                  'Meanwhile', ',', 'the', other, 'was', 'put', 'in',
                  'the', other_place, '.',
                  'Now', 'the', obj, 'is', 'in', 'the']
    else:
        prefix = ['The', obj, 'was', 'not', 'in', 'the', wrong, ';',
                  'it', 'was', 'in', 'the', right, '.',
                  'Later', ',', 'the', other, 'was', 'in', 'the',
                  other_place, '.',
                  'The', obj, 'was', 'in', 'the']
    return _example('negation_correction', prefix, right, PLACES,
        variant, object=obj, distractor_object=other, wrong=wrong,
        right=right, distractor_place=other_place)


def role_reversal(rng: random.Random, variant=0):
    agent, patient = rng.sample(NAMES, 2)
    ask_agent = rng.random() < .5
    if variant:
        prefix = [patient, 'was', 'thanked', 'by', agent, '.']
    else:
        prefix = [agent, 'thanked', patient, '.']
    prefix += ['The', ('agent' if ask_agent else 'recipient'), 'was']
    return _example('role_reversal', prefix,
        agent if ask_agent else patient, NAMES, variant,
        agent=agent, patient=patient, ask_agent=ask_agent)


GENERATORS = {
    'agreement': agreement,
    'reflexive': reflexive,
    'attribute_binding': attribute_binding,
    'state_update': state_update,
    'list_successor': list_successor,
    'weekday_successor': weekday_successor,
    'delimiter_stack': delimiter_stack,
    'comparison_chain': comparison_chain,
    'conditional_rule': conditional_rule,
    'speaker_binding': speaker_binding,
    'negation_correction': negation_correction,
    'role_reversal': role_reversal,
}


def generate(family: str, rng: random.Random, variant: int = 0) -> Example:
    if variant not in (0, 1):
        raise ValueError('variant must be 0 or 1')
    return GENERATORS[family](rng, variant)


def sample(n: int, seed: int, variant: int = 0,
           families: tuple[str, ...] | None = None) -> list[Example]:
    rng = random.Random(seed)
    names = families or tuple(GENERATORS)
    if not names:
        raise ValueError('at least one family is required')
    examples = [generate(names[i % len(names)], rng, variant)
                for i in range(n)]
    rng.shuffle(examples)
    return examples
