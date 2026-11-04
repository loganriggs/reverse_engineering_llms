"""Two independently editable infobox-to-prose value relations.

The source article is empirical Wikipedia text. Only the selected literal
relations are controlled; other article claims are not a symbolic oracle.
"""

from __future__ import annotations

from dataclasses import dataclass

from audit_structured_wikipedia_multifield import candidate, literal_mentions


@dataclass(frozen=True)
class Site:
    label: str
    answer: str
    span: tuple[int, int]


@dataclass(frozen=True)
class Arm:
    text: str
    sites: tuple[Site, Site]


@dataclass(frozen=True)
class FourArm:
    original: Arm
    edit_first: Arm
    edit_second: Arm
    edit_both: Arm


def _render(row, selected, alternatives, edits):
    title = row['name']
    abstract = row['abstract']
    values = [alternatives[i] if edits[i] else field['value']
              for i, field in enumerate(selected)]
    prefix = (title + '\n' +
              '\n'.join(field['label'] + ': ' + values[i]
                        for i, field in enumerate(selected)) + '\n\n')
    fragments = []
    spans = [None, None]
    cursor = 0
    out_len = len(prefix)
    for i, field in sorted(enumerate(selected),
                           key=lambda pair: pair[1]['span'][0]):
        lo, hi = field['span']
        before = abstract[cursor:lo]
        fragments.append(before)
        out_len += len(before)
        value = values[i]
        fragments.append(value)
        spans[i] = (out_len, out_len + len(value))
        out_len += len(value)
        cursor = hi
    fragments.append(abstract[cursor:])
    text = prefix + ''.join(fragments)
    sites = tuple(Site(field['label'], values[i], spans[i])
                  for i, field in enumerate(selected))
    return Arm(text, sites)


def visible_answers(arm):
    """Read field values from visible header and verify the prose targets."""
    header, _ = arm.text.split('\n\n', 1)
    rows = header.splitlines()[1:]
    values = {}
    for row in rows:
        label, sep, value = row.partition(': ')
        if not sep or label in values:
            raise ValueError('Ambiguous field header')
        values[label] = value
    answers = []
    for site in arm.sites:
        if site.label not in values:
            raise ValueError('Missing visible field')
        value = values[site.label]
        if arm.text[slice(*site.span)] != value:
            raise ValueError('Prose target disagrees with visible field')
        answers.append(value)
    return tuple(answers)


def make_four_arm(row, selected, alternatives):
    if len(selected) != 2 or len(alternatives) != 2:
        raise ValueError('Need exactly two independent fields')
    if selected[0]['label'].casefold() == selected[1]['label'].casefold():
        raise ValueError('Repeated field label')
    abstract = row['abstract']
    for i, field in enumerate(selected):
        alternate = alternatives[i]
        if (alternate.casefold() == field['value'].casefold() or
                len(literal_mentions(abstract, alternate)) or
                len(literal_mentions(abstract, field['value'])) != 1 or
                '\n' in alternate or ': ' in alternate):
            raise ValueError('Unsafe alternate value')
    arms = tuple(_render(row, selected, alternatives, edits)
                 for edits in ((False, False), (True, False),
                               (False, True), (True, True)))
    originals = visible_answers(arms[0])
    if any(visible_answers(arm) != expected for arm, expected in zip(
            arms, (originals,
                   (alternatives[0], originals[1]),
                   (originals[0], alternatives[1]),
                   tuple(alternatives)))):
        raise AssertionError('A field edit changed the wrong answer')
    # The observable prose outside the two literal fact values stays fixed.
    def strip_values(arm):
        text = arm.text.split('\n\n', 1)[1]
        base = len(arm.text) - len(text)
        spans = sorted((site.span[0]-base, site.span[1]-base)
                       for site in arm.sites)
        parts, cursor = [], 0
        for lo, hi in spans:
            parts.extend((text[cursor:lo], '<FACT>'))
            cursor = hi
        parts.append(text[cursor:])
        return ''.join(parts)
    if len({strip_values(arm) for arm in arms}) != 1:
        raise AssertionError('Edit changed unrelated prose')
    return FourArm(*arms)
