"""Controlled list-number continuation inside empirical web prose.

The source is a real document; this module supplies only a local list-number
rule and an optional independently editable prose parenthesis. It does not
claim an oracle for the rest of the document.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from corpus_transformed_delimiters import candidates as bracket_candidates


MARKER = re.compile(r'(?m)^(?P<indent>[ \t]{0,3})(?P<num>[1-9])\.(?=[ \t]+\S)')


@dataclass(frozen=True)
class Plan:
    source_text: str
    markers: tuple[tuple[int, int, int], ...]
    bracket: tuple[int, int] | None = None


@dataclass(frozen=True)
class Site:
    kind: str
    start: int
    end: int
    answer: str
    source_start: int
    source_end: int


@dataclass(frozen=True)
class Document:
    text: str
    sites: tuple[Site, ...]
    plan: Plan


def candidates(text, max_chars=8000, max_step_chars=1000,
               max_first_three_chars=1600):
    text = text[:max_chars]
    matches = list(MARKER.finditer(text))
    runs = []
    for i, start in enumerate(matches):
        if start.group('num') != '1':
            continue
        indent = start.group('indent')
        selected = [start]
        for next_match in matches[i+1:]:
            wanted = len(selected)+1
            if (wanted > 8 or int(next_match.group('num')) != wanted or
                    next_match.group('indent') != indent or
                    next_match.start()-selected[-1].start() > max_step_chars):
                break
            selected.append(next_match)
        if len(selected) < 3 or (selected[2].start()-selected[0].start()
                                 > max_first_three_chars):
            continue
        # A further sequential marker, even far away, would duplicate the
        # shifted final label if left untouched.
        if (i+len(selected) < len(matches) and
                int(matches[i+len(selected)].group('num')) ==
                len(selected)+1):
            continue
        runs.append(tuple((m.start('num'), m.end('num'),
                           int(m.group('num'))) for m in selected))
    return tuple(runs)


def plans(text, with_bracket=False, max_chars=8000):
    source = text[:max_chars]
    brackets = bracket_candidates(source)
    for markers in candidates(source, max_chars=max_chars):
        if not with_bracket:
            yield Plan(source, markers)
            continue
        first, target = markers[0][0], markers[2][0]
        for opener, closer, _ in brackets:
            if closer < target and opener >= max(0, first-1200):
                yield Plan(source, markers, (opener, closer))


def visible_next_number(prefix):
    seen = list(MARKER.finditer(prefix))
    if len(seen) < 2:
        raise ValueError('Need two visible list markers')
    a, b = (int(match.group('num')) for match in seen[-2:])
    if b != a+1 or b >= 9:
        raise ValueError('Visible list sequence is not consecutive')
    return str(b+1)


def realize(plan, shift=0, bracket_style='round'):
    if shift not in (0, 1) or bracket_style not in ('round', 'square'):
        raise ValueError('Invalid edit arm')
    chars = list(plan.source_text)
    for start, end, number in plan.markers:
        assert end == start+1 and chars[start] == str(number)
        chars[start] = str(number+shift)
    sites = []
    if plan.bracket is not None:
        opener, closer = plan.bracket
        assert chars[opener] == '(' and chars[closer] == ')'
        if bracket_style == 'square':
            chars[opener], chars[closer] = '[', ']'
        sites.append(Site('bracket', closer, closer+1, chars[closer],
                          opener, opener+1))
    target = plan.markers[2]
    sites.append(Site('list_next', target[0], target[1],
                      str(3+shift), plan.markers[1][0],
                      plan.markers[1][1]))
    text = ''.join(chars)
    assert visible_next_number(text[:target[0]]) == str(3+shift)
    for site in sites:
        assert text[site.start:site.end] == site.answer
    return Document(text, tuple(sites), plan)
