"""Controlled document-local name reuse inside unmodified natural prose.

This is a *corpus transformation*, not a symbolic oracle for the background
text. A source document is sampled first; a repeated first name is replaced
consistently by a randomly drawn, otherwise absent same-class name. The
later occurrence is a measurable induction target. All natural text remains
outside committed artifacts.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass


# High-precision subset of the existing V17 person-name bank cross-checked
# against NLTK names, with common ambiguous content words removed. The groups
# are a surface-agreement prior, not claims about any person's identity.
NAME_GROUPS = {
    'a': ('Bob', 'Dave', 'Henry', 'Jack', 'Leo', 'Nick', 'Paul', 'Aaron',
          'Ben', 'Felix', 'Isaac', 'Jules', 'Marco', 'Oscar', 'Victor',
          'Andrew', 'Benjamin', 'David', 'James', 'Lucas', 'Nathan',
          'Peter', 'Thomas', 'William', 'Arthur', 'Caleb', 'Edward',
          'Eli', 'Ethan', 'Jacob', 'Joseph', 'Logan', 'Mason',
          'Michael', 'Patrick', 'Robert', 'Simon', 'Walter'),
    'b': ('Alice', 'Carol', 'Eve', 'Iris', 'Kate', 'Mia', 'Abigail',
          'Clara', 'Elena', 'Hannah', 'Lena', 'Nadia', 'Rosa', 'Tara',
          'Wendy', 'Charlotte', 'Isabella', 'Katherine', 'Maria',
          'Olivia', 'Rachel', 'Sofia', 'Victoria', 'Zoe', 'Amelia',
          'Beatrice', 'Camila', 'Chloe', 'Eleanor', 'Elizabeth',
          'Emily', 'Fiona', 'Jasmine', 'Julia', 'Leah', 'Lily',
          'Nora', 'Rebecca', 'Sarah', 'Sophie', 'Stella', 'Susan',
          'Violet'),
}
NAME_TO_GROUP = {name: group for group, names in NAME_GROUPS.items()
                 for name in names}
assert len(NAME_TO_GROUP) == sum(map(len, NAME_GROUPS.values()))

NAME = re.compile(r'\b[A-Z][a-z]{2,}\b')
WORD = re.compile(r"\b[A-Za-z]+(?:'[A-Za-z]+)?\b")
FOLLOWING_TITLE = re.compile(r'^\s+([A-Z][a-z]{2,})\b')
FULL_NAME_END = re.compile(r'^\s+((?:[A-Z]\.\s+)?[A-Z][a-z]{2,})\b')
PRECEDING_TITLE = re.compile(r'([A-Z][a-z]{2,})\s+$')
PERSON_TITLES = {'Mr', 'Mrs', 'Ms', 'Dr', 'Professor', 'President',
                 'Senator', 'Governor', 'Coach', 'Captain', 'Sir'}


@dataclass(frozen=True)
class NameReuseExample:
    text: str
    source_name: str
    sampled_name: str
    name_group: str
    source_spans: tuple[tuple[int, int], ...]
    transformed_spans: tuple[tuple[int, int], ...]
    scored_occurrence: int
    gap_words: int
    target_pool_size: int

    @property
    def answer_span(self):
        return self.transformed_spans[self.scored_occurrence]

    @property
    def prefix(self):
        return self.text[:self.answer_span[0]]


def candidates(text, min_gap_words=20, max_gap_words=250,
               identity_policy='strict_full_name'):
    """Return supported later-mention pairs in one source window/document."""
    assert 0 <= min_gap_words <= max_gap_words
    assert identity_policy in ('strict_full_name', 'first_name')
    by_name = {}
    for match in NAME.finditer(text):
        if match.group() in NAME_TO_GROUP:
            by_name.setdefault(match.group(), []).append(match.span())
    options = []
    for source_name, spans in by_name.items():
        if len(spans) < 2:
            continue
        if identity_policy == 'strict_full_name':
            # Require every occurrence to belong to the *same repeated full
            # name*. Reject standalone first names and middle-name positions.
            full_names = []
            for start, end in spans:
                match = FULL_NAME_END.match(text[end:])
                prior = PRECEDING_TITLE.search(text[:start])
                if (not match or (prior and
                                  prior.group(1) not in PERSON_TITLES)):
                    full_names = []
                    break
                full_names.append(match.group(1))
            if len(full_names) != len(spans) or len(set(full_names)) != 1:
                continue
        else:
            # Useful as a lower-precision sensitivity control only.
            anchors = set()
            for _, end in spans:
                match = FOLLOWING_TITLE.match(text[end:])
                if match:
                    anchors.add(match.group(1))
            if len(anchors) > 1:
                continue
        for later in range(1, len(spans)):
            gap = len(WORD.findall(text[spans[later-1][1]:spans[later][0]]))
            if min_gap_words <= gap <= max_gap_words:
                options.append((source_name, later, gap))
    return tuple(options), by_name


def _replace(text, spans, sampled_name):
    pieces, transformed_spans, cursor, length = [], [], 0, 0
    for start, end in spans:
        chunk = text[cursor:start]
        pieces.extend((chunk, sampled_name))
        length += len(chunk)
        transformed_spans.append((length, length+len(sampled_name)))
        length += len(sampled_name)
        cursor = end
    pieces.append(text[cursor:])
    return ''.join(pieces), tuple(transformed_spans)


def transform(text, rng, min_gap_words=20, max_gap_words=250,
              identity_policy='strict_full_name',
              forced_source=None, forced_target=None,
              forced_occurrence=None):
    """Draw a consistent name substitution and a later prediction site.

    Returns None when the source text has no eligible repeated first name.
    Forced arguments support paired causal edits without a hidden-state leak.
    """
    options, spans_by_name = candidates(
        text, min_gap_words, max_gap_words, identity_policy)
    if forced_source is not None:
        options = tuple(row for row in options if row[0] == forced_source)
    if forced_occurrence is not None:
        options = tuple(row for row in options if row[1] == forced_occurrence)
    if not options:
        return None
    source_name, later, gap = rng.choice(options)
    group = NAME_TO_GROUP[source_name]
    present = {word.lower() for word in WORD.findall(text)}
    target_pool = tuple(name for name in NAME_GROUPS[group]
                        if name.lower() not in present and
                        name != source_name)
    if forced_target is not None:
        if forced_target not in target_pool:
            raise ValueError('Forced target is absent, incompatible, or collides')
        sampled_name = forced_target
    else:
        if not target_pool:
            return None
        sampled_name = rng.choice(target_pool)
    source_spans = tuple(spans_by_name[source_name])
    rewritten, transformed_spans = _replace(
        text, source_spans, sampled_name)
    return NameReuseExample(
        text=rewritten, source_name=source_name,
        sampled_name=sampled_name, name_group=group,
        source_spans=source_spans,
        transformed_spans=transformed_spans,
        scored_occurrence=later, gap_words=gap,
        target_pool_size=len(target_pool))


def reverse_transform(example):
    """Check that the only text changes are the declared name occurrences."""
    restored, _ = _replace(example.text,
                           example.transformed_spans,
                           example.source_name)
    return restored
