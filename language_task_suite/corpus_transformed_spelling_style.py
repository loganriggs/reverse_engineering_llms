"""Document-wide sampled US/UK spelling style in natural prose.

Only lowercase, whole-word spelling variants are rewritten. A scored target
follows an adjacent eligible style word from a different lexical family,
20–300 words away. The selected model window starts after any earlier style
word, leaving exactly one visible style cue before the scored target.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


FAMILIES = {
    'color': (('color', 'colour'), ('colors', 'colours'),
              ('colored', 'coloured'), ('colorful', 'colourful')),
    'favor': (('favor', 'favour'), ('favors', 'favours'),
              ('favored', 'favoured'), ('favoring', 'favouring'),
              ('favorite', 'favourite'), ('favorites', 'favourites')),
    'flavor': (('flavor', 'flavour'), ('flavors', 'flavours')),
    'honor': (('honor', 'honour'), ('honored', 'honoured')),
    'behavior': (('behavior', 'behaviour'),
                 ('behaviors', 'behaviours')),
    'labor': (('labor', 'labour'), ('labors', 'labours')),
    'rumor': (('rumor', 'rumour'), ('rumors', 'rumours')),
    'humor': (('humor', 'humour'),),
    'neighbor': (('neighbor', 'neighbour'),
                 ('neighbors', 'neighbours')),
    'center': (('center', 'centre'), ('centers', 'centres')),
    'theater': (('theater', 'theatre'), ('theaters', 'theatres')),
    'organize': (('organize', 'organise'),
                 ('organized', 'organised'),
                 ('organization', 'organisation'),
                 ('organizations', 'organisations')),
    'recognize': (('recognize', 'recognise'),
                  ('recognized', 'recognised')),
    'realize': (('realize', 'realise'),
                ('realized', 'realised')),
    'criticize': (('criticize', 'criticise'),
                  ('criticized', 'criticised')),
    'optimize': (('optimize', 'optimise'),
                 ('optimized', 'optimised')),
    'minimize': (('minimize', 'minimise'),
                 ('minimized', 'minimised')),
    'analyze': (('analyze', 'analyse'), ('analyzed', 'analysed')),
    'travel': (('traveling', 'travelling'), ('traveler', 'traveller')),
    'cancel': (('canceled', 'cancelled'), ('canceling', 'cancelling')),
    'label': (('labeled', 'labelled'), ('labeling', 'labelling')),
    'catalog': (('catalog', 'catalogue'),),
    'dialog': (('dialog', 'dialogue'),),
    'defense': (('defense', 'defence'),),
    'offense': (('offense', 'offence'),),
    'aluminum': (('aluminum', 'aluminium'),),
    'fiber': (('fiber', 'fibre'), ('fibers', 'fibres')),
    'jewelry': (('jewelry', 'jewellery'),),
    'liter': (('liter', 'litre'), ('liters', 'litres')),
    'sulfur': (('sulfur', 'sulphur'),),
    'gray': (('gray', 'grey'),),
}
LOOKUP = {word: (root, us, uk)
          for root, pairs in FAMILIES.items()
          for us, uk in pairs for word in (us, uk)}
assert len(LOOKUP) == 2*sum(len(rows) for rows in FAMILIES.values())
PATTERN = re.compile(r'\b(?:'+'|'.join(
    re.escape(word) for word in sorted(LOOKUP, key=len, reverse=True)
)+r')\b', re.I)
WORD = re.compile(r'\b[A-Za-z]+\b')


@dataclass(frozen=True)
class Occurrence:
    start: int
    end: int
    source_word: str
    root: str
    us: str
    uk: str


@dataclass(frozen=True)
class StyleExample:
    text: str
    source_text: str
    style: str
    cue_span: tuple[int, int]
    target_span: tuple[int, int]
    cue_root: str
    target_root: str
    target_word: str
    source_occurrences: tuple[Occurrence, ...]
    transformed_spans: tuple[tuple[int, int], ...]
    gap_words: int
    target_occurrence_index: int


def occurrences(text):
    result = []
    for match in PATTERN.finditer(text):
        source_word = match.group()
        if not source_word.islower():
            continue  # Avoid changing official names and sentence starts.
        root, us, uk = LOOKUP[source_word]
        result.append(Occurrence(match.start(), match.end(),
                                 source_word, root, us, uk))
    return tuple(result)


def candidate_indices(text, min_gap_words=20, max_gap_words=300):
    """Adjacent style words with different lexical roots and bounded gap."""
    rows = occurrences(text)
    result = []
    for index in range(1, len(rows)):
        cue, target = rows[index-1:index+1]
        if cue.root == target.root:
            continue
        gap = len(WORD.findall(text[cue.end:target.start]))
        if min_gap_words <= gap <= max_gap_words:
            result.append((index, gap))
    return rows, tuple(result)


def strict_candidate(text, min_gap_words=20, max_gap_words=300):
    rows, options = candidate_indices(text, min_gap_words, max_gap_words)
    return (rows, options[0]) if options else None


def transform(text, style, min_gap_words=20, max_gap_words=300,
              target_occurrence_index=None):
    if style not in ('us', 'uk'):
        raise ValueError(style)
    rows, options = candidate_indices(text, min_gap_words, max_gap_words)
    if not options:
        return None
    if target_occurrence_index is None:
        target_occurrence_index, gap = options[0]
    else:
        matching = [gap for index, gap in options
                    if index == target_occurrence_index]
        if not matching:
            raise ValueError('Target occurrence is not an eligible pair')
        gap = matching[0]
    pieces, spans, cursor, length = [], [], 0, 0
    for row in rows:
        prefix = text[cursor:row.start]
        replacement = row.us if style == 'us' else row.uk
        pieces.extend((prefix, replacement))
        length += len(prefix)
        spans.append((length, length+len(replacement)))
        length += len(replacement)
        cursor = row.end
    pieces.append(text[cursor:])
    index = target_occurrence_index
    return StyleExample(''.join(pieces), text, style,
                        spans[index-1], spans[index], rows[index-1].root,
                        rows[index].root,
                        rows[index].us if style == 'us' else rows[index].uk,
                        rows, tuple(spans), gap, index)


def reverse_transform(example):
    pieces, cursor = [], 0
    for row, (start, end) in zip(example.source_occurrences,
                                  example.transformed_spans):
        pieces.extend((example.text[cursor:start], row.source_word))
        cursor = end
    pieces.append(example.text[cursor:])
    return ''.join(pieces)
