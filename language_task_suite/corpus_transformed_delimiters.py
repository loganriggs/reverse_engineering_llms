"""Sample the delimiter style of existing natural parenthetical spans.

The enclosed words come from a source passage, not a synthetic template.
Only well-formed, nonnested prose parentheses are eligible. A random style
choice makes the later closing delimiter condition on the visible opener.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass


WORD = re.compile(r"\b[A-Za-z]+(?:'[A-Za-z]+)?\b")
STYLES = {'round': ('(', ')'), 'square': ('[', ']')}


@dataclass(frozen=True)
class DelimiterExample:
    text: str
    source_text: str
    pair_index: int
    style: str
    open_index: int
    close_index: int
    gap_words: int

    @property
    def answer(self):
        return STYLES[self.style][1]

    @property
    def prefix(self):
        return self.text[:self.close_index]


def candidates(text, min_words=3, max_words=30, max_characters=220):
    """Return parenthesis index pairs containing ordinary prose."""
    assert 1 <= min_words <= max_words and max_characters > 0
    stack, pairs = [], []
    for index, character in enumerate(text):
        if character == '(':
            if stack:
                parent_start, _ = stack[-1]
                stack[-1] = (parent_start, True)
            stack.append((index, bool(stack)))
        elif character == ')' and stack:
            start, nested = stack.pop()
            if nested or stack:
                continue
            inner = text[start+1:index]
            if ('\n' in inner or '\r' in inner or
                    any(mark in inner for mark in '[]()<>{}=\\') or
                    len(inner) > max_characters or
                    'http' in inner.lower()):
                continue
            word_count = len(WORD.findall(inner))
            if min_words <= word_count <= max_words:
                pairs.append((start, index, word_count))
    return tuple(pairs)


def transform(text, rng: random.Random, style=None, pair_index=None,
              min_words=3, max_words=30, square_rate=.5):
    """Draw one eligible pair and a style; return None without one."""
    if not 0 <= square_rate <= 1:
        raise ValueError('square_rate must be between zero and one')
    options = candidates(text, min_words, max_words)
    if not options:
        return None
    if pair_index is None:
        pair_index = rng.randrange(len(options))
    if not 0 <= pair_index < len(options):
        raise ValueError('Pair index outside eligible candidates')
    if style is None:
        style = 'square' if rng.random() < square_rate else 'round'
    if style not in STYLES:
        raise ValueError(style)
    start, end, gap = options[pair_index]
    opener, closer = STYLES[style]
    changed = text[:start]+opener+text[start+1:end]+closer+text[end+1:]
    return DelimiterExample(changed, text, pair_index, style,
                            start, end, gap)


def validate(example: DelimiterExample):
    """Independent string-level check of the localized transformation."""
    assert example.source_text[example.open_index] == '('
    assert example.source_text[example.close_index] == ')'
    assert example.text[example.open_index] == STYLES[example.style][0]
    assert example.text[example.close_index] == example.answer
    assert example.text[example.open_index+1:example.close_index] == \
        example.source_text[example.open_index+1:example.close_index]
    assert (example.text[:example.open_index] ==
            example.source_text[:example.open_index])
    assert (example.text[example.close_index+1:] ==
            example.source_text[example.close_index+1:])
    assert len(WORD.findall(example.text[
        example.open_index+1:example.close_index])) == example.gap_words
    return True
