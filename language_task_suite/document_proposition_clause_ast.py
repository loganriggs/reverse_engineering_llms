"""Typed, sampled clauses for comparison and current-fact statements.

The surface templates are a small grammar, not a claim of natural-language
realism. Parsing matches the visible text; it never reads the sampled plan.
"""

from dataclasses import dataclass

from document_world import VALUES


@dataclass(frozen=True)
class Clause:
    kind: str
    entity: str
    attr: str
    other: str | None = None


@dataclass(frozen=True)
class Frame:
    kind: str
    prefix: tuple[str, ...]
    suffix: tuple[str, ...]
    positive: str | None = None
    negative: str | None = None


COMPARISON_FRAMES = (
    Frame('comparison', ('With', 'respect', 'to', '{attr}', ',',
                         '{entity}', 'and', '{other}'), (), 'matched', 'differed'),
    Frame('comparison', ('The', '{attr}', 'readings', 'for', '{entity}',
                         'and', '{other}'), (), 'agreed', 'diverged'),
    Frame('comparison', ('On', '{attr}', ',', '{entity}', 'and', '{other}'),
          (), 'matched', 'differed'),
    Frame('comparison', ('For', '{attr}', ',', '{entity}', 'and',
                         '{other}', 'showed', 'a'),
          ('on', 'that', 'measure'), 'match', 'mismatch'),
    Frame('comparison', ('Between', '{entity}', 'and', '{other}', ',',
                         'there', 'was', 'a', '{attr}'),
          (), 'match', 'mismatch'),
    Frame('comparison', ('The', '{attr}', 'for', '{entity}', 'and',
                         '{other}', 'was', 'a'), (), 'match', 'mismatch'),
)

FACT_FRAMES = (
    Frame('fact', ('At', 'present', ',', 'the', '{attr}', 'for',
                   '{entity}', 'was'), ()),
    Frame('fact', ('For', '{entity}', ',', 'the', 'current', '{attr}',
                   'was'), ()),
    Frame('fact', ('Looking', 'at', '{entity}', ',', 'its', '{attr}',
                   'was'), ()),
    Frame('fact', ('The', '{attr}', 'currently', 'recorded', 'for',
                   '{entity}', 'was'), ()),
    Frame('fact', ('Current', 'records', 'gave', '{entity}', 'a',
                   '{attr}', 'of'), ()),
    Frame('fact', ('Reports', 'put', 'the', '{attr}', 'of', '{entity}',
                   'at'), ()),
    Frame('fact', ('For', '{entity}', ',', 'records', 'listed',
                   'its', '{attr}', 'as'),
          ('in', 'the', 'summary')),
)


def _fill(words, clause):
    fields = {'entity': clause.entity, 'attr': clause.attr,
              'other': clause.other}
    return tuple(fields[word[1:-1]] if word.startswith('{') else word
                 for word in words)


def render(clause: Clause, frame_index: int, value: str):
    frames = COMPARISON_FRAMES if clause.kind == 'comparison' else FACT_FRAMES
    assert clause.kind in ('comparison', 'fact')
    frame = frames[frame_index % len(frames)]
    prefix, suffix = _fill(frame.prefix, clause), _fill(frame.suffix, clause)
    if clause.kind == 'comparison':
        assert value in ('same', 'different')
        answer = frame.positive if value == 'same' else frame.negative
        candidates = (frame.positive, frame.negative)
    else:
        assert value in VALUES[clause.attr]
        answer = value
        candidates = tuple(VALUES[clause.attr])
    return prefix, answer, suffix, candidates


def _match(pattern, actual, captures):
    if len(pattern) != len(actual):
        return False
    for token, value in zip(pattern, actual):
        if token.startswith('{'):
            field = token[1:-1]
            if field in captures and captures[field] != value:
                return False
            captures[field] = value
        elif token != value:
            return False
    return True


def parse_prefix(raw):
    """Return a visible clause frame and captured semantic arguments, if any."""
    for frame in COMPARISON_FRAMES + FACT_FRAMES:
        captures = {}
        if _match(frame.prefix, raw, captures):
            if captures.get('attr') not in VALUES:
                continue
            return frame, captures
    return None


def parse_complete(raw):
    """Recognize a complete clause whose answer may be inside the sentence."""
    for frame in COMPARISON_FRAMES + FACT_FRAMES:
        split = len(frame.prefix)
        if len(raw) != split + 1 + len(frame.suffix):
            continue
        captures = {}
        if not _match(frame.prefix, raw[:split], captures):
            continue
        if not _match(frame.suffix, raw[split+1:], captures):
            continue
        if captures.get('attr') not in VALUES:
            continue
        answer = raw[split]
        if frame.kind == 'comparison' and answer not in (
                frame.positive, frame.negative):
            continue
        if frame.kind == 'fact' and answer not in VALUES[captures['attr']]:
            continue
        return frame, captures, answer
    return None
