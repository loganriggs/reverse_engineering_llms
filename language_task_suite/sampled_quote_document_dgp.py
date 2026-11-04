"""Sample source-bound phrase repetition from related natural claims.

The claims come from one sampled source document. The speaker assignment and
later phrase reference are sampled as a joint document process; the answer is
checked from rendered text, without access to sampled metadata. This is a
semi-parametric DGP pilot, not a claim of natural document-level discourse.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import re

from document_world import NAMES


WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
DEPENDENT = frozenset(('he', 'she', 'they', 'it', 'its', 'his', 'her', 'their',
                       'them', 'this', 'these', 'those', 'former', 'latter',
                       'we', 'our', 'you', 'your', 'i', 'my'))
QUOTE = frozenset(('"', '“', '”', '‘', '’'))
OPENERS = frozenset(('and', 'but', 'however', 'therefore', 'also', 'instead',
                     'where', 'when', 'while', 'because', 'although', 'if',
                     'which', 'who', 'whose'))
INTRO_RE = re.compile(r'^([A-Z][a-z]+) (said|wrote|noted), "(.+[.!?])"$')
RECAP_RE = re.compile(
    r'^Later, ([A-Z][a-z]+) returned to the phrase "(.+)"\.$')


@dataclass(frozen=True)
class Claim:
    text: str
    phrases: tuple[str, ...]


@dataclass(frozen=True)
class Sample:
    text: str
    source_span: tuple[int, int]
    target_span: tuple[int, int]
    answer: str
    speaker: str
    source_index: int


def _claim(sent):
    raw = sent.text.strip()
    words = WORD.findall(raw)
    if not (8 <= len(words) <= 32 and raw.endswith('.') and
            not raw.endswith('..') and
            not any(symbol in raw for symbol in QUOTE) and
            not any(symbol in raw for symbol in ('—', '–', '...')) and
            '\n' not in raw and 'http' not in raw.lower() and
            words[0].lower() not in OPENERS and
            not any(w.lower() in DEPENDENT for w in words)):
        return None
    root = sent.root
    if root.pos_ != 'VERB' or not any(
            child.dep_ == 'nsubj' for child in root.children):
        return None
    phrases = []
    for chunk in sent.noun_chunks:
        phrase = chunk.text.strip()
        phrase_words = WORD.findall(phrase)
        if not (2 <= len(phrase_words) <= 6 and
                all(w.lower() not in DEPENDENT for w in phrase_words) and
                phrase in raw and
                not any(token.like_num for token in chunk)):
            continue
        phrases.append(phrase)
    if not phrases:
        return None
    return Claim(raw, tuple(dict.fromkeys(phrases)))


def eligible_claim_pairs(doc):
    """Extract two standalone claims from one source, in source order."""
    claims = [claim for sent in doc.sents if (claim := _claim(sent))]
    pairs = []
    for i, first in enumerate(claims):
        for second in claims[i+1:]:
            first_unique = [p for p in first.phrases
                            if p.lower() not in second.text.lower()]
            second_unique = [p for p in second.phrases
                             if p.lower() not in first.text.lower()]
            if first_unique and second_unique:
                pairs.append((first, second, tuple(first_unique),
                              tuple(second_unique)))
    return tuple(pairs)


def _render(first: Claim, second: Claim, speakers, verbs, chosen, phrase,
            source_index):
    lines = [f'{name} {verb}, "{claim.text}"'
             for name, verb, claim in zip(
                 speakers, verbs, (first, second), strict=True)]
    recap = (f'Later, {speakers[chosen]} returned to the phrase '
             f'"{phrase}".')
    text = '\n'.join((*lines, recap))
    source_start = text.index(phrase, text.index(lines[chosen]))
    target_start = text.rindex(phrase)
    result = Sample(text, (source_start, source_start + len(phrase)),
                    (target_start, target_start + len(phrase)),
                    phrase, speakers[chosen], source_index)
    check_visible(result.text)
    return result


def sample(doc, source_index: int, rng: random.Random):
    pairs = eligible_claim_pairs(doc)
    if not pairs:
        return None
    first, second, first_phrases, second_phrases = rng.choice(pairs)
    speakers = rng.sample(NAMES, 2)
    chosen = rng.randrange(2)
    phrase = rng.choice(first_phrases if chosen == 0 else second_phrases)
    verbs = rng.choices(('said', 'wrote', 'noted'), (.55, .3, .15), k=2)
    return _render(first, second, speakers, verbs, chosen, phrase, source_index)


def paired_variants(doc, source_index: int, rng: random.Random,
                    max_trials=32):
    """Yield valid speaker-swap pairs with the same recap speaker.

    Each pair draws two distinct source-specific noun chunks. Their BPE
    first-token contrast is checked later by the cache builder.
    """
    pairs = list(eligible_claim_pairs(doc))
    rng.shuffle(pairs)
    trials = 0
    for first, second, first_phrases, second_phrases in pairs:
        options = [(a, b) for a in first_phrases for b in second_phrases]
        rng.shuffle(options)
        for phrase_a, phrase_b in options:
            speakers = rng.sample(NAMES, 2)
            verbs = rng.choices(('said', 'wrote', 'noted'),
                                (.55, .3, .15), k=2)
            a = _render(first, second, speakers, verbs, 0,
                        phrase_a, source_index)
            b = _render(first, second, speakers[::-1], verbs, 1,
                        phrase_b, source_index)
            assert a.speaker == b.speaker and a.answer != b.answer
            yield a, b
            trials += 1
            if trials >= max_trials:
                return


def check_visible(text: str):
    """Return the correct phrase from visible source lines alone."""
    lines = text.splitlines()
    if len(lines) != 3:
        raise ValueError('Expected two claims and one recap')
    sources = {}
    for line in lines[:2]:
        match = INTRO_RE.fullmatch(line)
        if match is None:
            raise ValueError('Malformed source line')
        speaker, _, claim = match.groups()
        if speaker in sources:
            raise ValueError('Speaker repeated in source lines')
        sources[speaker] = claim
    recap = RECAP_RE.fullmatch(lines[2])
    if recap is None:
        raise ValueError('Malformed recap')
    speaker, phrase = recap.groups()
    if speaker not in sources or phrase not in sources[speaker]:
        raise ValueError('Phrase is not in that speaker’s visible claim')
    other = next(value for key, value in sources.items() if key != speaker)
    if phrase.lower() in other.lower():
        raise ValueError('Phrase is not source-specific')
    return speaker, phrase


def swap_source_names(text: str):
    """Swap the source speakers, leaving claims and the recap untouched."""
    lines = text.splitlines()
    source_lines = [INTRO_RE.fullmatch(line) for line in lines[:2]]
    if len(lines) != 3 or any(match is None for match in source_lines):
        raise ValueError('Malformed document')
    first, second = source_lines
    lines[0] = second.group(1) + lines[0][len(first.group(1)):]
    lines[1] = first.group(1) + lines[1][len(second.group(1)):]
    return '\n'.join(lines)
