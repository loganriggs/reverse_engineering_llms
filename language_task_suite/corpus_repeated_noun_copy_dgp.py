"""Paired same-length noun substitutions in natural source documents.

The edits preserve local word shape and reuse an earlier sampled noun at a
later occurrence. They do not preserve every original factual claim. This is
a controlled surface-copy task, not a semantic entity-binding oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import re

import spacy

from corpus_rule_reference_dgp import Document, RuleSite, check_document


GROUPS = {
    'place': ('school', 'church', 'garden', 'market', 'office',
              'store', 'house', 'hotel', 'court', 'beach', 'river',
              'field', 'coast', 'route'),
    'document': ('report', 'letter', 'notice', 'record', 'paper', 'sheet'),
    'role': ('doctor', 'writer', 'singer', 'player', 'leader',
             'teacher', 'manager', 'officer'),
    'object': ('table', 'phone', 'chair', 'board', 'camera',
              'bottle', 'laptop', 'screen'),
    'event': ('event', 'party', 'trial', 'meeting', 'session'),
}
WORD_GROUP = {word: group for group, words in GROUPS.items()
              for word in words}


@dataclass(frozen=True)
class Plan:
    word: str
    group: str
    spans: tuple[tuple[int, int], ...]
    target_occurrence: int
    gap_words: int


def load_nlp():
    return spacy.load('en_core_web_sm',
                      disable=['parser', 'ner', 'lemmatizer'])


def candidates(text: str, parsed, min_gap=20, max_gap=160):
    by_word = {}
    for i, token in enumerate(parsed):
        if (token.text in WORD_GROUP and token.pos_ == 'NOUN' and
                'Sing' in token.morph.get('Number') and token.idx > 0 and
                text[token.idx-1] == ' '):
            by_word.setdefault(token.text, []).append(
                (i, token.idx, token.idx+len(token)))
    for word, locations in by_word.items():
        all_mentions = list(re.finditer(
            r'(?<![A-Za-z])'+re.escape(word)+r'(?![A-Za-z])', text, re.I))
        if (len(all_mentions) != len(locations) or
                any(match.group() != word for match in all_mentions)):
            continue
        spans = tuple((start, end) for _, start, end in locations)
        for target in range(1, len(locations)):
            gap = locations[target][0]-locations[target-1][0]
            if min_gap <= gap <= max_gap:
                yield Plan(word, WORD_GROUP[word], spans, target, gap)


def _vowel(word):
    return word[0] in 'aeiou'


def replacement_pool(plan: Plan, source_text: str, tokenizer):
    vocab = tokenizer.get_vocab()
    return tuple(word for word in GROUPS[plan.group]
                 if word != plan.word and len(word) == len(plan.word)
                 and _vowel(word) == _vowel(plan.word)
                 and 'Ġ'+word in vocab
                 and not re.search(r'(?<![A-Za-z])'+re.escape(word)+
                                   r'(?![A-Za-z])', source_text, re.I))


def realize(base: Document, plan: Plan, replacement: str):
    if (replacement == plan.word or
            len(replacement) != len(plan.word) or
            replacement not in GROUPS[plan.group] or
            _vowel(replacement) != _vowel(plan.word)):
        raise ValueError('Invalid noun replacement')
    chars = list(base.text)
    for start, end in plan.spans:
        assert base.text[start:end] == plan.word
        chars[start:end] = replacement
    text = ''.join(chars)
    assert len(text) == len(base.text)
    site = RuleSite('noun_copy', plan.spans[plan.target_occurrence-1],
                    plan.spans[plan.target_occurrence], replacement)
    result = Document(base.source_text, text,
                      base.sites+(site,), base.eligible+('noun_copy',),
                      base.applied+('noun_copy',))
    check_document(result)
    return result


def paired_document(base: Document, parsed, tokenizer,
                    rng: random.Random):
    eligible = []
    for plan in candidates(base.source_text, parsed):
        pool = replacement_pool(plan, base.source_text, tokenizer)
        if len(pool) >= 2:
            eligible.append((plan, pool))
    if not eligible:
        return None
    plan, pool = rng.choice(eligible)
    left, right = rng.sample(pool, 2)
    return realize(base, plan, left), realize(base, plan, right), plan
