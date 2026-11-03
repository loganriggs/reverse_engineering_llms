"""Sample agreement clauses from parsed corpus phrases with a known rule.

Subjects and present-tense VP continuations are sampled independently from
parsed treebank pools. A visible subject-phrase lexicon fixes number, and
the finite verb must agree. This is a syntax-task pilot, not a discourse DGP.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
import random

from audit_raw_pcfg_treebank_pilot import leaves, parse_tree
from audit_tree_substitution_pilot import walk


PAIRS = {
    'be': ('is', 'are'),
    'have': ('has', 'have'),
    'do': ('does', 'do'),
}
FORM_TO_PAIR = {form: (lemma, number)
                for lemma, forms in PAIRS.items()
                for number, form in enumerate(forms)}
BLOCKED = frozenset(('i', 'you', 'we', 'my', 'your', 'our'))
SINGULAR_DETERMINERS = frozenset(('a', 'an', 'this', 'that', 'every', 'each'))
PLURAL_DETERMINERS = frozenset(('these', 'those', 'many', 'several'))
ALLOWED_STARTS = frozenset(('the', 'a', 'an', 'this', 'that', 'these',
                            'those', 'his', 'her', 'their', 'some', 'all',
                            'many', 'several', 'every', 'each'))


@dataclass(frozen=True)
class Clause:
    tokens: tuple[str, ...]
    target_index: int
    lemma: str
    number: int  # 0 singular, 1 plural
    subject: tuple[str, ...]
    continuation: tuple[str, ...]
    attractor: bool
    subject_split: str


def noun_head(node):
    """Conservative direct-child NP head rule; reject coordination."""
    if node[0] != 'NP' or any(
            not isinstance(child, str) and child[0] == 'CC'
            for child in node[1]):
        return None
    children = [child for child in node[1]
                if not isinstance(child, str)]
    for child in reversed(children):
        if child[0] in ('NN', 'NNP'):
            return 0
        if child[0] in ('NNS', 'NNPS'):
            return 1
        if child[0] == 'PRP':
            words = [x.lower() for x in leaves(child)]
            if words in (['he'], ['she'], ['it']):
                return 0
            if words == ['they']:
                return 1
        if child[0] == 'NP':
            return noun_head(child)
    return None


def subject_candidate(tree):
    words = tuple(leaves(tree))
    if not 2 <= len(words) <= 9 or any(
            token.lower() in BLOCKED for token in words):
        return None
    if words[-1] in ("'s", "s'") or any(
            node[0] in ('VP', 'S', 'SBAR')
            for path, node in walk(tree) if path):
        return None
    if not all(token.isalpha() or token in ("'s", '-')
               for token in words):
        return None
    number = noun_head(tree)
    if number is None:
        return None
    first = words[0].lower()
    if first not in ALLOWED_STARTS:
        return None
    if first in SINGULAR_DETERMINERS and number != 0 or \
            first in PLURAL_DETERMINERS and number != 1:
        return None
    attractor = False
    for child in tree[1]:
        if isinstance(child, str) or child[0] != 'PP':
            continue
        for _, inside in walk(child):
            if inside[0] == 'NP' and noun_head(inside) is not None:
                attractor |= noun_head(inside) != number
                break
    return words, number, attractor


def vp_candidate(tree):
    if tree[0] != 'VP' or not tree[1] or \
            isinstance(tree[1][0], str):
        return None
    first = tree[1][0]
    word = tuple(leaves(first))
    if len(word) != 1 or word[0].lower() not in FORM_TO_PAIR or \
            first[0] not in ('VBZ', 'VBP'):
        return None
    lemma, number = FORM_TO_PAIR[word[0].lower()]
    if first[0] != ('VBZ' if number == 0 else 'VBP'):
        return None
    # Flatten non-terminal children too, preserving source order.
    rest = tuple(token for child in tree[1][1:]
                 for token in ((child,) if isinstance(child, str)
                               else tuple(leaves(child))))
    if not 2 <= len(rest) <= 22 or any(token in ('``', "''", '?', '!')
                                      for token in rest):
        return None
    if any(token.lower() in BLOCKED for token in rest):
        return None
    return lemma, rest


class AgreementWorld:
    def __init__(self, trees):
        subjects = defaultdict(set)
        subject_conditions = defaultdict(set)
        verbs = defaultdict(set)
        for tree in trees:
            for _, node in walk(tree):
                if node[0] == 'NP':
                    item = subject_candidate(node)
                    if item:
                        subjects[item[0]].add(item[1])
                        subject_conditions[item[0]].add(item[2])
                elif node[0] == 'VP':
                    item = vp_candidate(node)
                    if item:
                        verbs[item[0]].add(item[1])
        # A surface phrase with inconsistent parse labels is not an oracle.
        normalized = defaultdict(set)
        for words, numbers in subjects.items():
            normalized[tuple(token.lower() for token in words)].update(numbers)
        self.subject_number = {words: next(iter(numbers))
                               for words, numbers in normalized.items()
                               if len(numbers) == 1}
        self.subjects = {
            number: tuple(words for words, labels in subjects.items()
                          if self.subject_number.get(tuple(
                              token.lower() for token in words)) == number)
            for number in (0, 1)}
        self.subjects_by_condition = {
            (number, attractor): tuple(
                words for words, labels in subjects.items()
                if self.subject_number.get(tuple(
                    token.lower() for token in words)) == number and
                attractor in subject_conditions[words])
            for number in (0, 1) for attractor in (False, True)}
        self.subjects_by_split = {
            (split, number, attractor): tuple(
                words for words in self.subjects_by_condition[
                    number, attractor]
                if self._subject_split(words) == split)
            for split in ('train', 'heldout')
            for number in (0, 1) for attractor in (False, True)}
        self.continuations = {lemma: tuple(sorted(forms))
                              for lemma, forms in verbs.items()}
        if min(len(v) for v in self.subjects.values()) < 100:
            raise RuntimeError('Not enough clean subjects')
        if not self.continuations:
            raise RuntimeError('No finite verb continuations')

    @staticmethod
    def _subject_split(words):
        key = ' '.join(word.lower() for word in words).encode()
        value = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big')
        return 'heldout' if value % 5 == 0 else 'train'

    def sample(self, rng: random.Random, number=None, lemma=None,
               attractor=None, attractor_rate=.2,
               subject_split='all'):
        number = rng.randrange(2) if number is None else number
        lemma = rng.choice(tuple(self.continuations)) if lemma is None \
            else lemma
        attractor = rng.random() < attractor_rate if attractor is None \
            else attractor
        if subject_split == 'all':
            pool = self.subjects_by_condition[number, attractor]
        else:
            pool = self.subjects_by_split[
                subject_split, number, attractor]
        subject = rng.choice(pool)
        continuation = rng.choice(self.continuations[lemma])
        verb = PAIRS[lemma][number]
        subject = (subject[0].capitalize(),)+subject[1:]
        tokens = subject+(verb,)+continuation+('.',)
        return Clause(tokens, len(subject), lemma, number,
                      subject, continuation, attractor,
                      self._subject_split(subject))

    def number_from_visible_prefix(self, prefix):
        phrase = tuple(prefix)
        if not phrase:
            raise ValueError('Empty subject')
        original = tuple(token.lower() for token in phrase)
        if original not in self.subject_number:
            raise ValueError('Unknown or ambiguous subject phrase')
        return self.subject_number[original]

    def counterfactual(self, clause, rng):
        changed = self.sample(rng, 1-clause.number, clause.lemma,
                              attractor=clause.attractor,
                              subject_split=clause.subject_split)
        tokens = changed.subject+(PAIRS[clause.lemma][1-clause.number],)+ \
            clause.continuation+('.',)
        return Clause(tokens, len(changed.subject), clause.lemma,
                      1-clause.number, changed.subject,
                      clause.continuation, changed.attractor,
                      changed.subject_split)


@lru_cache(maxsize=2)
def load_world(path: str, fit_trees=20000):
    trees = []
    with Path(path).open() as handle:
        for _, line in zip(range(fit_trees), handle):
            trees.append(parse_tree(line))
    return AgreementWorld(trees)
