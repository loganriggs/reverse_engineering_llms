"""Corpus-derived joint agreement, parenthetical, and repetition DGP.

This is a multi-rule syntax/document candidate, not a semantic world model.
The treebank supplies variable NP/VP and PP surfaces. Rule gates, subject
reuse, and brackets are sampled together in each document. The corpus source
stays local; only aggregate audits belong in Git.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import random
from collections import defaultdict

from audit_raw_pcfg_treebank_pilot import leaves, parse_tree
from audit_tree_substitution_pilot import walk
from treebank_agreement_dgp import (
    PAIRS, AgreementWorld, load_world, subject_candidate, vp_candidate,
)


PREPOSITIONS = frozenset(('in', 'on', 'at', 'with', 'from', 'near',
                          'behind', 'beside', 'under', 'over', 'by'))


@dataclass(frozen=True)
class Site:
    family: str
    position: int
    answer: tuple[str, ...]
    source_span: tuple[int, int] | None = None
    lemma: str | None = None


@dataclass(frozen=True)
class Document:
    tokens: tuple[str, ...]
    sites: tuple[Site, ...]
    controls: dict
    clauses: tuple[tuple[str, ...], ...]

    def text(self):
        words = ' '.join(self.tokens)
        for before, after in ((' .', '.'), (' ,', ','), (' ;', ';'),
                              (' )', ')'), ('( ', '('), (' ]', ']'),
                              ('[ ', '[')):
            words = words.replace(before, after)
        return words.replace(' \n ', '\n')


@dataclass(frozen=True)
class SourceClause:
    subject: tuple[str, ...]
    number: int
    lemma: str
    continuation: tuple[str, ...]


@lru_cache(maxsize=2)
def source_clause_pool(path: str, fit_trees=100000):
    clauses = []
    seen = set()
    with Path(path).open() as handle:
        for _, line in zip(range(fit_trees), handle):
            tree = parse_tree(line)
            for _, node in walk(tree):
                if node[0] != 'S':
                    continue
                children = node[1]
                if (len(children) not in (2, 3) or
                        any(isinstance(child, str) for child in children) or
                        children[0][0] != 'NP' or
                        children[1][0] != 'VP' or
                        (len(children) == 3 and
                         tuple(leaves(children[2])) != ('.',))):
                    continue
                subject = subject_candidate(children[0])
                predicate = vp_candidate(children[1])
                if subject is None or predicate is None:
                    continue
                words, number, _ = subject
                lemma, rest = predicate
                if tuple(leaves(node)) != words + (
                        PAIRS[lemma][number],) + rest + (
                        ('.',) if len(children) == 3 else ()):
                    continue
                item = SourceClause(words, number, lemma, rest)
                if item not in seen:
                    seen.add(item)
                    clauses.append(item)
    if len(clauses) < 50:
        raise ValueError('Too few complete source clauses')
    return tuple(clauses)


@lru_cache(maxsize=2)
def pp_pool(path: str, fit_trees=100000):
    phrases = set()
    with Path(path).open() as handle:
        for _, line in zip(range(fit_trees), handle):
            tree = parse_tree(line)
            for _, node in walk(tree):
                if node[0] != 'PP':
                    continue
                children = node[1]
                if (len(children) != 2 or
                        any(isinstance(child, str) for child in children) or
                        children[0][0] not in ('IN', 'TO') or
                        children[1][0] != 'NP'):
                    continue
                words = tuple(leaves(node))
                if (2 <= len(words) <= 7 and
                        words[0].lower() in PREPOSITIONS and
                        all(word.isalpha() for word in words)):
                    phrases.add(tuple(word.lower() for word in words))
    if len(phrases) < 100:
        raise ValueError('Too few clean corpus PP phrases')
    return tuple(sorted(phrases))


@lru_cache(maxsize=2)
def head_matched_subjects(world: AgreementWorld):
    index = defaultdict(list)
    for number in (0, 1):
        for words in world.subjects[number]:
            if (2 <= len(words) <= 5 and
                    all(word.isalpha() for word in words) and
                    not any(word.lower() in PREPOSITIONS for word in words)):
                index[(number, words[-1].lower())].append(words)
    return {key: tuple(value) for key, value in index.items()}


def sample(world: AgreementWorld, parentheticals, seed, *,
           n_clauses=4, bracket_rate=.25, reuse_rate=.35,
           attractor_rate=.2, paragraph_rate=.2,
           source_clauses=(), source_rate=0., head_match_rate=0.):
    if not (2 <= n_clauses <= 8 and all(0 <= p <= 1 for p in
            (bracket_rate, reuse_rate, attractor_rate, paragraph_rate,
             source_rate, head_match_rate))):
        raise ValueError('Invalid sampling controls')
    if source_rate and not source_clauses:
        raise ValueError('Source clauses required for source_rate > 0')
    rng = random.Random(seed)
    tokens, sites, clauses = [], [], []
    previous = []
    head_swaps = 0
    for index in range(n_clauses):
        reuse = bool(previous) and rng.random() < reuse_rate
        if reuse:
            subject, number, source_span = rng.choice(previous)
            matching = [clause for clause in source_clauses
                        if clause.number == number] if source_rate and \
                rng.random() < source_rate else ()
            base = rng.choice(matching) if matching else world.sample(
                rng, number=number, attractor_rate=attractor_rate)
        else:
            base = (rng.choice(source_clauses)
                    if source_rate and rng.random() < source_rate else
                    world.sample(rng, attractor_rate=attractor_rate))
            subject, number, source_span = base.subject, base.number, None
            subject = (subject[0].capitalize(),) + subject[1:]
            if (head_match_rate and rng.random() < head_match_rate and
                    2 <= len(subject) <= 5 and
                    not any(word.lower() in PREPOSITIONS
                            for word in subject)):
                pool = head_matched_subjects(world).get(
                    (number, subject[-1].lower()), ())
                candidates = [words for words in pool
                              if tuple(word.lower() for word in words) !=
                              tuple(word.lower() for word in subject)]
                if candidates:
                    chosen = rng.choice(candidates)
                    subject = (chosen[0].capitalize(),) + chosen[1:]
                    head_swaps += 1
        if index and rng.random() < paragraph_rate:
            tokens.append('\n')
        start = len(tokens)
        tokens.extend(subject)
        if reuse:
            sites.append(Site('subject_reuse', start, subject,
                              source_span))
        else:
            previous.append((subject, number,
                             (start, start + len(subject))))
        if rng.random() < bracket_rate:
            phrase = rng.choice(parentheticals)
            tokens.append('(')
            tokens.extend(phrase)
            sites.append(Site('bracket_close', len(tokens), (')',)))
            tokens.append(')')
        sites.append(Site('agreement', len(tokens),
                          (PAIRS[base.lemma][number],),
                          (start, start+len(subject)), base.lemma))
        tokens.append(PAIRS[base.lemma][number])
        tokens.extend(base.continuation)
        tokens.append('.')
        clauses.append(tuple(tokens[start:]))
    doc = Document(tuple(tokens), tuple(sites),
                   {'n_clauses': n_clauses,
                    'bracket_rate': bracket_rate,
                    'reuse_rate': reuse_rate,
                    'attractor_rate': attractor_rate,
                    'paragraph_rate': paragraph_rate,
                    'source_rate': source_rate,
                    'head_match_rate': head_match_rate,
                    'head_swaps': head_swaps}, tuple(clauses))
    check_visible(doc, world)
    return doc


def check_visible(doc: Document, world: AgreementWorld):
    for site in doc.sites:
        actual = doc.tokens[site.position:site.position+len(site.answer)]
        if actual != site.answer:
            raise ValueError(('Site token mismatch', site, actual))
        prefix = doc.tokens[:site.position]
        if site.family == 'bracket_close':
            stack = []
            for token in prefix:
                if token in ('(', '['):
                    stack.append(token)
                elif token in (')', ']'):
                    if not stack or (stack.pop(), token) not in (
                            ('(', ')'), ('[', ']')):
                        raise ValueError('Bad visible bracket history')
            if stack != ['(']:
                raise ValueError('Closing site has no unique open bracket')
            if site.answer != (')',):
                raise ValueError('Closing token contradicts visible opener')
        elif site.family == 'agreement':
            assert site.source_span is not None
            lo, hi = site.source_span
            subject = prefix[lo:hi]
            number = world.number_from_visible_prefix(subject)
            if site.lemma not in PAIRS or site.answer != (
                    PAIRS[site.lemma][number],):
                raise ValueError('Verb does not agree with visible subject')
        elif site.family == 'subject_reuse':
            assert site.source_span is not None
            lo, hi = site.source_span
            if prefix[lo:hi] != site.answer:
                raise ValueError('Repeat differs from earlier visible text')
        else:
            raise ValueError(site.family)
    return True


def load(path, fit_trees=100000):
    world = load_world(str(path), fit_trees)
    source = tuple(clause for clause in source_clause_pool(str(path),
                                                          fit_trees)
                   if world.subject_number.get(tuple(
                       word.lower() for word in clause.subject)) ==
                   clause.number)
    return world, pp_pool(str(path), fit_trees), source
