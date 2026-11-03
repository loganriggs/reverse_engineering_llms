"""Natural-report surname reuse with controlled, document-wide aliases.

The report prose is human-written. The controlled intervention renames one
player consistently; a later surname mention is the scored site. The original
coreference is inferred by a conservative surface heuristic, so this is a
candidate task requiring manual validation, not a full symbolic LM oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import random
import re
from collections import Counter

from audit_rotowire_document_source import game_key
from sports_document_dgp import game_partition, resources


NAME = re.compile(r'^[A-Z][a-z]+$')


@dataclass(frozen=True)
class Plan:
    game_key: tuple
    source_tokens: tuple[str, ...]
    first: str
    original_surname: str
    source_token: int
    target_token: int
    alias_a: str
    alias_b: str


@dataclass(frozen=True)
class Document:
    text: str
    source_start: int
    source_end: int
    answer_start: int
    answer_end: int
    answer: str
    plan: Plan


@lru_cache(maxsize=1)
def surname_counts():
    names = Counter()
    for row in resources()[0]['rotowire'][0]:
        if game_partition(game_key(row)) != 'train':
            continue
        names.update(name.split()[-1]
                     for name in row['box_score']['PLAYER_NAME'].values()
                     if len(name.split()) == 2 and NAME.fullmatch(name.split()[-1]))
    return names


def surname_pool():
    return tuple(sorted(surname_counts()))


@lru_cache(maxsize=1)
def single_bpe_surname_pool():
    from tokenizers import Tokenizer
    from audit_fineweb_surfaces import DEFAULT_TOKENIZER
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    return tuple(name for name in surname_pool()
                 if len(tokenizer.encode(' '+name,
                                         add_special_tokens=False).ids) == 1)


@lru_cache(maxsize=1)
def eligible_target_surname_counts():
    allowed = set(single_bpe_surname_pool())
    counts = Counter()
    for row, options in eligible('train'):
        counts.update(surname for _, surname, _, _ in options
                      if surname in allowed)
    return counts


def candidates(row, min_gap=25, max_target=250):
    """Return full-name-to-later-surname candidates, never the source prose."""
    tokens = row['summary']
    roster = row['box_score']['PLAYER_NAME'].values()
    roster_surnames = [name.split()[-1] for name in roster
                       if len(name.split()) == 2]
    team_tokens = set((row['home_city']+' '+row['home_name']+' '+
                       row['vis_city']+' '+row['vis_name']).split())
    out = []
    for name in set(roster):
        parts = name.split()
        if len(parts) != 2 or not all(NAME.fullmatch(part) for part in parts):
            continue
        first, surname = parts
        if roster_surnames.count(surname) != 1 or surname in team_tokens:
            continue
        full = [i for i in range(len(tokens)-1)
                if tokens[i:i+2] == [first, surname]]
        if len(full) != 1:
            continue
        source = full[0]
        if surname in tokens[:source]:
            continue
        if any(tokens[i] == surname and i > 0 and
               NAME.fullmatch(tokens[i-1]) and tokens[i-1] != first
               for i in range(len(tokens))):
            # Another person's full name can share this surname even if
            # that person is absent from the box-score roster.
            continue
        if any(token != surname and token.startswith(surname)
               for token in tokens):
            # A possessive or inflected surname would remain unedited.
            continue
        later = [i for i in range(source+2, min(len(tokens), max_target))
                 if tokens[i] == surname and i-source >= min_gap]
        if later:
            out.append((first, surname, source, later[0]))
    return tuple(sorted(out, key=lambda item: (item[3], item[0], item[1])))


@lru_cache(maxsize=8)
def eligible(group, min_gap=25, max_target=250):
    if group not in ('train', 'heldout'):
        raise ValueError(group)
    return tuple((row, options) for row in resources()[0]['rotowire'][0]
                 if game_partition(game_key(row)) == group
                 if (options := candidates(row, min_gap, max_target)))


def sample_plan(rng: random.Random, group='train', min_gap=25,
                max_target=250, alias_exponent=0., alias_prior='roster'):
    if group not in ('train', 'heldout') or alias_exponent < 0 or \
            alias_prior not in ('roster', 'eligible_target'):
        raise ValueError('Invalid group or alias prior')
    rows = eligible(group, min_gap, max_target)
    for _ in range(500):
        row, choices = rng.choice(rows)
        key = game_key(row)
        first, surname, source, target = rng.choice(choices)
        prior_counts = surname_counts() if alias_prior == 'roster' else \
            eligible_target_surname_counts()
        pool = single_bpe_surname_pool() if alias_prior == 'roster' else \
            tuple(sorted(prior_counts))
        available = [word for word in pool
                     if word != surname and word not in row['summary'] and
                     word not in (row['home_city'], row['vis_city'])]
        if len(available) < 2:
            continue
        if alias_exponent == 0:
            alias_a, alias_b = rng.sample(available, 2)
        else:
            alias_a = rng.choices(available,
                                  weights=[prior_counts[word]**alias_exponent
                                           for word in available])[0]
            available.remove(alias_a)
            alias_b = rng.choices(available,
                                  weights=[prior_counts[word]**alias_exponent
                                           for word in available])[0]
        return Plan(key, tuple(row['summary']), first, surname,
                    source, target, alias_a, alias_b)
    raise RuntimeError('No eligible natural coreference candidate')


def _render_tokens(tokens):
    text = ''
    spans = []
    for token in tokens:
        if token in ('.', ',', '!', '?', ';', ':'):
            text = text.rstrip()+token
            spans.append((len(text)-len(token), len(text)))
        elif token == '-':
            text = text.rstrip()+'-'
            spans.append((len(text)-1, len(text)))
        else:
            if text and not text.endswith('-'):
                text += ' '
            start = len(text)
            text += token
            spans.append((start, len(text)))
    return text, spans


@lru_cache(maxsize=1)
def _rows():
    return {game_key(row): row for row in resources()[0]['rotowire'][0]}


def nuisance_rename(plan: Plan):
    """A different previously introduced player to rename, if available."""
    tokens = plan.source_tokens
    roster = _rows()[plan.game_key]['box_score']['PLAYER_NAME'].values()
    options = []
    for name in set(roster):
        parts = name.split()
        if len(parts) != 2 or not all(NAME.fullmatch(p) for p in parts):
            continue
        first, surname = parts
        if surname == plan.original_surname:
            continue
        full = [i for i in range(plan.target_token)
                if tokens[i:i+2] == (first, surname)]
        if not full or any(token != surname and token.startswith(surname)
                           for token in tokens):
            continue
        options.append((full[0], surname))
    if not options:
        return None
    _, old = min(options)
    available = [word for word in single_bpe_surname_pool()
                 if word not in tokens and word not in
                 (plan.original_surname, plan.alias_a, plan.alias_b)]
    if not available:
        return None
    index = int.from_bytes(hashlib.sha256(repr((plan.game_key, old,
                       plan.source_token)).encode()).digest()[:8], 'big')
    return old, available[index % len(available)]


def realize(plan: Plan, arm='a', nuisance=False):
    if arm not in ('a', 'b'):
        raise ValueError(arm)
    alias = plan.alias_a if arm == 'a' else plan.alias_b
    tokens = [alias if word == plan.original_surname else word
              for word in plan.source_tokens]
    if nuisance:
        rename = nuisance_rename(plan)
        if rename is None:
            raise ValueError('No nuisance player in report prefix')
        old, new = rename
        tokens = [new if word == old else word for word in tokens]
    text, spans = _render_tokens(tokens)
    assert tokens[plan.source_token:plan.source_token+2] == [plan.first, alias]
    assert tokens[plan.target_token] == alias
    source_start = spans[plan.source_token][0]
    source_end = spans[plan.source_token+1][1]
    answer_start, answer_end = spans[plan.target_token]
    assert text[source_start:source_end] == plan.first+' '+alias
    assert text[answer_start:answer_end] == alias
    return Document(text, source_start, source_end, answer_start,
                    answer_end, alias, plan)
