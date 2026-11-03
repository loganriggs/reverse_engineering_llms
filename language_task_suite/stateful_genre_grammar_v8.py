"""Finite PCFG fixture with world-conditioned mentions and attributed guesses.

Only facts already visible in task sentences alter the grammar. Independent
prior draws are explicitly attributed as guesses and do not assert a document
fact. This is a semantic-oracle fixture, not a natural-language renderer.
"""

from __future__ import annotations

from functools import lru_cache

from joint_document_v1 import ATTRIBUTES, NAMES, OBJECTS, ROLES
from small_genre_grammar_v7 import GRAMMARS, STARTS
from weighted_cfg_oracle import EOS, WeightedCFG


def _lex(words):
    return tuple((1/len(words), (word,)) for word in words)


@lru_cache(maxsize=4096)
def grammar(genre, profiles=(), claims=()):
    """Profiles: (name, role); claims: (item, attribute, speaker)."""
    assert genre in GRAMMARS
    base = GRAMMARS[genre]
    rules = dict(base.rules)
    rules['S'] = tuple((.7*p, rhs) for p, rhs in base.rules['S']) + (
        (.3, ('WORLD', '.', EOS)),)
    world = [(.25, ('A', 'witness', 'guessed', 'that',
                     'NAME', 'was', 'a', 'ROLE')),
             (.25, ('A', 'witness', 'guessed', 'that',
                     'the', 'OBJ', 'was', 'ATTR'))]
    if profiles:
        p = .25/len(profiles)
        world.extend((p, ('The', role, 'in', 'the', 'report',
                          'was', name)) for name, role in profiles)
    if claims:
        p = .25/len(claims)
        world.extend((p, ('The', item, 'was', 'described',
                          'as', attribute, 'by', speaker))
                     for item, attribute, speaker in claims)
    total = sum(p for p, _ in world)
    rules['WORLD'] = tuple((p/total, rhs) for p, rhs in world)
    rules['ROLE'] = _lex(ROLES)
    rules['OBJ'] = _lex(OBJECTS)
    rules['ATTR'] = _lex(ATTRIBUTES)
    rules['NAME'] = _lex(NAMES)
    return WeightedCFG(rules)


EXTRA_VOCAB = frozenset(('was', 'described', 'as', 'by', 'in',
                         'report', 'The', 'A', 'witness', 'guessed',
                         'that', 'a', 'the'))
assert all(set(grammar(genre).next_distribution(())) == set(STARTS)
           for genre in GRAMMARS)
