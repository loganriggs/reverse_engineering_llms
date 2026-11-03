"""Typed stat predicates extracted from training-game human reports.

The returned bank contains predicate templates, not full source sentences.
Each template says only that a named player had specified box-score stats.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import random
import re

from audit_player_predicate_fragments import extract
from audit_rotowire_document_source import game_key
from corpus_score_sentence_renderer import _sentences
from sports_document_dgp import game_partition, resources


BASIC = frozenset(('PTS', 'REB', 'AST'))


@lru_cache(maxsize=1)
def bank():
    forms = Counter()
    for source in ('rotowire', 'sbnation'):
        for row in resources()[0][source][0]:
            if game_partition(game_key(row)) != 'train':
                continue
            for sentence in _sentences(row['summary']):
                found = extract(sentence, row)
                if found and found['fields'][0] == 'PTS' and \
                        set(found['fields']) <= BASIC:
                    forms[(found['form'], found['fields'])] += 1
    if len(forms) < 30:
        raise RuntimeError('Empirical predicate bank unexpectedly small')
    return tuple((form, fields, count)
                 for (form, fields), count in sorted(forms.items()))


def render(name, stats, rng: random.Random):
    """Sample one whole, grounded player-stat sentence from the bank."""
    options = [(form, fields, count) for form, fields, count in bank()
               if all(field in stats for field in fields)]
    weights = [count**.5 for _, _, count in options]
    form, fields, _ = rng.choices(options, weights=weights)[0]
    for field, unit in (('PTS', 'points'), ('REB', 'rebounds'),
                        ('AST', 'assists')):
        form = re.sub(r'\{'+field+r'\}\s+'+unit.rstrip('s')+r's?\b',
                      '{'+field+'} '+unit, form, flags=re.I)
    text = form.format(**stats)
    for field, singular, plural in (
            ('REB', 'rebound', 'rebounds'),
            ('AST', 'assist', 'assists')):
        value = int(stats[field])
        unit = singular if value == 1 else plural
        text = re.sub(r'(?<!\d)'+str(value)+r'\s+'+plural+r'\b',
                      str(value)+' '+unit, text)
        text = re.sub(r'(?<!\d)'+str(value)+r'\s+'+singular+r'\b',
                      str(value)+' '+unit, text)
    return name+' '+text+'.'
