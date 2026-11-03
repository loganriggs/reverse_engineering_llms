"""Small semantic-themed PCFG pilot for document background prose.

The grammar is intentionally hand built and shallow. It validates the
grammar/oracle interface for the next corpus-induced, state-split version;
it is not claimed to match natural language or to encode document facts.
"""

from __future__ import annotations

from functools import lru_cache

from genre_document_v3 import GENRES
from weighted_cfg_oracle import EOS, WeightedCFG


TOPIC_WORDS = {
    'court': (('court', 'witness', 'record', 'statement'),
              ('reviewed', 'disputed', 'confirmed', 'examined')),
    'encyclopedia': (('species', 'region', 'researcher', 'collection'),
                     ('described', 'documented', 'classified', 'compared')),
    'procedure': (('operator', 'device', 'procedure', 'system'),
                  ('checked', 'adjusted', 'tested', 'recorded')),
    'math': (('student', 'number', 'quantity', 'example'),
             ('calculated', 'compared', 'estimated', 'recorded')),
}
PLURAL = {'species': 'species', 'quantity': 'quantities'}


def _plural(word):
    return PLURAL.get(word, word+'s')


def _lex(words, weights=None):
    if weights is None:
        weights = [1/len(words)]*len(words)
    return tuple((p, (word,)) for word, p in zip(words, weights, strict=True))


@lru_cache(maxsize=4)
def grammar(genre):
    assert genre in GENRES
    subjects, verbs = TOPIC_WORDS[genre]
    rules = {
        'S': ((.58, ('CLAUSE', '.', EOS)),
              (.14, ('ADV', ',', 'CLAUSE', '.', EOS)),
              (.12, ('CLAUSE', ';', 'CLAUSE', '.', EOS)),
              (.16, ('CLAUSE', 'PP', '.', EOS))),
        'CLAUSE': ((.54, ('SUBJ', 'PRED')),
                   (.23, ('SUBJ', 'PRED', 'PP')),
                   (.23, ('SUBJ', 'PRED_PP'))),
        'PRED': ((.72, ('V', 'OBJ')), (.28, ('VI',))),
        # Same string can arise through CLAUSE->PRED PP or PRED_PP.
        'PRED_PP': ((1., ('V', 'OBJ', 'PP')),),
        'SUBJ': ((.58, ('DET', 'N')),
                 (.24, ('DET', 'ADJ', 'N')),
                 (.18, ('QUANT', 'NPL'))),
        'OBJ': ((.57, ('DET_L', 'N')),
                (.25, ('DET_L', 'ADJ', 'N')),
                (.18, ('QUANT', 'NPL'))),
        'PP': ((.55, ('about', 'DET_L', 'N')),
               (.45, ('near', 'DET_L', 'N'))),
        'DET': _lex(('The', 'A', 'This'), (.62, .18, .20)),
        'DET_L': _lex(('the', 'a', 'this'), (.60, .18, .22)),
        'QUANT': _lex(('Several', 'Many'), (.55, .45)),
        'ADV': _lex(('Meanwhile', 'Recently', 'Later'), (.42, .35, .23)),
        'ADJ': _lex(('new', 'earlier', 'important', 'local', 'other')),
        'N': _lex(subjects),
        'NPL': _lex(tuple(_plural(word) for word in subjects)),
        'V': _lex(verbs),
        'VI': _lex(('changed', 'remained', 'continued')),
    }
    return WeightedCFG(rules)


GRAMMARS = {genre: grammar(genre) for genre in GENRES}
VOCAB = frozenset.union(*(g.terminals for g in GRAMMARS.values()))-{EOS}
STARTS = frozenset.union(*(
    frozenset(g.next_distribution(())) for g in GRAMMARS.values()))
