"""High-precision lexical evidence for a few infobox-to-clause relations.

These patterns assert a relation in the same sentence as the value. They
remain conservative candidate validators, not a general semantic parser.
"""

from __future__ import annotations

import re

from audit_structured_wikipedia_multifield import literal_mentions


PATTERNS = {
    'founded': (r'\b(?:founded|established|formed)\s+(?:in|on)\s*$',),
    'built': (r'\b(?:built|constructed|completed)\s+(?:in|on)\s*$',),
    'opened': (r'\b(?:opened|inaugurated)\s+(?:in|on)\s*$',),
    'education': (r'\b(?:studied at|graduated from|attended)\s*$',),
    'alma mater': (r'\b(?:studied at|graduated from|attended)\s*$',),
    'owner': (r'\b(?:owned by|ownership of|acquired by)\s+(?:the\s+)?$',),
    'parent': (r'\b(?:subsidiary of|owned by|acquired by)\s+(?:the\s+)?$',),
    'headquarters': (r'\b(?:headquartered|based)\s+in\s*$',),
    'manager': (r'\b(?:manager|managed by)\s*$',),
    'coach': (r'\b(?:coach|coached by)\s*$',),
    'venue': (r'\b(?:held|played|staged)\s+at\s*$',),
    'stadium': (r'\b(?:held|played|staged)\s+at\s*$',),
    'host city': (r'\b(?:held|hosted|staged)\s+in\s*$',),
    'host country': (r'\b(?:held|hosted|staged)\s+in\s*$',),
    'family': (r'\b(?:family|families)\s*$',),
    'genus': (r'\bgenus\s*$',),
    'occupation': (r'\b(?:is|was)\s+an?\s*$',),
    'released': (r'\b(?:released|published)\s+(?:in|on)\s*$',),
    'label': (r'\b(?:signed to|released by|on the label)\s*$',),
    'country': (r'\b(?:located|situated|based)\s+in\s*$',),
}


def sentence_bounds(text, offset):
    """Return an approximate visible sentence around a literal mention."""
    left = max(text.rfind('.', 0, offset), text.rfind('?', 0, offset),
               text.rfind('!', 0, offset), text.rfind('\n', 0, offset)) + 1
    ends = [i for char in '.?!\n'
            if (i := text.find(char, offset)) >= 0]
    right = min(ends) if ends else len(text)
    return left, right


def evidence(text, field, *, title=None, require_title=False):
    """Return a local lexical witness for one unique value mention."""
    label = field['label'].casefold()
    value = field['value']
    patterns = PATTERNS.get(label)
    if not patterns:
        return None
    mentions = literal_mentions(text, value)
    if len(mentions) != 1:
        return None
    lo, hi = mentions[0].span()
    sent_lo, sent_hi = sentence_bounds(text, lo)
    if hi > sent_hi:
        return None
    sentence = text[sent_lo:sent_hi]
    if require_title and title and title.casefold() not in sentence.casefold():
        return None
    before = text[max(sent_lo, lo-65):lo]
    if not any(re.search(pattern, before, re.I)
               for pattern in patterns):
        return None
    return {'label': field['label'], 'value': value,
            'span': (lo, hi), 'sentence_span': (sent_lo, sent_hi),
            'witness_prefix': before[-65:]}
