"""Fictional three-letter aliases for a literal copying task in web prose.

The source document is treated as an alternate world after every chosen
acronym occurrence receives the same opaque code. This preserves local
surface structure, not historical truth or the meaning of an acronym.
"""

from __future__ import annotations

from itertools import product
import random
import re

from corpus_uppercase_copy_dgp import candidates as broad_candidates, realize


CONSONANTS = 'BCDFGHJKLMNPQRSTVWXZ'
NON_ALIAS_CODES = {
    # Measurement, economic quantities, and currencies are not names whose
    # spelling can be changed without changing a unit or arithmetic claim.
    'MPH', 'KPH', 'KMH', 'MPG', 'PSI', 'RPM', 'BPM',
    'GDP', 'GNP', 'CPI', 'PPI', 'ROI', 'EPS',
    'USD', 'EUR', 'GBP', 'INR', 'JPY', 'CNY', 'AUD', 'CAD',
    'CHF', 'RUB', 'BRL', 'KRW', 'SGD', 'MYR', 'THB',
    'MIL', 'DIL', 'SIL', 'BIL', 'FIL',
}
ENTITY_SOURCE_CODES = {
    # High-precision organization, company, league, and party aliases.
    # This is a surface eligibility prior, not an entity linker: a code can
    # have another meaning in a particular document and still needs review.
    'ABC', 'AFP', 'BBC', 'CBS', 'CNN', 'FOX', 'NBC', 'NPR', 'PBS',
    'HBO', 'MTV', 'ESPN', 'IBM', 'AMD', 'NHS',
    'CIA', 'FBI', 'NSA', 'FTC', 'EPA', 'CDC', 'FDA', 'DOJ',
    'DHS', 'IRS', 'WHO', 'IMF', 'WTO', 'GOP', 'DNC', 'RNC',
    'NBA', 'NFL', 'NHL', 'MLB', 'MLS', 'UFC', 'WWE', 'PGA',
    'ATP', 'WTA', 'ATA', 'ABA', 'HSI', 'CWS', 'SSP',
}


def candidates(text, strict=True, entity_only=False, **kwargs):
    for plan in broad_candidates(text, **kwargs):
        if entity_only and plan.source_code not in ENTITY_SOURCE_CODES:
            continue
        if strict:
            if plan.source_code in NON_ALIAS_CODES:
                continue
            if any(text[max(0, start - 8):start].rstrip().endswith(
                    tuple(str(digit) for digit in range(10)))
                    for start, _ in plan.spans):
                continue
        yield plan


def codebook(tokenizer):
    """Opaque codes with three separate BPE character pieces after space."""
    codes = []
    for letters in product(CONSONANTS, repeat=3):
        code = ''.join(letters)
        if not any(letter in code for letter in 'QXZ'):
            continue
        encoding = tokenizer.encode(' ' + code, add_special_tokens=False)
        if len(encoding.ids) != 3:
            continue
        if not (encoding.tokens[0] == 'Ġ' + code[0] and
                encoding.tokens[1:] == list(code[1:])):
            continue
        codes.append(code)
    return tuple(codes)


def draw_pair(plan, pool, rng: random.Random, attempts=20):
    absent = [code for code in pool
              if not re.search(r'(?<![A-Za-z])' + code + r'(?![A-Za-z])',
                               plan.source_text)]
    if len(absent) < 2:
        return None
    for _ in range(attempts):
        a, b = rng.sample(absent, 2)
        if all(left != right for left, right in zip(a, b, strict=True)):
            return realize(plan, a), realize(plan, b)
    return None
