"""Pair same-length surname edits with existing FineWeb rule carriers.

Same-length edits allow the joint document's character-level rule sites to
retain their source offsets. The output is a conditional natural-text DGP,
not a complete oracle for the unmodified background text.
"""

from __future__ import annotations

import random
import re

from corpus_rule_reference_dgp import (
    Document, RuleSite, check_document, sample_document,
)
from fineweb_name_copy_dgp import WORD

KINDS = ('surname_copy', 'bracket', 'literal_copy', 'natural_style', 'list')


def _one_arm(dgp, index, original, replacement, surface_seed, nonce_pool,
             **rule_rates):
    sample = dgp.make(index, original, replacement)
    base = sample_document(sample.text, random.Random(surface_seed),
                           nonce_pool, **rule_rates)
    spans = [m.span() for m in re.finditer(
        r'(?<!\w)'+re.escape(replacement)+r'(?!\w)', base.text)]
    if len(spans) < 2 or any(base.text[a:b] != replacement for a,b in spans):
        raise AssertionError('Surname substitution was overwritten')
    site = RuleSite('surname_copy', spans[0], spans[1], replacement)
    result = Document(dgp.rows[index]['preview'][:len(base.text)], base.text,
                      base.sites+(site,), base.eligible+('surname_copy',),
                      base.applied+('surname_copy',))
    check_document(result)
    return result


def paired_document(dgp, index, rng, nonce_pool, min_gap_words=20,
                    **rule_rates):
    """Draw two counterfactual names with the same other sampled edits.

    Returns None when a source has no repeated PERSON surname with two
    available replacements of its own length. Callers still train the
    ordinary source document in that case, avoiding source selection bias.
    """
    candidates = []
    for original in dgp.candidates[index]:
        positions = dgp.candidates[index][original]
        gap = len(WORD.findall(dgp.rows[index]['preview'][
            positions[0]+len(original):positions[1]]))
        if gap < min_gap_words:
            continue
        options = [word for word in dgp.replacements(index, original)
                   if len(word) == len(original)]
        if len(options) >= 2:
            candidates.append((original, options))
    if not candidates:
        return None
    original, options = rng.choice(candidates)
    left, right = rng.sample(options, 2)
    surface_seed = rng.randrange(2**63)
    a = _one_arm(dgp,index,original,left,surface_seed,nonce_pool,**rule_rates)
    b = _one_arm(dgp,index,original,right,surface_seed,nonce_pool,**rule_rates)
    assert len(a.text) == len(b.text)
    # Both arms must differ only at the planted surname positions. The
    # independent rule sampler is replayed from the same seed in each arm.
    surname_spans = [m.span() for m in re.finditer(
        r'(?<!\w)'+re.escape(left)+r'(?!\w)', a.text)]
    masked = set(i for start,end in surname_spans for i in range(start,end))
    if any(char_a != char_b and i not in masked
           for i,(char_a,char_b) in enumerate(zip(a.text,b.text))):
        raise AssertionError('Non-surname difference between paired arms')
    if any(a_site.kind != b_site.kind or
           a_site.source != b_site.source or
           a_site.target != b_site.target
           for a_site,b_site in zip(a.sites,b.sites,strict=True)):
        raise AssertionError('Paired rule sites diverged')
    return a,b
