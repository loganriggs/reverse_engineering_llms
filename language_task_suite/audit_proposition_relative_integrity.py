"""Visible-truth and upstream-edit checks for proposition relative clauses."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random

from document_world import NAMES, VALUES
from document_world_v9_typed_lexicon import _KIND, _SURFACE_POOLS
from document_proposition_plan import FORMS, edit_initial_fact, sample_plan
from document_proposition_surface import realize
from document_proposition_relative import (
    _relative_claim, add_relative_clauses, answer_from_relative_prefix,
    expand_checked, parse_relative,
)


def clauses(tokens):
    for i in range(len(tokens)-1):
        if tokens[i:i+2] == (',', 'which'):
            attr, value, end = _relative_claim(tokens, i)
            yield i, tokens[i-1], attr, value, end


def false_claim(tokens, lexical):
    found = next(clauses(tokens), None)
    if found is None:
        return False
    i, _, attr, value, end = found
    if lexical:
        pool = _SURFACE_POOLS[_KIND[value]]
    else:
        pool = VALUES[attr]
    other = next(word for word in pool if word != value)
    bad = list(tokens)
    value_index = i+2 if attr == 'owner' else i+3 if attr == 'color' else i+5
    bad[value_index] = other
    try:
        expand_checked(tuple(bad), lexical)
    except AssertionError:
        return True
    raise AssertionError('False relative assertion accepted')


def owner_edit(plan, seed, changed):
    for _, obj, attr, value, _ in clauses(changed.tokens):
        if attr != 'owner' or obj not in plan['initial'] or \
                value != plan['initial'][obj]['owner']:
            continue
        updates = {node['value'] for node in plan['nodes']
                   if node['kind'] == 'update' and
                   node['entity'] == obj and node['attr'] == 'owner'}
        # A later owner update can shadow the initial fact before the
        # relative-clause site, so this intervention isolates untouched
        # owner histories only.
        if updates:
            continue
        alternatives = [name for name in NAMES
                        if name != value]
        if not alternatives:
            continue
        edited = edit_initial_fact(plan, obj, 'owner', alternatives[0])
        edited_base = realize(edited, seed=seed, surface_version=3)
        edited_doc, _ = add_relative_clauses(
            edited_base, seed=seed ^ 0x4E1A71B3, rate=1.)
        parse_relative(edited_doc)
        assert any((other_obj, other_attr, other_value) ==
                   (obj, 'owner', alternatives[0])
                   for _, other_obj, other_attr, other_value, _
                   in clauses(edited_doc.tokens))
        assert expand_checked(edited_doc.tokens) == edited_base.tokens
        return True
    return False


def audit(seed, per_form):
    rng = random.Random(seed)
    counts = Counter()
    for form in FORMS:
        for index in range(per_form):
            plan = sample_plan(rng, form=form, planning_style='threaded')
            surface_seed = seed + 10000 * FORMS.index(form) + index
            for lexical in (False, True):
                base = realize(plan, seed=surface_seed,
                               lexical_exponent=0. if lexical else None,
                               surface_version=3)
                changed, n = add_relative_clauses(
                    base, seed=surface_seed ^ 0x4E1A71B3,
                    rate=1., lexical_enabled=lexical)
                assert expand_checked(changed.tokens, lexical) == base.tokens
                parse_relative(changed, lexical)
                counts['documents'] += 1
                counts[f'form/{form}/documents'] += 1
                counts['relative_clauses'] += n
                for site in changed.sites:
                    assert answer_from_relative_prefix(
                        changed.tokens[:site.index], lexical) == site.answer
                    counts['answer_sites'] += 1
                counts['false_claims_rejected'] += false_claim(
                    changed.tokens, lexical)
                if not lexical:
                    counts['upstream_owner_edit_flips'] += owner_edit(
                        plan, surface_seed, changed)
    return dict(counts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=47701)
    p.add_argument('--per-form', type=int, default=100)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = {'config': {'seed': a.seed, 'per_form': a.per_form},
              'counts': audit(a.seed, a.per_form),
              'scope': 'False relative claims are modified in visible text '
                       'and must fail the independent checker. An edited '
                       'initial owner fact must propagate to a later '
                       'owner relative clause. This validates a local '
                       'causal relation, not naturalness or full-token oracle.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(result['counts'], flush=True)


if __name__ == '__main__':
    main()
