"""Visible-prefix and independent initial-fact audit for V11 documents."""

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path

from audit_document_world_v5_clause_realizations import _edited_world
from document_world import _preposition
from document_world_v11_initial_facts import (
    answer_from_visible_prefix, draw_varied, initial_sentences,
    parse_initial_fact, plan_initial, realize)


def check_sites(doc, lexical):
    for site in doc.sites:
        assert doc.tokens[site.index] == site.answer
        assert site.answer in site.candidates
        assert answer_from_visible_prefix(
            doc.tokens[:site.index], lexical_enabled=lexical) == site.answer


def check_initial_claims(world):
    claims = Counter()
    for item, sentence in zip(world['initial_fact_plan'],
                              initial_sentences(world), strict=True):
        obj, attr, value, prep = parse_initial_fact(sentence[:-1])
        assert (obj, attr) == (item['object'], item['attribute'])
        expected = (world['aliases'][obj] if attr == 'alias'
                    else world['initial'][obj][attr])
        assert value == expected
        assert prep == (_preposition(expected) if attr == 'place'
                        else None)
        claims[attr] += 1
    assert all(claims[attr] == len(world['initial'])
               for attr in ('owner', 'color', 'place', 'alias'))
    return claims


def audit(n, seed, lexical_exponent):
    rng = random.Random(seed)
    sites, edits, claims = Counter(), Counter(), Counter()
    changed = 0
    for _ in range(n):
        doc = draw_varied(rng, order_entropy=1., initial_form_entropy=1.,
                          act_rate=.8, lexical_exponent=lexical_exponent,
                          syntax_entropy=1., extra_source_recap_rate=.5,
                          source_delay_rate=.5)
        lexical = lexical_exponent is not None
        check_sites(doc, lexical)
        claims.update(check_initial_claims(doc.latent))
        sites.update(site.family for site in doc.sites)
        baseline = realize(doc.latent, initial_enabled=False,
                           lexical_enabled=lexical, syntax_entropy=1.)
        assert [(s.family, s.answer) for s in baseline.sites] == [
            (s.family, s.answer) for s in doc.sites]
        changed += baseline.tokens != doc.tokens
        check_sites(baseline, lexical)
        simple_world = copy.deepcopy(doc.latent)
        simple_world['initial_fact_plan'] = plan_initial(simple_world, 0., 0.)
        simple = realize(simple_world, lexical_enabled=lexical,
                         syntax_entropy=1.)
        assert [(s.family, s.answer) for s in simple.sites] == [
            (s.family, s.answer) for s in doc.sites]
        check_sites(simple, lexical)
        for j, plan in enumerate(doc.latent['recap_plan']):
            edited_world = _edited_world(doc.latent, plan)
            edited = realize(edited_world, lexical_enabled=lexical,
                             syntax_entropy=1.)
            check_sites(edited, lexical)
            assert edited.sites[j].family == doc.sites[j].family
            assert edited.sites[j].answer != doc.sites[j].answer
            edits[plan[1]] += 1
    assert changed == n
    return dict(n=n, seed=seed, lexical_exponent=lexical_exponent,
                sites=dict(sites), supported_edits=dict(edits),
                initial_claims=dict(claims),
                changed_from_complete_intro=changed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=300)
    p.add_argument('--seed', type=int, default=88121)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    panels = [audit(a.n, a.seed+i, lexical)
              for i, lexical in enumerate((None, 0.))]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(panels=panels), indent=2) + '\n')
    for panel in panels:
        print(panel)


if __name__ == '__main__':
    main()
