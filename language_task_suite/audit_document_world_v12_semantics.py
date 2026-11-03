"""Audit V12 sparse initial facts, supported edits, and hidden-fact neutrality."""

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path

from audit_document_world_v5_clause_realizations import _edited_world
from document_world import VALUES
from document_world_v3_multisite import ALIASES
from document_world_v12_partial_facts import (
    answer_from_visible_prefix, draw_varied, realize,
    required_initial_facts)
from document_world_v11_initial_facts import (
    initial_sentences, parse_initial_fact)


def check(doc, lexical):
    for site in doc.sites:
        assert doc.tokens[site.index] == site.answer
        assert site.answer in site.candidates
        assert answer_from_visible_prefix(
            doc.tokens[:site.index], lexical_enabled=lexical) == site.answer


def audit(n, seed, lexical_exponent):
    rng = random.Random(seed)
    sites, edits, visible, omitted = (Counter() for _ in range(4))
    omission_checks = 0
    for _ in range(n):
        doc = draw_varied(rng, optional_fact_rate=.35,
                          lexical_exponent=lexical_exponent,
                          syntax_entropy=1., extra_source_recap_rate=.5,
                          source_delay_rate=.5)
        lexical = lexical_exponent is not None
        check(doc, lexical)
        sites.update(site.family for site in doc.sites)
        world = doc.latent
        selected = {(item['object'], item['attribute'])
                    for item in world['initial_fact_plan']}
        assert required_initial_facts(world) <= selected
        for item, sentence in zip(world['initial_fact_plan'],
                                  initial_sentences(world), strict=True):
            obj, attr, value, _ = parse_initial_fact(sentence[:-1])
            assert (obj, attr) == (item['object'], item['attribute'])
            assert value == (world['aliases'][obj] if attr == 'alias'
                             else world['initial'][obj][attr])
            visible[attr] += 1
        for obj in world['initial']:
            for attr in ('owner', 'color', 'place', 'alias'):
                if (obj, attr) in selected:
                    continue
                omitted[attr] += 1
                altered = copy.deepcopy(world)
                if attr == 'alias':
                    original = altered['aliases'][obj]
                    altered['aliases'][obj] = next(
                        value for value in ALIASES if value != original)
                else:
                    original = altered['initial'][obj][attr]
                    altered['initial'][obj][attr] = next(
                        value for value in VALUES[attr] if value != original)
                other = realize(altered, lexical_enabled=lexical,
                                syntax_entropy=1.)
                assert other.tokens == doc.tokens
                assert [(s.family, s.answer) for s in other.sites] == [
                    (s.family, s.answer) for s in doc.sites]
                omission_checks += 1
        for j, plan in enumerate(world['recap_plan']):
            changed = realize(_edited_world(world, plan),
                              lexical_enabled=lexical,
                              syntax_entropy=1.)
            check(changed, lexical)
            assert changed.sites[j].family == doc.sites[j].family
            assert changed.sites[j].answer != doc.sites[j].answer
            edits[plan[1]] += 1
    return dict(n=n, seed=seed, lexical_exponent=lexical_exponent,
                sites=dict(sites), supported_edits=dict(edits),
                visible_initial_facts=dict(visible),
                omitted_initial_facts=dict(omitted),
                omission_neutrality_checks=omission_checks)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=300)
    p.add_argument('--seed', type=int, default=91121)
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
