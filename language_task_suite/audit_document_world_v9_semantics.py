"""Visible-only V9 answer and supported-edit audit across lexical priors."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from audit_document_world_v5_clause_realizations import _edited_world
from document_world_v9_typed_lexicon import (
    answer_from_visible_prefix, draw_varied, realize)


def check(doc):
    for site in doc.sites:
        assert doc.tokens[site.index] == site.answer
        assert site.answer in site.candidates
        assert answer_from_visible_prefix(doc.tokens[:site.index]) == site.answer


def audit(n, seed, exponent, source_rate, delay_rate):
    rng = random.Random(seed)
    families, edits = Counter(), Counter()
    syntax_changed = 0
    for _ in range(n):
        doc = draw_varied(rng, frequency_exponent=exponent,
                          syntax_entropy=1., source_delay_rate=delay_rate,
                          extra_source_recap_rate=source_rate)
        check(doc)
        families.update(site.family for site in doc.sites)
        simple = realize(doc.latent, syntax_entropy=0.)
        check(simple)
        assert [(s.family, s.answer) for s in simple.sites] == [
            (s.family, s.answer) for s in doc.sites]
        syntax_changed += simple.tokens != doc.tokens
        for j, plan in enumerate(doc.latent['recap_plan']):
            edited = realize(_edited_world(doc.latent, plan),
                             syntax_entropy=1.)
            check(edited)
            assert edited.sites[j].family == doc.sites[j].family
            assert edited.sites[j].answer != doc.sites[j].answer
            edits[plan[1]] += 1
    assert syntax_changed == n
    return dict(n=n, seed=seed, exponent=exponent,
                source_rate=source_rate, delay_rate=delay_rate,
                sites=dict(families), supported_edits=dict(edits),
                syntax_changed=syntax_changed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=300)
    p.add_argument('--seed', type=int, default=82719)
    p.add_argument('--source-rate', type=float, default=.5)
    p.add_argument('--delay-rate', type=float, default=.5)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    panels = [audit(a.n, a.seed + i, exponent, a.source_rate,
                    a.delay_rate)
              for i, exponent in enumerate((0., 1.))]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(panels=panels), indent=2) + '\n')
    for panel in panels:
        print(panel)


if __name__ == '__main__':
    main()
