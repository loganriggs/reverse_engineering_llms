"""Visible-text oracle and supported upstream edits for V8 documents."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from audit_document_world_v5_clause_realizations import _edited_world
from document_world_v8_discourse import (
    answer_from_visible_prefix, draw_varied, realize)


def check_sites(document):
    for site in document.sites:
        assert document.tokens[site.index] == site.answer
        assert site.answer in site.candidates
        assert answer_from_visible_prefix(
            document.tokens[:site.index]) == site.answer


def audit(n, seed):
    rng = random.Random(seed)
    families, edits = Counter(), Counter()
    syntax_changes = context_changes = 0
    for _ in range(n):
        document = draw_varied(rng)
        check_sites(document)
        families.update(site.family for site in document.sites)
        for index, plan in enumerate(document.latent['recap_plan']):
            edited = _edited_world(document.latent, plan)
            other = realize(edited)
            check_sites(other)
            assert other.sites[index].family == document.sites[index].family
            assert other.sites[index].answer != document.sites[index].answer
            edits[plan[1]] += 1
        plain = realize(document.latent, syntax_entropy=0., context_rate=0.)
        varied = realize(document.latent, syntax_entropy=1., context_rate=1.)
        check_sites(plain)
        check_sites(varied)
        assert [(site.family, site.answer) for site in plain.sites] == [
            (site.family, site.answer) for site in varied.sites]
        syntax_changes += plain.tokens != varied.tokens
        context_changes += (
            realize(document.latent, context_rate=0.).tokens !=
            realize(document.latent, context_rate=1.).tokens)
    assert syntax_changes and context_changes
    return dict(n=n, seed=seed, sites_by_family=dict(families),
                successful_upstream_edits=dict(edits),
                documents_changed_by_syntax=syntax_changes,
                documents_changed_by_context=context_changes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=300)
    parser.add_argument('--seed', type=int, default=9317)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    panels = [audit(args.n, args.seed + offset) for offset in (0, 1)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(panels=panels), indent=2) + '\n')
    for panel in panels:
        print(panel, flush=True)


if __name__ == '__main__':
    main()
