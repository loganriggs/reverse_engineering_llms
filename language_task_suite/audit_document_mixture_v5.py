"""Replay mixed-document task sites, paired edits, and lexical form gate."""

import argparse
import collections
import json
import random
from pathlib import Path

from document_mixture_v5 import (KINDS, sample_document,
                                 realize_record, realize_score, site_oracle)


def audit(n_per_kind, seed):
    rng = random.Random(seed)
    families = collections.Counter()
    kinds = collections.Counter()
    paired = collections.Counter()
    shortcuts = collections.Counter()
    examples = {}
    for kind in KINDS:
        for _ in range(n_per_kind):
            doc = sample_document(rng, kind)
            kinds[kind] += 1
            examples.setdefault(kind, doc.text()[:450])
            for site in doc.sites:
                assert doc.tokens[site.index] == site.target
                exact = site_oracle(doc, site)
                assert site.target in exact and abs(sum(exact.values()) - 1) < 1e-8
                assert set(exact) == set(site.distribution)
                assert max(abs(exact[k] - site.distribution[k]) for k in exact) < 1e-9
                families[site.family] += 1
            if kind == 'record_listing':
                if doc.latent['ships'][doc.latent['order'][-1]] == doc.latent['query_ship']:
                    shortcuts['record_query_is_last'] += 1
                else:
                    shortcuts['record_query_is_not_last'] += 1
                world = dict(doc.latent)
                world['query_ship'] = next(x for x in world['ships']
                                           if x != world['query_ship'])
                changed = realize_record(world)
                assert changed.sites[0].target != doc.sites[0].target
                assert site_oracle(changed, changed.sites[0]) == {
                    changed.sites[0].target: 1.}
                paired['record_scope_edit'] += 1
                reordered = dict(doc.latent)
                reordered['order'] = tuple(reversed(reordered['order']))
                same = realize_record(reordered)
                assert same.sites[0].target == doc.sites[0].target
                assert site_oracle(same, same.sites[0]) == {doc.sites[0].target: 1.}
                paired['record_order_nuisance'] += 1
            if kind == 'score_report':
                shortcuts['score_leaked' if doc.latent['headline_leak']
                          else 'score_unleaked'] += 1
                changed_world = dict(doc.latent)
                changed_world['headline_leak'] = False
                changed_world['increment'] = (
                    1 if changed_world['increment'] != 1 else 2)
                changed = realize_score(changed_world)
                assert changed.sites[0].target != doc.sites[0].target
                assert site_oracle(changed, changed.sites[0]) == {
                    changed.sites[0].target: 1.}
                paired['score_event_edit'] += 1
                leak_world = dict(doc.latent)
                leak_world['headline_leak'] = not leak_world['headline_leak']
                leak = realize_score(leak_world)
                assert leak.sites[0].target == doc.sites[0].target
                paired['score_leak_toggle'] += 1
    # A separate weighted draw avoids making the first 49 eligible documents
    # all narrative merely because the task audit loops over kinds.
    lexical_rng = random.Random(seed + 1)
    lexical = collections.Counter()
    lexical_kinds = collections.Counter()
    lexical_draws = 0
    while sum(lexical_kinds.values()) < 49:
        doc = sample_document(lexical_rng)
        lexical_draws += 1
        alpha = [t.lower() for t in doc.tokens if t.isalpha()]
        if len(alpha) >= 100:
            lexical.update(alpha[:100])
            lexical_kinds[doc.kind] += 1
    baseline_rng = random.Random(seed + 1)
    baseline = collections.Counter()
    for _ in range(49):
        doc = sample_document(baseline_rng, 'narrative')
        alpha = [t.lower() for t in doc.tokens if t.isalpha()]
        assert len(alpha) >= 100
        baseline.update(alpha[:100])
    return dict(version='document_mixture_v5_task_rich_pilot',
                seed=seed, n_per_kind=n_per_kind, kinds=dict(kinds),
                sites=dict(families), paired_checks=dict(paired),
                shortcut_conditions=dict(shortcuts),
                first49_eligible_kinds=dict(lexical_kinds),
                first49_eligible_draws=lexical_draws,
                first49_x100_alpha_types=len(lexical),
                first49_x100_alpha_singletons=sum(v == 1 for v in lexical.values()),
                first49_x100_alpha_top10_share=(
                    sum(v for _, v in lexical.most_common(10))/sum(lexical.values())),
                v14_diagnostic_49x100_control=dict(
                    types=len(baseline),
                    singletons=sum(v == 1 for v in baseline.values()),
                    top10_share=sum(v for _, v in baseline.most_common(10)) /
                    sum(baseline.values())),
                examples=examples,
                limitations=['task-rich weights are not measured natural prevalence',
                             'lexical and prose form still synthetic',
                             'full next-token oracle only for V14 narrative arm',
                             'document kinds use separate surface grammars',
                             'no model trained on this mixture yet'])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--n-per-kind', type=int, default=200)
    p.add_argument('--seed', type=int, default=23012)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    result = audit(args.n_per_kind, args.seed)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print('documents', sum(result['kinds'].values()),
          'sites', sum(result['sites'].values()),
          'paired', result['paired_checks'],
          'lexical types', result['first49_x100_alpha_types'])
