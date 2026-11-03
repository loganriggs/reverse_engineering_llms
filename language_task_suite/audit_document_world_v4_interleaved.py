"""Audit semantic replay and surface form of interleaved document pilot."""

import argparse
import collections
import copy
import json
from pathlib import Path

from document_world import NAMES, VALUES
from document_world_v4_interleaved import (answer_from_visible_prefix,
                                           realize, sample_documents)


def audit(n, seed):
    docs = sample_documents(n, seed)
    families = collections.Counter()
    docs_with = collections.Counter()
    site_counts = collections.Counter()
    first100 = collections.Counter()
    lexical_eligible = 0
    fractions = []
    causal_checks = collections.Counter()
    for i, doc in enumerate(docs):
        assert doc.sites
        site_counts[len(doc.sites)] += 1
        alpha = [token.lower() for token in doc.tokens if token.isalpha()]
        if len(alpha) >= 100 and lexical_eligible < 49:
            first100.update(alpha[:100])
            lexical_eligible += 1
        present = set()
        for site in doc.sites:
            assert site.answer == doc.tokens[site.index]
            assert site.answer in site.candidates
            assert answer_from_visible_prefix(doc.tokens[:site.index]) == site.answer
            families[site.family] += 1
            present.add(site.family)
            fractions.append(site.index / len(doc.tokens))
        docs_with.update(present)
        # Edit an upstream visible fact at one scored site, re-render the
        # document, and require the site's answer to change accordingly.
        for k, (site, plan) in enumerate(zip(doc.sites,
                                             doc.latent['recap_plan'], strict=True)):
            after, family, obj, attr = plan
            if family == 'alias_identity':
                continue
            edited = copy.deepcopy(doc.latent)
            if family == 'source_attribution':
                j = next(j for j in range(after, -1, -1)
                         if (edited['events'][j]['object'] == obj and
                             edited['events'][j]['attribute'] == attr))
                old = edited['sources'][j]
                edited['sources'][j] = next(name for name in NAMES if name != old)
            else:
                attr = family.removeprefix('world_')
                earlier = [j for j in range(after, -1, -1)
                           if (edited['events'][j]['object'] == obj and
                               edited['events'][j]['attribute'] == attr)]
                if earlier:
                    j = earlier[0]
                    old = edited['events'][j]['value']
                    edited['events'][j]['value'] = next(v for v in VALUES[attr]
                                                      if v != old)
                else:
                    old = edited['initial'][obj][attr]
                    edited['initial'][obj][attr] = next(v for v in VALUES[attr]
                                                        if v != old)
            changed = realize(edited)
            assert changed.sites[k].answer != site.answer, (i, k, plan, site.answer,
                                                             changed.sites[k].answer)
            assert answer_from_visible_prefix(
                changed.tokens[:changed.sites[k].index]) == changed.sites[k].answer
            causal_checks[family] += 1
            break
    assert lexical_eligible == 49
    return dict(version='document_world_v4_interleaved_structure_pilot',
                seed=seed, documents=n, total_sites=sum(families.values()),
                sites_by_family=dict(families), documents_with_family=dict(docs_with),
                paired_upstream_edits=dict(causal_checks),
                site_count_per_document=dict(sorted(site_counts.items())),
                site_position_fraction=dict(min=min(fractions),
                                            median=sorted(fractions)[len(fractions)//2],
                                            max=max(fractions)),
                first49_x100_alpha_types=len(first100),
                first49_eligible_selection='first 49 documents with at least 100 alphabetic tokens',
                first49_x100_alpha_singletons=sum(v == 1 for v in first100.values()),
                first49_x100_alpha_top10_share=(sum(v for _,v in first100.most_common(10))
                                               / sum(first100.values())),
                checks=['every site independently recovered from visible prefix',
                        'every target included in candidate set',
                        'first eligible site per document follows upstream edit'],
                limitations=['short fixed lexical inventory',
                             'repetitive initial profile and recap frames',
                             'no exact full next-token conditional oracle',
                             'original bigram/skip/induction/bracket mixture absent',
                             'natural incidence not estimated from these target rates'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=23012)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = audit(args.n, args.seed)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
