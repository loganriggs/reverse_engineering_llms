"""Paired discourse-order screen for the linked report DGP; no network data."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from audit_document_structure_vs_fineweb import structure
from audit_project_report_dgp import semantic, verify
from compare_v12_fineweb_fresh import first_words
from project_report_dgp import WORD, realize, sample_plan


def _site_window(doc, site):
    before = doc.text[:site.start].split()[-25:]
    after = doc.text[site.start:].split()[:16]
    return ' '.join(before + after)


def audit(seed, plans, n_form, words):
    rng = random.Random(seed)
    panels = {0.: [], 1.: []}
    sites = {value: {kind: [] for kind in ('total', 'status', 'source')}
             for value in panels}
    changed = 0
    draws = 0
    while len(panels[0.]) < n_form:
        draws += 1
        if draws > 100*n_form:
            raise RuntimeError('Too few reports of requested length')
        plan = sample_plan(rng, coupling_rate=.5,
                           frequency_exponent=1.)
        docs = {entropy: realize(plan, surface_entropy=1.,
                                 discourse_entropy=entropy)
                for entropy in panels}
        for doc in docs.values():
            verify(doc)
        crops = {entropy: first_words(doc.text, words)
                 for entropy, doc in docs.items()}
        if any(len(WORD.findall(crop)) < words for crop in crops.values()):
            continue
        changed += docs[0.].text != docs[1.].text
        for entropy, doc in docs.items():
            panels[entropy].append(crops[entropy])
            for site in doc.sites:
                sites[entropy][site.kind].append(_site_window(doc, site))
    semantic_counts = {}
    for entropy in panels:
        for coupling in (0., 1.):
            key = f'discourse{entropy:g}_coupling{coupling:g}'
            semantic_counts[key] = semantic(
                seed+10000+int(entropy)*100+int(coupling),
                plans, 1., coupling, entropy)['counts']
    return dict(config=dict(seed=seed, plans_per_condition=plans,
                            form_n=n_form, form_words=words,
                            paired_draws=draws),
                changed_documents=changed,
                semantic_counts=semantic_counts,
                form={str(entropy): structure(panel)
                      for entropy, panel in panels.items()},
                rule_window_form={str(entropy): {
                    kind: structure(rows) for kind, rows in site_panels.items()}
                    for entropy, site_panels in sites.items()},
                caveat='Permuting a few background clauses is an incremental '
                       'discourse control, not a realistic language DGP. '
                       'Rule-window form panels include answer-adjacent text.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=91041)
    p.add_argument('--plans', type=int, default=1000)
    p.add_argument('--form-n', type=int, default=64)
    p.add_argument('--form-words', type=int, default=100)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.seed, args.plans, args.form_n,
                   args.form_words)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    for entropy, panel in result['form'].items():
        print(entropy, panel['word_types'],
              panel['repeated_across_docs_trigram_occurrence_share'])
    print('changed', result['changed_documents'], '/', args.form_n)


if __name__ == '__main__':
    main()
