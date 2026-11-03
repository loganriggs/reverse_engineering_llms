"""Semantic/causal and natural-form audit for purpose-driven report DGP."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

from audit_document_structure_vs_fineweb import natural, structure
from compare_v12_fineweb_fresh import first_words
from document_world_v9_typed_lexicon import PEOPLE
from project_report_dgp import (
    Document, WORD, answer_from_visible_prefix, edit,
    realize, sample_plan,
)


def verify(document: Document):
    previous_end = 0
    for site in document.sites:
        assert previous_end <= site.start < site.end <= len(document.text)
        assert document.text[site.start:site.end] == site.answer
        assert answer_from_visible_prefix(
            document.text[:site.start], site.kind) == site.answer
        assert site.answer in site.candidates
        previous_end = site.end
    assert [site.kind for site in document.sites] == [
        'total', 'status', 'source']


def _values(document):
    return {site.kind: site.answer for site in document.sites}


def semantic(seed, plans, surface_entropy, coupling_rate,
             discourse_entropy=0.):
    rng = random.Random(seed)
    counts = Counter()
    samples = []
    for index in range(plans):
        plan = sample_plan(rng, coupling_rate=coupling_rate,
                           frequency_exponent=1.)
        doc = realize(plan, surface_entropy=surface_entropy,
                      paragraph_rate=.25,
                      discourse_entropy=discourse_entropy)
        verify(doc)
        counts['plans'] += 1
        counts['coupled' if plan.coupled else 'independent'] += 1
        counts[f'purpose/{plan.purpose}'] += 1
        counts[f'domain/{plan.project}'] += 1
        counts[f'status/{plan.status}'] += 1
        if len(samples) < 8:
            samples.append(dict(purpose=plan.purpose,
                                coupled=plan.coupled,
                                text=doc.text,
                                sites=[vars(site) for site in doc.sites]))
        original = _values(doc)
        choices = [name for name in PEOPLE
                   if name not in (plan.first_source, plan.second_source)]
        altered = realize(edit(plan, 'second_source', choices[0]),
                          surface_entropy=surface_entropy,
                          paragraph_rate=.25,
                          discourse_entropy=discourse_entropy)
        verify(altered)
        after = _values(altered)
        assert after['source'] != original['source']
        assert after['total'] == original['total']
        assert after['status'] == original['status']
        counts['source_edits_selective'] += 1
        altered = realize(edit(plan, 'first_count',
                               plan.first_count+3),
                          surface_entropy=surface_entropy,
                          paragraph_rate=.25,
                          discourse_entropy=discourse_entropy)
        verify(altered)
        after = _values(altered)
        assert int(after['total']) == int(original['total'])+3
        assert after['source'] == original['source']
        counts['count_edits_total_flip'] += 1
        if plan.coupled:
            # Force a threshold crossing while preserving all other fields.
            threshold = plan.total+1 if plan.status == 'open' \
                else plan.total-1
            altered = realize(edit(plan, 'target', threshold),
                              surface_entropy=surface_entropy,
                              paragraph_rate=.25,
                              discourse_entropy=discourse_entropy)
            verify(altered)
            after = _values(altered)
            assert after['status'] != original['status']
            assert after['total'] == original['total']
            counts['threshold_edits_status_flip'] += 1
        else:
            altered = realize(edit(plan, 'decision',
                                   'closed' if plan.status == 'open'
                                   else 'open'),
                              surface_entropy=surface_entropy,
                              paragraph_rate=.25,
                              discourse_entropy=discourse_entropy)
            verify(altered)
            after = _values(altered)
            assert after['status'] != original['status']
            assert after['total'] == original['total']
            counts['decision_edits_status_flip'] += 1
        counts['mean_words_sum'] += len(doc.text.split())
    return dict(counts=dict(counts), examples=samples)


def form(seed, n, words, surface_entropy, coupling_rate,
         discourse_entropy=0.):
    rng = random.Random(seed)
    rows = []
    draws = 0
    while len(rows) < n:
        draws += 1
        if draws > 100*n:
            raise RuntimeError('Too few long reports')
        plan = sample_plan(rng, coupling_rate=coupling_rate,
                           frequency_exponent=1.)
        doc = realize(plan, surface_entropy=surface_entropy,
                      paragraph_rate=.25,
                      discourse_entropy=discourse_entropy)
        crop = first_words(doc.text, words)
        if len(WORD.findall(crop)) < words:
            continue
        rows.append(crop)
    return dict(draws=draws, panel=structure(rows))


def audit(args):
    semantic_panels = {}
    form_panels = {}
    for entropy in (0., 1.):
        for coupling in (0., 1.):
            key = f'entropy{entropy:g}_coupling{coupling:g}'
            semantic_panels[key] = semantic(
                args.seed+int(entropy)*100+int(coupling),
                args.plans, entropy, coupling)
            form_panels[key] = form(
                args.seed+10_000+int(entropy)*100+int(coupling),
                args.form_n, args.form_words, entropy, coupling)
    reference, scanned = natural(args.fineweb_skip,
                                 args.form_n, args.form_words)
    return dict(config=dict(seed=args.seed, plans_per_condition=args.plans,
                            form_n=args.form_n,
                            form_words=args.form_words,
                            fineweb_skip=args.fineweb_skip,
                            fineweb_source='HuggingFaceFW/fineweb sample-10BT'),
                semantic=semantic_panels,
                form=form_panels,
                fineweb=dict(panel=reference, rows_scanned=scanned),
                caveat='Candidate report generator, not accepted as '
                       'realistic. Exact oracle is limited to the three '
                       'marked target sites; background next-token law '
                       'has not been marginalized. The FineWeb form panel '
                       'is broad web text, not genre-matched public works.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=71021)
    p.add_argument('--plans', type=int, default=1000)
    p.add_argument('--form-n', type=int, default=64)
    p.add_argument('--form-words', type=int, default=100)
    p.add_argument('--fineweb-skip', type=int, default=93000)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    for key, row in result['form'].items():
        panel = row['panel']
        print(key, 'types', panel['word_types'], 'trigram',
              panel['repeated_across_docs_trigram_occurrence_share'],
              flush=True)
    print('fineweb', result['fineweb']['panel']['word_types'],
          result['fineweb']['panel'][
              'repeated_across_docs_trigram_occurrence_share'],
          flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
