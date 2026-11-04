"""Count joint rule opportunities in local natural-text previews.

Eligibility is determined from visible text before any sampled edit. A
candidate is not proof that a model learns the rule or that the untouched
source has a unique next-token answer. Outputs contain aggregates only.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
from statistics import median

from audit_document_structure_vs_fineweb import structure
from audit_semantic_syntax_opportunities import first_words
from corpus_rule_reference_dgp import sample_document


WORD = re.compile(r"\b[A-Za-z]+(?:'[A-Za-z]+)?\b")
FAMILIES = ('list', 'bracket', 'nonce', 'literal_copy', 'natural_style')


def _aggregate_form(texts):
    row = structure(texts)
    return {key: value for key, value in row.items()
            if key not in ('top10_words', 'top_sentence_starts')}


def audit(preview: Path, n: int, max_chars: int):
    rows = json.loads(preview.read_text())
    if len(rows) < n:
        raise ValueError('Too few preview rows')
    counts = Counter()
    sites = Counter()
    distances = defaultdict(list)
    by_family_eligible = Counter()
    cooccurrence = Counter()
    word_lengths = []
    multi_rows, control_rows = [], []
    for row in rows[:n]:
        text = row['preview'][:max_chars]
        word_lengths.append(len(WORD.findall(text)))
        doc = sample_document(
            text, random.Random(700000+row['index']), (),
            max_chars=max_chars, list_rate=1., bracket_rate=1.,
            nonce_rate=0., list_shift_rate=.5, square_rate=.5,
            literal_copy_rate=1., natural_style_rate=1.)
        eligible = set(doc.eligible)
        assert eligible <= set(FAMILIES)
        counts[f'eligible_families_per_document/{len(eligible)}'] += 1
        summary = (len(WORD.findall(text)), first_words(text, 100))
        (multi_rows if len(eligible) >= 2 else control_rows).append(summary)
        for family in eligible:
            by_family_eligible[family] += 1
        for i, left in enumerate(FAMILIES):
            for right in FAMILIES[i+1:]:
                if left in eligible and right in eligible:
                    cooccurrence[f'{left}+{right}'] += 1
        applied = set(doc.applied)
        assert applied <= eligible
        counts[f'applied_families_per_document/{len(applied)}'] += 1
        counts['applied_sites_total'] += len(doc.sites)
        for site in doc.sites:
            sites[site.kind] += 1
            distances[site.kind].append(site.target[0]-site.source[1])
    matched = []
    available = list(control_rows)
    for length, _ in multi_rows:
        if not available:
            break
        j = min(range(len(available)),
                key=lambda k: (abs(available[k][0]-length), k))
        matched.append(available.pop(j))
    form = None
    if len(matched) >= 30 and len(matched) == len(multi_rows):
        form = {
            'multitask': _aggregate_form([text for _, text in multi_rows]),
            'length_matched_nonmultitask': _aggregate_form(
                [text for _, text in matched]),
            'median_visible_word_length_multitask': median(
                length for length, _ in multi_rows),
            'median_visible_word_length_control': median(
                length for length, _ in matched),
            'max_word_length_mismatch': max(
                abs(a[0]-b[0]) for a, b in zip(multi_rows, matched)),
        }
    return {
        'source': {'first_index': rows[0]['index'],
                   'last_index': rows[n-1]['index'],
                   'preview_sha256': hashlib.sha256(
                       preview.read_bytes()).hexdigest(),
                   'documents': n, 'max_chars': max_chars,
                   'median_visible_words': median(word_lengths)},
        'eligible_document_counts': dict(by_family_eligible),
        'eligible_family_count_histogram': {
            str(i): counts[f'eligible_families_per_document/{i}']
            for i in range(len(FAMILIES)+1)},
        'pairwise_eligible_document_counts': dict(cooccurrence),
        'applied_family_count_histogram': {
            str(i): counts[f'applied_families_per_document/{i}']
            for i in range(len(FAMILIES)+1)},
        'applied_site_counts': dict(sites),
        'median_source_target_char_gap': {
            family: median(values) for family, values in distances.items()},
        'multitask_selection_form': form,
        'scope': 'First max_chars characters of each local FineWeb preview. '
                 'List/bracket/literal-copy/natural-style eligible edits '
                 'were applied jointly and the document checker passed. '
                 'Nonce eligibility is counted, but nonce edits were off '
                 'because they require a separate code pool. These are '
                 'mechanical opportunities, not semantic rule prevalence '
                 'or model competence. The multi-opportunity form panel '
                 'uses greedy length-matched, distinct controls; it is not '
                 'a causal effect of task enrichment. No source text is saved.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preview', type=Path, required=True)
    p.add_argument('--documents', type=int, default=2000)
    p.add_argument('--max-chars', type=int, default=2500)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.preview, a.documents, a.max_chars)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['source']['first_index'],
          result['eligible_document_counts'],
          result['eligible_family_count_histogram'], flush=True)


if __name__ == '__main__':
    main()
