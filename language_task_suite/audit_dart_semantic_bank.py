"""Audit DART triples/text as a corpus-grounded realization resource.

The official DART data stays local; this writes only aggregates. Exact
entity mentions are a conservative visibility screen, not a semantic proof.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

from audit_document_structure_vs_fineweb import structure


DEFAULT_DATA = Path('/home/loganriggs/Coding/neuron_circuits/artifacts/dart/'
                    'data/v1.1.1/dart-v1.1.1-full-train.json')
SPECIAL = {'[TABLECONTEXT]'}


def _mention(text, value):
    if value in SPECIAL:
        return True
    return bool(re.search(r'(?<!\w)'+re.escape(value)+r'(?!\w)',
                          text, re.I))


def audit(path, seed=75931, panel_n=512):
    rows = json.loads(path.read_text())
    rng = random.Random(seed)
    by_source = defaultdict(list)
    counts = Counter()
    predicates = defaultdict(Counter)
    for row in rows:
        triples = row['tripleset']
        counts['triple_sets'] += 1
        for annotation in row['annotations']:
            source, text = annotation['source'], annotation['text']
            by_source[source].append(text)
            counts['annotations'] += 1
            visible = sum(_mention(text, subject) and _mention(text, obj)
                          for subject, _, obj in triples)
            counts[source+':annotations'] += 1
            counts[source+':all_triples_endpoints_visible'] += (
                visible == len(triples))
            counts[source+':two_or_more_triples'] += len(triples) >= 2
            counts[source+':three_or_more_triples'] += len(triples) >= 3
            for _, predicate, _ in triples:
                predicates[source][predicate] += 1
    panels = {}
    for source, texts in sorted(by_source.items()):
        sample = rng.sample(texts, min(panel_n, len(texts)))
        panels[source] = {
            'total_annotations': len(texts),
            'panel_n': len(sample),
            'distinct_predicates': len(predicates[source]),
            'top_predicates': predicates[source].most_common(10),
            'all_endpoints_visible_fraction':
                counts[source+':all_triples_endpoints_visible']/len(texts),
            'two_plus_triples_fraction':
                counts[source+':two_or_more_triples']/len(texts),
            'three_plus_triples_fraction':
                counts[source+':three_or_more_triples']/len(texts),
            'form': structure(sample)}
    return {
        'data': str(path),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'seed': seed, 'panel_n': panel_n,
        'triple_sets': counts['triple_sets'],
        'annotations': counts['annotations'],
        'sources': panels,
        'caveat': 'DART is a supervised meaning/text corpus, not a sampled '
                  'document language distribution. Endpoint-string presence '
                  'does not prove that all relations are expressed correctly '
                  'or that other claims are absent. Source pools vary greatly '
                  'in style and repetition; only aggregates are exported.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--seed', type=int, default=75931)
    parser.add_argument('--panel-n', type=int, default=512)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.data, args.seed, args.panel_n)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    for source, row in result['sources'].items():
        print(source, row['total_annotations'],
              round(row['all_endpoints_visible_fraction'], 3),
              round(row['form'][
                  'repeated_across_docs_trigram_occurrence_share'], 3))


if __name__ == '__main__':
    main()
