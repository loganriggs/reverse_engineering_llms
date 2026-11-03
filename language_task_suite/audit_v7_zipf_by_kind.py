"""Measure whether V7 document-kind reweighting can fix its word head."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random

from audit_surface_repetition import WORD
from document_mixture_v7_cross import KINDS, sample_document


HERE = Path(__file__).resolve().parent
REFERENCE = HERE/'results/v14_zipf_seed99851_n480/v14_vs_fineweb_word_rank.json'


def run(n: int, seed: int) -> dict:
    reference = json.loads(REFERENCE.read_text())['summary']['FineWeb']
    arms = {}
    for kind in KINDS:
        rng = random.Random(seed)
        counts = Counter()
        first100 = Counter()
        lengths = []
        for _ in range(n):
            doc = sample_document(rng, kind=kind)
            words = [w.lower() for w in WORD.findall(' '.join(doc.tokens))]
            counts.update(words)
            lengths.append(len(words))
            if len(words) >= 100:
                first100.update(words[:100])
        total = sum(counts.values())
        arms[kind] = {
            'documents': n, 'words': total,
            'mean_words_per_document': total/n,
            'documents_with_100_words': sum(length >= 100 for length in lengths),
            'the_share': counts['the']/total,
            'top10_share': sum(value for _, value in counts.most_common(10))/total,
            'observed_vocabulary': len(counts),
            'top10': counts.most_common(10),
            'first100_the_share': (first100['the']/sum(first100.values())
                                   if first100 else None),
        }
    minimum = min(row['the_share'] for row in arms.values())
    return {
        'config': {'n_per_kind': n, 'seed': seed,
                   'word_rule': 'lowercase WORD regex; all words in each document'},
        'fineweb_reference': {
            'word_windows': reference['n_windows'],
            'words': reference['total_words'],
            'the_share': dict(reference['top25'])['the']/reference['total_words'],
            'top10_share': reference['top10_mass'],
        },
        'by_kind': arms,
        'min_kind_the_share': minimum,
        'interpretation': ('A nonnegative mixture of fixed kind-level word '
                           'distributions cannot fall below its least '
                           'the-heavy kind. The 100-word comparison applies '
                           'only to long V7 kinds; record and score reports '
                           'are too short. A renderer change is required '
                           'to match the reference common-word head. The '
                           'panel is an independent sample, not paired '
                           'FineWeb documents.'),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=500)
    parser.add_argument('--seed', type=int, default=31900)
    parser.add_argument('--output', type=Path,
                        default=HERE/'results/v7_zipf_by_kind_n500.json')
    args = parser.parse_args()
    result = run(args.n, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('FineWeb the share:', result['fineweb_reference']['the_share'])
    for kind, row in result['by_kind'].items():
        print(kind, row['the_share'], row['top10_share'])
    print('Saved', args.output)


if __name__ == '__main__':
    main()
