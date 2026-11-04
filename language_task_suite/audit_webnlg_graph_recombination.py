"""Form and structural screens for novel WebNLG role-graph recombination."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random

from audit_document_structure_vs_fineweb import structure
from audit_webnlg_finite_oracle import DEFAULT_PREVIEW, first_words
from webnlg_graph_recombination import (
    _clean, _lexical_name, load_forms, render, sample,
)


def audit(preview, seed=76001, n=512, form_words=20,
          cross_role_rate=.5):
    groups = load_forms()
    original_graphs = set()
    for key, forms in groups.items():
        for form in forms:
            values = {tag: _lexical_name(entity) for tag, entity in
                      form.roles}
            original_graphs.add(tuple(sorted(
                (values[s], predicate, values[o])
                for s, predicate, o in key[1])))
    rng = random.Random(seed)
    counts = Counter()
    texts, source_texts, category_counts = [], [], Counter()
    for _ in range(n):
        row = sample(groups, rng, cross_role_rate=cross_role_rate)
        counts['draws'] += 1
        counts['crossed'] += row['crossed_tag'] is not None
        counts['novel_graph'] += (tuple(sorted(row['triples'])) not in
                                  original_graphs)
        assert render(row['template'], row['roles']) == row['text']
        assert all(_clean(value) in row['text']
                   for value in row['roles'].values())
        category_counts[row['category']] += 1
        cropped = first_words(row['text'], form_words)
        if cropped is not None:
            texts.append(cropped)
            source_texts.append(first_words(row['source_text'], form_words))
    source_texts = [text for text in source_texts if text is not None]
    natural_rows = json.loads(preview.read_text())
    natural_pool = [first_words(row['preview'], form_words)
                    for row in natural_rows]
    natural_pool = [text for text in natural_pool if text is not None]
    if len(natural_pool) < len(texts):
        raise ValueError('Natural panel too small')
    natural = rng.sample(natural_pool, len(texts))
    return {
        'config': {'seed': seed, 'n': n, 'form_words': form_words,
                   'cross_role_rate': cross_role_rate,
                   'natural_preview': str(preview)},
        'bank': {'graph_keys': len(groups),
                 'forms': sum(map(len, groups.values())),
                 'original_normalized_graphs': len(original_graphs)},
        'counts': dict(counts),
        'category_counts': dict(category_counts),
        'form_eligible': len(texts),
        'form': {'generated': structure(texts),
                 'source_human': structure(source_texts),
                 'natural_matched_n': structure(natural)},
        'caveat': 'Role-graph structure and slot fill are checked. A source '
                  'template can assert unsupported English predicates or '
                  'be ungrammatical for a target value. This is a candidate '
                  'generator, not an approved semantic DGP; the natural '
                  'panel is not topic or task matched.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview', type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument('--seed', type=int, default=76001)
    parser.add_argument('--n', type=int, default=512)
    parser.add_argument('--form-words', type=int, default=20)
    parser.add_argument('--cross-role-rate', type=float, default=.5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.preview, args.seed, args.n, args.form_words,
                   args.cross_role_rate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['bank'], result['counts'], result['form_eligible'])
    for key, form in result['form'].items():
        print(key, round(form[
            'repeated_across_docs_trigram_occurrence_share'], 3))


if __name__ == '__main__':
    main()
