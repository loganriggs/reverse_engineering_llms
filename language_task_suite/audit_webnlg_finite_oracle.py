"""Check the finite WebNLG oracle and quantify corpus-support shortcuts."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re

from audit_document_structure_vs_fineweb import structure
from webnlg_finite_oracle import build


DEFAULT_PREVIEW = Path('/home/loganriggs/Coding/neuron_circuits/artifacts/'
                       'corpus_surname_fresh_140000_2000_preview.json')
WORDS = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def first_words(text, n):
    matches = list(WORDS.finditer(text))
    return text[:matches[n-1].end()] if len(matches) >= n else None


def audit(preview, seed=75981, sample_n=512, form_words=20,
          category_exponent=0.):
    rng = random.Random(seed)
    oracle = build(category_exponent=category_exponent)
    assert abs(oracle.root.mass-1.) < 1e-10
    n_by_category = Counter()
    generated, items = [], []
    for _ in range(sample_n):
        entry, text = oracle.sample(rng)
        n_by_category[entry.category] += 1
        crop = first_words(text, form_words)
        if crop is not None:
            generated.append(crop)
        ids = tuple(oracle.tokenizer.encode(
            text, add_special_tokens=False).ids)+(oracle.eos,)
        items.append(ids)
    natural_rows = json.loads(preview.read_text())
    natural_pool = [first_words(row['preview'], form_words)
                    for row in natural_rows]
    natural_pool = [text for text in natural_pool if text is not None]
    if len(natural_pool) < len(generated):
        raise ValueError('Natural preview too small for matched panel')
    natural = rng.sample(natural_pool, len(generated))
    prefix_stats = {}
    for length in (0, 1, 2, 4, 8, 16, 32):
        panel = [seq for seq in items if len(seq)-1 >= length]
        stats = [oracle.prefix_stats(seq[:length]) for seq in panel]
        prefix_stats[str(length)] = {
            'n': len(panel),
            'mean_support': sum(x['support'] for x in stats)/len(stats),
            'median_support': sorted(x['support'] for x in stats)[len(stats)//2],
            'one_branch_fraction': sum(x['branches'] == 1 for x in stats)/
                                   len(stats),
            'mean_entropy_bits': sum(x['entropy_bits'] for x in stats)/len(stats)}
    # Compare the trie to a direct finite-sum calculation for prefixes from
    # independently sampled support items; this verifies weighted duplicate
    # texts and nonuniform category/entry/lex weights.
    for seq in rng.sample(items, min(16, len(items))):
        length = rng.randrange(min(9, len(seq)))
        prefix = seq[:length]
        direct = Counter()
        mass = 0.
        for _, candidate, weight, _ in oracle.items:
            if candidate[:length] == prefix:
                mass += weight
                direct[candidate[length]] += weight
        got = oracle.next_distribution(prefix)
        assert set(got) == set(direct)
        assert max(abs(got[token]-value/mass)
                   for token, value in direct.items()) < 1e-10
    return {
        'config': {'seed': seed, 'sample_n': sample_n,
                   'form_words': form_words,
                   'category_exponent': category_exponent,
                   'natural_preview': str(preview)},
        'support_items': len(oracle.items),
        'unique_texts': len({text for _, _, _, text in oracle.items}),
        'trie_nodes': oracle.node_count,
        'category_counts': dict(n_by_category),
        'form_eligible': len(generated),
        'form': {'generated': structure(generated),
                 'natural_matched_n': structure(natural)},
        'prefix_stats': prefix_stats,
        'oracle_check': '16 sampled weighted finite sums match the trie to '
                        'absolute error <1e-10; root mass 1',
        'caveat': 'The law samples original finite corpus texts. Long prefix '
                  'determinism reflects support memorization, not a learned '
                  'linguistic rule. Natural comparison is a short matched-'
                  'length form screen, not matched by topic/task.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview', type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument('--seed', type=int, default=75981)
    parser.add_argument('--sample-n', type=int, default=512)
    parser.add_argument('--form-words', type=int, default=20)
    parser.add_argument('--category-exponent', type=float, default=0.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.preview, args.seed, args.sample_n,
                   args.form_words, args.category_exponent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('support', result['support_items'], 'trie', result['trie_nodes'],
          'form eligible', result['form_eligible'])
    for length, stats in result['prefix_stats'].items():
        print(length, stats['median_support'],
              round(stats['one_branch_fraction'], 3))


if __name__ == '__main__':
    main()
