"""Held-out local n-gram screen for raw PCFG and tree substitution."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

from audit_raw_pcfg_treebank_pilot import Grammar, leaves, parse_tree
from audit_tree_substitution_pilot import eligible, sample as substitute


def grams(sentence, n):
    tokens = [token.lower() for token in sentence]
    return zip(*(tokens[i:] for i in range(n)))


def score(sentences, reference):
    out = {}
    for n in (1, 2, 3, 4):
        observed = {gram for sentence in reference
                    for gram in grams(sentence, n)}
        numerator = denominator = 0
        for sentence in sentences:
            for gram in grams(sentence, n):
                denominator += 1
                numerator += gram in observed
        out[str(n)] = {'seen_occurrences': numerator,
                       'total_occurrences': denominator,
                       'share': round(numerator/denominator, 4)}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--treebank', type=Path, required=True)
    parser.add_argument('--fit-trees', type=int, default=20000)
    parser.add_argument('--reference-trees', type=int, default=3000)
    parser.add_argument('--test-sentences', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=41001)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    fit, reference, heldout = [], [], []
    with args.treebank.open() as handle:
        for index, line in enumerate(handle):
            if index >= args.fit_trees+2*args.reference_trees:
                break
            tree = parse_tree(line)
            if index < args.fit_trees:
                fit.append(tree)
            elif index < args.fit_trees+args.reference_trees:
                reference.append(tuple(leaves(tree)))
            else:
                heldout.append(tuple(leaves(tree)))
    grammar = Grammar(fit)
    pool = defaultdict(list)
    for tree in fit:
        for _, node, key, _ in eligible(tree):
            pool[key].append(node)
    raw = []
    while len(raw) < args.test_sentences:
        try:
            raw.append(tuple(grammar.sample(rng)))
        except OverflowError:
            pass
    changed = [substitute(fit, pool, rng)
               for _ in range(args.test_sentences)]
    filtered = [sentence for sentence in heldout
                if 6 <= len(sentence) <= 40]
    natural = rng.sample(filtered, args.test_sentences)
    out = {
        'config': {key: value for key, value in vars(args).items()
                   if key not in ('treebank', 'output')} |
                  {'treebank_file': args.treebank.name,
                   'treebank_sha256': hashlib.sha256(
                       args.treebank.read_bytes()).hexdigest()},
        'reference_pool_size': len(reference),
        'arms': {name: {
            'mean_tokens': sum(map(len, sentences))/len(sentences),
            'ngrams': score(sentences, reference),
        } for name, sentences in (
            ('independent_treebank', natural),
            ('raw_pcfg', raw),
            ('tree_substitution', changed))},
        'caveat': 'Reference n-gram bank and independent natural candidates '
                  'are disjoint treebank slices. This is a local form proxy, '
                  'not semantic coherence or document realism. No source '
                  'or generated text is saved.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2)+'\n')
    for name, value in out['arms'].items():
        print(name, 'mean', round(value['mean_tokens'], 2),
              {n: v['share'] for n, v in value['ngrams'].items()})


if __name__ == '__main__':
    main()
