"""Corpus-tree substitution as a conservative probabilistic syntax prior.

Samples a source parse, then swaps one short phrase with another from the
training treebank matched by phrase category and approximate NP number.
This is a surface prior experiment, not a document or semantic-task DGP.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

from audit_document_structure_vs_fineweb import structure
from audit_raw_pcfg_treebank_pilot import (
    documents, leaves, parse_tree,
)


PHRASES = frozenset(('NP', 'PP', 'ADVP', 'ADJP'))


def walk(tree, path=()):
    yield path, tree
    for index, child in enumerate(tree[1]):
        if not isinstance(child, str):
            yield from walk(child, path+(index,))


def signature(tree):
    label = tree[0]
    if label != 'NP':
        return label
    preterminals = [node[0] for _, node in walk(tree)
                    if all(isinstance(child, str) for child in node[1])]
    for pos in reversed(preterminals):
        if pos in ('NN', 'NNP', 'PRP'):
            return 'NP/singular'
        if pos in ('NNS', 'NNPS'):
            return 'NP/plural'
    return 'NP/other'


def eligible(tree):
    return [(path, node, signature(node), len(tuple(leaves(node))))
            for path, node in walk(tree) if node[0] in PHRASES
            and 2 <= len(tuple(leaves(node))) <= 10]


def replace(tree, path, donor):
    if not path:
        return donor
    children = list(tree[1])
    children[path[0]] = replace(children[path[0]], path[1:], donor)
    return tree[0], tuple(children)


def sample(trees, pool, rng):
    for _ in range(100):
        base = rng.choice(trees)
        candidates = eligible(base)
        if not candidates:
            continue
        path, current, key, length = rng.choice(candidates)
        donors = pool[key]
        for _ in range(30):
            donor = rng.choice(donors)
            donor_length = len(tuple(leaves(donor)))
            if donor != current and length/2 <= donor_length <= length*2:
                changed = replace(base, path, donor)
                words = tuple(leaves(changed))
                if 6 <= len(words) <= 40:
                    return words
    raise RuntimeError('Could not make a compatible substitution')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--treebank', type=Path, required=True)
    parser.add_argument('--fit-trees', type=int, default=20000)
    parser.add_argument('--reference-trees', type=int, default=3000)
    parser.add_argument('--generated-sentences', type=int, default=5000)
    parser.add_argument('--documents', type=int, default=256)
    parser.add_argument('--sentences-per-document', type=int, default=5)
    parser.add_argument('--words', type=int, default=55)
    parser.add_argument('--seed', type=int, default=40901)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    fit, reference = [], []
    with args.treebank.open() as handle:
        for index, line in enumerate(handle):
            if index >= args.fit_trees+args.reference_trees:
                break
            tree = parse_tree(line)
            if index < args.fit_trees:
                fit.append(tree)
            else:
                reference.append(tuple(leaves(tree)))
    pool = defaultdict(list)
    for tree in fit:
        for _, node, key, _ in eligible(tree):
            pool[key].append(node)
    generated = [sample(fit, pool, rng)
                 for _ in range(args.generated_sentences)]
    generated_docs = documents(generated, args.documents,
                               args.sentences_per_document, rng, args.words)
    reference_docs = documents(reference, args.documents,
                               args.sentences_per_document, rng, args.words)
    train_sentences = {tuple(leaves(tree)) for tree in fit}
    out = {
        'config': {key: value for key, value in vars(args).items()
                   if key not in ('treebank', 'output')} |
                  {'treebank_file': args.treebank.name,
                   'treebank_sha256': hashlib.sha256(
                       args.treebank.read_bytes()).hexdigest()},
        'pool_sizes': {key: len(value) for key, value in pool.items()},
        'generated_exact_training_sentence_share': round(
            sum(sentence in train_sentences for sentence in generated) /
            len(generated), 4),
        'reference': structure(reference_docs),
        'substitution': structure(generated_docs),
        'caveat': 'One typed phrase swap per source parse. Matching phrase '
                  'type does not ensure grammatical agreement or semantics. '
                  'Independent sentences are concatenated into pseudo-docs; '
                  'no source or generated excerpts are saved.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2)+'\n')
    print('pools', out['pool_sizes'], 'copy',
          out['generated_exact_training_sentence_share'])
    for arm in ('reference', 'substitution'):
        print(arm, 'docs', out[arm]['n'], 'top10',
              round(out[arm]['top10_word_share'], 3), 'repeat',
              round(out[arm][
                  'repeated_across_docs_trigram_occurrence_share'], 3))


if __name__ == '__main__':
    main()
