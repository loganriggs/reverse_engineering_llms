"""Small raw-PCFG form pilot from Jumelet & Zuidema's public treebank.

This is deliberately *not* their state-split PCFG. It tests whether a cheap
syntax prior merits integration into the rule-known toy language. The source
treebank stays outside this repository; aggregate form statistics are saved.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

from audit_document_structure_vs_fineweb import structure
from compare_v12_fineweb_fresh import first_words


TOKENS = re.compile(r'\(|\)|[^\s()]+')


def parse_tree(line):
    tokens = TOKENS.findall(line)
    def node(index):
        assert tokens[index] == '('
        label = tokens[index+1]
        index += 2
        children = []
        while tokens[index] != ')':
            if tokens[index] == '(':
                child, index = node(index)
                children.append(child)
            else:
                children.append(tokens[index])
                index += 1
        return (label, tuple(children)), index+1
    tree, end = node(0)
    assert end == len(tokens) and tree[0] == 'ROOT'
    return tree


def leaves(tree):
    for child in tree[1]:
        if isinstance(child, str):
            yield child
        else:
            yield from leaves(child)


class Grammar:
    def __init__(self, trees):
        counts = defaultdict(Counter)
        for tree in trees:
            self._add(tree, counts)
        self.rules = {}
        for label, alternatives in counts.items():
            choices = list(alternatives.items())
            cumulative = []
            running = 0
            for _, count in choices:
                running += count
                cumulative.append(running)
            self.rules[label] = ([rule for rule, _ in choices],
                                 cumulative, running)
        self.production_types = sum(len(x[0]) for x in self.rules.values())

    @classmethod
    def _add(cls, tree, counts):
        label, children = tree
        rule = tuple(('T', child) if isinstance(child, str)
                     else ('N', child[0]) for child in children)
        counts[label][rule] += 1
        for child in children:
            if not isinstance(child, str):
                cls._add(child, counts)

    def sample(self, rng, max_depth=18, max_tokens=40):
        output = []
        def expand(label, depth):
            if depth > max_depth or len(output) > max_tokens:
                raise OverflowError
            choices, cumulative, total = self.rules[label]
            rule = choices[bisect_right(cumulative, rng.randrange(total))]
            for kind, value in rule:
                if kind == 'T':
                    output.append(value)
                else:
                    expand(value, depth+1)
        expand('ROOT', 0)
        if not 6 <= len(output) <= max_tokens:
            raise OverflowError
        return output


def documents(sentences, n, n_sentences, rng, words):
    results = []
    for _ in range(n):
        pieces = [rng.choice(sentences) for _ in range(n_sentences)]
        text = ' '.join(' '.join(sentence) for sentence in pieces)
        crop = first_words(text, words)
        if len(crop.split()) < words:
            continue
        results.append(crop)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--treebank', type=Path, required=True)
    parser.add_argument('--fit-trees', type=int, default=20000)
    parser.add_argument('--reference-trees', type=int, default=2000)
    parser.add_argument('--generated-sentences', type=int, default=5000)
    parser.add_argument('--documents', type=int, default=256)
    parser.add_argument('--sentences-per-document', type=int, default=5)
    parser.add_argument('--words', type=int, default=55)
    parser.add_argument('--seed', type=int, default=40801)
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
    grammar = Grammar(fit)
    generated, rejected = [], 0
    while len(generated) < args.generated_sentences:
        try:
            generated.append(tuple(grammar.sample(rng)))
        except OverflowError:
            rejected += 1
        if rejected > 100*args.generated_sentences:
            raise RuntimeError('Grammar rejection rate too high')
    generated_docs = documents(generated, args.documents,
                               args.sentences_per_document, rng, args.words)
    reference_docs = documents(reference, args.documents,
                               args.sentences_per_document, rng, args.words)
    out = {
        'config': {key: value for key, value in vars(args).items()
                   if key not in ('treebank', 'output')} |
                  {'treebank_file': args.treebank.name,
                   'treebank_sha256': hashlib.sha256(
                       args.treebank.read_bytes()).hexdigest()},
        'grammar_nonterminals': len(grammar.rules),
        'grammar_production_types': grammar.production_types,
        'rejected_derivations': rejected,
        'reference': structure(reference_docs),
        'raw_pcfg': structure(generated_docs),
        'caveat': 'Raw count PCFG, not the paper\'s state-split grammar. '
                  'Independent sentences are concatenated into pseudo-docs; '
                  'this does not model discourse or semantic task relations. '
                  'Original corpus text and generated excerpts are omitted.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2)+'\n')
    print('grammar', out['grammar_nonterminals'],
          out['grammar_production_types'], 'rejections', rejected)
    for arm in ('reference', 'raw_pcfg'):
        print(arm, 'docs', out[arm]['n'], 'top10',
              round(out[arm]['top10_word_share'], 3), 'repeat',
              round(out[arm][
                  'repeated_across_docs_trigram_occurrence_share'], 3))


if __name__ == '__main__':
    main()
