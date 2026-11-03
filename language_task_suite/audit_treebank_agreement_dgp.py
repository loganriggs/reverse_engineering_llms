"""Semantic-rule and local-form screen for parsed-corpus agreement clauses."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from audit_raw_pcfg_treebank_pilot import leaves, parse_tree
from compare_syntax_priors_ngrams import score
from treebank_agreement_dgp import PAIRS, load_world


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--treebank', type=Path, required=True)
    parser.add_argument('--fit-trees', type=int, default=20000)
    parser.add_argument('--reference-trees', type=int, default=3000)
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--attractor-rate', type=float, default=.2)
    parser.add_argument('--seed', type=int, default=41201)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    world = load_world(str(args.treebank), args.fit_trees)
    reference = []
    with args.treebank.open() as handle:
        for index, line in enumerate(handle):
            if index < args.fit_trees:
                continue
            if index >= args.fit_trees+args.reference_trees:
                break
            reference.append(tuple(leaves(parse_tree(line))))
    rng = random.Random(args.seed)
    examples = []
    originals = []
    counts = Counter()
    seen_subjects = set()
    for _ in range(args.n):
        clause = world.sample(rng, attractor_rate=args.attractor_rate)
        changed = world.counterfactual(clause, rng)
        for item in (clause, changed):
            assert world.number_from_visible_prefix(
                item.tokens[:item.target_index]) == item.number
            assert item.tokens[item.target_index] == PAIRS[item.lemma][
                item.number]
            assert item.tokens[-1] == '.'
            counts['visible_rule_checks'] += 1
        assert clause.lemma == changed.lemma
        assert clause.continuation == changed.continuation
        assert clause.number != changed.number
        assert clause.attractor == changed.attractor
        counts['causal_pairs_checked'] += 1
        counts['attractor_pairs'] += clause.attractor
        counts['lemma/'+clause.lemma] += 1
        seen_subjects.add(tuple(token.lower() for token in clause.subject))
        originals.append(clause.tokens)
        if len(examples) < 5:
            examples.append({'base': ' '.join(clause.tokens),
                             'changed': ' '.join(changed.tokens),
                             'target_base': clause.tokens[clause.target_index],
                             'target_changed': changed.tokens[
                                 changed.target_index]})
    out = {
        'config': {key: value for key, value in vars(args).items()
                   if key not in ('treebank', 'output')} |
                  {'treebank_file': args.treebank.name,
                   'treebank_sha256': hashlib.sha256(
                       args.treebank.read_bytes()).hexdigest()},
        'subject_pool_sizes': {f'{number}_{attractor}': len(pool)
                               for (number, attractor), pool in
                               world.subjects_by_condition.items()},
        'vp_continuation_sizes': {lemma: len(forms) for lemma, forms in
                                  world.continuations.items()},
        'counts': dict(counts),
        'distinct_sampled_subjects': len(seen_subjects),
        'local_ngrams_seen_in_disjoint_reference': score(originals,
                                                        reference),
        'generated_examples': examples,
        'caveat': 'Agreement rule is verified by a visible subject-phrase '
                  'lexicon induced from parse labels, not a full semantic '
                  'parser. Recombining noun phrases and predicates can '
                  'produce implausible meanings. A verb lemma is sampled '
                  'latently, so evaluate the correct versus incorrect '
                  'inflection of that lemma, not exact next-token accuracy.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2)+'\n')
    print('pools', out['subject_pool_sizes'],
          'checks', counts['visible_rule_checks'],
          'seen trigram', out['local_ngrams_seen_in_disjoint_reference'][
              '3']['share'])


if __name__ == '__main__':
    main()
