"""Semantic-site and form audit for the joint treebank syntax candidate."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from audit_raw_pcfg_treebank_pilot import leaves, parse_tree
from audit_totto_table_description_dgp import form
from compare_syntax_priors_ngrams import score
from treebank_joint_syntax_document import (
    Document, Site, check_visible, load, sample,
)


def false_site(doc, site):
    tokens = list(doc.tokens)
    if site.family == 'bracket_close':
        wrong = (']',)
    elif site.family == 'agreement':
        wrong = ({'is':'are','are':'is','has':'have','have':'has',
                  'does':'do','do':'does'}[site.answer[0]],)
    else:
        wrong = ('other',) * len(site.answer)
    tokens[site.position:site.position+len(site.answer)] = wrong
    sites = tuple(Site(item.family, item.position,
                       wrong if item is site else item.answer,
                       item.source_span, item.lemma)
                  for item in doc.sites)
    return Document(tuple(tokens), sites, doc.controls, doc.clauses)


def first_words(text, n):
    import re
    matches = list(re.finditer(r"[A-Za-z]+(?:'[A-Za-z]+)?", text))
    return text[:matches[n-1].end()] if len(matches) >= n else text


def audit(path, fit_trees=100000, reference_trees=3000,
          n=512, seed=75001):
    world, pps, source = load(path, fit_trees)
    reference, heldout = [], []
    with Path(path).open() as handle:
        for i, line in enumerate(handle):
            if i < fit_trees:
                continue
            if i >= fit_trees + 2*reference_trees:
                break
            words = tuple(leaves(parse_tree(line)))
            if i < fit_trees + reference_trees:
                reference.append(words)
            else:
                heldout.append(words)
    rng = random.Random(seed)
    if len(heldout) < 4*n:
        raise ValueError('Need four disjoint natural sentences per document')
    sampled_reference = rng.sample(heldout, 4*n)
    natural = [' '.join(' '.join(sentence) for sentence in
                        sampled_reference[4*i:4*i+4]) for i in range(n)]
    arms = {
        'recombined_joint': {'source_rate': 0., 'bracket_rate': .25,
                             'reuse_rate': .35},
        'source_joint': {'source_rate': 1., 'bracket_rate': .25,
                         'reuse_rate': .35},
        'head_matched_joint': {'source_rate': 1., 'head_match_rate': 1.,
                               'bracket_rate': .25, 'reuse_rate': .35},
        'source_plain': {'source_rate': 1., 'bracket_rate': 0.,
                         'reuse_rate': 0.},
    }
    result = {
        'config': {'fit_trees': fit_trees,
                   'reference_trees': reference_trees, 'n': n,
                   'seed': seed,
                   'treebank_sha256': hashlib.sha256(
                       Path(path).read_bytes()).hexdigest()},
        'pool_sizes': {'pp': len(pps), 'source_complete_clause': len(source),
                       'singular_subject': len(world.subjects[0]),
                       'plural_subject': len(world.subjects[1])},
        'natural_treebank_pseudodocs': form(
            [first_words(x, 60) for x in natural]),
        'arms': {},
        'caveat': 'Four independently sampled treebank sentences are only '
                  'a local-form reference, not coherent natural documents. '
                  'Source-complete clauses preserve local syntax; recombined '
                  'NP/VP clauses and inserted PPs can be semantically odd. '
                  'Agreement site labels require the sampled verb lemma; '
                  'the visible subject determines number, not lemma.'}
    for arm, controls in arms.items():
        docs = [sample(world, pps, seed+i, source_clauses=source,
                       **controls) for i in range(n)]
        counts = Counter()
        windows = {family: [] for family in
                   ('agreement', 'bracket_close', 'subject_reuse')}
        for doc in docs:
            check_visible(doc, world)
            counts['head_swaps'] += doc.controls['head_swaps']
            families = {site.family for site in doc.sites}
            counts['docs_all_three_families'] += len(families) == 3
            counts['docs_bracket_and_agreement'] += (
                'bracket_close' in families and 'agreement' in families)
            for site in doc.sites:
                counts['sites/'+site.family] += 1
                try:
                    check_visible(false_site(doc, site), world)
                except (ValueError, AssertionError):
                    counts['false_sites_rejected'] += 1
                else:
                    raise AssertionError(('False site accepted', site))
            for family in windows:
                chosen = next((site for site in doc.sites
                               if site.family == family), None)
                if chosen is not None:
                    lo = max(0, chosen.position-20)
                    hi = min(len(doc.tokens), chosen.position+21)
                    windows[family].append(' '.join(doc.tokens[lo:hi]))
        texts = [first_words(doc.text(), 60) for doc in docs]
        clauses = [clause for doc in docs for clause in doc.clauses]
        result['arms'][arm] = {
            'counts': dict(counts),
            'first60_form': form(texts),
            'site_windows': {family: form(items) if items else None
                             for family, items in windows.items()},
            'local_ngrams_seen_in_disjoint_reference':
                score(clauses, reference),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--treebank', type=Path, required=True)
    parser.add_argument('--fit-trees', type=int, default=100000)
    parser.add_argument('--reference-trees', type=int, default=3000)
    parser.add_argument('--n', type=int, default=512)
    parser.add_argument('--seed', type=int, default=75001)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.treebank, args.fit_trees, args.reference_trees,
                   args.n, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('pools', result['pool_sizes'], flush=True)
    for arm, data in result['arms'].items():
        print(arm, data['counts'], data['first60_form'], flush=True)


if __name__ == '__main__':
    main()
