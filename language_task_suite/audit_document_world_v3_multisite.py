"""Audit deterministic answer sites and basic form of the multisite pilot."""

import argparse
import collections
import json
import random
from pathlib import Path

from document_world import VALUES
from document_world_v3_multisite import (GENRES, answer_from_visible_prefix,
                                         draw_document, realize)


def audit(n, seed):
    rng = random.Random(seed)
    counts = collections.Counter()
    words = collections.Counter()
    first100_words = collections.Counter()
    lengths = []
    for i in range(n):
        world = draw_document(rng, GENRES[i % len(GENRES)])
        doc = realize(world)
        lengths.append(len(doc.tokens))
        words.update(doc.tokens)
        if i < 49:
            first100_words.update([token.lower() for token in doc.tokens
                                   if token.isalpha()][:100])
        assert len(doc.sites) == 5
        for site in doc.sites:
            assert doc.tokens[site.index] == site.answer
            assert site.answer in site.candidates
            assert answer_from_visible_prefix(doc.tokens[:site.index]) == site.answer
            counts[site.family] += 1
        # Re-rendering the same semantic world in every genre preserves answers.
        for genre in GENRES:
            variant = {**world, 'genre': genre}
            assert [s.answer for s in realize(variant).sites] == [
                s.answer for s in doc.sites]
        # An edit to the last relevant event changes the final answer. The
        # source answer remains fixed because no source identity was edited.
        edited = {**world, 'events': [dict(e) for e in world['events']]}
        obj, attr = world['query_object'], world['query_attribute']
        index = [j for j, event in enumerate(edited['events'])
                 if event['object'] == obj and event['attribute'] == attr][-1]
        old = edited['events'][index]['value']
        edited['events'][index]['value'] = next(v for v in VALUES[attr] if v != old)
        changed = realize(edited)
        original = {s.family: s.answer for s in doc.sites}
        new = {s.family: s.answer for s in changed.sites}
        assert new[f'world_{attr}'] != original[f'world_{attr}']
        assert new['alias_identity'] == original['alias_identity']
        assert new['source_attribution'] == original['source_attribution']
        for site in changed.sites:
            assert answer_from_visible_prefix(changed.tokens[:site.index]) == site.answer
    return dict(version='document_world_v3_multisite_pilot', documents=n, seed=seed,
                genres=GENRES, sites=dict(counts), min_tokens=min(lengths),
                median_tokens=sorted(lengths)[len(lengths)//2],
                max_tokens=max(lengths), vocabulary=len(words),
                top10_share=sum(v for _, v in words.most_common(10))/sum(words.values()),
                first49_x100_alpha_types=len(first100_words),
                first49_x100_alpha_singletons=sum(v == 1 for v in first100_words.values()),
                first49_x100_alpha_top10_share=(sum(v for _, v in first100_words.most_common(10))
                                               / sum(first100_words.values())),
                checks=['visible-prefix answer replay', 'candidate membership',
                        'genre-invariant semantic answers', 'last-event causal edit'],
                limitations=['fixed five-query ending', 'tiny lexicon',
                             'no exact full next-token conditional distribution',
                             'no training or natural-form validation'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=23012)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = audit(args.n, args.seed)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
