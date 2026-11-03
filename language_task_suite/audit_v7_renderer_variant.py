"""Paired lexical and rule audit for optional V7 event-frame variation."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random

from audit_document_mixture_v7_cross import check, edits
from audit_surface_repetition import WORD
from document_mixture_v7_cross import EVENT_KWARGS, draw_cross, realize_cross
from document_world_v5_clause_realizations import (
    answer_from_visible_prefix, draw_varied, realize,
)


HERE = Path(__file__).resolve().parent


def words(tokens):
    return [word.lower() for word in WORD.findall(' '.join(tokens))]


def check_event(doc):
    for site in doc.sites:
        assert doc.tokens[site.index] == site.answer
        assert answer_from_visible_prefix(doc.tokens[:site.index]) == site.answer


def summary(rows):
    counts = Counter(word for row in rows for word in row[:100])
    total = sum(counts.values())
    assert total == len(rows)*100
    return {'documents': len(rows), 'words': total,
            'observed_vocabulary': len(counts),
            'the_share': counts['the']/total,
            'top10_share': sum(v for _, v in counts.most_common(10))/total,
            'hapax_types': sum(v == 1 for v in counts.values()),
            'top20': counts.most_common(20)}


def audit(n, seed, rate):
    rng = random.Random(seed)
    results = {}
    for kind in ('event_report', 'cross_report'):
        baseline, variant = [], []
        changed = 0
        edit_checks = Counter()
        attempts = 0
        while len(baseline) < n and attempts < 10*n:
            attempts += 1
            if kind == 'event_report':
                world = draw_varied(rng, **EVENT_KWARGS).latent
                old = realize(world, **EVENT_KWARGS)
                new = realize(world, **{**EVENT_KWARGS,
                                        'variant_rate': rate})
                check_event(old)
                check_event(new)
                assert [(s.family, s.answer) for s in old.sites] == [
                    (s.family, s.answer) for s in new.sites]
            else:
                world = draw_cross(rng)
                old = realize_cross(world)
                new = realize_cross(world, {**EVENT_KWARGS,
                                            'variant_rate': rate})
                check(old)
                check(new)
                assert [(s.family, s.target) for s in old.sites] == [
                    (s.family, s.target) for s in new.sites]
                if len(baseline) < 40:
                    target = next(s.target for s in new.sites
                                  if s.family == 'cross_source_record')
                    for name, altered, flips in edits(world):
                        other = realize_cross(altered,
                            {**EVENT_KWARGS, 'variant_rate': rate})
                        check(other)
                        answer = next(s.target for s in other.sites
                                      if s.family == 'cross_source_record')
                        assert (answer != target) == flips, name
                        edit_checks[name] += 1
            before, after = words(old.tokens), words(new.tokens)
            if len(before) < 100 or len(after) < 100:
                continue
            baseline.append(before)
            variant.append(after)
            changed += old.tokens != new.tokens
        assert len(baseline) == n
        results[kind] = {'attempts': attempts, 'changed_documents': changed,
                         'paired_edit_checks': dict(edit_checks),
                         'baseline': summary(baseline),
                         'variant': summary(variant)}
    return {'config': {'n_per_long_kind': n, 'seed': seed,
                       'variant_rate': rate,
                       'window': 'first 100 lowercase alphabetic words; '
                                 'same sampled worlds in both arms'},
            'results': results,
            'caveat': ('Controlled form intervention with exact visible-answer '
                       'checks. It does not demonstrate natural-language '
                       'realism, a full next-token oracle, or model learning.')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=480)
    parser.add_argument('--seed', type=int, default=34901)
    parser.add_argument('--rate', type=float, default=1.)
    parser.add_argument('--output', type=Path,
                        default=HERE/'results/v7_renderer_variant_n480.json')
    args = parser.parse_args()
    result = audit(args.n, args.seed, args.rate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    for kind, row in result['results'].items():
        print(kind, row['changed_documents'],
              'the', row['baseline']['the_share'],
              row['variant']['the_share'],
              'top10', row['baseline']['top10_share'],
              row['variant']['top10_share'], flush=True)
    print('Saved', args.output)


if __name__ == '__main__':
    main()
