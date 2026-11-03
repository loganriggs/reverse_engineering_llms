"""Semantic/edit and paired form screens for the variable sports act plan."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import random

from audit_document_structure_vs_fineweb import structure
from audit_rotowire_document_source import game_key
from compare_v12_fineweb_fresh import first_words
from sports_document_actplan_v1 import (
    answer_from_visible_prefix, edit, realize, record_facts_from_visible,
    sample_plan,
)
from sports_document_dgp import (
    game_partition, realize as base_realize, resources,
)


KINDS = ('winner', 'margin', 'top_player')


def check(doc):
    seen = {}
    for site in doc.sites:
        assert site.kind not in seen
        assert doc.text[site.start:site.end] == site.answer
        assert answer_from_visible_prefix(doc.text[:site.start],
                                          site.kind) == site.answer
        seen[site.kind] = site.answer
    assert seen and set(seen) <= set(KINDS)
    records = record_facts_from_visible(doc.text)
    expected_before = {doc.plan.base.home: doc.plan.home_before,
                       doc.plan.base.away: doc.plan.away_before}
    for team, value in records['before'].items():
        assert value == expected_before[team]
    for team, value in records['after'].items():
        assert team in records['before']
        assert value == doc.plan.after[team]
        before = records['before'][team]
        delta = (value[0]-before[0], value[1]-before[1])
        assert delta in ((1, 0), (0, 1))
        assert (delta == (1, 0)) == (team == doc.plan.base.winner)
    return seen, records


def semantic(seed, n, source, group):
    rng = random.Random(seed)
    counts = Counter()
    examples = []
    for _ in range(n):
        plan = sample_plan(rng, sbnation_rate=int(source == 'sbnation'),
                           source_group=group)
        assert game_partition(plan.base.source_game_key) == group
        doc = realize(plan)
        original, records = check(doc)
        counts['documents'] += 1
        counts['visible_pregame_records'] += len(records['before'])
        counts['visible_postgame_records'] += len(records['after'])
        first_sentence = doc.text.split('.')[0]+'.'
        counts['score_leads'] += (plan.base.home in first_sentence and
                                 plan.base.away in first_sentence)
        for kind in original:
            counts['site/'+kind] += 1
        counts['site_set/'+','.join(sorted(original))] += 1
        if len(examples) < 2:
            examples.append(doc.text)
        for field, value in (
                ('swap_teams', True),
                ('high_score', plan.base.high_score+1)):
            changed, new_records = check(realize(edit(plan, field, value)))
            assert set(changed) == set(original)
            if field == 'swap_teams':
                if 'winner' in original:
                    assert changed['winner'] != original['winner']
                for kind in ('margin', 'top_player'):
                    if kind in original:
                        assert changed[kind] == original[kind]
            else:
                if 'margin' in original:
                    assert int(changed['margin']) == int(original['margin'])+1
                for kind in ('winner', 'top_player'):
                    if kind in original:
                        assert changed[kind] == original[kind]
            counts['edit/'+field] += 1
        if 'top_player' in original:
            points = list(plan.base.points)
            i = next(i for i, p in enumerate(points)
                     if p != max(points))
            points[i] = max(points)+1
            changed, _ = check(realize(edit(plan, 'points', points)))
            assert changed['top_player'] == plan.base.players[i]
            for kind in ('winner', 'margin'):
                if kind in original:
                    assert changed[kind] == original[kind]
            counts['edit/points'] += 1
        if records['before']:
            w, l = plan.home_before
            changed_plan = replace(plan, home_before=(w+1, l))
            changed, changed_records = check(realize(changed_plan))
            assert changed == original
            if plan.base.home in records['before']:
                assert changed_records['before'][plan.base.home] == (w+1, l)
                if plan.base.home in records['after']:
                    assert changed_records['after'][plan.base.home][0] == \
                        records['after'][plan.base.home][0]+1
            counts['edit/record_prior'] += 1
    return {'counts': dict(counts), 'generated_examples': examples}


def form(seed, n, words, source, group):
    rng = random.Random(seed)
    generated, fixed = [], []
    windows = {kind: {'fixed': [], 'variable': []} for kind in KINDS}
    attempts = 0
    while len(generated) < n:
        attempts += 1
        if attempts > 100*n:
            raise RuntimeError('Insufficient matched word crops')
        plan = sample_plan(rng, sbnation_rate=int(source == 'sbnation'),
                           source_group=group)
        candidate = realize(plan)
        baseline = base_realize(plan.base, pair_rate=0.,
                                stat_richness=1., discourse_entropy=1.,
                                lead_stat_rate=.25,
                                empirical_stat_rate=1.)
        c, b = first_words(candidate.text, words), \
            first_words(baseline.text, words)
        if min(len(c.split()), len(b.split())) >= words:
            generated.append(c)
            fixed.append(b)
            baseline_sites = {site.kind: site for site in baseline.sites}
            for site in candidate.sites:
                before = candidate.text[:site.start].split()[-25:]
                after = candidate.text[site.start:].split()[:16]
                windows[site.kind]['variable'].append(' '.join(before+after))
                old = baseline_sites[site.kind]
                before = baseline.text[:old.start].split()[-25:]
                after = baseline.text[old.start:].split()[:16]
                windows[site.kind]['fixed'].append(' '.join(before+after))
    natural_by_game = {}
    for row in resources()[0][source][0]:
        if game_partition(game_key(row)) != group:
            continue
        crop = first_words(' '.join(row['summary']), words)
        if len(crop.split()) >= words:
            natural_by_game.setdefault(game_key(row), crop)
    human = random.Random(seed+50000).sample(
        list(natural_by_game.values()), n)
    return {'matched_crops': n, 'draw_attempts': attempts,
            'fixed_empirical': structure(fixed),
            'variable_actplan': structure(generated),
            'human_reference': structure(human),
            'rule_windows': {kind: {arm: structure(texts)
                                   for arm, texts in groups.items()}
                             for kind, groups in windows.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=38101)
    parser.add_argument('--semantic-n', type=int, default=100)
    parser.add_argument('--form-n', type=int, default=64)
    parser.add_argument('--words', type=int, default=55)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = {'config': vars(args) | {'output': str(args.output)},
           'arms': {},
           'caveat': 'Synthetic pregame records are drawn from empirical W-L '
                     'values; this is not a certified realistic corpus or '
                     'full-token next-word oracle. Form uses jointly eligible '
                     'generated crops and unpaired human games.'}
    for source in ('rotowire', 'sbnation'):
        for group in ('train', 'heldout'):
            key = source+'_'+group
            seed = args.seed+100*len(out['arms'])
            out['arms'][key] = {
                'semantic': semantic(seed, args.semantic_n, source, group),
                'form': form(seed+10000, args.form_n, args.words,
                             source, group),
            }
            p = out['arms'][key]['form']
            print(key, p['draw_attempts'],
                  p['fixed_empirical']['repeated_across_docs_trigram_occurrence_share'],
                  p['variable_actplan']['repeated_across_docs_trigram_occurrence_share'],
                  p['human_reference']['repeated_across_docs_trigram_occurrence_share'],
                  flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2)+'\n')


if __name__ == '__main__':
    main()
