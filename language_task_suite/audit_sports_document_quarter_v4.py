"""Semantic, intervention, and paired-form audit of the quarter-rule arm."""

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import random
import re

from audit_document_structure_vs_fineweb import structure
from compare_v12_fineweb_fresh import first_words
from sports_document_dgp import realize as base_realize, game_partition
from sports_document_quarter_v4 import (
    QUARTER, WEEKDAYS, answer_from_visible_prefix, edit, realize,
    sample_plan,
)


KINDS = {'winner', 'margin', 'top_player', 'halftime_leader', 'comeback'}


def answers(doc):
    result = {}
    previous = 0
    for site in doc.sites:
        assert previous <= site.start < site.end <= len(doc.text)
        assert doc.text[site.start:site.end] == site.answer
        assert site.answer in site.candidates
        found = answer_from_visible_prefix(doc.text[:site.start], site.kind)
        assert found == site.answer, (site.kind, found, site.answer)
        result[site.kind] = found
        previous = site.end
    assert set(result) == KINDS
    return result


def check_quarters(doc):
    rows = QUARTER.findall(doc.text)
    assert len(rows) == 4 and [int(row[0]) for row in rows] == [1, 2, 3, 4]
    assert all(row[1] == doc.plan.base.home and
               row[3] == doc.plan.base.away for row in rows)
    home, away = sum(int(row[2]) for row in rows), \
        sum(int(row[4]) for row in rows)
    expected = (doc.plan.base.high_score, doc.plan.base.low_score) \
        if doc.plan.base.home_win else \
        (doc.plan.base.low_score, doc.plan.base.high_score)
    assert (home, away) == expected
    expected_day = datetime.strptime(
        doc.plan.base.source_game_key[0], '%m_%d_%y').strftime('%A')
    stated_days = WEEKDAYS.intersection(re.findall(r'\b[A-Za-z]+\b',
                                                 doc.text))
    assert not stated_days or stated_days == {expected_day}
    assert re.search(r'\b1 (?:assists|rebounds)\b', doc.text) is None


def _render(plan):
    return realize(plan, form_entropy=1., paragraph_rate=.12,
                   pair_rate=1., discourse_entropy=1.,
                   lead_stat_rate=.25, stat_richness=1.)


def audit(seed, n, form_n, source, group, words):
    rng = random.Random(seed)
    counts = Counter()
    samples = []
    paired_base, paired_quarter = [], []
    windows = {kind: [] for kind in KINDS}
    for index in range(n):
        plan = sample_plan(rng, source_group=group,
                           sbnation_rate=int(source == 'sbnation'))
        assert game_partition(plan.base.source_game_key) == group
        assert game_partition(plan.donor_game_key) == group
        doc = _render(plan)
        original = answers(doc)
        check_quarters(doc)
        counts['plans'] += 1
        counts['comeback/'+original['comeback']] += 1
        if len(samples) < 2:
            samples.append({'text': doc.text, 'answers': original})
        variants = {
            'swap_teams': edit(plan, 'swap_teams', True),
            'high_score': edit(plan, 'high_score',
                               plan.base.high_score+1),
            'flip_halftime': edit(plan, 'flip_halftime'),
        }
        alt = next(j for j, points in enumerate(plan.base.points)
                   if points != max(plan.base.points))
        points = list(plan.base.points)
        points[alt] = max(points)+1
        variants['points'] = edit(plan, 'points', points)
        for field, changed in variants.items():
            transformed = _render(changed)
            after = answers(transformed)
            check_quarters(transformed)
            if field == 'swap_teams':
                assert after['winner'] != original['winner']
                assert after['halftime_leader'] != original['halftime_leader']
                assert all(after[k] == original[k]
                           for k in ('margin', 'top_player', 'comeback'))
            elif field == 'high_score':
                assert int(after['margin']) == int(original['margin'])+1
                assert all(after[k] == original[k] for k in
                           ('winner', 'top_player', 'halftime_leader',
                            'comeback'))
            elif field == 'flip_halftime':
                assert after['halftime_leader'] != original['halftime_leader']
                assert after['comeback'] != original['comeback']
                assert all(after[k] == original[k] for k in
                           ('winner', 'margin', 'top_player'))
            else:
                assert after['top_player'] == plan.base.players[alt]
                assert all(after[k] == original[k] for k in
                           ('winner', 'margin', 'halftime_leader',
                            'comeback'))
            counts['edit/'+field] += 1
        if len(paired_quarter) < form_n:
            base = base_realize(plan.base, form_entropy=1.,
                                paragraph_rate=.12, pair_rate=1.,
                                discourse_entropy=1.,
                                lead_stat_rate=.25, stat_richness=1.)
            base_crop, quarter_crop = (first_words(text, words)
                                       for text in (base.text, doc.text))
            if len(base_crop.split()) >= words and \
                    len(quarter_crop.split()) >= words:
                paired_base.append(base_crop)
                paired_quarter.append(quarter_crop)
                for site in doc.sites:
                    windows[site.kind].append(
                        ' '.join(doc.text[:site.start].split()[-25:]+
                                 doc.text[site.start:].split()[:16]))
    if len(paired_quarter) != form_n:
        raise RuntimeError('Insufficient form panel')
    return {'counts': dict(counts), 'examples': samples,
            'form': {'base_matched': structure(paired_base),
                     'quarter_matched': structure(paired_quarter),
                     'quarter_rule_windows': {
                         kind: structure(texts)
                         for kind, texts in windows.items()}}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=94101)
    p.add_argument('--plans', type=int, default=100)
    p.add_argument('--form-n', type=int, default=64)
    p.add_argument('--words', type=int, default=55)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = {'config': vars(a) | {'output': str(a.output)}, 'arms': {},
              'caveat': 'Quarter V4 is a semantic fixture and form test, not '
                        'an accepted realistic training distribution. '
                        'Source score templates retain their guard limits; '
                        'quarter table and answer clauses are synthetic. '
                        'Form comparison is paired by sampled world.'}
    for source in ('rotowire', 'sbnation'):
        for group in ('train', 'heldout'):
            key = source+'_'+group
            result['arms'][key] = audit(
                a.seed+len(result['arms'])*100, a.plans,
                a.form_n, source, group, a.words)
            row = result['arms'][key]
            print(key, row['counts'],
                  'trigram base/quarter',
                  row['form']['base_matched']
                     ['repeated_across_docs_trigram_occurrence_share'],
                  row['form']['quarter_matched']
                     ['repeated_across_docs_trigram_occurrence_share'],
                  flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
