"""Matched semantic and form screen for distributed quarter-event prose."""

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import random
import re

from audit_document_structure_vs_fineweb import structure
from compare_v12_fineweb_fresh import first_words
from sports_document_dgp import game_partition
from sports_document_quarter_v4 import WEEKDAYS, realize as v4_realize
from sports_document_quarter_v5 import (
    answer_from_visible_prefix, edit, realize, sample_plan,
    visible_quarters,
)


KINDS = {'winner', 'margin', 'top_player', 'halftime_leader',
         'winner_halftime_state'}


def answers(doc):
    result = {}
    previous = 0
    for site in doc.sites:
        assert previous <= site.start < site.end <= len(doc.text)
        assert doc.text[site.start:site.end] == site.answer
        assert site.answer in site.candidates
        found = answer_from_visible_prefix(doc.text[:site.start],
                                           site.kind)
        assert found == site.answer, (site.kind, found, site.answer)
        result[site.kind] = found
        previous = site.end
    assert set(result) == KINDS
    return result


def check_facts(doc):
    rows = visible_quarters(doc.text)
    assert len(rows) == 4
    assert all(row[1] == doc.plan.base.home and
               row[3] == doc.plan.base.away for row in rows)
    totals = (sum(row[2] for row in rows), sum(row[4] for row in rows))
    expected = (doc.plan.base.high_score, doc.plan.base.low_score) \
        if doc.plan.base.home_win else \
        (doc.plan.base.low_score, doc.plan.base.high_score)
    assert totals == expected
    expected_day = datetime.strptime(
        doc.plan.base.source_game_key[0], '%m_%d_%y').strftime('%A')
    days = WEEKDAYS.intersection(re.findall(r'\b[A-Za-z]+\b', doc.text))
    assert not days or days == {expected_day}
    assert re.search(r'\b1 (?:assists|rebounds)\b', doc.text) is None


def render(plan, spread):
    return realize(plan, act_spread=spread, form_entropy=1.,
                   paragraph_rate=.12, pair_rate=1.,
                   discourse_entropy=1., lead_stat_rate=.25,
                   stat_richness=1.)


def window(doc, site):
    return ' '.join(doc.text[:site.start].split()[-25:]+
                    doc.text[site.start:].split()[:16])


def audit(seed, n, form_n, words, source, group, spread):
    rng = random.Random(seed)
    counts = Counter()
    examples = []
    matched_v4, matched_v5 = [], []
    windows = {kind: [] for kind in KINDS}
    for _ in range(n):
        plan = sample_plan(rng, source_group=group,
                           sbnation_rate=int(source == 'sbnation'))
        assert game_partition(plan.base.source_game_key) == group
        assert game_partition(plan.donor_game_key) == group
        doc = render(plan, spread)
        original = answers(doc)
        check_facts(doc)
        if spread == 1.:
            fixed = render(plan, 0.)
            sentence_bag = lambda text: Counter(
                piece.strip() for piece in
                re.findall(r'[^.!?]+[.!?]', text))
            assert sentence_bag(doc.text) == sentence_bag(fixed.text)
            counts['placement_only_sentence_multiset'] += 1
            counts['placement_changed_text'] += doc.text != fixed.text
        counts['plans'] += 1
        counts['state/'+original['winner_halftime_state']] += 1
        if len(examples) < 2:
            examples.append({'text': doc.text, 'answers': original})
        alt = next(j for j, value in enumerate(plan.base.points)
                   if value != max(plan.base.points))
        points = list(plan.base.points)
        points[alt] = max(points)+1
        variants = {
            'swap_teams': edit(plan, 'swap_teams', True),
            'high_score': edit(plan, 'high_score',
                               plan.base.high_score+1),
            'flip_halftime': edit(plan, 'flip_halftime'),
            'points': edit(plan, 'points', points),
        }
        for field, changed in variants.items():
            revised = render(changed, spread)
            after = answers(revised)
            check_facts(revised)
            if field == 'swap_teams':
                assert after['winner'] != original['winner']
                assert after['halftime_leader'] != original['halftime_leader']
                assert all(after[k] == original[k] for k in
                           ('margin', 'top_player', 'winner_halftime_state'))
            elif field == 'high_score':
                assert int(after['margin']) == int(original['margin'])+1
                assert all(after[k] == original[k] for k in
                           ('winner', 'top_player', 'halftime_leader',
                            'winner_halftime_state'))
            elif field == 'flip_halftime':
                assert after['halftime_leader'] != original['halftime_leader']
                assert after['winner_halftime_state'] != \
                    original['winner_halftime_state']
                assert all(after[k] == original[k] for k in
                           ('winner', 'margin', 'top_player'))
            else:
                assert after['top_player'] == plan.base.players[alt]
                assert all(after[k] == original[k] for k in
                           ('winner', 'margin', 'halftime_leader',
                            'winner_halftime_state'))
            counts['edit/'+field] += 1
        if len(matched_v5) < form_n:
            old = v4_realize(plan, form_entropy=1.,
                             paragraph_rate=.12, pair_rate=1.,
                             discourse_entropy=1., lead_stat_rate=.25,
                             stat_richness=1.)
            old_crop, new_crop = (first_words(text, words)
                                  for text in (old.text, doc.text))
            if len(old_crop.split()) >= words and \
                    len(new_crop.split()) >= words:
                matched_v4.append(old_crop)
                matched_v5.append(new_crop)
                for site in doc.sites:
                    windows[site.kind].append(window(doc, site))
    if len(matched_v5) != form_n:
        raise RuntimeError('Insufficient matched form panel')
    return {'counts': dict(counts), 'examples': examples,
            'form': {'v4_matched': structure(matched_v4),
                     'v5_matched': structure(matched_v5),
                     'v5_windows': {kind: structure(rows)
                                    for kind, rows in windows.items()}}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=95201)
    p.add_argument('--plans', type=int, default=100)
    p.add_argument('--form-n', type=int, default=64)
    p.add_argument('--words', type=int, default=55)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = {'config': {**vars(a), 'output': str(a.output)},
              'arms': {}, 'caveat': 'Lexical form only, matched within each '
                                   'arm; target-site windows matter more than '
                                   'a fixed first-word crop. The V5 quarter '
                                   'prose remains a typed grammar, not a '
                                   'natural-report distribution.'}
    for source_index, source in enumerate(('rotowire', 'sbnation')):
        for group_index, group in enumerate(('train', 'heldout')):
            for spread in (0., 1.):
                key = f'{source}_{group}_spread{spread:g}'
                # Same sampled worlds across placement arms. Rendering
                # changes only the act-placement intervention.
                row = audit(a.seed+source_index*1000+group_index*100,
                            a.plans, a.form_n, a.words,
                            source, group, spread)
                result['arms'][key] = row
                form = row['form']
                print(key, row['counts'],
                      'trigrams',
                      round(form['v4_matched']
                            ['repeated_across_docs_trigram_occurrence_share'], 3),
                      round(form['v5_matched']
                            ['repeated_across_docs_trigram_occurrence_share'], 3),
                      'state-window',
                      round(form['v5_windows']['winner_halftime_state']
                            ['repeated_across_docs_trigram_occurrence_share'], 3),
                      flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
