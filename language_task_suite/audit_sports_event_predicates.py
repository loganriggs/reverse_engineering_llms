"""Audit game-flow predicates available for an event-grounded sports DGP.

Uses local NBA season archives; emits only aggregate statistics. A source
report's game is matched by date, full team names and final home/away score.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from audit_rotowire_document_source import game_key
from audit_sports_pbp_join import csv_rows, metadata, report_key, sha256
from sports_document_dgp import game_partition, resources


def team_codes(path):
    codes = {}
    for row in csv_rows(path, 'shotdetail_2016.csv'):
        game_id = row['GAME_ID']
        pair = (row['HTM'], row['VTM'])
        if game_id in codes and codes[game_id] != pair:
            raise ValueError(f'Conflicting team codes: {game_id}')
        codes[game_id] = pair
    return codes


def scoring_events(path, codes):
    games = defaultdict(list)
    for row in csv_rows(path, 'nbastatsv3_2016.csv'):
        game_id = row['gameId']
        if game_id not in codes:
            continue
        try:
            score = (int(row['scoreHome']), int(row['scoreAway']))
        except ValueError:
            continue
        if score == (0, 0):
            continue
        games[game_id].append((int(row['period']), int(row['actionNumber']),
                               score, row['teamTricode'],
                               row['actionType'].strip()))
    return games


def event_facts(rows, codes):
    """Extract facts, rejecting trajectories with inconsistent score steps.

    A lead change is a reversal of the non-tied leader; ties preserve the
    previous leader. An unanswered run is consecutive points by one team
    within a period, terminated by an opponent score or period boundary.
    """
    previous = (0, 0)
    period = 0
    previous_leader = 0
    lead_changes = 0
    fourth_period_lead_changes = 0
    leader_ties = 0
    scoring_steps = 0
    attribution_errors = 0
    bad_steps = 0
    runs = []
    run_side = None
    run_points = 0
    run_period = None
    quarters = {}
    for p, _, score, team, action_type in rows:
        if p < period:
            bad_steps += 1
        if p != period and run_side is not None:
            runs.append((run_period, run_side, run_points))
            run_side, run_points = None, 0
        period = p
        delta = (score[0]-previous[0], score[1]-previous[1])
        if not any(delta):
            quarters[p] = score
            continue
        if min(delta) < 0 or max(delta) > 4 or all(delta):
            bad_steps += 1
            previous = score
            quarters[p] = score
            continue
        side = 0 if delta[0] else 1
        points = delta[side]
        if team != codes[side]:
            attribution_errors += 1
        if action_type not in ('Made Shot', 'Free Throw'):
            attribution_errors += 1
        scoring_steps += 1
        if side == run_side:
            run_points += points
        else:
            if run_side is not None:
                runs.append((run_period, run_side, run_points))
            run_side, run_points, run_period = side, points, p
        leader = (score[0] > score[1]) - (score[0] < score[1])
        if leader == 0:
            leader_ties += 1
        elif previous_leader and leader != previous_leader:
            lead_changes += 1
            fourth_period_lead_changes += p >= 4
            previous_leader = leader
        elif not previous_leader:
            previous_leader = leader
        previous = score
        quarters[p] = score
    if run_side is not None:
        runs.append((run_period, run_side, run_points))
    return {'score': previous, 'quarters': quarters,
            'scoring_steps': scoring_steps, 'bad_steps': bad_steps,
            'attribution_errors': attribution_errors,
            'lead_changes': lead_changes,
            'fourth_period_lead_changes': fourth_period_lead_changes,
            'ties_after_scoring': leader_ties,
            'max_unanswered_points': max((r[2] for r in runs), default=0),
            'runs_ge8': sum(r[2] >= 8 for r in runs),
            'runs_ge10': sum(r[2] >= 10 for r in runs),
            'fourth_period_runs_ge8': sum(r[0] >= 4 and r[2] >= 8
                                           for r in runs)}


def audit_source(rows, index, facts):
    matched = {'train': set(), 'heldout': set()}
    for row in rows:
        key = report_key(row)
        if key is None or not '20161001' <= key[0] <= '20170531':
            continue
        ids = index.get(key, ())
        if len(ids) == 1:
            matched[game_partition(game_key(row))].add(ids[0])
    out = {}
    for partition, ids in matched.items():
        counts = Counter()
        for game_id in ids:
            fact = facts[game_id]
            counts['matched_distinct_games'] += 1
            counts['usable_score_trajectory'] += fact['bad_steps'] == 0
            counts['clean_scoring_attribution'] += fact['attribution_errors'] == 0
            counts['has_lead_change'] += fact['lead_changes'] > 0
            counts['has_4q_or_ot_lead_change'] += fact['fourth_period_lead_changes'] > 0
            counts['has_unanswered_run_ge8'] += fact['runs_ge8'] > 0
            counts['has_unanswered_run_ge10'] += fact['runs_ge10'] > 0
            counts['has_4q_or_ot_unanswered_run_ge8'] += \
                fact['fourth_period_runs_ge8'] > 0
            counts['max_unanswered_points_sum'] += \
                fact['max_unanswered_points']
            counts['lead_changes_sum'] += fact['lead_changes']
            counts['scoring_steps_sum'] += fact['scoring_steps']
            counts['bad_steps_sum'] += fact['bad_steps']
            counts['attribution_errors_sum'] += fact['attribution_errors']
        out[partition] = dict(counts)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pbp', type=Path, required=True)
    parser.add_argument('--shotdetail', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    meta = metadata(args.shotdetail)
    codes = team_codes(args.shotdetail)
    trajectories = scoring_events(args.pbp, codes)
    facts = {game_id: event_facts(rows, codes[game_id])
             for game_id, rows in trajectories.items()}
    index = defaultdict(list)
    for game_id, item in meta.items():
        if game_id in facts and len(item['teams']) == 2:
            key = (item['date'], frozenset(item['teams']),
                   facts[game_id]['score'])
            index[key].append(game_id)
    out = {
        'config': {'pbp_file': args.pbp.name,
                   'pbp_sha256': sha256(args.pbp),
                   'shotdetail_file': args.shotdetail.name,
                   'shotdetail_sha256': sha256(args.shotdetail),
                   'scope': 'distinct matched games by whole-game partition',
                   'run_definition': 'consecutive scoring by one side within '
                                     'one period, terminated by opponent '
                                     'score or period boundary',
                   'lead_change_definition': 'reversal of last non-tied '
                                             'leader after a scoring action'},
        'event_games': len(facts),
        'event_game_quality': {
            'bad_score_step_games': sum(f['bad_steps'] > 0 for f in facts.values()),
            'attribution_error_games': sum(f['attribution_errors'] > 0
                                           for f in facts.values())},
        'sources': {name: audit_source(resources()[0][name][0], index, facts)
                    for name in ('rotowire', 'sbnation')},
        'caveat': 'Rule prevalence and event availability only. Event claims '
                  'must still be checked against visible text and held-out '
                  'human form; this does not certify prose entailment.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps({'quality': out['event_game_quality'],
                      'sources': out['sources']}, indent=2))


if __name__ == '__main__':
    main()
