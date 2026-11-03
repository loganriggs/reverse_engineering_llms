"""Strictly join 2016/17 report games to NBA event trajectories.

Requires local NBA data-archive nbastatsv3 and shotdetail season tarballs.
Saves aggregate counts only, not play descriptions or human report prose.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tarfile

from audit_rotowire_document_source import game_key
from sports_document_dgp import game_partition, resources


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path, member):
    with tarfile.open(path) as archive:
        handle = archive.extractfile(member)
        if handle is None:
            raise FileNotFoundError(member)
        yield from csv.DictReader(line.decode('utf-8') for line in handle)


def metadata(path):
    games = {}
    for row in csv_rows(path, 'shotdetail_2016.csv'):
        game = games.setdefault(row['GAME_ID'],
                                {'date': row['GAME_DATE'], 'teams': set()})
        if game['date'] != row['GAME_DATE']:
            raise ValueError('Conflicting game dates')
        game['teams'].add(row['TEAM_NAME'])
    return games


def events(path):
    games = {}
    for row in csv_rows(path, 'nbastatsv3_2016.csv'):
        game = games.setdefault(row['gameId'], {
            'actions': 0, 'last_score': (0, 0), 'period': 0,
            'period_ends': {}, 'negative_score_steps': 0,
            'large_score_steps': 0, 'backward_period_steps': 0,
            'action_types': Counter(), 'player_names': set(),
        })
        game['actions'] += 1
        game['action_types'][row['actionType']] += 1
        if row['playerName']:
            game['player_names'].add(row['playerName'].lower())
        period = int(row['period'])
        if period < game['period']:
            game['backward_period_steps'] += 1
        game['period'] = period
        home, away = row['scoreHome'], row['scoreAway']
        # This archive encodes non-scoring actions as 0/0, even after the
        # game has points. Treat that pair as "no score supplied" and carry
        # the last observed cumulative score forward.
        if home.isdigit() and away.isdigit() and \
                (int(home), int(away)) != (0, 0):
            score = (int(home), int(away))
            delta = (score[0]-game['last_score'][0],
                     score[1]-game['last_score'][1])
            game['negative_score_steps'] += min(delta) < 0
            game['large_score_steps'] += max(delta) > 4
            game['last_score'] = score
        game['period_ends'][period] = game['last_score']
    return games


def report_key(row):
    try:
        score = (int(row['home_line']['TEAM-PTS']),
                 int(row['vis_line']['TEAM-PTS']))
        date = datetime.strptime(row['day'], '%m_%d_%y').strftime('%Y%m%d')
    except (KeyError, TypeError, ValueError):
        return None
    teams = frozenset((row['home_city']+' '+row['home_name'],
                       row['vis_city']+' '+row['vis_name']))
    return date, teams, score


def quarter_score_matches(row, game):
    try:
        expected = [
            (int(row['home_line'][f'TEAM-PTS_QTR{i}']),
             int(row['vis_line'][f'TEAM-PTS_QTR{i}']))
            for i in range(1, 5)]
    except (KeyError, TypeError, ValueError):
        return None
    ends = game['period_ends']
    if any(i not in ends for i in range(1, 5)):
        return None
    previous = (0, 0)
    for i, target in enumerate(expected, 1):
        score = ends[i]
        observed = (score[0]-previous[0], score[1]-previous[1])
        if observed != target:
            return False
        previous = score
    return True


def audit_source(rows, index, games):
    counts = Counter()
    matched_ids = set()
    for row in rows:
        key = report_key(row)
        if key is None or not '20161001' <= key[0] <= '20170531':
            continue
        counts['period_report_rows'] += 1
        game_ids = index.get(key, ())
        counts['strict_join_ambiguous'] += len(game_ids) > 1
        if len(game_ids) != 1:
            continue
        game_id = game_ids[0]
        game = games[game_id]
        matched_ids.add(game_id)
        counts['strict_join_rows'] += 1
        counts['source_partition/'+game_partition(game_key(row))] += 1
        counts['no_negative_score_steps'] += game['negative_score_steps'] == 0
        counts['no_large_score_steps'] += game['large_score_steps'] == 0
        counts['no_backward_period_steps'] += \
            game['backward_period_steps'] == 0
        counts['at_least_400_actions'] += game['actions'] >= 400
        quarter = quarter_score_matches(row, game)
        counts['quarter_fields_and_periods_available'] += quarter is not None
        counts['quarter_scores_agree'] += quarter is True
        surnames = {str(name).split()[-1].lower()
                    for name in row['box_score']['PLAYER_NAME'].values()
                    if name}
        overlap = len(surnames & game['player_names'])
        counts['roster_overlap_ge8'] += overlap >= 8
        counts['roster_overlap_sum'] += overlap
    return {'counts': dict(counts),
            'distinct_joined_games': len(matched_ids),
            'strict_join_row_share': round(
                counts['strict_join_rows']/counts['period_report_rows'], 4)
            if counts['period_report_rows'] else 0,
            'joined_game_ids': matched_ids}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pbp', type=Path, required=True)
    parser.add_argument('--shotdetail', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    meta = metadata(args.shotdetail)
    games = events(args.pbp)
    index = defaultdict(list)
    for game_id, item in meta.items():
        if game_id in games and len(item['teams']) == 2:
            key = (item['date'], frozenset(item['teams']),
                   games[game_id]['last_score'])
            index[key].append(game_id)
    out = {'config': {'pbp_file': args.pbp.name,
                      'pbp_sha256': sha256(args.pbp),
                      'shotdetail_file': args.shotdetail.name,
                      'shotdetail_sha256': sha256(args.shotdetail),
                      'join': 'exact date + unordered full team names + '
                              'ordered home/away final score'},
           'pbp_games': len(games),
           'metadata_games': len(meta),
           'indexed_games': sum(map(len, index.values())),
           'action_type_counts': dict(sum((g['action_types']
                                           for g in games.values()), Counter())),
           'sources': {},
           'caveat': 'This tests event availability and exact report/game '
                     'alignment. It does not certify that human prose is '
                     'entailed by the event feed or that a sampled renderer '
                     'is language-like. Raw reports and play descriptions '
                     'are not saved.'}
    joined = {}
    for source in ('rotowire', 'sbnation'):
        result = audit_source(resources()[0][source][0], index, games)
        joined[source] = result.pop('joined_game_ids')
        out['sources'][source] = result
        print(source, result, flush=True)
    out['joined_game_overlap'] = len(joined['rotowire'] & joined['sbnation'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2)+'\n')


if __name__ == '__main__':
    main()
