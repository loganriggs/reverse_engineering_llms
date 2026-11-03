"""Event-grounded latent game worlds for a sampled sports-language DGP.

Worlds sample empirical 2016/17 games, not human report sentences. The
source-game partition prevents the same game leaking across train/heldout.
This module supplies facts; it does not claim a realistic text renderer.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import random

from audit_rotowire_document_source import game_key
from audit_sports_event_predicates import (event_facts, scoring_events,
                                           team_codes)
from audit_sports_pbp_join import metadata, report_key
from sports_document_dgp import game_partition, resources


ARCHIVE_DIR = Path('/home/loganriggs/Coding/neuron_circuits/artifacts/nba_pbp')


@dataclass(frozen=True)
class ScoringInterval:
    period: int
    side: str
    points_for: int
    points_against: int
    start_score: tuple[int, int]
    end_score: tuple[int, int]
    start_action: int
    end_action: int


@dataclass(frozen=True)
class EventWorld:
    source: str
    partition: str
    game_id: str
    home: str
    away: str
    final_score: tuple[int, int]
    period_end_scores: tuple[tuple[int, int], ...]
    top_player: str
    top_player_points: int
    interval: ScoringInterval


def candidate_intervals(rows):
    """Within-period two-sided runs with a substantial score swing."""
    by_period = defaultdict(list)
    previous = (0, 0)
    for period, action, score, _, _ in rows:
        if period not in by_period:
            by_period[period].append((action-1, previous))
        if score != previous:
            by_period[period].append((action, score))
        previous = score
    out = []
    for period, steps in by_period.items():
        for i, (start_action, start) in enumerate(steps):
            for end_action, end in steps[i+1:min(i+21, len(steps))]:
                delta = (end[0]-start[0], end[1]-start[1])
                if min(delta) < 0:
                    continue
                side = 0 if delta[0] > delta[1] else 1
                points_for, against = delta[side], delta[1-side]
                if 8 <= points_for <= 25 and 1 <= against <= 12 and \
                        points_for-against >= 6:
                    out.append(ScoringInterval(period, 'home' if side == 0
                                               else 'away', points_for,
                                               against, start, end,
                                               start_action, end_action))
    return tuple(out)


def top_scorer(row):
    box = row['box_score']
    players = []
    for player_id, name in box['PLAYER_NAME'].items():
        points = box['PTS'].get(player_id, '')
        if name and str(points).isdigit():
            players.append((int(points), name))
    return max(players, default=None)


@lru_cache(maxsize=1)
def world_pool():
    pbp = ARCHIVE_DIR/'nbastatsv3_2016.tar.xz'
    shotdetail = ARCHIVE_DIR/'shotdetail_2016.tar.xz'
    meta = metadata(shotdetail)
    codes = team_codes(shotdetail)
    trajectories = scoring_events(pbp, codes)
    facts = {game_id: event_facts(rows, codes[game_id])
             for game_id, rows in trajectories.items()}
    index = defaultdict(list)
    for game_id, item in meta.items():
        if game_id in facts and len(item['teams']) == 2 and \
                facts[game_id]['bad_steps'] == 0:
            index[(item['date'], frozenset(item['teams']),
                   facts[game_id]['score'])].append(game_id)
    intervals = {game_id: candidate_intervals(rows)
                 for game_id, rows in trajectories.items()
                 if facts[game_id]['bad_steps'] == 0}
    pools = {(source, group): [] for source in ('rotowire', 'sbnation')
             for group in ('train', 'heldout')}
    seen = set()
    for source in ('rotowire', 'sbnation'):
        for row in resources()[0][source][0]:
            key = report_key(row)
            if key is None:
                continue
            ids = index.get(key, ())
            if len(ids) != 1:
                continue
            game_id = ids[0]
            group = game_partition(game_key(row))
            if (source, game_id) in seen or not intervals[game_id]:
                continue
            seen.add((source, game_id))
            scorer = top_scorer(row)
            if scorer is None:
                continue
            home = row['home_city']+' '+row['home_name']
            away = row['vis_city']+' '+row['vis_name']
            if row['home_city'] == row['vis_city']:
                continue
            period_end_scores = tuple(score for _, score in
                                      sorted(facts[game_id]['quarters'].items()))
            pools[source, group].append((game_id, home, away,
                                         facts[game_id]['score'],
                                         period_end_scores, scorer,
                                         intervals[game_id]))
    return {key: tuple(value) for key, value in pools.items()}


def sample_world(rng: random.Random, source='rotowire',
                 partition='train'):
    if (source, partition) not in world_pool():
        raise ValueError('Unknown source/partition')
    pool = world_pool()[source, partition]
    if not pool:
        raise RuntimeError('No joined event worlds available')
    game_id, home, away, final, periods, scorer, intervals = rng.choice(pool)
    # Favor stronger swings while retaining stochastic selection. This is a
    # DGP prior, not a claim about which interval the human report mentioned.
    ranked = sorted(intervals, key=lambda run: (
        run.points_for-run.points_against, run.points_for), reverse=True)
    run = rng.choice(ranked[:min(20, len(ranked))])
    return EventWorld(source, partition, game_id, home, away, final,
                      periods, scorer[1], scorer[0], run)
