"""Sample short, real NBA scoring stretches as visible live-score ledgers.

Play descriptions come from the local 2016/17 NBA event archive. Score
arithmetic and answer sites are exact; descriptions are empirical carriers
without a symbolic all-token oracle. Whole games stay in one split.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import random
import re

from audit_sports_event_predicates import team_codes
from audit_sports_pbp_join import csv_rows
from document_mixture_v5 import Document, Site
from sports_event_world_dgp import ARCHIVE_DIR, world_pool


TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]")


@dataclass(frozen=True)
class Action:
    period: int
    action: int
    clock: str
    team: str
    points: int
    before: tuple[int, int]
    after: tuple[int, int]
    description: str


@lru_cache(maxsize=1)
def _codes():
    return team_codes(ARCHIVE_DIR/'shotdetail_2016.tar.xz')


@lru_cache(maxsize=1)
def scoring_actions():
    path = ARCHIVE_DIR/'nbastatsv3_2016.tar.xz'
    codes = _codes()
    previous = defaultdict(lambda: (0, 0))
    games = defaultdict(list)
    bad = set()
    for row in csv_rows(path, 'nbastatsv3_2016.csv'):
        game_id = row['gameId']
        if game_id not in codes:
            continue
        try:
            score = (int(row['scoreHome']), int(row['scoreAway']))
        except ValueError:
            continue
        if score == (0, 0) or score == previous[game_id]:
            continue
        before = previous[game_id]
        delta = (score[0]-before[0], score[1]-before[1])
        previous[game_id] = score
        if min(delta) < 0 or max(delta) > 4 or all(delta):
            bad.add(game_id)
            continue
        side = 0 if delta[0] else 1
        if row['teamTricode'] != codes[game_id][side] or \
                row['actionType'].strip() not in ('Made Shot', 'Free Throw'):
            bad.add(game_id)
            continue
        games[game_id].append(Action(int(row['period']),
                                     int(row['actionNumber']), row['clock'],
                                     codes[game_id][side], delta[side],
                                     before, score, row['description']))
    return {game_id: tuple(rows) for game_id, rows in games.items()
            if game_id not in bad}


def short_windows(rows):
    out = []
    for start in range(len(rows)):
        for span in range(4, 8):
            end = start+span
            if end > len(rows):
                break
            selected = rows[start:end]
            if selected[0].period != selected[-1].period:
                continue
            before, after = selected[0].before, selected[-1].after
            delta = (after[0]-before[0], after[1]-before[1])
            if min(delta) < 1 or max(delta) < 6 or \
                    abs(delta[0]-delta[1]) < 3:
                continue
            out.append((start, end))
    return tuple(out)


@lru_cache(maxsize=1)
def ledger_pool():
    events = scoring_actions()
    out = {}
    for source in ('rotowire', 'sbnation'):
        for group in ('train', 'heldout'):
            entries = []
            for world in world_pool()[source, group]:
                game_id = world[0]
                if game_id not in events or events[game_id][-1].after != \
                        world[3]:
                    continue
                windows = short_windows(events[game_id])
                if windows:
                    entries.append((world, windows))
            out[source, group] = tuple(entries)
    return out


def sample_document(rng: random.Random, source='rotowire',
                    group='train', include_descriptions=True):
    pool = ledger_pool()[source, group]
    if not pool:
        raise RuntimeError('No eligible live-score ledger games')
    world, windows = rng.choice(pool)
    game_id, home, away = world[:3]
    codes = _codes()[game_id]
    rows = scoring_actions()[game_id]
    start, end = rng.choice(windows)
    selected = rows[start:end]
    before, after = selected[0].before, selected[-1].after
    delta = (after[0]-before[0], after[1]-before[1])
    favored = 0 if delta[0] > delta[1] else 1
    order = (favored, 1-favored)
    tokens = ['Live', 'score', ':']+TOKEN.findall(home)+['vs']+ \
        TOKEN.findall(away)+['.', '\n']

    def score_line(score):
        return ['Score', ':', codes[0], str(score[0]), ',', codes[1],
                str(score[1]), '.', '\n']

    tokens.extend(score_line(before))
    for action in selected:
        if include_descriptions:
            tokens.extend(['Play', ':']+TOKEN.findall(action.description)+
                          ['.', '\n'])
        else:
            tokens.extend(['Play', ':', action.team, 'scored',
                           str(action.points), '.', '\n'])
        tokens.extend(score_line(action.after))
    tokens.extend(['Latest', ':', codes[0]])
    copy_index = len(tokens)
    copy_value = str(after[0])
    tokens.extend([copy_value, '.', '\n'])
    tokens.extend(['Run', ':', codes[order[0]]])
    first_index = len(tokens)
    first = str(delta[order[0]])
    tokens.extend([first, ',', codes[order[1]]])
    second_index = len(tokens)
    second = str(delta[order[1]])
    tokens.extend([second, '.', '\n'])
    sites = (Site(copy_index, 'live_score_copy', copy_value,
                  {copy_value: 1.}),
             Site(first_index, 'live_run_first', first, {first: 1.}),
             Site(second_index, 'live_run_second', second, {second: 1.}))
    return Document(tuple(tokens), sites, 'live_score_ledger',
                    {'source': source, 'split': group, 'game_id': game_id,
                     'period': selected[0].period,
                     'start_score': before, 'end_score': after,
                     'actions': len(selected), 'descriptions':
                     include_descriptions})


def answer_from_visible_prefix(prefix, kind='run'):
    """Read only printed score lines and the queried team before target."""
    score_lines = []
    for i in range(len(prefix)-8):
        if tuple(prefix[i:i+2]) != ('Score', ':'):
            continue
        line = prefix[i:i+9]
        if len(line) != 9 or line[4] != ',' or \
                tuple(line[7:9]) != ('.', '\n'):
            continue
        score_lines.append((line[2], int(line[3]), line[5], int(line[6])))
    if len(score_lines) < 2:
        raise ValueError('Need visible start and end scores')
    first, last = score_lines[0], score_lines[-1]
    if first[0] != last[0] or first[2] != last[2]:
        raise ValueError('Team codes changed')
    if prefix[-1] not in (first[0], first[2]):
        raise ValueError('No queried team before target')
    side = 0 if prefix[-1] == first[0] else 1
    if kind == 'copy':
        return str(last[1] if side == 0 else last[3])
    if kind != 'run':
        raise ValueError(kind)
    value = (last[1]-first[1]) if side == 0 else (last[3]-first[3])
    if value < 0:
        raise ValueError('Negative score change')
    return str(value)
