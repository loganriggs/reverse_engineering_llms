"""Screen human x-y run mentions against matched play-by-play score windows.

This is an opportunity and consistency audit, not a full semantic entailment
parser. It writes aggregate counts, never source sentences.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from audit_rotowire_document_source import game_key
from audit_sports_event_predicates import (event_facts, scoring_events,
                                           team_codes)
from audit_sports_pbp_join import metadata, report_key, sha256
from corpus_score_sentence_renderer import _sentences
from sports_document_dgp import game_partition, resources


RUN = re.compile(r'\b(\d{1,2})\s*-\s*(\d{1,2})\s+run\b', re.I)
PERIOD = re.compile(r'\b(first|second|third|fourth)\s+'
                    r'(?:quarter|period)\b', re.I)
ORDINAL = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4}


def run_windows(rows):
    """Point deltas over within-period windows and whole-game windows (0)."""
    by_period = defaultdict(list)
    all_scores = [(0, 0)]
    previous = (0, 0)
    for period, _, score, _, _ in rows:
        if period not in by_period:
            by_period[period].append(previous)
        if score != previous:
            by_period[period].append(score)
            all_scores.append(score)
        previous = score
    by_period[0] = all_scores
    out = defaultdict(set)
    for period, scores in by_period.items():
        for start_index, start in enumerate(scores):
            for end in scores[start_index+1:]:
                delta = (end[0]-start[0], end[1]-start[1])
                if min(delta) < 0:
                    continue
                out[period].add(delta)
                out[period].add((delta[1], delta[0]))
    return out


def audit(rows, index, facts, windows):
    counts = Counter()
    examples_by_pair = Counter()
    for row in rows:
        key = report_key(row)
        if key is None or not '20161001' <= key[0] <= '20170531':
            continue
        ids = index.get(key, ())
        if len(ids) != 1:
            continue
        game_id = ids[0]
        if facts[game_id]['bad_steps']:
            continue
        counts['matched_clean_report_rows'] += 1
        game_windows = windows[game_id]
        row_mentions = 0
        for sentence in _sentences(row['summary']):
            text = ' '.join(sentence)
            period_match = PERIOD.search(text)
            explicit_period = ORDINAL[period_match.group(1).lower()] \
                if period_match else None
            for match in RUN.finditer(text):
                pair = (int(match.group(1)), int(match.group(2)))
                if pair[0] < 5 or pair[0] > 40 or pair[1] > 30:
                    counts['out_of_scope_run_pair'] += 1
                    continue
                counts['candidate_run_mentions'] += 1
                row_mentions += 1
                examples_by_pair[pair] += 1
                found = [p for p, candidates in game_windows.items()
                         if p != 0 and pair in candidates]
                counts['any_period_score_window_match'] += bool(found)
                counts['unique_period_score_window_match'] += len(found) == 1
                counts['any_game_score_window_match'] += \
                    pair in game_windows[0]
                counts['cross_period_only_match'] += \
                    not found and pair in game_windows[0]
                if explicit_period is not None:
                    counts['explicit_period_mentions'] += 1
                    counts['explicit_period_window_match'] += \
                        explicit_period in found
                counts['unanswered_run_mentions'] += pair[1] == 0
        counts['report_rows_with_run_mentions'] += row_mentions > 0
    return {'counts': dict(counts),
            'common_pairs': [[f'{a}-{b}', n] for (a, b), n in
                             examples_by_pair.most_common(12)]}


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
    windows = {game_id: run_windows(rows)
               for game_id, rows in trajectories.items()
               if not facts[game_id]['bad_steps']}
    index = defaultdict(list)
    for game_id, item in meta.items():
        if game_id in windows and len(item['teams']) == 2:
            index[(item['date'], frozenset(item['teams']),
                   facts[game_id]['score'])].append(game_id)
    out = {'config': {'pbp_file': args.pbp.name,
                      'pbp_sha256': sha256(args.pbp),
                      'shotdetail_file': args.shotdetail.name,
                      'shotdetail_sha256': sha256(args.shotdetail),
                      'screen': 'numeric x-y run mention, x 5..40, y 0..30; '
                                'contiguous scoring windows within one '
                                'period or across the game, either team '
                                'orientation'},
           'caveat': 'Finding an exact score window does not establish the '
                     'named team, temporal wording, or full-sentence '
                     'entailment. Missing matches can reflect cross-period '
                     'runs or archive/report errors.',
           'sources': {}}
    for source in ('rotowire', 'sbnation'):
        rows = resources()[0][source][0]
        out['sources'][source] = {}
        for group in ('train', 'heldout'):
            selected = [row for row in rows
                        if game_partition(game_key(row)) == group]
            result = audit(selected, index, facts, windows)
            out['sources'][source][group] = result
            print(source, group, result, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2)+'\n')


if __name__ == '__main__':
    main()
