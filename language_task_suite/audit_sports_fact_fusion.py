"""Screen natural report sentences for grounded fact combinations.

This is an opportunity audit, not a semantic parser. It saves aggregates only.
All source rows retain their existing whole-game train/heldout partition.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from audit_rotowire_document_source import game_key
from corpus_score_sentence_renderer import _find, _sentences
from sports_document_dgp import game_partition, resources


WIN_WORDS = {'win', 'wins', 'victory', 'victories'}
LOSS_WORDS = {'loss', 'losses', 'defeat'}


def point_mentions(sentence, row):
    """Return names with a nearby exact box-score `N points` phrase."""
    box = row['box_score']
    found = set()
    for player_id, name in box['PLAYER_NAME'].items():
        points = str(box['PTS'].get(player_id, ''))
        if not name or not points.isdigit():
            continue
        for name_start in _find(sentence, name.split()):
            after_name = name_start+len(name.split())
            for point_start in _find(sentence, [points, 'points']):
                if 0 <= point_start-after_name <= 14:
                    found.add(name)
    return found


def audit(rows):
    counts = Counter()
    for row in rows:
        try:
            scores = [int(row[side+'_line']['TEAM-PTS'])
                      for side in ('home', 'vis')]
        except (KeyError, TypeError, ValueError):
            continue
        if scores[0] == scores[1]:
            continue
        score_pair = [str(max(scores)), '-', str(min(scores))]
        counts['documents'] += 1
        document_hits = set()
        for sentence in list(_sentences(row['summary']))[:12]:
            counts['sentences_first12'] += 1
            score = bool(_find(sentence, score_pair))
            names = point_mentions(sentence, row)
            words = {token.lower() for token in sentence}
            outcome = bool(words & (WIN_WORDS | LOSS_WORDS))
            categories = {
                'exact_final_score': score,
                'player_exact_points': bool(names),
                'final_score_and_player_points': score and bool(names),
                'player_points_and_outcome_word': bool(names) and outcome,
                'two_player_point_facts': len(names) >= 2,
                'player_points_and_win_word': bool(names) and
                    bool(words & WIN_WORDS),
                'player_points_and_loss_word': bool(names) and
                    bool(words & LOSS_WORDS),
            }
            for category, present in categories.items():
                if present:
                    counts['sentence/'+category] += 1
                    document_hits.add(category)
        for category in document_hits:
            counts['document/'+category] += 1
    documents = counts['documents']
    return {'counts': dict(sorted(counts.items())),
            'document_shares': {
                name: round(counts['document/'+name]/documents, 4)
                for name in ('exact_final_score', 'player_exact_points',
                             'final_score_and_player_points',
                             'player_points_and_outcome_word',
                             'two_player_point_facts',
                             'player_points_and_win_word',
                             'player_points_and_loss_word')}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source_rows = resources()[0]
    out = {
        'scope': 'First 12 sentences per report, existing whole-game split; '
                 'exact final-score pair and exact player PTS near "points".',
        'caveat': 'Outcome words are lexical opportunities, not certified '
                  'winner/loss semantics. Point mentions may be incidental '
                  'despite exact box-score and proximity checks. No source '
                  'sentences are saved.',
        'arms': {},
    }
    for source in ('rotowire', 'sbnation'):
        rows = source_rows[source][0]
        for group in ('train', 'heldout'):
            selected = [row for row in rows if
                        game_partition(game_key(row)) == group]
            out['arms'][source+'_'+group] = audit(selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2)+'\n')
    for arm, result in out['arms'].items():
        print(arm, result['document_shares'], flush=True)


if __name__ == '__main__':
    main()
