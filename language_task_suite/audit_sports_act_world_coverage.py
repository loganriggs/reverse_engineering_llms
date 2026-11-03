"""Measure sports report act coverage and box-score record opportunities.

Lexical act tags are proxies, not validated rhetorical annotations. Output
contains aggregate counts only; human source text is never written.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from audit_sports_discourse_acts import labels
from corpus_score_sentence_renderer import _sentences
from sports_document_dgp import game_partition, resources
from audit_rotowire_document_source import game_key


RECORD = re.compile(r'(?<!\d)(\d{1,2})\s*-\s*(\d{1,2})(?!\d)')
SUPPORTED = {'result', 'player_stat', 'game_flow'}


def record_pairs(row):
    pairs = set()
    for field in ('home_line', 'vis_line'):
        line = row[field]
        try:
            wins = int(line['TEAM-WINS'])
            losses = int(line['TEAM-LOSSES'])
        except (TypeError, KeyError, ValueError):
            continue
        if 0 <= wins <= 99 and 0 <= losses <= 99:
            pairs.add((wins, losses))
    return pairs


def audit(rows):
    counts = Counter()
    for row in rows:
        pairs = record_pairs(row)
        counts['documents'] += 1
        counts['documents_with_record_fields'] += bool(pairs)
        seen_record_match = False
        for sentence in list(_sentences(row['summary']))[:12]:
            tags = labels(sentence, row)
            primary = tags[0]
            counts['sentences'] += 1
            counts['primary/'+primary] += 1
            counts['primary_outside_current_world'] += primary not in SUPPORTED
            text = ' '.join(sentence)
            observed = {(int(a), int(b)) for a, b in RECORD.findall(text)}
            if observed:
                counts['record_pair_surface'] += 1
            if observed & pairs:
                counts['record_pair_matches_box_score'] += 1
                counts['record_match_primary/'+primary] += 1
                seen_record_match = True
        counts['documents_with_visible_record_match'] += seen_record_match
    n = counts['sentences']
    return {'counts': dict(counts),
            'primary_outside_current_world_share': round(
                counts['primary_outside_current_world']/n, 4),
            'visible_record_match_document_share': round(
                counts['documents_with_visible_record_match'] /
                counts['documents'], 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = {'scope': 'first twelve sentences; train archive partitioned '
                    'by stable game key',
           'caveat': 'Act labels are lexical proxies. A matching W-L pair '
                     'can be incidental, pregame, or another team record; '
                     'this is an opportunity screen, not a certified '
                     'record-clause bank or exact next-token oracle.',
           'arms': {}}
    for source in ('rotowire', 'sbnation'):
        rows = resources()[0][source][0]
        for group in ('train', 'heldout'):
            selected = [row for row in rows
                        if game_partition(game_key(row)) == group]
            out['arms'][source+'_'+group] = audit(selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2)+'\n')
    for key, arm in out['arms'].items():
        print(key, arm['primary_outside_current_world_share'],
              arm['visible_record_match_document_share'])


if __name__ == '__main__':
    main()
