"""Screen real report sentences for checkable halftime-score opportunities.

Exact score matching near a halftime word is an opportunity proxy, not a
certification that the sentence can be edited or used as a task oracle.
Only aggregate counts are saved; source prose stays local.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from audit_rotowire_document_source import game_key
from audit_sports_discourse_acts import player_stat, score_pair
from sports_alias_bracket_dgp import bracket_pairs
from sports_document_dgp import game_partition, resources
from sports_entity_alias_dgp import candidates as alias_candidates


MARKERS = frozenset(('halftime', 'half', 'intermission'))
SEPARATORS = frozenset(('-', 'to', '–'))


def quarter_values(row):
    try:
        home = [int(row['home_line'][f'TEAM-PTS_QTR{i}'])
                for i in range(1, 5)]
        away = [int(row['vis_line'][f'TEAM-PTS_QTR{i}'])
                for i in range(1, 5)]
        final = (int(row['home_line']['TEAM-PTS']),
                 int(row['vis_line']['TEAM-PTS']))
    except (KeyError, TypeError, ValueError):
        return None
    if min(home+away) < 0 or (sum(home), sum(away)) != final:
        return None
    return (sum(home[:2]), sum(away[:2])), final


def half_pair_positions(tokens, halftime):
    a, b = halftime
    return tuple(i for i in range(len(tokens)-2)
                 if tokens[i].isdigit() and tokens[i+2].isdigit()
                 and tokens[i+1] in SEPARATORS
                 and (int(tokens[i]), int(tokens[i+2]))
                 in ((a, b), (b, a)))


def screen(row, source):
    values = quarter_values(row)
    if values is None:
        return None
    halftime, final = values
    words = row['summary']
    markers = [i for i, token in enumerate(words)
               if token.lower() in MARKERS]
    pairs = half_pair_positions(words, halftime)
    near = [i for i in pairs
            if any(abs(i-j) <= 20 for j in markers)]
    outcome = {
        'half_marker': bool(markers),
        'half_pair_anywhere': bool(pairs),
        'half_pair_near_marker': bool(near),
        'half_leader_reverses': (halftime[0]-halftime[1])*
                                (final[0]-final[1]) < 0,
        'final_pair_anywhere': score_pair(words, row),
        'player_stat_proxy': player_stat(words, row),
    }
    if source == 'rotowire':
        outcome['alias_eligible'] = bool(alias_candidates(row))
        outcome['bracket_eligible'] = bool(bracket_pairs(words))
    return outcome


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    output = {}
    source_rows = resources()[0]
    for source in ('rotowire', 'sbnation'):
        for group in ('train', 'heldout'):
            counts = Counter()
            for row in source_rows[source][0]:
                if game_partition(game_key(row)) != group:
                    continue
                counts['documents'] += 1
                flags = screen(row, source)
                if flags is None:
                    continue
                counts['quarter_consistent'] += 1
                for key, value in flags.items():
                    counts[key] += value
                if flags['half_pair_near_marker']:
                    for key in ('half_leader_reverses', 'final_pair_anywhere',
                                'player_stat_proxy', 'alias_eligible',
                                'bracket_eligible'):
                        if key in flags:
                            counts['near_half_and_'+key] += flags[key]
                    if source == 'rotowire':
                        counts['near_half_alias_bracket'] += (
                            flags['alias_eligible'] and
                            flags['bracket_eligible'])
            output[source+'/'+group] = dict(counts)
            print(source, group, dict(counts), flush=True)
    output['definitions'] = {
        'near_half': 'An exact box-score-derived first-half score pair, in '
                     'either orientation and separated by -, to, or –, '
                     'within 20 source tokens of half/halftime/intermission.',
        'alias_eligible': 'Existing surface heuristic: unique full player '
                          'name followed later by surname in first 250 '
                          'tokens, with at least 25-token gap.',
        'bracket_eligible': 'Existing natural parenthetical pair with a '
                            '3–30-token gap and closer before token 250.',
        'caveat': 'Near-half is a surface match, not a certified halftime '
                  'claim or a safe editable sentence. Full-document score '
                  'and player proxies are likewise lexical. Raw reports '
                  'and source examples are not saved.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
