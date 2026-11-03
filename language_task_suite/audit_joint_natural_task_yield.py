"""Screen co-occurring natural sports task sites before building a joint DGP.

Broad numeric matches are *opportunities*, not verified editable rules.
Strict counts require a whole-clause player grammar and a guarded first score
sentence; neither grants a full oracle for the report.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from audit_player_stat_grammar_bank import accepted as tally_accepted
from audit_rotowire_document_source import (
    game_key, player_point_mentions, score_pair_present,
)
from corpus_score_sentence_renderer import _sentences, _template
from score_template_guard import safe_for
from sports_document_dgp import game_partition, resources
from sports_entity_alias_dgp import candidates as alias_candidates


def safe_first_score(row):
    try:
        home_score = int(row['home_line']['TEAM-PTS'])
        away_score = int(row['vis_line']['TEAM-PTS'])
    except (KeyError, TypeError, ValueError):
        return False
    if home_score == away_score:
        return False
    home = row['home_city']+' '+row['home_name']
    away = row['vis_city']+' '+row['vis_name']
    winner, loser = (home, away) if home_score > away_score else \
        (away, home)
    first = next(_sentences(row['summary']), [])
    high, low = max(home_score, away_score), min(home_score, away_score)
    template = _template(first, winner, loser, str(high), str(low),
                         home_score > away_score, Counter())
    return bool(template and safe_for(template, high, low) and
                safe_for(template, high+1, low))


def safe_player_points(row):
    return any((result := tally_accepted(sentence, row)) is not None
               and 'PTS' in result['fields']
               for sentence in _sentences(row['summary']))


def audit(group):
    counts = Counter()
    combinations = Counter()
    strict_keys = []
    for row in resources()[0]['rotowire'][0]:
        key = game_key(row)
        if game_partition(key) != group:
            continue
        broad_score = score_pair_present(row)
        broad_player = player_point_mentions(row) > 0
        alias = bool(alias_candidates(row))
        strict_score = safe_first_score(row)
        strict_player = safe_player_points(row)
        flags = dict(broad_score=broad_score, broad_player=broad_player,
                     alias=alias, strict_score=strict_score,
                     strict_player=strict_player)
        counts['reports'] += 1
        for name, value in flags.items():
            counts[name] += value
        for score_name, player_name in (
                ('broad_score', 'broad_player'),
                ('broad_score', 'strict_player'),
                ('strict_score', 'broad_player'),
                ('strict_score', 'strict_player')):
            label = score_name+'__'+player_name
            combinations[label] += flags[score_name] and flags[player_name]
            combinations[label+'__alias'] += \
                flags[score_name] and flags[player_name] and alias
        combinations['strict_score__alias'] += strict_score and alias
        combinations['strict_player__alias'] += strict_player and alias
        if strict_score and strict_player and alias:
            strict_keys.append(hashlib.sha256(repr(key).encode()).hexdigest())
    return {'counts': dict(counts),
            'intersections': dict(combinations),
            'strict_triple_game_sha256': strict_keys}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = {'source': 'RotoWire, grouped by date and teams',
              'groups': {group: audit(group)
                         for group in ('train', 'heldout')},
              'caveat': 'Broad score/player surface matches are not semantic '
                        'or edit-safe. Strict first-score and whole-clause '
                        'player screens are conservative candidate guards, '
                        'not a full-report oracle. Alias relation remains '
                        'surface-screened. No raw report text is saved.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    for group, row in result['groups'].items():
        print(group, row['counts'], row['intersections'], flush=True)


if __name__ == '__main__':
    main()
