"""Audit complete natural player clauses with box-score-verified relations.

The relation must be explicit and entirely checkable from one game record.
This extends the plain tally grammar without accepting arbitrary prose.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

from audit_grounded_player_sentence_bank import _unique_player
from audit_player_stat_grammar_bank import accepted as tally_accepted
from audit_rotowire_document_source import game_key, load
from corpus_score_sentence_renderer import _sentences
from sports_document_dgp import ARCHIVE_DIR, _detokenize, game_partition


RELATION_SUFFIX = re.compile(r'(?P<body>.+?)\s+(?P<relation>'
                             r'in\s+the\s+(?:win|loss)|off\s+the\s+bench)\.$', re.I)
RANK = re.compile(r'(?P<name>.+?)\s+scored\s+a\s+'
                  r'(?P<rank>team.high|game.high)\s+'
                  r'(?P<points>\d+)\s+points\.$', re.I)


def _score(row, key):
    value = row['box_score']['PTS'].get(key)
    return int(value) if str(value).isdigit() else None


def _player_relation(row, key, relation):
    box = row['box_score']
    if relation == 'off the bench':
        minute = box['MIN'].get(key)
        return box['START_POSITION'].get(key) == 'N/A' and \
            str(minute).isdigit() and int(minute) > 0
    city = box['TEAM_CITY'].get(key)
    home_pts = int(row['home_line']['TEAM-PTS'])
    away_pts = int(row['vis_line']['TEAM-PTS'])
    if home_pts == away_pts:
        return False
    if city == row['home_city']:
        won = home_pts > away_pts
    elif city == row['vis_city']:
        won = away_pts > home_pts
    else:
        return False
    return won if relation == 'in the win' else not won


def _rank_relation(row, key, rank):
    box = row['box_score']
    value = _score(row, key)
    if value is None:
        return False
    if rank == 'team-high':
        city = box['TEAM_CITY'].get(key)
        if city not in (row['home_city'], row['vis_city']):
            return False
        peer_keys = [k for k, c in box['TEAM_CITY'].items() if c == city]
    else:
        peer_keys = list(box['PLAYER_NAME'])
    peers = [_score(row, k) for k in peer_keys]
    # Missing score is permitted for an inactive player only if MIN is N/A.
    if any(_score(row, k) is None and str(box['MIN'].get(k)).isdigit()
           for k in peer_keys):
        return False
    known = [n for n in peers if n is not None]
    return bool(known) and value == max(known)


def accepted(sentence, row):
    text = _detokenize(sentence)
    player = _unique_player(sentence, row)
    if player is None:
        return None
    key, name = player
    suffix = RELATION_SUFFIX.fullmatch(text)
    if suffix:
        relation = suffix['relation'].lower()
        shorter = (suffix['body']+'.').replace(',', ' ,').replace('.', ' .').split()
        base = tally_accepted(shorter, row)
        if base is not None and _player_relation(row, key, relation):
            return {'relation': relation, 'form': base['form']+' <'+
                    relation.upper().replace(' ', '_')+'>', 'text': text}
    rank_match = RANK.fullmatch(text)
    if rank_match and rank_match['name'] == name:
        rank = rank_match['rank'].lower().replace(' ', '-')
        if int(rank_match['points']) != _score(row, key) or \
                not _rank_relation(row, key, rank):
            return None
        shorter = (name+' scored '+rank_match['points']+' points .').split()
        base = tally_accepted(shorter, row)
        if base is not None:
            return {'relation': rank, 'form': 'scored a <'+
                    rank.upper().replace('-', '_')+'> <PTS> points',
                    'text': text}
    return None


def audit(rows, group, seed):
    counts = Counter()
    forms = Counter()
    examples = []
    for row in rows:
        if game_partition(game_key(row)) != group:
            continue
        counts['games'] += 1
        in_game = 0
        for sentence in _sentences(row['summary']):
            result = accepted(sentence, row)
            if result is None:
                continue
            counts['accepted_clauses'] += 1
            counts['relation/'+result['relation']] += 1
            forms[result['form']] += 1
            in_game += 1
            if len(examples) < 100:
                examples.append(result['text'])
        counts['games_with_clause'] += in_game > 0
    rng = random.Random(seed)
    return {'counts': dict(counts), 'distinct_forms': len(forms),
            'top_form_hashes': [
                {'sha256': hashlib.sha256(form.encode()).hexdigest(), 'count': n}
                for form, n in forms.most_common(10)],
            'manual_examples': rng.sample(examples, min(8, len(examples)))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=96501)
    a = p.parse_args()
    result = {'config': {'seed': a.seed}, 'arms': {},
              'caveat': 'Narrow whole-clause relation grammar. No source '
                        'prose saved. Team link, minutes/position and score '
                        'rank are checked against the same game record.'}
    for source in ('rotowire', 'sbnation'):
        rows = load(ARCHIVE_DIR/f'{source}.tar.bz2', source)['train']
        for group in ('train', 'heldout'):
            key = source+'_'+group
            record = audit(rows, group, a.seed)
            examples = record.pop('manual_examples')
            result['arms'][key] = record
            print(key, record['counts'], 'examples', examples, flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
