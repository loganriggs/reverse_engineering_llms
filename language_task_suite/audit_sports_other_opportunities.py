"""Quantify what the coarse 'other' sports sentence proxy may contain.

Every label here is an opportunity screen, not semantic certification. The
script writes aggregate counts only and keeps source prose out of artifacts.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from audit_rotowire_document_source import game_key
from audit_sports_discourse_acts import labels
from corpus_score_sentence_renderer import _sentences
from sports_document_dgp import game_partition, resources


PERCENT = re.compile(r'\b(\d{1,3})\s*(?:percent|%)\b', re.I)
COUNT = re.compile(r'\b(\d{1,3})\s+(rebounds?|assists?|turnovers?)\b', re.I)
FUTURE = re.compile(r'\b(?:next|upcoming|will|face|faces|host|hosts|'
                    r'visit|visits|travel|travels|conclude|concludes)\b', re.I)
HISTORY = re.compile(r'\b(?:last|previous|season|career|streak|straight|'
                     r'prior|earlier|again|already)\b', re.I)
EDITORIAL = re.compile(r'\b(?:I|we|my|our|think|seems|seemed|perhaps|'
                       r'questionable|unbelievable|incredible|amazing|'
                       r'clearly|should|could|would)\b', re.I)
POSSESSION = re.compile(r'\b(?:foul|fouled|turnaround|buzzer|layup|'
                        r'dunk|steal|block|possession|clock|timeout|'
                        r'free throw|three-pointer)\b', re.I)
ANAPHORA = re.compile(r'\b(?:he|him|his|they|them|their|it|its)\b', re.I)


def team_mentions(sentence, row):
    """Nonoverlapping complete-name or nickname spans for the two teams."""
    raw = []
    for side in ('home', 'vis'):
        name = row[side+'_city']+' '+row[side+'_name']
        for alias in (name, row[side+'_name']):
            for match in re.finditer(r'(?<!\w)'+re.escape(alias)+r'(?!\w)',
                                     sentence):
                raw.append((match.start(), match.end(), side))
    raw.sort(key=lambda item: (item[0], -(item[1]-item[0])))
    out = []
    for mention in raw:
        if not any(mention[0] < old[1] and old[0] < mention[1]
                   for old in out):
            out.append(mention)
    return out


def numeric_team_match(sentence, row):
    """Nearest preceding team plus exact box-score value and field cue."""
    mentions = team_mentions(sentence, row)
    matched = set()
    for pattern in (PERCENT, COUNT):
        for fact in pattern.finditer(sentence):
            previous = [item for item in mentions
                        if item[1] <= fact.start() and
                        fact.start()-item[1] <= 85]
            if not previous:
                continue
            mention = max(previous, key=lambda item: item[1])
            side = mention[2]
            between = sentence[mention[1]:fact.start()].lower()
            tail = sentence[fact.end():fact.end()+45].lower()
            context = (between+' '+tail)
            if pattern is PERCENT:
                if any(cue in context for cue in ('three', '3-point',
                                                  'beyond', 'from deep',
                                                  'long range')):
                    field = 'TEAM-FG3_PCT'
                elif 'free throw' in context or 'charity stripe' in context:
                    field = 'TEAM-FT_PCT'
                elif any(cue in context for cue in ('field', 'floor',
                                                    'shoot', 'shot')):
                    field = 'TEAM-FG_PCT'
                else:
                    continue
            else:
                unit = fact.group(2).lower().rstrip('s')
                field = {'rebound': 'TEAM-REB', 'assist': 'TEAM-AST',
                         'turnover': 'TEAM-TOV'}[unit]
            value = row[side+'_line'].get(field)
            if value is not None and str(value).isdigit() and \
                    int(value) == int(fact.group(1)):
                matched.add(field)
    return matched


def audit(rows):
    counts = Counter()
    for row in rows:
        counts['documents'] += 1
        for sentence_tokens in list(_sentences(row['summary']))[:12]:
            if labels(sentence_tokens, row)[0] != 'other':
                continue
            counts['other_sentences'] += 1
            sentence = ' '.join(sentence_tokens)
            found = numeric_team_match(sentence, row)
            counts['any_exact_team_stat_opportunity'] += bool(found)
            for field in found:
                counts['team_stat/'+field] += 1
            for name, pattern in (('future', FUTURE), ('historical', HISTORY),
                                  ('editorial', EDITORIAL),
                                  ('possession', POSSESSION),
                                  ('anaphora', ANAPHORA)):
                counts['cue/'+name] += bool(pattern.search(sentence))
    total = counts['other_sentences']
    return {'counts': dict(counts),
            'exact_team_stat_opportunity_share': round(
                counts['any_exact_team_stat_opportunity']/total, 4),
            'cue_shares': {name: round(counts['cue/'+name]/total, 4)
                           for name in ('future', 'historical', 'editorial',
                                        'possession', 'anaphora')}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = {'scope': 'Primary other proxy, first twelve sentences, stable '
                    'game-key train/heldout partition',
           'caveat': 'All categories overlap and are lexical or exact-value '
                     'opportunity proxies. Nearest-team alignment can be '
                     'wrong; none is a certified claim or causal rule.',
           'arms': {}}
    for source in ('rotowire', 'sbnation'):
        rows = resources()[0][source][0]
        for group in ('train', 'heldout'):
            selected = [row for row in rows
                        if game_partition(game_key(row)) == group]
            key = source+'_'+group
            out['arms'][key] = audit(selected)
            result = out['arms'][key]
            print(key, result['counts']['other_sentences'],
                  result['exact_team_stat_opportunity_share'],
                  result['cue_shares'], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2)+'\n')


if __name__ == '__main__':
    main()
