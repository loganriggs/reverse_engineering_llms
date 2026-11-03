"""Inventory which natural player-stat quantities are grounded in box scores.

This is an opportunity screen, not a semantic-safe renderer. It counts
explicit quantities around exactly one full player name and compares them
with the structured record, including fields beyond PTS/REB/AST.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re

from audit_grounded_player_sentence_bank import _unique_player
from audit_rotowire_document_source import game_key, load
from corpus_score_sentence_renderer import _sentences
from sports_document_dgp import ARCHIVE_DIR, _detokenize, game_partition


NUMBER_WORDS = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
    'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
    'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
    'fourteen': 14, 'fifteen': 15, 'sixteen': 16,
    'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
    'twenty': 20,
}
NUM = r'(?:\d+|'+'|'.join(NUMBER_WORDS)+r')'
UNIT = (r'(?:points?|rebounds?|assists?|steals?|blocks?|'
        r'turnovers?|threes|three.pointers?|3.pointers?)')
FACT = re.compile(r'\b(?P<number>'+NUM+r')\s+(?P<unit>'+UNIT+r')\b',
                  flags=re.I)
NUMBER_TOKEN = re.compile(r'\b\d+\b')
SPELLED_NUMBER = re.compile(r'\b(?:'+'|'.join(NUMBER_WORDS)+r')\b',
                            flags=re.I)
CONTEXT = {
    'history': re.compile(r'\b(?:season|career|streak|record|previous|'
                          r'last|first|consecutive|playoff|series)\b',
                          flags=re.I),
    'game_flow': re.compile(r'\b(?:quarter|half|halftime|overtime|'
                            r'late|lead|led|trailing|run)\b',
                            flags=re.I),
    'outcome': re.compile(r'\b(?:win|loss|victory|defeat|beat|'
                          r'losing|winning)\b', flags=re.I),
}
FIELD = {'point': 'PTS', 'rebound': 'REB', 'assist': 'AST',
         'steal': 'STL', 'block': 'BLK', 'turnover': 'TO',
         'three': 'FG3M', 'three.pointer': 'FG3M',
         '3.pointer': 'FG3M'}
BASIC = frozenset(('PTS', 'REB', 'AST'))


def _field(unit):
    cleaned = re.sub(r'[-–]', '.', unit.lower().rstrip('s'))
    if cleaned == 'three.pointer':
        return 'FG3M'
    return FIELD.get(cleaned)


def inspect(sentence, row):
    player = _unique_player(sentence, row)
    if player is None:
        return None
    key, name = player
    text = _detokenize(sentence)
    matches = list(FACT.finditer(text))
    if not matches:
        return {'has_fact': False}
    fields = []
    matched = []
    numeric_spans = set()
    for match in matches:
        raw = match['number'].lower()
        number = int(raw) if raw.isdigit() else NUMBER_WORDS[raw]
        field = _field(match['unit'])
        if field is None:
            continue
        expected = row['box_score'].get(field, {}).get(key)
        equal = str(expected).isdigit() and int(expected) == number
        fields.append(field)
        matched.append(equal)
        numeric_spans.add(match.span('number'))
        unit = match['unit'].lower()
        if unit.startswith(('three-', 'three ', '3-')):
            unit_start = match.start('unit')
            numeric_spans.add((unit_start, unit_start+
                               (5 if unit.startswith('three') else 1)))
    all_numbers = [match.span() for expression in
                   (NUMBER_TOKEN, SPELLED_NUMBER)
                   for match in expression.finditer(text)]
    complete = len(fields) == len(matches) and \
        all(span in numeric_spans for span in all_numbers)
    flags = {label: bool(pattern.search(text))
             for label, pattern in CONTEXT.items()}
    return {'has_fact': True, 'fields': tuple(fields),
            'all_fact_values_match': bool(fields) and all(matched),
            'all_number_tokens_accounted': complete,
            'context_flags': flags,
            'text': text, 'player': name}


def summarize(rows, group, seed):
    counts = Counter()
    examples = defaultdict(list)
    for row in rows:
        if game_partition(game_key(row)) != group:
            continue
        counts['games'] += 1
        any_licensed = False
        for sentence in _sentences(row['summary']):
            counts['sentences'] += 1
            record = inspect(sentence, row)
            if record is None:
                continue
            counts['one_full_player'] += 1
            if not record['has_fact']:
                continue
            counts['one_player_with_explicit_stat'] += 1
            for field in set(record['fields']):
                counts['mention/'+field] += 1
            if not record['all_fact_values_match']:
                counts['fact_value_mismatch'] += 1
                continue
            counts['all_matched_values'] += 1
            if not record['all_number_tokens_accounted']:
                counts['extra_unaccounted_number'] += 1
                continue
            counts['all_numeric_claims_accounted'] += 1
            if set(record['fields']) <= BASIC:
                counts['basic_only'] += 1
            else:
                counts['needs_extended_field'] += 1
                for field in set(record['fields'])-BASIC:
                    counts['extended_rescue/'+field] += 1
            for label, present in record['context_flags'].items():
                counts['context/'+label] += present
            any_licensed = True
            category = ('extended' if set(record['fields'])-BASIC
                        else 'basic')
            if len(examples[category]) < 100:
                examples[category].append(record['text'])
        counts['games_with_numeric_candidate'] += any_licensed
    rng = random.Random(seed)
    selected = {key: rng.sample(values, min(5, len(values)))
                for key, values in examples.items()}
    return {'counts': dict(counts), 'manual_examples': selected}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=96301)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = {'config': {'seed': a.seed}, 'arms': {},
              'caveat': 'Numeric opportunity only. Matching box-score '
                        'quantities does not validate comparative, outcome, '
                        'history, injury, shooting-efficiency, or discourse '
                        'claims. Spelled numbers beyond twenty and complex '
                        'stat constructions are not parsed. No source prose '
                        'is saved in the JSON.'}
    for source in ('rotowire', 'sbnation'):
        rows = load(ARCHIVE_DIR/f'{source}.tar.bz2', source)['train']
        for group in ('train', 'heldout'):
            key = source+'_'+group
            record = summarize(rows, group, a.seed)
            examples = record.pop('manual_examples')
            result['arms'][key] = record
            print(key, record['counts'], flush=True)
            print('manual examples', examples, flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
