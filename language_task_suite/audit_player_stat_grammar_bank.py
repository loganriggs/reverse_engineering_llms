"""Conservative complete-clause bank for box-score-grounded player statistics.

This accepts only whole, player-led tally clauses from a small explicit grammar.
It is a measured lower bound for reusable natural clauses, not a general parser.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

from audit_boxscore_relation_inventory import FACT, inspect
from audit_rotowire_document_source import game_key, load
from corpus_score_sentence_renderer import _sentences
from sports_document_dgp import ARCHIVE_DIR, _detokenize, game_partition


STAT_SEQUENCE = r'<STAT>(?:,?\s+(?:and\s+)?<STAT>)*'
PREDICATE = re.compile(
    r'(?:also\s+)?(?:'
    r'(?:finished|ended)(?:\s+(?:the\s+game|the\s+night|the\s+contest))?\s+with|'
    r'(?:had|posted|tallied|totaled|totalled|recorded|notched|collected|'
    r'contributed|added|accrued|registered|amassed)'
    r'(?:\s+(?:a|an))?|'
    r'put\s+up|chipped\s+in\s+with)\s+'
    +STAT_SEQUENCE+r'|(?:also\s+)?scored\s+<PTS>', re.I)


def accepted(sentence, row):
    if not 6 <= len(sentence) <= 55 or sentence[-1] != '.' or \
            sentence.count('.') != 1:
        return None
    record = inspect(sentence, row)
    if not record or not record.get('has_fact') or \
            not record['all_fact_values_match'] or \
            not record['all_number_tokens_accounted']:
        return None
    text = record['text']
    name = record['player']
    if not text.startswith(name+' '):
        return None
    # A full-name match alone does not rule out another player's surname.
    remaining = text[len(name):]
    surnames = {other.split()[-1] for other in
                row['box_score']['PLAYER_NAME'].values()
                if other != name and len(other.split()) >= 2}
    if any(re.search(r'\b'+re.escape(surname)+r'\b', remaining, re.I)
           for surname in surnames):
        return None
    # Accept no outcome or discourse suffix; its truth would require a
    # player-to-team relation and an explicit sampled discourse context.
    body = text[len(name)+1:-1]
    matches = list(FACT.finditer(body))
    if len(matches) != len(record['fields']) or len(set(record['fields'])) != len(matches):
        return None
    form = body
    for match, field in reversed(list(zip(matches, record['fields']))):
        form = form[:match.start()]+'<'+field+'>'+form[match.end():]
    # PREDICATE uses a generic stat marker except the scored-PTS special case.
    generic = re.sub(r'<(?:PTS|REB|AST|STL|BLK|TO|FG3M)>', '<STAT>', form)
    if not PREDICATE.fullmatch(generic) and not \
            (record['fields'] == ('PTS',) and
             re.fullmatch(r'(?:also\s+)?scored\s+<PTS>', form, re.I)):
        return None
    return {'form': form, 'fields': record['fields'], 'text': text}


def audit(rows, group, seed):
    counts = Counter()
    forms = Counter()
    examples = []
    for row in rows:
        if game_partition(game_key(row)) != group:
            continue
        counts['games'] += 1
        accepted_in_game = 0
        for sentence in _sentences(row['summary']):
            result = accepted(sentence, row)
            if result is None:
                continue
            counts['accepted_clauses'] += 1
            accepted_in_game += 1
            forms[result['form']] += 1
            if any(field not in ('PTS', 'REB', 'AST') for field in result['fields']):
                counts['extended_field_clauses'] += 1
            if len(examples) < 200:
                examples.append(result['text'])
        counts['games_with_clause'] += accepted_in_game > 0
        counts['games_with_two_clauses'] += accepted_in_game >= 2
    rng = random.Random(seed)
    return {'counts': dict(counts), 'distinct_forms': len(forms),
            'forms_in_two_games': sum(n >= 2 for n in forms.values()),
            'top_form_hashes': [
                {'sha256': hashlib.sha256(f.encode()).hexdigest(), 'count': n}
                for f, n in forms.most_common(10)],
            'manual_examples': rng.sample(examples, min(12, len(examples)))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--seed', type=int, default=96401)
    a = p.parse_args()
    result = {'config': {'seed': a.seed}, 'arms': {},
              'caveat': 'Whole-clause grammar screen, not semantic certification '
                        'of arbitrary corpus prose. No source text saved.'}
    for source in ('rotowire', 'sbnation'):
        rows = load(ARCHIVE_DIR/f'{source}.tar.bz2', source)['train']
        for group in ('train', 'heldout'):
            key = source+'_'+group
            record = audit(rows, group, a.seed)
            examples = record.pop('manual_examples')
            result['arms'][key] = record
            print(key, record['counts'], 'forms', record['distinct_forms'],
                  'examples', examples, flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
