"""Screen human-written stat predicates for a typed, reusable renderer.

Only a player-led stat predicate is extracted; surrounding prose is discarded.
This is a factual *fragment* screen, not a claim that an entire source sentence
can be safely transplanted. Source strings stay local; output contains counts.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

from audit_boxscore_relation_inventory import FACT, _field
from audit_grounded_player_sentence_bank import _unique_player
from audit_rotowire_document_source import game_key, load
from corpus_score_sentence_renderer import _sentences
from sports_document_dgp import ARCHIVE_DIR, _detokenize, game_partition


VERB = re.compile(
    r'(?P<verb>scored|recorded|registered|tallied|posted|collected|'
    r'accumulated|amassed|notched|managed|totaled|totalled|contributed|'
    r'added|chipped in with|put up|finished with|ended with|had)\s+', re.I)
CONNECTOR = re.compile(r'\s*(?:,\s*(?:and\s+)?|and\s+|plus\s+)')
UNSAFE_NEXT = re.compile(r'\s*(?:and|plus|with)\b', re.I)


def extract(sentence, row):
    """Return an exact typed stat phrase after one named player's safe verb."""
    player = _unique_player(sentence, row)
    if player is None:
        return None
    key, name = player
    text = _detokenize(sentence)
    name_match = re.search(r'(?<!\w)'+re.escape(name)+r'(?!\w)', text)
    if name_match is None:
        return None
    rest = text[name_match.end():].lstrip()
    rest = re.sub(r'^(?:also|quietly)\s+', '', rest, flags=re.I)
    verb = VERB.match(rest)
    if verb is None:
        return None
    cursor = verb.end()
    facts = []
    while True:
        fact = FACT.match(rest, cursor)
        if fact is None:
            break
        field = _field(fact['unit'])
        raw = fact['number'].lower()
        from audit_boxscore_relation_inventory import NUMBER_WORDS
        number = int(raw) if raw.isdigit() else NUMBER_WORDS[raw]
        expected = row['box_score'].get(field, {}).get(key)
        if field is None or not str(expected).isdigit() or \
                int(expected) != number or field in [f[0] for f in facts]:
            return None
        facts.append((field, fact))
        cursor = fact.end()
        connector = CONNECTOR.match(rest, cursor)
        if connector is None:
            break
        next_fact = FACT.match(rest, connector.end())
        if next_fact is None:
            if connector.group().strip().lower().startswith(('and', 'plus')):
                return None
            break
        cursor = connector.end()
    if not facts or UNSAFE_NEXT.match(rest[cursor:]):
        return None
    # Do not claim the source sentence's remaining clauses as part of the
    # portable fact. They may mention a different player, injury or history.
    fragment = rest[:cursor].strip()
    form = fragment
    for field, match in reversed(facts):
        form = form[:match.start()]+'{'+field+'} '+match['unit']+form[match.end():]
    return {'form': form, 'fields': tuple(field for field, _ in facts),
            'fragment': name+' '+fragment+'.', 'source_sentence': text}


def audit(rows, group, seed):
    counts, forms = Counter(), Counter()
    examples = []
    for row in rows:
        if game_partition(game_key(row)) != group:
            continue
        counts['games'] += 1
        found = 0
        for sentence in _sentences(row['summary']):
            counts['sentences'] += 1
            result = extract(sentence, row)
            if result is None:
                continue
            found += 1
            counts['accepted_fragments'] += 1
            counts['stat_count/'+str(len(result['fields']))] += 1
            for field in result['fields']:
                counts['field/'+field] += 1
            forms[result['form']] += 1
            if len(examples) < 1000:
                examples.append(result)
        counts['games_with_fragment'] += found > 0
        counts['games_with_2plus_fragments'] += found >= 2
    rng = random.Random(seed)
    return {'counts': dict(counts), 'distinct_forms': len(forms),
            'forms_in_2plus_games': sum(v >= 2 for v in forms.values()),
            'top_form_hashes': [
                {'sha256': hashlib.sha256(k.encode()).hexdigest(), 'count': v}
                for k, v in forms.most_common(10)],
            'manual_examples': rng.sample(examples, min(30, len(examples)))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=37101)
    args = p.parse_args()
    result = {'config': {'seed': args.seed}, 'arms': {},
              'caveat': 'Typed stat fragments only, not complete natural '
                        'sentences or verified discourse. Source text omitted.'}
    for source in ('rotowire', 'sbnation'):
        rows = load(ARCHIVE_DIR/f'{source}.tar.bz2', source)['train']
        for group in ('train', 'heldout'):
            key = source+'_'+group
            record = audit(rows, group, args.seed)
            examples = record.pop('manual_examples')
            result['arms'][key] = record
            print(key, record['counts'], 'forms', record['distinct_forms'],
                  flush=True)
            print('Manual fragments:', [x['fragment'] for x in examples],
                  flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
