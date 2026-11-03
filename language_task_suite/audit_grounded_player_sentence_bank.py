"""Conservative yield screen for corpus player-stat realizations.

Accept a source sentence only when exactly one roster player is mentioned,
every digit is an explicit points/rebounds/assists quantity matching that
player's box score, and a set of context claims absent from the world is
excluded. This is a candidate bank, not proof of complete semantic fidelity.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

from audit_rotowire_document_source import game_key, load
from corpus_score_sentence_renderer import _sentences
from sports_document_dgp import ARCHIVE_DIR, PLAYER, _detokenize, game_partition


NUMBER = re.compile(r'\b\d+\b')
STAT = re.compile(r'\b(\d+)\s+(points?|rebounds?|assists?)\b',
                  flags=re.I)
UNSUPPORTED = re.compile(
    r'\b(?:career|season|streak|straight|previous|last|first|'
    r'overtime|quarter|half|minute|minutes|shooting|field|'
    r'free throw|three.pointer|percentage|percent|'
    r'injur\w*|start\w*|bench|record|'
    r'win|loss|victory|defeat|double.double|triple.double|'
    r'game.high|season.high|playoff|series|consecutive)\b',
    flags=re.I)
SAFE_WORDS = frozenset((
    'scored', 'scoring', 'posted', 'had', 'finished', 'contributed',
    'added', 'adding', 'tallied', 'registered', 'recorded', 'collected',
    'amassed', 'totaled', 'totalled', 'grabbed', 'pulled', 'down',
    'hauled', 'in', 'dished', 'out', 'handed', 'put', 'up', 'chipped',
    'notched', 'with', 'and', 'also', 'to', 'go', 'along', 'a', 'an',
    'the', 'of', 'for', 'points', 'point', 'rebounds', 'rebound',
    'assists', 'assist', 'while', 'game', 'night', 'from', 'on',
    'as', 'plus', 'by', 'he', 'his',
))


def _unique_player(sentence, row):
    tokens = sentence
    table = row['box_score']
    matches = []
    for key, name in table['PLAYER_NAME'].items():
        if not PLAYER.fullmatch(name):
            continue
        words = name.split()
        for i in range(len(tokens)-len(words)+1):
            if tokens[i:i+len(words)] == words:
                matches.append((key, name))
    if len(matches) != 1:
        return None
    return matches[0]


def accepted(sentence, row):
    if not 8 <= len(sentence) <= 65:
        return None, 'length'
    if sentence[-1] != '.' or sentence.count('.') != 1 or \
            any(mark in sentence for mark in ('?', '!', '...')):
        return None, 'fragment_or_nonstatement'
    player = _unique_player(sentence, row)
    if player is None:
        return None, 'player_count'
    key, name = player
    text = _detokenize(sentence)
    if re.search(r'\b(?:and|while|with|to|for|as|also)\.$', text,
                 flags=re.I):
        return None, 'dangling_clause'
    if not re.match(r'^\W*'+re.escape(name)+r'\b', text):
        return None, 'not_player_led'
    if UNSUPPORTED.search(text):
        return None, 'unsupported_context_word'
    remainder = re.sub(r'(?<!\w)'+re.escape(name)+r'(?!\w)', '', text)
    words = [word.lower() for word in re.findall(r'[A-Za-z]+', remainder)]
    if any(word not in SAFE_WORDS for word in words):
        return None, 'unlicensed_word'
    values = {metric: row['box_score'][column].get(key)
              for metric, column in (('points', 'PTS'),
                                     ('rebounds', 'REB'),
                                     ('assists', 'AST'))}
    matches = list(STAT.finditer(text))
    if not matches:
        return None, 'no_stat'
    if len(matches) != len(NUMBER.findall(text)):
        return None, 'unlicensed_number'
    found = set()
    for match in matches:
        number = int(match[1])
        metric = match[2].lower().rstrip('s')
        if metric == 'point':
            metric = 'points'
        elif metric == 'rebound':
            metric = 'rebounds'
        elif metric == 'assist':
            metric = 'assists'
        if metric in found or values[metric] is None or \
                not str(values[metric]).isdigit() or \
                number != int(values[metric]):
            return None, 'fact_mismatch'
        found.add(metric)
    # Exclude other roster names shortened to surnames only, a common source
    # of multi-person claims that the full-name screen would miss.
    surnames = {other.split()[-1] for other in
                row['box_score']['PLAYER_NAME'].values()
                if other != name and len(other.split()) >= 2}
    remaining = re.sub(r'(?<!\w)'+re.escape(name)+r'(?!\w)', '', text)
    if any(re.search(r'\b'+re.escape(surname)+r'\b', remaining)
           for surname in surnames):
        return None, 'other_player_surname'
    form = re.sub(r'(?<!\w)'+re.escape(name)+r'(?!\w)', '<PLAYER>', text)
    form = STAT.sub(lambda m: '<'+
                    {'point': 'PTS', 'points': 'PTS',
                     'rebound': 'REB', 'rebounds': 'REB',
                     'assist': 'AST', 'assists': 'AST'}
                    [m[2].lower()]+'> '+m[2].lower(), form)
    return {'name': name, 'metrics': tuple(sorted(found)),
            'form': form, 'tokens': len(sentence),
            'text': text}, None


def audit(rows, group, sample_seed):
    counts = Counter()
    form_counts = Counter()
    examples = defaultdict(list)
    seen_game_forms = set()
    for row in rows:
        if game_partition(game_key(row)) != group:
            continue
        counts['games'] += 1
        accepted_in_game = 0
        for sentence in _sentences(row['summary']):
            counts['sentences'] += 1
            result, reason = accepted(sentence, row)
            if reason:
                counts['reject/'+reason] += 1
                continue
            counts['accepted_occurrences'] += 1
            accepted_in_game += 1
            form_counts[result['form']] += 1
            signature = (game_key(row), result['form'])
            if signature not in seen_game_forms:
                seen_game_forms.add(signature)
                counts['distinct_game_form_occurrences'] += 1
            key = '+'.join(result['metrics'])
            counts['metric_set/'+key] += 1
            # Sample for manual QA only; never save verbatim source prose.
            if len(examples[key]) < 50:
                examples[key].append(result['text'])
        counts['games_with_accepted'] += accepted_in_game > 0
        counts['games_with_2plus_accepted'] += accepted_in_game >= 2
    rng = random.Random(sample_seed)
    selected = []
    for key, texts in sorted(examples.items()):
        selected.extend((key, text) for text in rng.sample(
            texts, min(3, len(texts))))
    return {'counts': dict(counts),
            'distinct_forms': len(form_counts),
            'forms_seen_2plus_games': sum(n >= 2 for n in form_counts.values()),
            'forms_seen_10plus_games': sum(n >= 10 for n in form_counts.values()),
            'top_form_hashes': [
                {'sha256': hashlib.sha256(form.encode()).hexdigest(),
                 'count': count}
                for form, count in form_counts.most_common(10)],
            'manual_examples': selected}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=96201)
    a = p.parse_args()
    result = {'config': {'seed': a.seed}, 'arms': {},
              'caveat': 'This is a conservative surface/fact screen, not a '
                        'semantic certificate. No source text is saved in '
                        'the committed JSON; manual examples are printed '
                        'only and removed before writing.'}
    for source in ('rotowire', 'sbnation'):
        rows = load(ARCHIVE_DIR/f'{source}.tar.bz2', source)['train']
        for group in ('train', 'heldout'):
            key = source+'_'+group
            record = audit(rows, group, a.seed)
            examples = record.pop('manual_examples')
            result['arms'][key] = record
            print(key, record['counts'], 'forms',
                  record['distinct_forms'], flush=True)
            print('manual examples', examples, flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
