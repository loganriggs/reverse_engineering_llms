"""Aggregate sentence-act and transition screen for full sports reports.

These are transparent lexical proxies, not certified discourse annotations.
The output contains counts only; source prose remains outside Git.
"""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import statistics

from audit_rotowire_document_source import DEFAULT_ARCHIVE, game_key, load
from corpus_score_sentence_renderer import _sentences
from sports_document_dgp import game_partition


FLOW = frozenset(('quarter', 'quarters', 'half', 'halftime', 'period',
                  'run', 'runs', 'lead', 'led', 'trailing', 'trailed',
                  'overtime', 'fourth', 'third', 'second', 'first'))
OUTLOOK = frozenset(('next', 'upcoming', 'will', 'face', 'faces', 'host',
                     'hosts', 'visit', 'visits', 'travel', 'travels'))
CONTEXT = frozenset(('season', 'record', 'streak', 'straight', 'previous',
                     'career', 'injury', 'injured', 'starter', 'bench'))
NUMERIC = frozenset(('points', 'point', 'rebounds', 'rebound', 'assists',
                     'assist', 'pts'))


def contains_phrase(tokens, phrase):
    return any(tokens[i:i+len(phrase)] == phrase
               for i in range(len(tokens)-len(phrase)+1))


def score_pair(sentence, row):
    a, b = row['home_line']['TEAM-PTS'], row['vis_line']['TEAM-PTS']
    return any(middle in ('-', 'to', '–') and
               (left, right) in ((a, b), (b, a))
               for left, middle, right in
               zip(sentence, sentence[1:], sentence[2:]))


def player_stat(sentence, row):
    table = row['box_score']
    lower = frozenset(token.lower() for token in sentence)
    if not lower & NUMERIC:
        return False
    for name in table['PLAYER_NAME'].values():
        if len(name.split()) >= 2 and contains_phrase(sentence, name.split()):
            return True
    return False


def labels(sentence, row):
    lower = frozenset(token.lower() for token in sentence)
    found = []
    if score_pair(sentence, row):
        found.append('result')
    if player_stat(sentence, row):
        found.append('player_stat')
    if lower & FLOW:
        found.append('game_flow')
    if lower & OUTLOOK:
        found.append('outlook')
    if lower & CONTEXT:
        found.append('background')
    return tuple(found) or ('other',)


def quarter_facts(row):
    try:
        home = [int(row['home_line'][f'TEAM-PTS_QTR{i}'])
                for i in range(1, 5)]
        away = [int(row['vis_line'][f'TEAM-PTS_QTR{i}'])
                for i in range(1, 5)]
        total_home = int(row['home_line']['TEAM-PTS'])
        total_away = int(row['vis_line']['TEAM-PTS'])
    except (KeyError, TypeError, ValueError):
        return None
    if min(home+away) < 0 or sum(home) != total_home or \
            sum(away) != total_away:
        return None
    margins = []
    h, a = 0, 0
    for hq, aq in zip(home, away):
        h, a = h+hq, a+aq
        margins.append(h-a)
    signs = [1 if value > 0 else -1 if value < 0 else 0
             for value in margins]
    return {'halftime_reversal': signs[1] * signs[3] < 0,
            'any_period_leader_reversal': any(
                a*b < 0 for a, b in zip(signs, signs[1:])),
            'halftime_tie': signs[1] == 0,
            'final_tie': signs[3] == 0}


def summarize(rows, max_sentences):
    counts = Counter()
    first = Counter()
    transition = Counter()
    position = [Counter() for _ in range(max_sentences)]
    lengths = []
    unique_keys = set()
    for row in rows:
        key = game_key(row)
        unique_keys.add(key)
        sentences = list(_sentences(row['summary']))
        quarter = quarter_facts(row)
        counts['quarter_facts_consistent'] += quarter is not None
        if quarter is not None:
            for key, value in quarter.items():
                counts['quarter/'+key] += value
        lengths.append(len(sentences))
        acts = []
        for index, sentence in enumerate(sentences[:max_sentences]):
            tags = labels(sentence, row)
            counts['sentences'] += 1
            counts['multi_label_sentences'] += len(tags) > 1
            for tag in tags:
                counts['label/'+tag] += 1
                position[index][tag] += 1
            # A fixed primary order is needed only for a transition proxy.
            primary = tags[0]
            acts.append(primary)
            if index == 0:
                first[primary] += 1
        for a, b in zip(acts, acts[1:]):
            transition[a+'→'+b] += 1
        counts['documents'] += 1
        counts['documents_with_result'] += 'result' in acts
        counts['documents_with_player_stat'] += 'player_stat' in acts
        counts['documents_with_flow'] += 'game_flow' in acts
        counts['documents_with_outlook'] += 'outlook' in acts
    return {'counts': dict(counts), 'unique_game_keys': len(unique_keys),
            'sentences_per_document': {
                'mean': statistics.mean(lengths),
                'median': statistics.median(lengths)},
            'first_primary_act': dict(first),
            'primary_transitions': dict(transition),
            'position_label_counts': [dict(x) for x in position]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rotowire-archive', type=Path, default=DEFAULT_ARCHIVE)
    p.add_argument('--sbnation-archive', type=Path, default=DEFAULT_ARCHIVE.with_name('sbnation.tar.bz2'))
    p.add_argument('--max-sentences', type=int, default=12)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.max_sentences < 2:
        raise ValueError('Need at least two sentences for transitions')
    output = {'config': {'max_sentences': a.max_sentences,
                         'sources': ('rotowire', 'sbnation'),
                         'grouping': 'stable game-key hash'},
              'archive_sha256': {},
              'arms': {},
              'caveat': 'Lexical multi-label proxies are not semantic truth. '
                        'Score-pair and player-name matches can be coincidental; '
                        'game-flow/outlook/background tags can be false positives. '
                        'Use these to design and audit a sampler, not as a rule oracle.'}
    for name, archive in (('rotowire', a.rotowire_archive),
                          ('sbnation', a.sbnation_archive)):
        output['archive_sha256'][name] = sha256(archive.read_bytes()).hexdigest()
        rows = load(archive, name)['train']
        for group in ('train', 'heldout'):
            selected = [row for row in rows
                        if game_partition(game_key(row)) == group]
            output['arms'][name+'_'+group] = summarize(selected, a.max_sentences)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(output, indent=2)+'\n')
    for name, arm in output['arms'].items():
        print(name, arm['counts'], 'first', arm['first_primary_act'], flush=True)


if __name__ == '__main__':
    main()
