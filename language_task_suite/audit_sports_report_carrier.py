"""Semantic/edit and natural-form screen for the empirical sports carrier."""

import argparse
import hashlib
import json
import random
from datetime import datetime
from collections import Counter
from pathlib import Path

from audit_document_structure_vs_fineweb import structure
from compare_v12_fineweb_fresh import first_words
from sports_document_dgp import game_partition, resources
from sports_report_carrier_dgp import (
    eligible, increment_winning_score, realize, sample_plan, visible_answer,
)
from audit_rotowire_document_source import game_key


def _window(doc):
    site = doc.site
    return ' '.join(doc.text[:site.start].split()[-25:]+
                    doc.text[site.start:].split()[:16])


def audit_one(source, group, seed, n, form_n, words,
              carrier_words=100, query_kind='margin'):
    available = eligible(source, group, carrier_words)
    rng = random.Random(seed)
    counts = Counter()
    texts = []
    paired_originals = []
    windows = []
    seen_games = set()
    hashes = []
    originals_by_game = {game_key(row): row['summary']
                         for row in resources()[0][source][0]}
    for _ in range(n):
        plan = sample_plan(rng, source, group, carrier_words,
                           query_kind=query_kind)
        assert game_partition(plan.source_game_key) == group
        doc = realize(plan)
        weekday = datetime.strptime(plan.source_game_key[0],
                                    '%m_%d_%y').strftime('%A')
        assert weekday in doc.text.split('.')[0]
        assert doc.text[doc.site.start:doc.site.end] == doc.site.answer
        assert visible_answer(doc.text[:doc.site.start], query_kind) == \
            doc.site.answer
        changed = realize(increment_winning_score(plan))
        assert changed.site.answer == str(int(doc.site.answer)+1)
        assert visible_answer(changed.text[:changed.site.start],
                              query_kind) == \
            changed.site.answer
        assert changed.plan.carrier == plan.carrier
        counts['validated_original_and_score_edit'] += 1
        if plan.source_game_key not in seen_games and \
                len(first_words(doc.text, words).split()) >= words and \
                len(texts) < form_n:
            seen_games.add(plan.source_game_key)
            texts.append(first_words(doc.text, words))
            paired_originals.append(first_words(
                ' '.join(originals_by_game[plan.source_game_key]), words))
            windows.append(_window(doc))
            hashes.append(hashlib.sha256(
                repr(plan.source_game_key).encode()).hexdigest())
    if len(texts) < form_n:
        raise RuntimeError('Too few unique carrier games for form panel')
    return {'eligible_rows': len(available),
            'checks': dict(counts), 'form': structure(texts),
            'paired_original_form': structure(paired_originals),
            'rule_window': structure(windows),
            'form_game_sha256': hashes}


def natural_reference(source, group, seed, n, words):
    rows = resources()[0][source][0]
    by_game = {}
    for row in rows:
        key = game_key(row)
        if game_partition(key) == group and \
                len(row['summary']) >= words:
            by_game.setdefault(key, row)
    chosen = random.Random(seed).sample(list(by_game.values()), n)
    return structure([first_words(' '.join(row['summary']), words)
                      for row in chosen])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=93301)
    p.add_argument('--checks', type=int, default=200)
    p.add_argument('--form-n', type=int, default=64)
    p.add_argument('--words', type=int, default=55)
    p.add_argument('--sources', default='rotowire')
    p.add_argument('--carrier-words', type=int, default=100)
    p.add_argument('--query-kind', choices=('margin', 'winning_score'),
                   default='margin')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = {'config': {'seed': a.seed, 'checks': a.checks,
                         'form_n': a.form_n, 'words': a.words,
                         'sources': a.sources,
                         'carrier_words': a.carrier_words,
                         'query_kind': a.query_kind},
              'arms': {}, 'natural': {},
              'coverage': {source: {group: len(eligible(
                  source, group, a.carrier_words))
                                   for group in ('train', 'heldout')}
                           for source in ('rotowire', 'sbnation')},
              'caveat': 'Only the inserted score and final margin are '
                        'explicitly generated. Numeric screening of the '
                        'empirical carrier does not prove that a score '
                        'edit preserves every claim in human prose.'}
    for source in a.sources.split(','):
        if source not in ('rotowire', 'sbnation'):
            raise ValueError(source)
        for group in ('train', 'heldout'):
            key = f'{source}_{group}'
            seed = a.seed+len(result['arms'])*100
            result['arms'][key] = audit_one(source, group, seed,
                                            a.checks, a.form_n, a.words,
                                            a.carrier_words, a.query_kind)
            result['natural'][key] = natural_reference(
                source, group, seed+1, a.form_n, a.words)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    for key, row in result['arms'].items():
        print(key, row['eligible_rows'], row['form']['word_types'],
              row['form']['repeated_across_docs_trigram_occurrence_share'],
              row['rule_window']['repeated_across_docs_trigram_occurrence_share'])


if __name__ == '__main__':
    main()
