"""Paired fixed-versus-interleaved semantic and form audit."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

from audit_document_structure_vs_fineweb import structure
from audit_rotowire_document_source import game_key
from compare_v12_fineweb_fresh import first_words
from sports_document_dgp import game_partition, resources
from sports_report_multirule_dgp import (
    increment_player, increment_score, realize as fixed_realize,
    sample_plan, visible_site_answer,
)
from sports_report_multirule_interleaved import (
    placement, realize as spread_realize, realize_mixed,
)


def window(doc, site):
    return ' '.join(doc.text[:site.start].split()[-25:]+
                    doc.text[site.start:].split()[:16])


def sentence_multiset(text):
    return sorted(part.strip() for part in re.split(r'(?<=[.!?])\s+', text)
                  if part.strip())


def check_triplet(plan, renderer):
    original, score, player = (renderer(variant) for variant in
                               (plan, increment_score(plan),
                                increment_player(plan)))
    documents = (original, score, player)
    for doc in documents:
        for site in doc.sites:
            assert doc.text[site.start:site.end] == site.answer
            assert visible_site_answer(doc, site) == site.answer
    answers = [{site.kind: site.answer for site in doc.sites}
               for doc in documents]
    assert int(answers[1]['winning_score']) == \
        int(answers[0]['winning_score'])+1
    assert int(answers[1]['margin']) == int(answers[0]['margin'])+1
    assert answers[1]['player_points'] == answers[0]['player_points']
    assert answers[2]['winning_score'] == answers[0]['winning_score']
    assert answers[2]['margin'] == answers[0]['margin']
    assert int(answers[2]['player_points']) == \
        int(answers[0]['player_points'])+1
    return original


def audit(group, seed, n, form_n):
    rng = random.Random(seed)
    rows = {game_key(row): row for row in resources()[0]['rotowire'][0]}
    counts = Counter()
    chosen = set()
    game_hashes = []
    forms = {placement: {'first100': [], 'full': [],
                         'windows': {kind: [] for kind in
                                     ('winning_score', 'margin', 'player_points')},
                         'position_fraction': {kind: [] for kind in
                                               ('winning_score', 'margin',
                                                'player_points')}}
             for placement in ('fixed', 'spread')}
    natural = []
    natural_full = []
    natural_windows = {name: {kind: [] for kind in
                              ('winning_score', 'margin', 'player_points')}
                       for name in ('fixed', 'spread')}
    examples = []
    for _ in range(n):
        plan = sample_plan(rng, group)
        key = plan.carrier_plan.source_game_key
        assert game_partition(key) == group
        fixed = check_triplet(plan, fixed_realize)
        spread = check_triplet(plan, spread_realize)
        mixed = check_triplet(plan, realize_mixed)
        assert mixed.text in (fixed.text, spread.text)
        counts['mixed_spread' if mixed.text == spread.text else
               'mixed_fixed'] += 1
        counts['visible_checks'] += 27
        counts['selective_triplets'] += 3
        assert sentence_multiset(fixed.text) == sentence_multiset(spread.text)
        counts['sentence_multiset_equal'] += 1
        counts['text_order_changed'] += fixed.text != spread.text
        assert placement(plan) == placement(increment_score(plan)) == \
            placement(increment_player(plan))
        if key in chosen or len(chosen) >= form_n:
            continue
        chosen.add(key)
        game_hashes.append(hashlib.sha256(repr(key).encode()).hexdigest())
        source = ' '.join(rows[key]['summary'])
        natural.append(first_words(source, 100))
        natural_full.append(first_words(source, len(fixed.text.split())))
        for name, doc in (('fixed', fixed), ('spread', spread)):
            forms[name]['first100'].append(first_words(doc.text, 100))
            forms[name]['full'].append(doc.text)
            for site in doc.sites:
                forms[name]['windows'][site.kind].append(window(doc, site))
                fraction = len(doc.text[:site.start].split())/len(doc.text.split())
                forms[name]['position_fraction'][site.kind].append(fraction)
                source_words = source.split()
                index = max(25, min(len(source_words)-16,
                                    round(fraction*len(source_words))))
                natural_windows[name][site.kind].append(
                    ' '.join(source_words[index-25:index+16]))
        if len(examples) < 2:
            examples.append((fixed.text, spread.text))
    if len(chosen) != form_n:
        raise RuntimeError('Insufficient unique games for paired form panel')
    output = {'counts': dict(counts), 'form_game_sha256': game_hashes,
              'paired_natural_first100': structure(natural),
              'paired_natural_length_matched': structure(natural_full),
              'position_matched_natural_windows': {
                  name: {kind: structure(texts) for kind, texts in rows.items()}
                  for name, rows in natural_windows.items()},
              'arms': {}, 'manual_examples': examples}
    for name, rows in forms.items():
        output['arms'][name] = {
            'first100': structure(rows['first100']),
            'full': structure(rows['full']),
            'windows': {kind: structure(texts)
                        for kind, texts in rows['windows'].items()},
            'position_fraction': {kind: {
                'min': min(values), 'mean': sum(values)/len(values),
                'max': max(values)}
                for kind, values in rows['position_fraction'].items()},
        }
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=97101)
    p.add_argument('--n', type=int, default=300)
    p.add_argument('--form-n', type=int, default=64)
    a = p.parse_args()
    result = {'config': {**vars(a), 'output': str(a.output)},
              'groups': {},
              'caveat': 'Same worlds and sentence multiset; only placement '
                        'changes. Natural carrier has no full symbolic oracle. '
                        'Uncommitted manual examples contain source prose.'}
    for group in ('train', 'heldout'):
        row = audit(group, a.seed+(group == 'heldout')*100, a.n, a.form_n)
        examples = row.pop('manual_examples')
        result['groups'][group] = row
        print(group, row['counts'], 'examples', examples, flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
